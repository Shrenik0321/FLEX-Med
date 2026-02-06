import torch
import numpy as np
import os
import json
import time
import uuid
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from typing import Tuple, Optional, Iterable, Dict, List

from flwr.common import Message, Metadata, RecordDict, ArrayRecord, ConfigRecord
from flwr.server import Grid
from flwr.serverapp.strategy import Strategy
from flwr_datasets.partitioner import DirichletPartitioner
from datasets import Dataset

from flex_med.utils.config import (
    BASE_PATH, CLIENT_INFO_FILE_PATH, DATASET_FILE_PATH,
    PUBLIC_ANCHOR_DATASET_PATH, PUBLIC_TEST_DATASET_PATH, LOCAL_TRAIN_DATASET_PATH,
    MODEL_CHECKPOINT_FILE_PATH, GRAPHS_OUTPUT_DIR, NUM_CLASSES, IMG_SIZE,
    TRAIN_LOSS_WEIGHT, DISTILL_LOSS_WEIGHT, CONSENSUS_MOMENTUM,
    DIRICHLET_ALPHA, DIRICHLET_SEED, DIRICHLET_MIN_PARTITION_SIZE,
    FOCAL_ALPHA_PER_ARCH, FOCAL_ALPHA_DEFAULT, FOCAL_GAMMA,
)

from flex_med.utils.helpers import (
    sanitize_client_paths, get_weighted_sampler,
    get_partition_stats, print_client_data_distribution_summary,
    extract_training_metrics_for_persistence, save_global_post_fl_metrics,
    save_round_training_metrics, save_round_validation_metrics,
    check_for_degradation_warnings, SUPABASE_CLIENT, SIMULATION_ID,
    apply_freeze_strategy,
)

# <----------------------------- CONSTANTS & GLOBAL VARIABLES ----------------------------->

# Partitioner cache for Dirichlet data distribution
_PARTITIONER_CACHE = {}

# <----------------------------- DATA TRANSFORMS ----------------------------->

COMMON_TRANSFORM = Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

PRIVATE_TRAIN_TRANSFORM = Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# <----------------------------- CLIENT CONFIGURATION ----------------------------->

def load_client_config(config_path: str = CLIENT_INFO_FILE_PATH) -> List[Dict]:
    """Load client configuration from database (preferred) or file (fallback)."""
    if SUPABASE_CLIENT is not None and SIMULATION_ID is not None:
        try:
            response = SUPABASE_CLIENT.from_('client_simulation_metrics') \
                .select('client_id, clients(*)') \
                .eq('simulation_id', SIMULATION_ID) \
                .execute()
            if response.data:
                clients = [r.get('clients') for r in response.data if r.get('clients')]
                if clients:
                    return sanitize_client_paths(clients)
        except Exception:
            pass

    env_config_path = os.getenv('FLEX_MED_CONFIG_FILE')
    if env_config_path:
        config_path = env_config_path

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, 'r') as f:
        config_data = json.load(f)

    clients = config_data.get("clients", config_data) if isinstance(config_data, dict) else config_data
    return sanitize_client_paths(clients)


def get_client_by_partition_id(partition_id: int, config_path: str = CLIENT_INFO_FILE_PATH) -> Dict:
    """Retrieve individual client config by partition ID."""
    clients = load_client_config(config_path)
    if partition_id >= len(clients):
        raise ValueError(f"Partition ID {partition_id} exceeds number of clients ({len(clients)})")
    return clients[partition_id]


# <----------------------------- MODEL CREATION & MANAGEMENT ----------------------------->

def get_model_by_type(model_type: str, use_pretrained: bool = True, dropout_rate: float = None):
    """Create a model instance by type with dropout-enhanced classifier."""
    model_type = model_type.lower()
    if dropout_rate is None:
        dropout_rate = get_initial_dropout_rate(model_type)

    model_map = {
        'resnet50': models.resnet50, 'resnet18': models.resnet18,
        'mobilenet_v2': models.mobilenet_v2, 'densenet121': models.densenet121,
        'efficientnet_b0': models.efficientnet_b0,
    }

    if model_type not in model_map:
        raise ValueError(f"Unsupported model type: {model_type}. Supported: {list(model_map.keys())}")

    if use_pretrained:
        weights = "DEFAULT"
    else:
        weights = None

    model = model_map[model_type](weights=weights)
    return add_dropout_to_classifier(model, model_type, dropout_rate)


def get_initial_dropout_rate(model_type: str) -> float:
    """Get default dropout rate for model architecture."""
    rates = {'resnet50': 0.35, 'mobilenet_v2': 0.45, 'densenet121': 0.35, 'efficientnet_b0': 0.40, 'resnet18': 0.35}
    return rates.get(model_type.lower(), 0.3)


def add_dropout_to_classifier(model, model_type: str, dropout_rate: float = 0.3):
    """Add/update dropout layer before the final classifier head (idempotent)."""
    model_type = model_type.lower()

    if model_type in ('resnet50', 'resnet18'):
        if isinstance(model.fc, nn.Sequential) and isinstance(model.fc[0], nn.Dropout):
            model.fc[0].p = dropout_rate
        else:
            in_features = model.fc.in_features
            model.fc = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(in_features, NUM_CLASSES))

    elif model_type == 'mobilenet_v2':
        if isinstance(model.classifier[1], nn.Sequential) and isinstance(model.classifier[1][0], nn.Dropout):
            model.classifier[1][0].p = dropout_rate
        else:
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(in_features, NUM_CLASSES))

    elif model_type == 'densenet121':
        if isinstance(model.classifier, nn.Sequential) and isinstance(model.classifier[0], nn.Dropout):
            model.classifier[0].p = dropout_rate
        else:
            in_features = model.classifier.in_features
            model.classifier = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(in_features, NUM_CLASSES))

    elif model_type == 'efficientnet_b0':
        if isinstance(model.classifier[1], nn.Sequential) and isinstance(model.classifier[1][0], nn.Dropout):
            model.classifier[1][0].p = dropout_rate
        else:
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(in_features, NUM_CLASSES))

    return model


def update_dropout_rate(model, model_type: str, new_dropout_rate: float):
    """Update dropout rate in existing model architecture."""
    model_type = model_type.lower()
    try:
        if model_type in ('resnet50', 'resnet18'):
            if isinstance(model.fc, nn.Sequential) and isinstance(model.fc[0], nn.Dropout):
                model.fc[0].p = new_dropout_rate
        elif model_type in ('mobilenet_v2', 'efficientnet_b0'):
            if isinstance(model.classifier[1], nn.Sequential) and isinstance(model.classifier[1][0], nn.Dropout):
                model.classifier[1][0].p = new_dropout_rate
        elif model_type == 'densenet121':
            if isinstance(model.classifier, nn.Sequential) and isinstance(model.classifier[0], nn.Dropout):
                model.classifier[0].p = new_dropout_rate
    except Exception:
        pass
    return model


def load_existing_model(model, model_path, device):
    """Load model from checkpoint (handles legacy and new formats)."""
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        metadata = {
            'model_type': checkpoint.get('model_type'), 'num_classes': checkpoint.get('num_classes'),
            'round': checkpoint.get('round'), 'adaptive_state': checkpoint.get('adaptive_state'),
            'scheduler_state': checkpoint.get('scheduler_state')
        }
        return model, metadata
    else:
        model.load_state_dict(checkpoint)
        return model, {}


# <----------------------------- LOAD DATA ----------------------------->

def create_dirichlet_partitioner(
    dataset_path: str, num_partitions: int, alpha: float = DIRICHLET_ALPHA,
    seed: int = DIRICHLET_SEED, min_partition_size: int = DIRICHLET_MIN_PARTITION_SIZE
) -> Tuple[DirichletPartitioner, datasets.ImageFolder]:
    """Create a Dirichlet partitioner for heterogeneous data distribution."""
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=None)
    image_paths = [path for path, _ in full_dataset.samples]
    labels = [label for _, label in full_dataset.samples]

    hf_dataset = Dataset.from_dict({"image_path": image_paths, "label": labels})
    partitioner = DirichletPartitioner(
        num_partitions=num_partitions, partition_by="label", alpha=alpha,
        min_partition_size=min_partition_size, self_balancing=False, shuffle=True, seed=seed
    )
    partitioner.dataset = hf_dataset
    return partitioner, full_dataset


def load_public_dataset(batch_size=64, round_num=1, total_rounds=10):
    """Load public anchor dataset for consensus generation."""
    if not os.path.exists(PUBLIC_ANCHOR_DATASET_PATH):
        raise FileNotFoundError(f"Public data not found at {PUBLIC_ANCHOR_DATASET_PATH}")

    full_dataset = datasets.ImageFolder(root=PUBLIC_ANCHOR_DATASET_PATH, transform=COMMON_TRANSFORM)
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(full_dataset), generator=generator).tolist()
    subset = torch.utils.data.Subset(full_dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)


def load_private_dataset(partition_id: int, num_partitions: int, batch_size=64,
                         config_path: str = CLIENT_INFO_FILE_PATH):
    """Load private dataset for a client using Dirichlet partitioning."""
    if not os.path.exists(LOCAL_TRAIN_DATASET_PATH):
        raise FileNotFoundError(f"Shared training dataset not found at {LOCAL_TRAIN_DATASET_PATH}")

    cache_key = (LOCAL_TRAIN_DATASET_PATH, num_partitions, DIRICHLET_ALPHA, DIRICHLET_SEED)

    if cache_key not in _PARTITIONER_CACHE:
        partitioner, full_dataset = create_dirichlet_partitioner(
            LOCAL_TRAIN_DATASET_PATH, num_partitions, DIRICHLET_ALPHA, DIRICHLET_SEED
        )
        _PARTITIONER_CACHE[cache_key] = (partitioner, full_dataset)
    else:
        partitioner, full_dataset = _PARTITIONER_CACHE[cache_key]

    partition_dataset = partitioner.load_partition(partition_id)
    partition_paths = partition_dataset["image_path"]

    path_to_idx = {path: idx for idx, (path, _) in enumerate(full_dataset.samples)}
    client_indices = [path_to_idx[path] for path in partition_paths]

    client_dataset = torch.utils.data.Subset(
        datasets.ImageFolder(root=LOCAL_TRAIN_DATASET_PATH, transform=PRIVATE_TRAIN_TRANSFORM),
        client_indices
    )

    # Stratified train/validation split (85/15)
    class_indices = {0: [], 1: []}
    for subset_idx, global_idx in enumerate(client_indices):
        class_indices[full_dataset.targets[global_idx]].append(subset_idx)

    train_indices, val_indices = [], []
    generator = torch.Generator().manual_seed(42)

    for indices in class_indices.values():
        if not indices:
            continue
        perm = torch.randperm(len(indices), generator=generator)
        shuffled = torch.tensor(indices)[perm].tolist()
        split = int(0.85 * len(shuffled))
        train_indices.extend(shuffled[:split])
        val_indices.extend(shuffled[split:])

    train_ds = torch.utils.data.Subset(client_dataset, train_indices)
    test_ds = torch.utils.data.Subset(client_dataset, val_indices)

    train_targets = [full_dataset.targets[client_indices[i]] for i in train_indices]
    train_sampler = get_weighted_sampler(train_targets)

    trainloader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, shuffle=False, num_workers=2)
    testloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader


def load_public_test_dataset(batch_size=64):
    """Load public test dataset for evaluation."""
    if not os.path.exists(PUBLIC_TEST_DATASET_PATH):
        raise FileNotFoundError(f"Public test data not found at {PUBLIC_TEST_DATASET_PATH}")
    dataset = datasets.ImageFolder(root=PUBLIC_TEST_DATASET_PATH, transform=COMMON_TRANSFORM)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)


def get_public_logits(model, public_loader, device):
    """Generate logits on the public dataset for consensus computation."""
    if public_loader is None:
        raise ValueError("public_loader is None.")
    model.to(device)
    model.eval()
    all_logits = []
    with torch.no_grad():
        for images, _ in public_loader:
            images = images.to(device)
            outputs = model(images)
            all_logits.append(outputs.cpu().numpy())
    return np.concatenate(all_logits)


# <----------------------------- CHECKPOINT MANAGEMENT ----------------------------->

def save_checkpoint(checkpoint_dir: str, current_round: int, consensus_logits: np.ndarray,
                    eval_history: List[Dict], training_metrics: Dict):
    """Save FL training checkpoint with consensus logits and metrics."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_round_{current_round}.pt")
    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")

    checkpoint = {
        'round': current_round, 'consensus_logits': consensus_logits,
        'eval_history': eval_history, 'training_metrics': training_metrics,
        'timestamp': time.time()
    }
    torch.save(checkpoint, checkpoint_path)
    torch.save(checkpoint, latest_path)


def load_checkpoint(checkpoint_dir: str) -> Optional[Dict]:
    """Load the latest FL training checkpoint for resumption."""
    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    if not os.path.exists(latest_path):
        return None
    try:
        return torch.load(latest_path, map_location='cpu', weights_only=False)
    except Exception:
        return None


# <----------------------------- ADAPTIVE TRAINING STATE ----------------------------->

class AdaptiveTrainingState:
    """Manages per-client adaptive dropout based on validation loss trends."""

    def __init__(self, client_id: int, model_type: str, initial_dropout: float = 0.3,
                 max_dropout: float = 0.6, min_dropout: float = 0.2,
                 dropout_increment: float = 0.1, patience_rounds: int = 2):
        self.client_id = client_id
        self.model_type = model_type
        self.dropout_rate = initial_dropout
        self.max_dropout = max_dropout
        self.min_dropout = min_dropout
        self.dropout_increment = dropout_increment
        self.patience_rounds = patience_rounds
        self.val_loss_history = []
        self.scheduler_state = None

    def should_increase_dropout(self) -> bool:
        if len(self.val_loss_history) < 3:
            return False
        recent = self.val_loss_history[-3:]
        return recent[1] > recent[0] and recent[2] > recent[1]

    def should_decrease_dropout(self) -> bool:
        if len(self.val_loss_history) < 3:
            return False
        recent = self.val_loss_history[-3:]
        return recent[1] < recent[0] and recent[2] < recent[1]

    def update_dropout(self, model) -> float:
        old_dropout = self.dropout_rate
        if self.should_increase_dropout() and self.dropout_rate < self.max_dropout:
            self.dropout_rate = min(self.max_dropout, self.dropout_rate + self.dropout_increment)
        elif self.should_decrease_dropout() and self.dropout_rate > self.min_dropout:
            self.dropout_rate = max(self.min_dropout, self.dropout_rate - self.dropout_increment)

        if self.dropout_rate != old_dropout:
            update_dropout_rate(model, self.model_type, self.dropout_rate)
        return self.dropout_rate

    def add_val_loss(self, val_loss: float):
        self.val_loss_history.append(val_loss)
        if len(self.val_loss_history) > 5:
            self.val_loss_history.pop(0)

    def to_dict(self) -> dict:
        return {
            'client_id': self.client_id, 'model_type': self.model_type,
            'dropout_rate': self.dropout_rate, 'max_dropout': self.max_dropout,
            'min_dropout': self.min_dropout, 'dropout_increment': self.dropout_increment,
            'patience_rounds': self.patience_rounds, 'val_loss_history': self.val_loss_history,
            'scheduler_state': self.scheduler_state
        }

    @classmethod
    def from_dict(cls, state_dict: dict) -> 'AdaptiveTrainingState':
        state = cls(
            client_id=state_dict['client_id'], model_type=state_dict['model_type'],
            initial_dropout=state_dict['dropout_rate'],
            max_dropout=state_dict.get('max_dropout', 0.6),
            min_dropout=state_dict.get('min_dropout', 0.2),
            dropout_increment=state_dict.get('dropout_increment', 0.1),
            patience_rounds=state_dict.get('patience_rounds', 2)
        )
        state.val_loss_history = state_dict.get('val_loss_history', [])
        state.scheduler_state = state_dict.get('scheduler_state')
        return state


# <----------------------------- LOSS FUNCTIONS ----------------------------->

class FocalLoss(nn.Module):
    """Focal Loss: FL(pt) = -alpha * (1 - pt)^gamma * log(pt). Focuses on hard examples."""

    def __init__(self, alpha: float = 0.35, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma
        alpha_t = torch.where(targets == 0, self.alpha, 1 - self.alpha)
        return (alpha_t * focal_term * ce_loss).mean()


# <----------------------------- TRAINING & EVALUATION ----------------------------->

def _validate_epoch(model, valloader, criterion, device):
    """Perform validation for a single epoch, returns average loss."""
    model.eval()
    val_loss, val_batches = 0.0, 0
    with torch.no_grad():
        for images, labels in valloader:
            images, labels = images.to(device), labels.to(device)
            val_loss += criterion(model(images), labels).item()
            val_batches += 1
    model.train()
    return val_loss / val_batches if val_batches > 0 else 0.0


def train(model, trainloader, epochs, lr, device, model_type: str = None,
          valloader=None, adaptive_state=None, server_round: int = 1, total_rounds: int = 10):
    """Train model with Focal Loss and gradual unfreeze strategy.

    Args:
        model: PyTorch model to train
        trainloader: Training data loader
        epochs: Number of local epochs
        lr: Learning rate
        device: Device to train on
        model_type: Model architecture name (for freeze strategy)
        valloader: Optional validation data loader
        adaptive_state: Optional adaptive training state
        server_round: Current FL round (1-indexed, for freeze strategy)
        total_rounds: Total number of FL rounds (for freeze strategy)

    Returns:
        (avg_train_loss, avg_val_loss, train_accuracy, scheduler_state)
    """
    if trainloader is None:
        return 0.0, 0.0, 0.0, None

    model.to(device)

    # Apply freeze/unfreeze strategy based on current round
    lr_multiplier = 1.0
    if model_type is not None:
        model, lr_multiplier = apply_freeze_strategy(model, model_type, server_round, total_rounds)

    # Apply LR multiplier (reduced in Phase 3 for "polish" fine-tuning)
    effective_lr = lr * lr_multiplier

    # Get per-architecture focal alpha (configurable in app/config.py)
    focal_alpha = FOCAL_ALPHA_DEFAULT
    if model_type is not None:
        focal_alpha = FOCAL_ALPHA_PER_ARCH.get(model_type.lower(), FOCAL_ALPHA_DEFAULT)
    criterion = FocalLoss(alpha=focal_alpha, gamma=FOCAL_GAMMA)

    # Only optimize parameters that require gradients (respects freeze strategy)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=effective_lr, betas=(0.9, 0.999), weight_decay=0.01
    )
    has_validation = valloader is not None
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6
    ) if has_validation else None

    total_train_loss, total_val_loss = 0.0, 0.0
    total_train_batches, total_val_epochs = 0, 0
    total_train_correct, total_train_samples = 0, 0

    model.train()
    for epoch in range(epochs):
        epoch_train_loss, epoch_train_batches = 0.0, 0
        epoch_correct, epoch_samples = 0, 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item()
            epoch_train_batches += 1
            _, predicted = torch.max(outputs.data, 1)
            epoch_samples += labels.size(0)
            epoch_correct += (predicted == labels).sum().item()

        total_train_loss += epoch_train_loss
        total_train_batches += epoch_train_batches
        total_train_correct += epoch_correct
        total_train_samples += epoch_samples

        if has_validation:
            epoch_val_loss = _validate_epoch(model, valloader, criterion, device)
            total_val_loss += epoch_val_loss
            total_val_epochs += 1
            scheduler.step(epoch_val_loss)

    avg_train_loss = total_train_loss / total_train_batches if total_train_batches > 0 else 0.0
    avg_val_loss = total_val_loss / total_val_epochs if total_val_epochs > 0 else 0.0
    train_accuracy = total_train_correct / total_train_samples if total_train_samples > 0 else 0.0
    return avg_train_loss, avg_val_loss, train_accuracy, scheduler.state_dict() if scheduler else {}


def test(model, testloader, device, return_detailed=False):
    """Evaluate model with comprehensive metrics. Returns (loss, acc) or detailed dict."""
    from flwr.common import log
    from logging import INFO, WARNING

    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct, total, total_loss = 0, 0, 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            probs = F.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().tolist())
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy = correct / total if total > 0 else 0.0
    loss = total_loss / len(testloader) if len(testloader) > 0 else 0.0
    all_preds, all_labels, all_probs = torch.tensor(all_preds), torch.tensor(all_labels), torch.tensor(all_probs)

    TP = ((all_preds == 0) & (all_labels == 0)).sum().item()
    FP = ((all_preds == 0) & (all_labels == 1)).sum().item()
    FN = ((all_preds == 1) & (all_labels == 0)).sum().item()
    TN = ((all_preds == 1) & (all_labels == 1)).sum().item()

    leukemia_acc = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    healthy_acc = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = leukemia_acc
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = healthy_acc
    class_gap = abs(healthy_acc - leukemia_acc)

    try:
        from sklearn.metrics import roc_auc_score
        roc_auc = roc_auc_score(all_labels.numpy(), all_probs[:, 0].numpy())
    except Exception:
        roc_auc = (recall + specificity) / 2

    log(INFO, f"[EVAL] Overall: {accuracy:.1%} | Leukemia: {leukemia_acc:.1%} | Healthy: {healthy_acc:.1%} | Gap: {class_gap:.1%}")
    if class_gap > 0.3:
        log(WARNING, f"[EVAL] Class imbalance detected! Gap: {class_gap:.1%}")

    if return_detailed:
        return {
            "loss": round(loss, 6), "accuracy": round(accuracy, 6),
            "precision": round(precision, 6), "recall": round(recall, 6),
            "f1_score": round(f1_score, 6), "specificity": round(specificity, 6),
            "roc_auc": round(roc_auc, 6),
            "leukemia_accuracy": round(leukemia_acc, 6), "healthy_accuracy": round(healthy_acc, 6),
            "class_gap": round(class_gap, 6),
            "confusion_matrix": {"TP": TP, "FP": FP, "FN": FN, "TN": TN},
            "num_samples": total,
            "num_leukemia_samples": (all_labels == 0).sum().item(),
            "num_healthy_samples": (all_labels == 1).sum().item()
        }
    return loss, accuracy


# <----------------------------- KNOWLEDGE DISTILLATION ----------------------------->

def distill_knowledge(model, public_loader, consensus_logits, device, epochs, lr, temperature,
                      current_round=1, total_rounds=10, adaptive=True):
    """Distill consensus knowledge into local model using KL divergence with adaptive weighting."""
    BASE_DISTILL_WEIGHT = 0.40

    if adaptive and total_rounds > 1:
        progress = (current_round - 1) / (total_rounds - 1)
        decay_rate = 0.5
        adaptive_factor = np.exp(-decay_rate * progress)
        DISTILL_WEIGHT = max(0.15, min(0.50, BASE_DISTILL_WEIGHT * adaptive_factor))
    else:
        DISTILL_WEIGHT = BASE_DISTILL_WEIGHT

    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    consensus_tensor = torch.from_numpy(consensus_logits).float()

    total_loss, idx = 0.0, 0
    for _epoch in range(epochs):
        for images, _ in public_loader:
            images = images.to(device)
            batch_size = images.size(0)
            if idx + batch_size > len(consensus_tensor):
                break

            batch_consensus = consensus_tensor[idx:idx+batch_size].to(device)
            student_logits = model(images)
            kl_loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(batch_consensus / temperature, dim=1),
                reduction='batchmean'
            ) * (temperature ** 2)
            loss = DISTILL_WEIGHT * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            idx += batch_size
        idx = 0

    return total_loss / (len(public_loader) * epochs)


# <----------------------------- CONSENSUS COMPUTATION ----------------------------->

def compute_consensus(
    logits_list: List[np.ndarray], client_metrics: List[Dict], client_configs: List[Dict],
    server_round: int, last_consensus: Optional[np.ndarray] = None,
    eval_history: Optional[List[Dict]] = None, momentum: float = CONSENSUS_MOMENTUM,
) -> Tuple[Optional[np.ndarray], Dict]:
    """Compute weighted consensus from client logits with momentum and class reweighting."""
    if not logits_list:
        return None, {"error": "No client logits provided"}
    if len(logits_list) != len(client_metrics) or len(logits_list) != len(client_configs):
        return None, {"error": "Mismatched input lengths"}

    # ========== QUALITY GATING: Filter poorly performing clients ==========
    ACCURACY_THRESHOLD = 0.55  # Minimum accuracy to contribute to consensus
    GAP_THRESHOLD = 0.80       # Maximum prediction gap (class imbalance in predictions)

    def calculate_prediction_gap(logits: np.ndarray) -> float:
        """Calculate how skewed a client's predictions are towards one class."""
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        predicted_classes = np.argmax(probs, axis=1)
        unique, counts = np.unique(predicted_classes, return_counts=True)
        if len(counts) < 2:
            return 1.0  # All predictions same class = 100% gap
        total = sum(counts)
        majority_ratio = max(counts) / total
        minority_ratio = min(counts) / total
        return majority_ratio - minority_ratio  # Range: 0.0 (balanced) to 1.0 (all one class)

    # Evaluate each client for quality gating
    qualified_indices = []
    excluded_clients = []

    for i in range(len(logits_list)):
        metrics = client_metrics[i]
        config = client_configs[i]
        client_name = config.get("client_name", f"client_{i}")

        # Get accuracy (prefer eval_acc, fallback to train_accuracy)
        accuracy = metrics.get("eval_acc", metrics.get("train_accuracy", 0.5))

        # Calculate prediction gap from logits
        pred_gap = calculate_prediction_gap(logits_list[i])

        exclusion_reasons = []
        if accuracy < ACCURACY_THRESHOLD:
            exclusion_reasons.append(f"accuracy={accuracy:.2%} < {ACCURACY_THRESHOLD:.0%}")
        if pred_gap > GAP_THRESHOLD:
            exclusion_reasons.append(f"pred_gap={pred_gap:.2%} > {GAP_THRESHOLD:.0%}")

        if exclusion_reasons:
            excluded_clients.append({
                "client_name": client_name,
                "index": i,
                "accuracy": accuracy,
                "prediction_gap": pred_gap,
                "reasons": exclusion_reasons
            })
            print(f"[Consensus] EXCLUDED {client_name}: {', '.join(exclusion_reasons)}")
        else:
            qualified_indices.append(i)
            print(f"[Consensus] QUALIFIED {client_name}: acc={accuracy:.2%}, gap={pred_gap:.2%}")

    # Fallback: If all clients excluded, use the best one
    if not qualified_indices:
        print("[Consensus] WARNING: All clients excluded! Falling back to best client.")
        # Score = accuracy - prediction_gap (higher is better)
        best_idx = max(
            range(len(logits_list)),
            key=lambda i: client_metrics[i].get("eval_acc", client_metrics[i].get("train_accuracy", 0.5))
                          - calculate_prediction_gap(logits_list[i])
        )
        qualified_indices = [best_idx]
        best_name = client_configs[best_idx].get("client_name", f"client_{best_idx}")
        print(f"[Consensus] Using fallback client: {best_name}")

    # Filter to qualified clients only
    logits_list = [logits_list[i] for i in qualified_indices]
    client_metrics = [client_metrics[i] for i in qualified_indices]
    client_configs = [client_configs[i] for i in qualified_indices]

    gating_info = {
        "accuracy_threshold": ACCURACY_THRESHOLD,
        "gap_threshold": GAP_THRESHOLD,
        "total_clients": len(qualified_indices) + len(excluded_clients),
        "qualified_clients": len(qualified_indices),
        "excluded_clients": excluded_clients,
    }
    # ========== END QUALITY GATING ==========

    num_clients = len(logits_list)
    weights, weight_breakdown = [], []

    for i in range(num_clients):
        metrics, config = client_metrics[i], client_configs[i]
        num_samples = metrics.get("num-examples", 1)
        train_loss = metrics.get("train_loss", 0.0)
        distill_loss = metrics.get("distill_loss", 0.0)
        model_type = config.get("model_type", "unknown").lower()
        client_name = config.get("client_name", f"client_{i}")

        base_weight = max(num_samples, 1)
        combined_loss = TRAIN_LOSS_WEIGHT * train_loss + DISTILL_LOSS_WEIGHT * distill_loss
        quality_multiplier = 1.0 / (1.0 + combined_loss)
        val_acc = metrics.get("eval_acc", metrics.get("train_accuracy", 0.5))
        accuracy_factor = 0.5 + val_acc
        round_trust = min(server_round / 5.0, 1.0)
        stabilized_quality = 0.5 + 0.5 * round_trust * quality_multiplier
        final_weight = base_weight * stabilized_quality * accuracy_factor

        weights.append(final_weight)
        weight_breakdown.append({
            "client_name": client_name, "model_type": model_type, "num_samples": num_samples,
            "train_loss": train_loss, "distill_loss": distill_loss, "combined_loss": combined_loss,
            "base_weight": base_weight, "quality_multiplier": quality_multiplier,
            "accuracy_factor": accuracy_factor, "round_trust": round_trust,
            "stabilized_quality": stabilized_quality, "final_weight": final_weight,
        })

    total_weight = sum(weights)
    if total_weight == 0:
        return None, {"error": "All clients have zero weight", "weight_breakdown": weight_breakdown}

    normalized_weights = [w / total_weight for w in weights]
    for i, breakdown in enumerate(weight_breakdown):
        breakdown["normalized_weight"] = normalized_weights[i]

    new_consensus = np.average(logits_list, axis=0, weights=normalized_weights)

    # Class-based reweighting: boost minority class predictions
    consensus_probs = np.exp(new_consensus) / np.sum(np.exp(new_consensus), axis=1, keepdims=True)
    predicted_classes = np.argmax(consensus_probs, axis=1)
    unique_classes, class_counts = np.unique(predicted_classes, return_counts=True)

    total_samples = len(predicted_classes)
    class_weights = {cls: total_samples / (len(unique_classes) * count) for cls, count in zip(unique_classes, class_counts)}
    weight_sum = sum(class_weights.values())
    class_weights = {cls: w / weight_sum for cls, w in class_weights.items()}

    def reweight_consensus_probs(logits: np.ndarray, weights: dict) -> np.ndarray:
        """Reweight consensus in probability space (mathematically correct).
        
        Args:
            logits: Raw logits array of shape [num_samples, num_classes]
            weights: Dict mapping class index to weight value
            
        Returns:
            Reweighted logits with proper probability scaling
        """
        # Stable softmax: subtract max for numerical stability
        logits_stable = logits - np.max(logits, axis=1, keepdims=True)
        probs = np.exp(logits_stable)
        probs = probs / probs.sum(axis=1, keepdims=True)
        
        # Build weight vector for all classes
        num_classes = probs.shape[1]
        weight_vector = np.ones(num_classes)
        for cls, w in weights.items():
            if 0 <= cls < num_classes:
                weight_vector[cls] = w
        
        # Apply weights to probabilities
        probs = probs * weight_vector
        
        # Renormalize to valid probability distribution
        probs = probs / probs.sum(axis=1, keepdims=True)
        
        # Convert back to logits (with epsilon for numerical stability)
        return np.log(probs + 1e-10)

    class_weighted_consensus = reweight_consensus_probs(new_consensus, class_weights)
    new_consensus = class_weighted_consensus

    class_weighting_info = {
        "class_distribution": {int(cls): int(count) for cls, count in zip(unique_classes, class_counts)},
        "class_weights": {int(cls): float(w) for cls, w in class_weights.items()},
        "minority_class": int(unique_classes[np.argmin(class_counts)]) if len(unique_classes) > 0 else None,
        "majority_class": int(unique_classes[np.argmax(class_counts)]) if len(unique_classes) > 0 else None,
    }

    # Momentum smoothing across FL rounds
    smoothing_applied = False
    if last_consensus is not None and server_round > 1:
        if last_consensus.shape != new_consensus.shape:
            if new_consensus.shape[0] > last_consensus.shape[0]:
                if new_consensus.shape[1] != last_consensus.shape[1]:
                    return None, {"error": "Class count mismatch in consensus resizing"}
                old_len = last_consensus.shape[0]
                balanced_part = momentum * last_consensus + (1 - momentum) * new_consensus[:old_len]
                consensus_logits = np.concatenate([balanced_part, new_consensus[old_len:]], axis=0)
            else:
                consensus_logits = new_consensus
        else:
            consensus_logits = momentum * last_consensus + (1 - momentum) * new_consensus
        smoothing_applied = True
    else:
        consensus_logits = new_consensus

    aggregation_metadata = {
        "server_round": server_round, "num_clients": num_clients,
        "num_contributing_clients": sum(1 for w in weights if w > 0),
        "total_raw_weight": total_weight,
        "momentum": momentum if smoothing_applied else None,
        "smoothing_applied": smoothing_applied,
        "weight_breakdown": weight_breakdown, "normalized_weights": normalized_weights,
        "weight_statistics": {
            "min": min(normalized_weights) if normalized_weights else 0,
            "max": max(normalized_weights) if normalized_weights else 0,
            "mean": float(np.mean(normalized_weights)) if normalized_weights else 0,
            "std": float(np.std(normalized_weights)) if normalized_weights else 0,
        },
        "class_weighting": class_weighting_info,
        "quality_gating": gating_info,
        "parameters": {"train_loss_weight": TRAIN_LOSS_WEIGHT, "distill_loss_weight": DISTILL_LOSS_WEIGHT}
    }
    return consensus_logits, aggregation_metadata


# <----------------------------- FL STRATEGY ----------------------------->

class FLEXMedStrategy(Strategy):
    """Coordinates federated learning rounds with model-agnostic knowledge distillation."""

    def __init__(self, config_path: str = CLIENT_INFO_FILE_PATH, checkpoint_dir: str = MODEL_CHECKPOINT_FILE_PATH):
        super().__init__()
        self.config_path = config_path
        self.checkpoint_dir = checkpoint_dir
        self.client_configs = load_client_config(config_path)
        self.num_clients = len(self.client_configs)
        self.eval_history = []
        self.start_round = 1
        self.last_consensus_logits = None
        self.early_stopping_enabled = True
        self.patience = 2
        self.best_round = 0
        self.best_avg_val_loss = float('inf')
        self.best_avg_val_acc = 0.0
        self.degradation_count = 0
        self.best_checkpoints_saved = False
        self.client_history = {}

        checkpoint = load_checkpoint(checkpoint_dir)
        if checkpoint:
            self.start_round = checkpoint['round'] + 1
            self.eval_history = checkpoint['eval_history']
            self.last_consensus_logits = checkpoint['consensus_logits']

    def start(self, grid: Grid, initial_arrays: ArrayRecord, num_rounds: int = 3,
              timeout: float = 14400, train_config: Optional[ConfigRecord] = None,
              evaluate_config: Optional[ConfigRecord] = None, evaluate_fn=None):
        """Execute FL with resume support and per-round evaluation."""
        from flwr.common import log
        from flwr.serverapp.strategy.result import Result
        from logging import INFO, WARNING

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.start_round > 1:
            remaining = num_rounds - (self.start_round - 1)
            if remaining <= 0:
                result = Result()
                result.arrays = initial_arrays
                return result

        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config

        result = Result()
        arrays = initial_arrays
        t_start = time.time()
        self.round_metrics_history = {}

        print_client_data_distribution_summary(self.client_configs, _PARTITIONER_CACHE)

        for current_round in range(self.start_round, num_rounds + 1):
            log(INFO, f"\n{'='*70}\n[ROUND {current_round}/{num_rounds}]\n{'='*70}")

            train_msgs = self.configure_train(current_round, arrays, train_config, grid)
            train_replies = grid.send_and_receive(messages=train_msgs, timeout=timeout)
            agg_arrays, agg_metrics, training_metrics = self.aggregate_train(current_round, train_replies)

            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays
            if agg_metrics:
                result.train_metrics_clientapp[current_round] = agg_metrics
            if training_metrics:
                save_round_training_metrics(current_round, training_metrics, self.client_configs, _PARTITIONER_CACHE)

            try:
                round_val_metrics = self._evaluate_all_clients_on_validation(device)
                save_round_validation_metrics(current_round, round_val_metrics, self.client_configs)
                self.round_metrics_history[f"round_{current_round}_validation"] = round_val_metrics

                for client_id, metrics in round_val_metrics.items():
                    if client_id not in self.client_history:
                        self.client_history[client_id] = []
                    self.client_history[client_id].append({'round': current_round, 'metrics': metrics})

                check_for_degradation_warnings(round_val_metrics, current_round, self.client_history)
            except Exception as e:
                log(WARNING, f"[ROUND {current_round}] Validation evaluation failed: {e}")

            eval_msgs = self.configure_evaluate(current_round, arrays, evaluate_config, grid)
            eval_replies = grid.send_and_receive(messages=eval_msgs, timeout=timeout)
            eval_metrics = self.aggregate_evaluate(current_round, eval_replies)
            if eval_metrics:
                result.evaluate_metrics_clientapp[current_round] = eval_metrics

        log(INFO, f"\n{'='*70}\n[GLOBAL] Final Federated Model Evaluation\n{'='*70}")

        try:
            global_post_fl_metrics = self._evaluate_all_clients_on_public_test(device)
            save_global_post_fl_metrics(global_post_fl_metrics, self.client_configs)
        except Exception as e:
            log(WARNING, f"[GLOBAL] Post-FL Evaluation failed: {e}")

        log(INFO, f"Strategy execution finished in {time.time() - t_start:.2f}s")
        return result

    def _evaluate_all_clients_on_validation(self, device: torch.device) -> Dict:
        """Evaluate all clients on their private validation sets."""
        from sklearn.metrics import precision_recall_fscore_support

        client_metrics = {}
        for i, client in enumerate(self.client_configs):
            try:
                model = get_model_by_type(client['model_type'])
                if os.path.exists(client['model_path']):
                    model, _ = load_existing_model(model, client['model_path'], device)
                model.to(device)
                _, valloader = load_private_dataset(i, len(self.client_configs), batch_size=64)

                if valloader is None:
                    valloader = load_public_test_dataset(batch_size=64)

                model.eval()
                all_preds, all_labels = [], []
                total_loss = 0.0
                criterion = nn.CrossEntropyLoss()

                with torch.no_grad():
                    for images, labels in valloader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        total_loss += criterion(outputs, labels).item() * images.size(0)
                        _, preds = torch.max(outputs, 1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                all_preds, all_labels = np.array(all_preds), np.array(all_labels)
                accuracy = (all_preds == all_labels).mean()
                loss = total_loss / len(all_labels)
                precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)

                leukemia_mask, healthy_mask = (all_labels == 0), (all_labels == 1)
                leukemia_acc = (all_preds[leukemia_mask] == all_labels[leukemia_mask]).mean() if leukemia_mask.any() else 0
                healthy_acc = (all_preds[healthy_mask] == all_labels[healthy_mask]).mean() if healthy_mask.any() else 0

                client_metrics[str(i)] = {
                    'loss': float(loss), 'accuracy': float(accuracy),
                    'precision': float(precision), 'recall': float(recall), 'f1_score': float(f1),
                    'class_gap': float(abs(leukemia_acc - healthy_acc)),
                    'leukemia_accuracy': float(leukemia_acc), 'healthy_accuracy': float(healthy_acc),
                }
            except Exception as e:
                client_metrics[str(i)] = {"accuracy": None, "loss": None, "error": str(e)}
        return client_metrics

    def _evaluate_all_clients_on_public_test(self, device: torch.device) -> Dict:
        """Evaluate all clients on public test dataset."""
        client_metrics = {}
        for i, client in enumerate(self.client_configs):
            try:
                model = get_model_by_type(client['model_type'])
                if os.path.exists(client['model_path']):
                    model, _ = load_existing_model(model, client['model_path'], device)
                model.to(device)
                test_loader = load_public_test_dataset(batch_size=64)
                metrics = test(model, test_loader, device, return_detailed=True)
                client_metrics[str(i)] = metrics
            except Exception as e:
                client_metrics[str(i)] = {"accuracy": None, "loss": None, "error": str(e)}
        return client_metrics

    def configure_evaluate(self, server_round: int, arrays: Optional[ArrayRecord],
                           config: ConfigRecord, grid: Grid) -> Iterable[Message]:
        messages = []
        for node_id in grid.get_node_ids():
            msg = Message(
                metadata=Metadata(run_id=0, message_id=str(uuid.uuid4()), src_node_id=0, dst_node_id=node_id,
                                  reply_to_message_id="", group_id=str(server_round), ttl=86400.0,
                                  message_type="evaluate", created_at=time.time()),
                content=RecordDict({"config": ConfigRecord({"round": server_round})}),
            )
            messages.append(msg)
        return messages

    def aggregate_evaluate(self, server_round: int, results: Iterable[Message], **kwargs):
        results_list = list(results)
        if not results_list:
            return {"loss": None, "metrics": {}}

        clients_with_data = []
        total_loss, total_acc, total_examples = 0.0, 0.0, 0

        for i, msg in enumerate(results_list):
            if not msg.has_content():
                continue
            metrics = msg.content.get("metrics", {})
            eval_loss = metrics.get("eval_loss", 0.0)
            eval_acc = metrics.get("eval_acc", 0.0)
            num_examples = metrics.get("num-examples", 0)
            clients_with_data.append({"client_id": metrics.get("client_id", -1), "loss": eval_loss, "acc": eval_acc, "examples": num_examples})
            total_loss += eval_loss * num_examples
            total_acc += eval_acc * num_examples
            total_examples += num_examples

        if total_examples == 0:
            return {"loss": None, "metrics": {}}

        avg_loss, avg_acc = total_loss / total_examples, total_acc / total_examples
        round_metrics = {"round": server_round, "avg_loss": avg_loss, "avg_acc": avg_acc,
                         "total_examples": total_examples, "clients_with_data": clients_with_data}
        self.eval_history.append(round_metrics)

        if self.last_consensus_logits is not None:
            try:
                save_checkpoint(self.checkpoint_dir, server_round, self.last_consensus_logits, self.eval_history, round_metrics)
            except Exception:
                pass

        return {"loss": avg_loss, "metrics": {"eval_acc": avg_acc, "num_clients": len(results_list),
                                               "total_examples": total_examples, "clients_with_data": len(clients_with_data)}}

    def aggregate_train(self, server_round: int, results: Iterable[Message], **kwargs):
        results_list = list(results)
        if not results_list:
            return None, {}, {}

        logits_list, client_metrics_list, client_names = [], [], []

        for i, msg in enumerate(results_list):
            if not msg.has_content():
                continue
            try:
                client_logits = msg.content["arrays"]["0"].numpy()
                logits_list.append(client_logits)
                client_metrics_list.append(msg.content.get("metrics", {}))
                if i < len(self.client_configs):
                    client_names.append(self.client_configs[i]['client_name'])
            except (KeyError, IndexError):
                pass

        consensus_logits, aggregation_metadata = compute_consensus(
            logits_list, client_metrics_list, self.client_configs[:len(logits_list)],
            server_round, self.last_consensus_logits, self.eval_history, CONSENSUS_MOMENTUM
        )

        if consensus_logits is None:
            return None, {}, {}

        self.last_consensus_logits = consensus_logits

        metrics_aggregated = {
            "consensus_round": server_round, "num_clients": len(logits_list),
            "client_names": client_names, "weights": aggregation_metadata['normalized_weights'],
            "aggregation_details": aggregation_metadata
        }

        training_metrics_for_persistence = extract_training_metrics_for_persistence(
            client_metrics_list, aggregation_metadata, self.client_configs
        )
        return ArrayRecord([consensus_logits]), metrics_aggregated, training_metrics_for_persistence

    # Configure the training process for each client and passes it to each client as a message
    def configure_train(self, server_round: int, arrays: Optional[ArrayRecord],
        config: ConfigRecord, grid: Grid) -> Iterable[Message]:
        messages = []
        for node_id in grid.get_node_ids():
            msg = Message(
                metadata=Metadata(run_id=0, message_id=str(uuid.uuid4()), src_node_id=0, dst_node_id=node_id,
                                  reply_to_message_id="", group_id=str(server_round), ttl=86400.0,
                                  message_type="train", created_at=time.time()),
                content=RecordDict({"arrays": arrays, "config": config}),
            )
            messages.append(msg)
        return messages

    def save_evaluation_history(self):
        return None

    def summary(self) -> str:
        return ""
