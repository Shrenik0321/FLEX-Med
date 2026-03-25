import torch
import numpy as np
import os
import json
import time
import uuid
import shutil
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import Compose, ToTensor, Normalize
from typing import Tuple, Optional, Iterable, Dict, List
from flwr.common import Message, Metadata, RecordDict, ArrayRecord, ConfigRecord, log
from logging import INFO, WARNING
from flwr.serverapp.strategy import Strategy
from flwr.server import Grid
from flwr_datasets.partitioner import DirichletPartitioner
from datasets import Dataset
from flex_med.utils.config import (
    CLIENT_INFO_FILE_PATH,
    PUBLIC_ANCHOR_DATASET_PATH, PUBLIC_TEST_DATASET_PATH, LOCAL_TRAIN_DATASET_PATH,
    NUM_CLASSES, IMG_SIZE,
    DIRICHLET_ALPHA, DIRICHLET_SEED, DIRICHLET_MIN_PARTITION_SIZE,
    MINORITY_BOOST, WEIGHT_DECAY, DISTILL_WEIGHT_BASE, DISTILL_DECAY_RATE
)
from flex_med.utils.helpers import (
    sanitize_client_paths,
    get_partition_stats, print_client_data_distribution_summary,
    extract_training_metrics_for_persistence, save_global_post_fl_metrics,
    save_round_training_metrics, save_round_validation_metrics,
    check_for_degradation_warnings, SUPABASE_CLIENT, SIMULATION_ID,
    save_all_clients_data_heterogeneity,
    load_client_config, get_client_by_partition_id,
    get_model_by_type, load_model_weights, set_bn_eval
)

# <----------------------------- CONSTANTS & GLOBAL VARIABLES ----------------------------->
PARTITIONER_CACHE = {}

# <----------------------------- DATA TRANSFORMS ----------------------------->
COMMON_TRANSFORM = Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

PRIVATE_TRAIN_TRANSFORM = Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), # Standardize resolution
    transforms.RandomHorizontalFlip(p=0.5), # Handle arbitrary orientation
    transforms.RandomVerticalFlip(p=0.5),   # Cells aren't axis-aligned
    transforms.RandomRotation(15),          # Slight angular variations
    ToTensor(), # Convert to tensor
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet standards
])

class TransformOverrideSubset(torch.utils.data.Dataset):
    """Wraps a nested Subset to apply a different transform (e.g. no augmentation for val)."""

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        # Resolve the root ImageFolder through nested Subsets
        ds = subset
        self._index_chain = []
        while isinstance(ds, torch.utils.data.Subset):
            self._index_chain.append(ds.indices)
            ds = ds.dataset
        self._root_dataset = ds  # The ImageFolder

    def _resolve_index(self, idx):
        resolved = idx
        for indices in self._index_chain:
            resolved = indices[resolved]
        return resolved

    def __getitem__(self, idx):
        root_idx = self._resolve_index(idx)
        path, label = self._root_dataset.samples[root_idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), label

    def __len__(self):
        return len(self.subset)

# <----------------------------- LOAD DATA ----------------------------->
# Create a Dirichlet partitioner for heterogeneous data distribution.
def create_dirichlet_partitioner(
    dataset_path: str, num_partitions: int, alpha: float = DIRICHLET_ALPHA,
    seed: int = DIRICHLET_SEED, min_partition_size: int = DIRICHLET_MIN_PARTITION_SIZE
) -> Tuple[DirichletPartitioner, datasets.ImageFolder]:
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

# Load the public anchor dataset for consensus generation.
def load_public_dataset(batch_size=32):
    if not os.path.exists(PUBLIC_ANCHOR_DATASET_PATH):
        raise FileNotFoundError(f"Public data not found at {PUBLIC_ANCHOR_DATASET_PATH}")

    full_dataset = datasets.ImageFolder(root=PUBLIC_ANCHOR_DATASET_PATH, transform=COMMON_TRANSFORM)
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(full_dataset), generator=generator).tolist()
    subset = torch.utils.data.Subset(full_dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)

# Load and assign the data distribution to client models using Dirichlet partitioning.
def load_private_dataset(partition_id: int, num_partitions: int, batch_size=32,
                         config_path: str = CLIENT_INFO_FILE_PATH):
    if not os.path.exists(LOCAL_TRAIN_DATASET_PATH):
        raise FileNotFoundError(f"Shared training dataset not found at {LOCAL_TRAIN_DATASET_PATH}")

    cache_key = (LOCAL_TRAIN_DATASET_PATH, num_partitions, DIRICHLET_ALPHA, DIRICHLET_SEED)

    if cache_key not in PARTITIONER_CACHE:
        partitioner, full_dataset = create_dirichlet_partitioner(
            LOCAL_TRAIN_DATASET_PATH, num_partitions, DIRICHLET_ALPHA, DIRICHLET_SEED
        )
        PARTITIONER_CACHE[cache_key] = (partitioner, full_dataset)
    else:
        partitioner, full_dataset = PARTITIONER_CACHE[cache_key]

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
    test_ds = TransformOverrideSubset(
        torch.utils.data.Subset(client_dataset, val_indices), COMMON_TRANSFORM
    )

    train_labels = [full_dataset.targets[client_indices[i]] for i in train_indices]
    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)

    class_weights = 1.0 / np.maximum(class_counts, 1) ** MINORITY_BOOST
    sample_weights = [class_weights[label] for label in train_labels]

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    trainloader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=2)
    testloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader, class_counts

# Load public test dataset for post_fl evaluation.
def load_public_test_dataset(batch_size=32):
    if not os.path.exists(PUBLIC_TEST_DATASET_PATH):
        raise FileNotFoundError(f"Public test data not found at {PUBLIC_TEST_DATASET_PATH}")
    dataset = datasets.ImageFolder(root=PUBLIC_TEST_DATASET_PATH, transform=COMMON_TRANSFORM)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Generate logits on the public anchor dataset for consensus computation.
def get_public_logits(model, public_loader, device):
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

# <----------------------------- LOCAL MODEL Training : Train client model with Focal Loss and gradual unfreeze strategy. ----------------------------->
# Validation function during local training
def validate_training(model, valloader, criterion, device):
    model.eval()
    val_loss, val_batches = 0.0, 0
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in valloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            val_batches += 1
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = val_loss / val_batches if val_batches > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy

def train(model, trainloader, epochs, classifier_lr, backbone_lr, device,
          valloader=None, server_round: int = 1, total_rounds: int = 10):

    if trainloader is None:
        return 0.0, 0.0, 0.0, 0.0

    model.to(device)

    # Standard CrossEntropyLoss with label smoothing (WRS handles class balance)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    classifier_decay = []
    classifier_no_decay = []
    backbone_decay = []
    backbone_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        is_classifier = "classifier" in name or "fc" in name
        if is_classifier:
            if len(param.shape) == 1:
                classifier_no_decay.append(param)
            else:
                classifier_decay.append(param)
        else:
            if len(param.shape) == 1:
                backbone_no_decay.append(param)
            else:
                backbone_decay.append(param)
    
    optimizer = torch.optim.AdamW([
        {"params": classifier_decay, "lr": classifier_lr, "weight_decay": WEIGHT_DECAY},
        {"params": classifier_no_decay, "lr": classifier_lr, "weight_decay": 0.0},
        {"params": backbone_decay, "lr": backbone_lr, "weight_decay": WEIGHT_DECAY * 2.0},
        {"params": backbone_no_decay, "lr": backbone_lr, "weight_decay": 0.0}
    ], betas=(0.9, 0.999))

    has_validation = valloader is not None

    total_train_loss, total_val_loss = 0.0, 0.0
    total_train_batches, total_val_epochs = 0, 0
    total_train_correct, total_train_samples = 0, 0
    total_val_accuracy = 0.0

    for epoch in range(epochs):
        model.train()
        model.apply(set_bn_eval)
        
        epoch_train_loss, epoch_train_batches = 0.0, 0
        epoch_correct, epoch_samples = 0, 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
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
            epoch_val_loss, epoch_val_acc = validate_training(model, valloader, criterion, device)
            total_val_loss += epoch_val_loss
            total_val_accuracy += epoch_val_acc
            total_val_epochs += 1

    avg_train_loss = total_train_loss / total_train_batches if total_train_batches > 0 else 0.0
    avg_val_loss = total_val_loss / total_val_epochs if total_val_epochs > 0 else 0.0
    train_accuracy = total_train_correct / total_train_samples if total_train_samples > 0 else 0.0
    val_accuracy = total_val_accuracy / total_val_epochs if total_val_epochs > 0 else 0.0
    
    return avg_train_loss, avg_val_loss, train_accuracy, val_accuracy

# <----------------------------- LOCAL MODEL Testing & Evaluation : Post FL testing of each client model with public test dataset. ----------------------------->
def test(model, testloader, device, return_detailed=False):
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

    # Calculate confusion matrix (0 = Leukemia [Positive class], 1 = Healthy [Negative class])
    TP = ((all_preds == 0) & (all_labels == 0)).sum().item()  # True Positive: Predicted Leukemia, Actual Leukemia
    FP = ((all_preds == 0) & (all_labels == 1)).sum().item()  # False Positive: Predicted Leukemia, Actual Healthy
    FN = ((all_preds == 1) & (all_labels == 0)).sum().item()  # False Negative: Predicted Healthy, Actual Leukemia
    TN = ((all_preds == 1) & (all_labels == 1)).sum().item()  # True Negative: Predicted Healthy, Actual Healthy

    # Standard ML Metrics (Focusing on Leukemia detection)
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0        # Also known as Sensitivity or True Positive Rate
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0   # True Negative Rate
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0     # Positive Predictive Value

    # F1 Score: Harmonic mean of Precision and Recall
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Aliases for domain-specific logging and returning
    leukemia_acc = recall
    healthy_acc = specificity
    class_gap = abs(healthy_acc - leukemia_acc)

    log(INFO, f"[EVAL] Overall: {accuracy:.1%} | Leukemia: {leukemia_acc:.1%} | Healthy: {healthy_acc:.1%} | Gap: {class_gap:.1%}")
    if class_gap > 0.3:
        log(WARNING, f"[EVAL] Class imbalance detected! Gap: {class_gap:.1%}")

    if return_detailed:
        return {
            "loss": round(loss, 6), "accuracy": round(accuracy, 6),
            "precision": round(precision, 6), "recall": round(recall, 6),
            "f1_score": round(f1_score, 6), "specificity": round(specificity, 6),
            "leukemia_accuracy": round(leukemia_acc, 6), "healthy_accuracy": round(healthy_acc, 6),
            "class_gap": round(class_gap, 6),
            "confusion_matrix": {"TP": TP, "FP": FP, "FN": FN, "TN": TN},
            "num_samples": total,
            "num_leukemia_samples": (all_labels == 0).sum().item(),
            "num_healthy_samples": (all_labels == 1).sum().item()
        }
    return loss, accuracy

# <----------------------------- KNOWLEDGE DISTILLATION : Distill consensus knowledge into local model using KL divergence with adaptive weighting. ----------------------------->
def distill_knowledge(model, public_loader, consensus_logits, device, epochs, classifier_lr, backbone_lr, temperature,
                      current_round=1, total_rounds=10, adaptive=True):
    base_weight = DISTILL_WEIGHT_BASE
    decay_rate = DISTILL_DECAY_RATE
    weight_decay = WEIGHT_DECAY

    if adaptive and total_rounds > 1:
        progress = (current_round - 1) / (total_rounds - 1)
        adaptive_factor = np.exp(-decay_rate * progress)
        DISTILL_WEIGHT = max(0.20, min(1.0, base_weight * adaptive_factor))
    else:
        DISTILL_WEIGHT = base_weight

    model.to(device)
    model.train()
    model.apply(set_bn_eval)
    
    # Separate parameters into four groups for Discriminative LRs + Weight Decay optimization
    # Rule: Exclude 1D parameters (biases and BN layers) from weight decay.
    classifier_decay = []
    classifier_no_decay = []
    backbone_decay = []
    backbone_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        is_classifier = "classifier" in name or "fc" in name
        if is_classifier:
            if len(param.shape) == 1:
                classifier_no_decay.append(param)
            else:
                classifier_decay.append(param)
        else:
            if len(param.shape) == 1:
                backbone_no_decay.append(param)
            else:
                backbone_decay.append(param)
                
    # Optimzer called to updated weights as the model tries to match the conensus value
    optimizer = torch.optim.AdamW([
        {"params": classifier_decay, "lr": classifier_lr, "weight_decay": weight_decay},
        {"params": classifier_no_decay, "lr": classifier_lr, "weight_decay": 0.0},
        {"params": backbone_decay, "lr": backbone_lr, "weight_decay": weight_decay * 2.0},
        {"params": backbone_no_decay, "lr": backbone_lr, "weight_decay": 0.0}
    ], betas=(0.9, 0.999))

    consensus_tensor = torch.from_numpy(consensus_logits).float()

    total_loss, idx = 0.0, 0
    for _epoch in range(epochs):
        for images, _ in public_loader:
            images = images.to(device)
            batch_size = images.size(0)
            if idx + batch_size > len(consensus_tensor):
                break

            batch_consensus = consensus_tensor[idx:idx+batch_size].to(device)

            # Run the public anchor dataset through the client model to obtain the student logits
            student_logits = model(images)

            # Kullback Leibler Divergence
            kl_loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(batch_consensus / temperature, dim=1),
                reduction='batchmean'
            ) * (temperature ** 2)
            loss = DISTILL_WEIGHT * kl_loss

            # Backpropagate and update
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            total_loss += loss.item()
            idx += batch_size
        idx = 0

    return total_loss / (len(public_loader) * epochs)

# <----------------------------- CONSENSUS COMPUTATION ----------------------------->
def compute_consensus(
    logits_list: List[np.ndarray], client_metrics: List[Dict], client_configs: List[Dict],
    server_round: int,
) -> Tuple[Optional[np.ndarray], Dict]:
    if not logits_list:
        return None, {"error": "No client logits provided"}

    num_clients = len(logits_list)

    # Weighted aggregation
    raw_weights = []
    for i in range(num_clients):
        # Safely extract num_examples if it exists in the nested dict structure from client_app.py
        if i < len(client_metrics) and "num-examples" in client_metrics[i]:
            raw_weights.append(float(client_metrics[i]["num-examples"]) ** 0.5)
        else:
            raw_weights.append(1.0) # Fallback

    total_samples = sum(raw_weights)
    normalized_weights = [w / total_samples for w in raw_weights]
    
    # Current weighted average
    current_avg_logits = np.average(logits_list, axis=0, weights=normalized_weights)
    
    consensus_logits = current_avg_logits

    weight_breakdown = []
    for i in range(num_clients):
        config = client_configs[i] if i < len(client_configs) else {}
        weight_breakdown.append({
            "client_name": config.get("client_name", f"client_{i}"),
            "model_type": config.get("model_type", "unknown"),
            "normalized_weight": normalized_weights[i],
        })

    aggregation_metadata = {
        "server_round": server_round, "num_clients": num_clients,
        "weight_breakdown": weight_breakdown, "normalized_weights": normalized_weights,
    }

    return consensus_logits, aggregation_metadata

# <----------------------------- FL STRATEGY ----------------------------->
class FLEXMedStrategy(Strategy):
    def __init__(self, config_path: str = CLIENT_INFO_FILE_PATH, batch_size: int = 32):
        super().__init__()
        self.config_path = config_path
        self.batch_size = batch_size
        self.client_configs = load_client_config(config_path)
        self.num_clients = len(self.client_configs)
        self.client_history = {}
        self.total_rounds = 0

# <----------------------------- HELPER METHODS ----------------------------->
    # Evaluate all clients on their private validation sets (pos class is class 0 (all) and healthy is class 1)
    def evaluate_all_clients_on_validation(self, device: torch.device) -> Dict:
        """Evaluate all clients on their private validation sets."""
        from sklearn.metrics import precision_recall_fscore_support

        client_metrics = {}
        for i, client in enumerate(self.client_configs):
            try:
                model = get_model_by_type(client['model_type'])
                model = load_model_weights(model, client['model_path'], device)
                model.to(device)
                _, valloader, _ = load_private_dataset(i, len(self.client_configs), batch_size=self.batch_size)

                if valloader is None:
                    raise ValueError(f"Validation loader for client {i} is None")

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
                precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', pos_label=0, zero_division=0)

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
                log(WARNING, f"[VAL] Client {i} validation failed: {e}")
                client_metrics[str(i)] = {"accuracy": None, "loss": None, "error": str(e)}
        return client_metrics

    # Called only once, for the post FL evaluation on unseen test data to evaluate all clients on public test dataset
    def evaluate_all_clients_on_public_test(self, device: torch.device) -> Dict:
        client_metrics = {}
        for i, client in enumerate(self.client_configs):
            try:
                model = get_model_by_type(client['model_type'])
                model = load_model_weights(model, client['model_path'], device)
                model.to(device)
                test_loader = load_public_test_dataset(batch_size=self.batch_size)
                metrics = test(model, test_loader, device, return_detailed=True)
                client_metrics[str(i)] = metrics
            except Exception as e:
                client_metrics[str(i)] = {"accuracy": None, "loss": None, "error": str(e)}
        return client_metrics

# <----------------------------- Start FL simulation ----------------------------->
    def start(self, grid: Grid, initial_arrays: ArrayRecord, num_rounds: int = 3,
              timeout: float = 14400, train_config: Optional[ConfigRecord] = None):

        from flwr.common import log
        from flwr.serverapp.strategy.result import Result
        from logging import INFO, WARNING

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        train_config = ConfigRecord() if train_config is None else train_config

        result = Result()
        arrays = initial_arrays
        t_start = time.time()
        self.round_metrics_history = {}

        # Best model tracking
        self.best_avg_accuracy = -1.0
        self.best_round = 0

        # <----------------------------- DATA DISTRIBUTION ASSIGNMENT TO EACH CLIENT ----------------------------->
        try:
            log(INFO, "[FL] Pre-initializing Dirichlet partitioner...")
            cache_key = (LOCAL_TRAIN_DATASET_PATH, self.num_clients, DIRICHLET_ALPHA, DIRICHLET_SEED)
            if cache_key not in PARTITIONER_CACHE:
                partitioner, full_dataset = create_dirichlet_partitioner(
                    LOCAL_TRAIN_DATASET_PATH, self.num_clients, DIRICHLET_ALPHA, DIRICHLET_SEED
                )
                PARTITIONER_CACHE[cache_key] = (partitioner, full_dataset)
                log(INFO, f"[FL] Partitioner initialized: {self.num_clients} partitions, alpha={DIRICHLET_ALPHA}")
            
            save_all_clients_data_heterogeneity(self.client_configs, PARTITIONER_CACHE) # Save data heterogeneity for all clients at FL start
        except Exception as e:
            log(WARNING, f"[FL] Failed to pre-initialize partitioner: {e}")

        print_client_data_distribution_summary(self.client_configs, PARTITIONER_CACHE)

        # <----------------------------- TRAINING STARTS BASED ON THE NUMBER OF ROUNDS ----------------------------->
        for current_round in range(1, num_rounds + 1):
            log(INFO, f"\n{'='*70}\n[ROUND {current_round}/{num_rounds}]\n{'='*70}")

            # <----------------------------- Configure and aggregate training ----------------------------->
            train_msgs = self.configure_train(current_round, arrays, train_config, grid)
            train_replies = grid.send_and_receive(messages=train_msgs, timeout=timeout)
            agg_arrays, agg_metrics, training_metrics = self.aggregate_train(current_round, train_replies) # agg_arrays is the new consensus which will then be passed back to the clients

            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays
            if agg_metrics:
                result.train_metrics_clientapp[current_round] = agg_metrics
            if training_metrics:
                try:
                    save_round_training_metrics(current_round, training_metrics, self.client_configs, PARTITIONER_CACHE)
                except Exception as e:
                    log(WARNING, f"[ROUND {current_round}] Failed to save training metrics: {e}")

            try:
                # <----------------------------- Evaluate model on validation set ----------------------------->
                round_val_metrics = self.evaluate_all_clients_on_validation(device)
                save_round_validation_metrics(current_round, round_val_metrics, self.client_configs)
                self.round_metrics_history[f"round_{current_round}_validation"] = round_val_metrics

                for client_id, metrics in round_val_metrics.items():
                    if client_id not in self.client_history:
                        self.client_history[client_id] = []
                    self.client_history[client_id].append({'round': current_round, 'metrics': metrics})

                check_for_degradation_warnings(round_val_metrics, current_round, self.client_history)

                # <----------------------------- Best model checkpoint tracking ----------------------------->
                valid_accs = [
                    (m.get('leukemia_accuracy', 0) + m.get('healthy_accuracy', 0)) / 2
                    for m in round_val_metrics.values()
                    if m.get('leukemia_accuracy') is not None and m.get('healthy_accuracy') is not None
                ]
                if valid_accs:
                    avg_acc = sum(valid_accs) / len(valid_accs)
                    if avg_acc > self.best_avg_accuracy:
                        self.best_avg_accuracy = avg_acc
                        self.best_round = current_round
                        log(INFO, f"[BEST] New best avg balanced accuracy {avg_acc:.2%} at round {current_round}")
                        for client_cfg in self.client_configs:
                            src = client_cfg.get('model_path', '')
                            if src and os.path.exists(src):
                                base, ext = os.path.splitext(src)
                                dst = f"{base}_best{ext}"
                                shutil.copy2(src, dst)
                    else:
                        log(INFO, f"[BEST] Round {current_round} avg balanced accuracy {avg_acc:.2%} "
                                  f"(best: {self.best_avg_accuracy:.2%} at round {self.best_round})")

            except Exception as e:
                log(WARNING, f"[ROUND {current_round}] Validation evaluation failed: {e}")

        # <----------------------------- Restore best model checkpoints before final evaluation ----------------------------->
        if self.best_round > 0 and self.best_round < num_rounds:
            log(INFO, f"[BEST] Restoring best models from round {self.best_round} "
                      f"(avg accuracy: {self.best_avg_accuracy:.2%})")
            for client_cfg in self.client_configs:
                src = client_cfg.get('model_path', '')
                if src:
                    base, ext = os.path.splitext(src)
                    best_path = f"{base}_best{ext}"
                    if os.path.exists(best_path):
                        shutil.copy2(best_path, src)
                        log(INFO, f"[BEST] Restored {os.path.basename(src)} from round {self.best_round}")
        else:
            log(INFO, f"[BEST] Final round {num_rounds} was the best — no restore needed")

        log(INFO, f"\n{'='*70}\n[GLOBAL] Final Federated Model Evaluation\n{'='*70}")

        try:
            # <----------------------------- Evaluate model on public test set (unseen data) ----------------------------->
            global_post_fl_metrics = self.evaluate_all_clients_on_public_test(device)
            save_global_post_fl_metrics(global_post_fl_metrics, self.client_configs, self.best_round)
        except Exception as e:
            log(WARNING, f"[GLOBAL] Post-FL Evaluation failed: {e}")

        log(INFO, f"Strategy execution finished in {time.time() - t_start:.2f}s")
        return result

    # <----------------------------- Aggregate training results ----------------------------->
    def aggregate_train(self, server_round: int, results: Iterable[Message], **kwargs):
        results_list = list(results)
        if not results_list:
            return None, {}, {}

        # FIX: Sort the results by client_id to prevent metrics from swapping between clients
        results_list.sort(key=lambda msg: msg.content.get("metrics", {}).get("client_id", 999) if msg.has_content() else 999)

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
            server_round, 
        )

        if consensus_logits is None:
            return None, {}, {}

        metrics_aggregated = {
            "consensus_round": server_round, "num_clients": len(logits_list),
            "client_names": client_names, "weights": aggregation_metadata['normalized_weights'],
            "aggregation_details": aggregation_metadata
        }

        training_metrics_for_persistence = extract_training_metrics_for_persistence(
            client_metrics_list, aggregation_metadata, self.client_configs
        )

        return ArrayRecord([consensus_logits]), metrics_aggregated, training_metrics_for_persistence

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

    def configure_evaluate(self, server_round: int, arrays, config, grid) -> Iterable[Message]:
        return []

    def aggregate_evaluate(self, server_round: int, results: Iterable[Message], **kwargs):
        return None, None

    def summary(self) -> str:
        return ""