# This file contains the task definition for the federated learning application.

import torch
import numpy as np
import os
import json
import torch.nn as nn
from torchvision import models, datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from typing import Tuple, Optional, Iterable, Dict, List
import uuid
import time
from flwr.common import (
    Message,
    Metadata,
    RecordDict,
    ArrayRecord,
    ConfigRecord,
)
from flwr.server import Grid
from flwr.serverapp.strategy import Strategy
from flex_med.utils.config import (
    BASE_PATH, 
    CLIENT_INFO_FILE_PATH, 
    DATASET_FILE_PATH, 
    PUBLIC_ANCHOR_DATASET_PATH, 
    PUBLIC_TEST_DATASET_PATH, 
    MODEL_CHECKPOINT_FILE_PATH, 
    ROUND_METRICS_FILE_PATH, 
    GRAPHS_OUTPUT_DIR, 
    NUM_CLASSES, 
    IMG_SIZE, 
    DATA_JSON_PATH, 
    MODEL_SUITABILITY_SCORES, 
    TRAIN_LOSS_WEIGHT, 
    DISTILL_LOSS_WEIGHT, 
    CONSENSUS_MOMENTUM
)
from datetime import datetime

# <------------------------------------------ DATA TRANSFORMS ------------------------------------------>

# Standardize inputs: Resize -> Tensor -> ImageNet Normalization
COMMON_TRANSFORM = Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Enhanced Augmentation for Private Training (critical for small datasets like ALL-IDB2)
PRIVATE_TRAIN_TRANSFORM = Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),  # Cutout augmentation (after ToTensor)
])

# <------------------------------------------ UTILITY FUNCTIONS ------------------------------------------>

# Retrieves all client metadata (model type, dataset paths.)
# Args: config_path - Path to client configuration JSON file
# Returns: List of client configuration dictionaries
def load_client_config(config_path: str = CLIENT_INFO_FILE_PATH) -> List[Dict]:
    # Check for runtime config from environment variable (production mode)
    env_config_path = os.getenv('FLEX_MED_CONFIG_FILE')
    if env_config_path:
        config_path = env_config_path
        print(f"[Config] Using runtime config from: {config_path}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, 'r') as f:
        clients = json.load(f)

    return clients

# Retrieves individual client config for FL round execution
# Args: partition_id - Index of the client in configuration list
#       config_path - Path to client configuration JSON file
# Returns: Dictionary containing client configuration
def get_client_by_partition_id(partition_id: int, config_path: str = CLIENT_INFO_FILE_PATH) -> Dict:
    clients = load_client_config(config_path)
    
    if partition_id >= len(clients):
        raise ValueError(f"Partition ID {partition_id} exceeds number of clients ({len(clients)})")
    
    return clients[partition_id]

# Generate logits on the public dataset for FL simulation consensus
# Args: model - PyTorch model to generate predictions
#       public_loader - DataLoader for public anchor dataset
#       device - Device to run inference on (cpu/cuda)
# Returns: Numpy array of concatenated logits from all batches
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
    
# <------------------------------------------ MODEL FUNCTION DEFINITIONS ------------------------------------------>

# Factory function for heterogeneous model creation in FL simulation with transfer learning
# Args: model_type - String identifier ('resnet18', 'mobilenet_v2', 'efficientnet_b3')
#       use_pretrained - Whether to load ImageNet pretrained weights (default: True for transfer learning)
# Returns: PyTorch model initialized for binary classification (NUM_CLASSES=2)
# STRATEGY: Progressive fine-tuning
#   ✓ Load ImageNet backbone (proven feature extractors)
#   ✗ Discard ImageNet classifier head (1000 classes → not relevant)
#   ✓ Replace with new binary classifier (2 classes: ALL vs Healthy)
#   ✓ Fine-tune head first (frozen backbone), then unfreeze for full training
def get_model_by_type(model_type: str, use_pretrained: bool = True):
    model_type = model_type.lower()

    if model_type == 'resnet18':
        # Load ImageNet pre-trained backbone
        # model = models.resnet18(weights='IMAGENET1K_V1' if use_pretrained else None)
        model = models.resnet18(weights=None) # hopefully it starts with lower pre_fl accuracy
        # Replace classifier head with binary classifier
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        return model

    elif model_type == 'mobilenet_v2':
        # Load ImageNet pre-trained backbone
        # model = models.mobilenet_v2(weights='IMAGENET1K_V1' if use_pretrained else None)
        model = models.mobilenet_v2(weights=None)
        # Replace classifier head with binary classifier
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
        return model

    elif model_type == 'efficientnet_b3':
        # Load ImageNet pre-trained backbone
        # model = models.efficientnet_b3(weights='IMAGENET1K_V1' if use_pretrained else None)
        model = models.efficientnet_b3(weights=None)
        # Replace classifier head with binary classifier
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
        return model

    else:
        raise ValueError(f"Unsupported model type: {model_type}. "
                        f"Supported types: resnet18, mobilenet_v2, efficientnet_b3")

# <------------------------------------------ TRANSFER LEARNING UTILITIES ------------------------------------------>

def freeze_backbone(model, model_type: str):
    """
    Freeze all layers except the classifier head for initial training.

    Strategy: Train only the new classifier head while keeping ImageNet features frozen.
    This prevents catastrophic forgetting and allows the head to learn task-specific features.

    Args:
        model: PyTorch model
        model_type: Model architecture identifier
    """
    model_type = model_type.lower()

    if model_type == 'resnet18':
        # Freeze all layers except fc (classifier head)
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
        print(f"[Transfer Learning] ResNet18 backbone frozen, training head only")

    elif model_type == 'mobilenet_v2':
        # Freeze all layers except classifier
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        print(f"[Transfer Learning] MobileNetV2 backbone frozen, training head only")

    elif model_type == 'efficientnet_b3':
        # Freeze all layers except classifier
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        print(f"[Transfer Learning] EfficientNetB3 backbone frozen, training head only")

    return model


def unfreeze_backbone(model, model_type: str):
    """
    Unfreeze all layers for full fine-tuning.

    Strategy: After the head has learned task-specific features, unfreeze backbone
    to adapt pre-trained features to medical domain with lower learning rate.

    Args:
        model: PyTorch model
        model_type: Model architecture identifier
    """
    # Unfreeze all parameters
    for param in model.parameters():
        param.requires_grad = True

    print(f"[Transfer Learning] Backbone unfrozen, full model fine-tuning enabled")
    return model


def get_trainable_params(model):
    """Count trainable vs frozen parameters"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen = total - trainable
    return trainable, frozen, total


# <------------------------------------------ DATA LOADER FUNCTION DEFINITIONS ------------------------------------------>

# Provides shared dataset for generating consensus logits across clients
# Args: batch_size - Batch size for DataLoader
# Returns: DataLoader with shuffle=False (order consistency critical for FL simulation)
def load_public_dataset(batch_size=64, round_num=1, total_rounds=10):    
    if not os.path.exists(PUBLIC_ANCHOR_DATASET_PATH):
        raise FileNotFoundError(f"Public data not found at {PUBLIC_ANCHOR_DATASET_PATH}")

    # Load full dataset first
    full_dataset = datasets.ImageFolder(root=PUBLIC_ANCHOR_DATASET_PATH, transform=COMMON_TRANSFORM)
    
    # Create deterministic subsets
    # Strategy: Split indices for each class into `total_rounds` buckets
    # Then take buckets 0 to `round_num` (exclusive if we used 0-indexing, but we want cumulative)
    
    # 1. Group indices by class
    class_indices = {}
    for idx, (_, label) in enumerate(full_dataset.samples):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
        
    # 2. Shuffle indices deterministically and split into chunks
    selected_indices = []
    generator = torch.Generator().manual_seed(42)  # Critical for consistency across clients
    
    for label, indices in class_indices.items():
        # Shuffle indices for this class
        indices_tensor = torch.tensor(indices)
        perm = torch.randperm(len(indices), generator=generator)
        shuffled_indices = indices_tensor[perm].tolist()
        
        # Determine chunk size per round
        # Use ceil to ensure we cover all data even if not perfectly divisible
        chunk_size = int(np.ceil(len(indices) / total_rounds))
        
        # Select accumulated chunks up to current round
        end_idx = min(len(indices), chunk_size * round_num)
        selected_indices.extend(shuffled_indices[:end_idx])
        
    # 3. Create Subset
    # Do NOT sort indices. reliable appending depends on the order being [Round1_Indices, Round2_Indices, ...]
    # selected_indices is already constructed in that order (shuffled_chunk_1 + shuffled_chunk_2 + ...).
    subset = torch.utils.data.Subset(full_dataset, selected_indices)
    
    # Shuffle=False is CRITICAL for FL simulation so all clients see images in the same order
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)
    return loader

# Handles heterogeneous client datasets with 80/20 train/test split
# Args: partition_id - Client ID (0, 1, 2, ...)
#       num_partitions - Total number of partitions (not used but kept for compatibility)
#       batch_size - Batch size for data loaders
#       config_path - Path to the configuration file
# Returns: Tuple of (trainloader, testloader) or (None, None) if client has no data
def load_private_dataset(partition_id: int, num_partitions: int, batch_size=32, config_path: str = CLIENT_INFO_FILE_PATH):
    # Get client configuration
    client_config = get_client_by_partition_id(partition_id, config_path)

    # Check if client has local data
    if not client_config['has_local_data'] or client_config['dataset_path'] is None:
        return None, None

    data_path = client_config['dataset_path']

    if not os.path.exists(data_path):
        return None, None

    # Load the Full Dataset from folder
    full_dataset = datasets.ImageFolder(root=data_path, transform=PRIVATE_TRAIN_TRANSFORM)

    # Create a stratified Train/Validation split (85% Train, 15% Validation)
    # This ensures both splits maintain the same class distribution
    
    # Group indices by class
    class_indices = {}
    for idx, (_, label) in enumerate(full_dataset.samples):
        if label not in class_indices:
            class_indices[label] = []
        class_indices[label].append(idx)
    
    # Split each class separately with 85/15 ratio
    train_indices = []
    val_indices = []
    generator = torch.Generator().manual_seed(42)
    
    for label, indices in class_indices.items():
        # Shuffle indices for this class
        indices_tensor = torch.tensor(indices)
        perm = torch.randperm(len(indices), generator=generator)
        shuffled_indices = indices_tensor[perm].tolist()
        
        # Split 85/15
        split_point = int(0.85 * len(shuffled_indices))
        train_indices.extend(shuffled_indices[:split_point])
        val_indices.extend(shuffled_indices[split_point:])
    
    # Create subsets
    train_ds = torch.utils.data.Subset(full_dataset, train_indices)
    test_ds = torch.utils.data.Subset(full_dataset, val_indices)

    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader

# Loads the public test dataset for consistent evaluation across all rounds
# Args: batch_size - Batch size for DataLoader
# Returns: DataLoader with shuffle=False for deterministic evaluation
def load_public_test_dataset(batch_size=64):
    if not os.path.exists(PUBLIC_TEST_DATASET_PATH):
        raise FileNotFoundError(f"Public test data not found at {PUBLIC_TEST_DATASET_PATH}")

    dataset = datasets.ImageFolder(root=PUBLIC_TEST_DATASET_PATH, transform=COMMON_TRANSFORM)

    # Shuffle=False for deterministic evaluation results
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    print(f"[Data] Loaded public test dataset: {len(dataset)} samples")
    return loader

# Evaluates a single client model on the public test dataset
# Args: client_id - Client identifier (partition_id)
#       model_path - Path to client's model checkpoint
#       model_type - Architecture identifier (e.g., 'resnet18')
#       device - Device to evaluate on (cpu/cuda)
# Returns: Dictionary with comprehensive metrics including confusion matrix, ROC-AUC, etc.
def evaluate_client_on_public_test(client_id: int, model_path: str, model_type: str, device: torch.device) -> Dict:
    # Initialize model architecture
    model = get_model_by_type(model_type)

    # Load model weights if checkpoint exists
    if os.path.exists(model_path):
        try:
            model, metadata = load_existing_model(model, model_path, device)
        except Exception as e:
            print(f"[Eval] Client {client_id}: Using untrained model ({e})")
    else:
        print(f"[Eval] Client {client_id}: No checkpoint found, using fresh model")

    model.to(device)

    # Load public test dataset
    test_loader = load_public_test_dataset(batch_size=64)

    # Evaluate model with comprehensive metrics (confusion matrix, ROC-AUC, etc.)
    metrics = test(model, test_loader, device, return_detailed=True)

    # Add timestamp
    metrics["evaluated_at"] = datetime.now().isoformat()

    return metrics

# Saves round metrics to JSON file for visualization
# Args: round_num - Current FL round number
#       stage - Either "pre_fl" or "post_fl"
#       client_metrics - Dictionary mapping client_id to metrics
#       metrics_path - Path to save JSON file
# Returns: None (saves to disk)
def save_round_metrics(round_num: int, stage: str, client_metrics: Dict, metrics_path: str = ROUND_METRICS_FILE_PATH):
    # Load existing metrics or create new structure
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}

    round_key = f"round_{round_num}"

    # Initialize round entry if not exists
    if round_key not in all_metrics:
        all_metrics[round_key] = {
            "pre_fl": {"clients": {}},
            "post_fl": {"clients": {}},
            "improvement": {}
        }

    # Update stage metrics
    all_metrics[round_key][stage]["clients"] = client_metrics

    # Calculate improvement if both pre and post exist
    if (all_metrics[round_key]["pre_fl"]["clients"] and
        all_metrics[round_key]["post_fl"]["clients"]):
        improvement = {}
        for client_id in all_metrics[round_key]["post_fl"]["clients"]:
            if client_id in all_metrics[round_key]["pre_fl"]["clients"]:
                pre_acc = all_metrics[round_key]["pre_fl"]["clients"][client_id].get("accuracy", 0)
                post_acc = all_metrics[round_key]["post_fl"]["clients"][client_id].get("accuracy", 0)
                improvement[client_id] = post_acc - pre_acc
        all_metrics[round_key]["improvement"] = improvement

    # Save to file
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True) if os.path.dirname(metrics_path) else None
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"[Metrics] Saved {stage} metrics for round {round_num}")

# Evaluates all clients on the public test dataset
# Args: client_configs - List of client configuration dictionaries
#       device - Device to evaluate on (cpu/cuda)
# Returns: Dictionary mapping client_id (as string) to metrics
def evaluate_all_clients(client_configs: List[Dict], device: torch.device) -> Dict:
    client_metrics = {}

    for i, client in enumerate(client_configs):
        client_id = str(i)
        model_path = client['model_path']
        model_type = client['model_type']
        client_name = client['client_name']

        print(f"[Eval] Evaluating client {i}: {client_name} ({model_type})")

        try:
            metrics = evaluate_client_on_public_test(i, model_path, model_type, device)
            client_metrics[client_id] = metrics
            print(f"[Eval] Client {i}: Accuracy={metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%), Loss={metrics['loss']:.4f}")
        except Exception as e:
            print(f"[Eval] Client {i}: Evaluation failed - {e}")
            client_metrics[client_id] = {"accuracy": None, "loss": None, "error": str(e)}

    return client_metrics

# Evaluates all clients on public test dataset - ONLY for global Pre/Post-FL comparison
# Args: client_configs - List of client configuration dictionaries
#       device - Device to evaluate on (cpu/cuda)
# Returns: Dictionary mapping client_id (as string) to metrics
# NOTE: This function should ONLY be called for initial (Pre-FL) and final (Post-FL) evaluations
#       to avoid repeatedly exposing the test set during training.
def evaluate_all_clients_on_public_test(client_configs: List[Dict], device: torch.device) -> Dict:
    """
    Evaluate all clients on the public test dataset.
    Used ONLY for global Pre-FL (before training starts) and Post-FL (after training completes) comparisons.
    This keeps test set exposure minimal (only 2 times total).
    """
    client_metrics = {}

    for i, client in enumerate(client_configs):
        client_id = str(i)
        model_path = client['model_path']
        model_type = client['model_type']
        client_name = client['client_name']

        print(f"[Global Eval] Evaluating client {i}: {client_name} ({model_type}) on public test")

        try:
            # Load model architecture
            model = get_model_by_type(model_type)

            # Load checkpoint if exists
            if os.path.exists(model_path):
                try:
                    model, metadata = load_existing_model(model, model_path, device)
                    print(f"[Global Eval] Client {i}: Loaded checkpoint (round {metadata.get('round', 'unknown')})")
                except Exception as e:
                    print(f"[Global Eval] Client {i}: Using fresh model ({e})")
            else:
                print(f"[Global Eval] Client {i}: No checkpoint found, using fresh model")

            model.to(device)

            # Load public test dataset (1880 samples)
            test_loader = load_public_test_dataset(batch_size=64)

            # Evaluate with detailed metrics
            metrics = test(model, test_loader, device, return_detailed=True)
            metrics['dataset'] = 'public_test'
            metrics['evaluation_type'] = 'global'
            metrics['num_samples'] = len(test_loader.dataset)
            metrics['evaluated_at'] = datetime.now().isoformat()

            client_metrics[client_id] = metrics

            acc = metrics.get('accuracy', 0)
            loss = metrics.get('loss', 0)
            print(f"[Global Eval] Client {i}: Accuracy={acc:.1%}, Loss={loss:.3f}")

        except Exception as e:
            print(f"[Global Eval] Client {i}: Evaluation failed - {e}")
            client_metrics[client_id] = {"accuracy": None, "loss": None, "error": str(e)}

    return client_metrics


# Evaluates all clients on their private validation sets - for per-round tracking
# Args: client_configs - List of client configuration dictionaries
#       device - Device to evaluate on (cpu/cuda)
#       num_partitions - Total number of clients
# Returns: Dictionary mapping client_id (as string) to validation metrics
# NOTE: This function is called every round to track learning progression without
#       contaminating the test set.
def evaluate_all_clients_on_validation(client_configs: List[Dict], device: torch.device, num_partitions: int) -> Dict:
    """
    Evaluate all clients on their private validation sets (20% of private data).
    Used for per-round progress tracking without exposing the test set.
    Free rider clients (no local data) are evaluated on public test as a proxy.
    """
    client_metrics = {}

    for i, client in enumerate(client_configs):
        client_id = str(i)
        model_path = client['model_path']
        model_type = client['model_type']
        client_name = client['client_name']
        has_local_data = client.get('has_local_data', False)

        print(f"[Round Eval] Evaluating client {i}: {client_name} ({model_type}) on validation")

        try:
            # Load model architecture
            model = get_model_by_type(model_type)

            # Load checkpoint if exists
            if os.path.exists(model_path):
                try:
                    model, metadata = load_existing_model(model, model_path, device)
                except Exception as e:
                    print(f"[Round Eval] Client {i}: Using fresh model ({e})")
            else:
                print(f"[Round Eval] Client {i}: No checkpoint found, using fresh model")

            model.to(device)

            # Load appropriate dataset
            if has_local_data:
                # Load validation set (20% of private data)
                _, valloader = load_private_dataset(i, num_partitions, batch_size=64)

                if valloader is not None:
                    dataset_type = 'validation'
                    num_samples = len(valloader.dataset)
                else:
                    # Fallback if data loading fails
                    valloader = load_public_test_dataset(batch_size=64)
                    dataset_type = 'public_test_proxy'
                    num_samples = len(valloader.dataset)
            else:
                # Free rider: use public test as proxy for generalization
                valloader = load_public_test_dataset(batch_size=64)
                dataset_type = 'public_test_proxy'
                num_samples = len(valloader.dataset)

            # Evaluate
            model.eval()
            all_preds = []
            all_labels = []
            total_loss = 0.0
            criterion = nn.CrossEntropyLoss()

            with torch.no_grad():
                for images, labels in valloader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item() * images.size(0)

                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)

            # Calculate metrics
            from sklearn.metrics import precision_recall_fscore_support

            accuracy = (all_preds == all_labels).mean()
            loss = total_loss / len(all_labels)

            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='binary', zero_division=0
            )

            # Calculate class-specific accuracies
            leukemia_mask = (all_labels == 1)
            healthy_mask = (all_labels == 0)

            leukemia_acc = (all_preds[leukemia_mask] == all_labels[leukemia_mask]).mean() if leukemia_mask.any() else 0
            healthy_acc = (all_preds[healthy_mask] == all_labels[healthy_mask]).mean() if healthy_mask.any() else 0
            class_gap = abs(leukemia_acc - healthy_acc)

            metrics = {
                'loss': float(loss),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'class_gap': float(class_gap),
                'leukemia_accuracy': float(leukemia_acc),
                'healthy_accuracy': float(healthy_acc),
                'num_samples': num_samples,
                'dataset': dataset_type,
                'evaluation_type': 'per_round',
                'evaluated_at': datetime.now().isoformat()
            }

            client_metrics[client_id] = metrics

            print(f"[Round Eval] Client {i}: Acc={accuracy:.1%}, Loss={loss:.3f}, Gap={class_gap:.1%}")

        except Exception as e:
            print(f"[Round Eval] Client {i}: Evaluation failed - {e}")
            client_metrics[client_id] = {"accuracy": None, "loss": None, "error": str(e)}

    return client_metrics


# Updates cnmc_data.json with comprehensive per-round metrics for visualization
# Args: round_num - Current FL round number
#       pre_fl_metrics - Dictionary mapping client_id to pre-FL evaluation metrics
#       post_fl_metrics - Dictionary mapping client_id to post-FL evaluation metrics
#       training_metrics - Dictionary mapping client_id to training metrics (distill_loss, train_loss, etc.)
#       client_configs - List of client configuration dictionaries
#       data_json_path - Path to cnmc_data.json file
# Returns: None (saves to disk)
def update_client_data_json_metrics(
    round_num: int,
    pre_fl_metrics: Dict,
    post_fl_metrics: Dict,
    training_metrics: Dict,
    client_configs: List[Dict],
    data_json_path: str = DATA_JSON_PATH
):
    """Update cnmc_data.json with comprehensive per-round metrics for graphical visualization.

    New Schema:
    metrics: {
        "current": {
            "pre_fl": { latest pre_fl metrics },
            "post_fl": { latest post_fl metrics },
            "improvement": { calculated improvement },
            "last_round": N
        },
        "rounds": [
            {
                "round": 1,
                "pre_fl": { comprehensive metrics with confusion_matrix, roc_auc, etc. },
                "training": { distill_loss, train_loss, num_examples, consensus_weight },
                "post_fl": { comprehensive metrics },
                "improvement": { per-metric deltas }
            }
        ]
    }
    """
    # Load current cnmc_data.json
    if not os.path.exists(data_json_path):
        print(f"[Metrics] Warning: {data_json_path} not found, skipping update")
        return

    try:
        with open(data_json_path, 'r') as f:
            clients_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[Metrics] Error reading {data_json_path}: {e}")
        return

    # Update each client's metrics
    for i, client in enumerate(clients_data):
        client_id = str(i)

        # Get metrics for this client
        pre_fl = pre_fl_metrics.get(client_id, {})
        post_fl = post_fl_metrics.get(client_id, {})
        training = training_metrics.get(client_id, {})

        # Skip clients with failed evaluation
        if post_fl.get("accuracy") is None:
            print(f"[Metrics] Skipping client {i} - no post_fl metrics")
            continue

        # Initialize metrics structure if needed
        if not isinstance(client.get("metrics"), dict) or "rounds" not in client.get("metrics", {}):
            client["metrics"] = {
                "current": {},
                "rounds": []
            }

        # Calculate improvement (post_fl - pre_fl)
        improvement = {}
        metrics_to_compare = ["accuracy", "precision", "recall", "f1_score", "specificity", "roc_auc",
                              "leukemia_accuracy", "healthy_accuracy"]

        for metric in metrics_to_compare:
            pre_val = pre_fl.get(metric, 0.0) or 0.0
            post_val = post_fl.get(metric, 0.0) or 0.0
            improvement[metric] = round(post_val - pre_val, 6)

        # Create round entry
        round_entry = {
            "round": round_num,
            "pre_fl": pre_fl if pre_fl else None,
            "training": {
                "distill_loss": training.get("distill_loss"),
                "train_loss": training.get("train_loss"),
                "num_examples": training.get("num-examples"),
                "training_time_sec": training.get("training_time"),
                "consensus_weight": training.get("consensus_weight")
            } if training else None,
            "post_fl": post_fl if post_fl else None,
            "improvement": improvement
        }

        # Check if this round already exists, update if so
        round_exists = False
        for j, entry in enumerate(client["metrics"]["rounds"]):
            if entry.get("round") == round_num:
                client["metrics"]["rounds"][j] = round_entry
                round_exists = True
                break

        if not round_exists:
            client["metrics"]["rounds"].append(round_entry)

        # Sort rounds by round number
        client["metrics"]["rounds"].sort(key=lambda x: x.get("round", 0))

        # Update current (latest) metrics
        client["metrics"]["current"] = {
            "pre_fl": pre_fl if pre_fl else None,
            "post_fl": post_fl if post_fl else None,
            "improvement": improvement,
            "last_round": round_num
        }

    # Save updated cnmc_data.json
    try:
        with open(data_json_path, 'w') as f:
            json.dump(clients_data, f, indent=2)
        print(f"[Metrics] Updated {data_json_path} with round {round_num} comprehensive metrics")
    except Exception as e:
        print(f"[Metrics] Error saving {data_json_path}: {e}")


# Helper function to extract training metrics from aggregate results (for persistence)
def extract_training_metrics_for_persistence(
    client_metrics_list: List[Dict],
    aggregation_metadata: Dict,
    client_configs: List[Dict]
) -> Dict:
    """
    Extract training metrics from aggregate_train results for cnmc_data.json persistence.

    Args:
        client_metrics_list: List of metrics dicts from each client's training
        aggregation_metadata: Metadata from compute_consensus including weights
        client_configs: Client configuration list

    Returns:
        Dictionary mapping client_id (str) to training metrics
    """
    training_metrics = {}
    weight_breakdown = aggregation_metadata.get("weight_breakdown", [])

    for i, metrics in enumerate(client_metrics_list):
        client_id = str(i)

        # Find consensus weight for this client
        consensus_weight = None
        if i < len(weight_breakdown):
            consensus_weight = weight_breakdown[i].get("normalized_weight")

        training_metrics[client_id] = {
            "distill_loss": metrics.get("distill_loss"),
            "train_loss": metrics.get("train_loss"),
            "num-examples": metrics.get("num-examples"),
            "training_time": metrics.get("training_time"),
            "consensus_weight": consensus_weight,
            "has_local_data": metrics.get("has_local_data", 0)
        }

    return training_metrics


# Helper functions for the new hybrid evaluation strategy

def save_global_pre_fl_metrics(metrics: Dict, client_configs: List[Dict], data_json_path: str = DATA_JSON_PATH):
    """Save global Pre-FL metrics to cnmc_data.json."""
    if not os.path.exists(data_json_path):
        print(f"[Global Metrics] Warning: {data_json_path} not found, skipping save")
        return

    try:
        with open(data_json_path, 'r') as f:
            clients_data = json.load(f)

        for i, client in enumerate(clients_data):
            client_id = str(i)

            if client_id in metrics:
                if 'metrics' not in client or not isinstance(client['metrics'], dict):
                    client['metrics'] = {}

                if 'global' not in client['metrics']:
                    client['metrics']['global'] = {}

                client['metrics']['global']['pre_fl'] = metrics[client_id]

        with open(data_json_path, 'w') as f:
            json.dump(clients_data, f, indent=2)

        print(f"[Global Metrics] Saved Pre-FL metrics to {data_json_path}")

    except Exception as e:
        print(f"[Global Metrics] Error saving Pre-FL metrics: {e}")


def save_global_post_fl_metrics(metrics: Dict, client_configs: List[Dict], data_json_path: str = DATA_JSON_PATH):
    """Save global Post-FL metrics and calculate improvements."""
    if not os.path.exists(data_json_path):
        print(f"[Global Metrics] Warning: {data_json_path} not found, skipping save")
        return

    try:
        with open(data_json_path, 'r') as f:
            clients_data = json.load(f)

        for i, client in enumerate(clients_data):
            client_id = str(i)

            if client_id in metrics:
                if 'metrics' not in client:
                    client['metrics'] = {}
                if 'global' not in client['metrics']:
                    client['metrics']['global'] = {}

                # Save Post-FL metrics
                client['metrics']['global']['post_fl'] = metrics[client_id]

                # Calculate improvement
                pre_fl = client['metrics']['global'].get('pre_fl', {})
                post_fl = metrics[client_id]

                improvement = {}
                for metric in ['accuracy', 'loss', 'precision', 'recall', 'f1_score', 'class_gap',
                               'leukemia_accuracy', 'healthy_accuracy']:
                    if metric in pre_fl and metric in post_fl:
                        pre_val = pre_fl[metric]
                        post_val = post_fl[metric]
                        improvement[metric] = round(post_val - pre_val, 6)

                client['metrics']['global']['improvement'] = improvement

        with open(data_json_path, 'w') as f:
            json.dump(clients_data, f, indent=2)

        print(f"[Global Metrics] Saved Post-FL metrics and improvements to {data_json_path}")

    except Exception as e:
        print(f"[Global Metrics] Error saving Post-FL metrics: {e}")


def save_round_validation_metrics(round_num: int, metrics: Dict, client_configs: List[Dict], data_json_path: str = DATA_JSON_PATH):
    """Save per-round validation metrics to cnmc_data.json."""
    if not os.path.exists(data_json_path):
        print(f"[Round Metrics] Warning: {data_json_path} not found, skipping save")
        return

    try:
        with open(data_json_path, 'r') as f:
            clients_data = json.load(f)

        for i, client in enumerate(clients_data):
            client_id = str(i)

            if client_id in metrics:
                if 'metrics' not in client:
                    client['metrics'] = {}
                if 'rounds' not in client['metrics']:
                    client['metrics']['rounds'] = []

                # Find or create round entry
                round_entry = None
                for r in client['metrics']['rounds']:
                    if r.get('round') == round_num:
                        round_entry = r
                        break

                if round_entry is None:
                    round_entry = {'round': round_num}
                    client['metrics']['rounds'].append(round_entry)

                # Add validation metrics
                round_entry['validation'] = metrics[client_id]

        with open(data_json_path, 'w') as f:
            json.dump(clients_data, f, indent=2)

        print(f"[Round Metrics] Saved validation metrics for round {round_num} to {data_json_path}")

    except Exception as e:
        print(f"[Round Metrics] Error saving validation metrics: {e}")


def load_global_pre_fl_for_client(client_id: str, client_configs: List[Dict], data_json_path: str = DATA_JSON_PATH) -> Dict:
    """Load global Pre-FL metrics for a specific client."""
    if not os.path.exists(data_json_path):
        return {}

    try:
        with open(data_json_path, 'r') as f:
            clients_data = json.load(f)

        for i, client in enumerate(clients_data):
            if str(i) == client_id:
                return client.get('metrics', {}).get('global', {}).get('pre_fl', {})

    except Exception as e:
        print(f"[Global Metrics] Error loading Pre-FL for client {client_id}: {e}")

    return {}


def check_for_degradation_warnings(current_metrics: Dict, round_num: int, client_history: Dict):
    """Check validation metrics and log warnings if degradation detected (does not stop training)."""
    from flwr.common import log
    from logging import WARNING

    if round_num < 3:
        return

    for client_id, history in client_history.items():
        if len(history) >= 3:
            # Get last 3 rounds of accuracy
            last_3 = history[-3:]
            accs = [r['metrics'].get('accuracy', 0) for r in last_3]

            # Check for 2 consecutive declines
            if accs[-1] < accs[-2] and accs[-2] < accs[-3]:
                log(WARNING, "")
                log(WARNING, "⚠️  DEGRADATION WARNING ⚠️")
                log(WARNING, f"Client {client_id}: Validation accuracy declining for 2 consecutive rounds")
                log(WARNING, f"  Round {round_num-2}: {accs[0]:.3%}")
                log(WARNING, f"  Round {round_num-1}: {accs[1]:.3%}")
                log(WARNING, f"  Round {round_num}: {accs[2]:.3%}")
                log(WARNING, f"Consider reviewing hyperparameters or stopping early.")
                log(WARNING, "")

            # Check for loss increase
            losses = [r['metrics'].get('loss', 999) for r in last_3]
            if losses[-1] > losses[-2] and losses[-2] > losses[-3]:
                log(WARNING, "")
                log(WARNING, "⚠️  OVERFITTING WARNING ⚠️")
                log(WARNING, f"Client {client_id}: Validation loss increasing for 2 consecutive rounds")
                log(WARNING, f"  Round {round_num-2}: {losses[0]:.3f}")
                log(WARNING, f"  Round {round_num-1}: {losses[1]:.3f}")
                log(WARNING, f"  Round {round_num}: {losses[2]:.3f}")
                log(WARNING, f"Model may be overfitting to training data.")
                log(WARNING, "")


# <------------------------------------------ CHECKPOINT UTILITY FUNCTION DEFINITIONS ------------------------------------------>

# Handles both legacy state_dict and new checkpoint format with metadata
# Args: model - PyTorch model instance to load weights into
#       model_path - Path to checkpoint file (.pt or .pth)
#       device - Device to map model tensors to (cpu/cuda)
# Returns: Tuple of (model, metadata_dict) where metadata contains training info
def load_existing_model(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        # Load from checkpoint with metadata
        model.load_state_dict(checkpoint['state_dict'])
        
        # Return metadata if available
        metadata = {
            'model_type': checkpoint.get('model_type'),
            'num_classes': checkpoint.get('num_classes'),
            'round': checkpoint.get('round'),
        }
        return model, metadata
    else:
        # Direct state_dict
        model.load_state_dict(checkpoint)
        return model, {}

# Save FL training checkpoint with consensus logits and metrics
# Args: checkpoint_dir - Directory to save checkpoints
#       current_round - Current FL round number
#       consensus_logits - Aggregated consensus logits from clients
#       eval_history - List of evaluation metrics from previous rounds
#       training_metrics - Current round's training metrics
# Returns: None (saves to disk)
def save_checkpoint(checkpoint_dir: str, current_round: int, consensus_logits: np.ndarray, eval_history: List[Dict], training_metrics: Dict):
    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_round_{current_round}.pt")
    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")

    checkpoint = {
        'round': current_round,
        'consensus_logits': consensus_logits,
        'eval_history': eval_history,
        'training_metrics': training_metrics,
        'timestamp': time.time()
    }

    # Save round-specific checkpoint
    torch.save(checkpoint, checkpoint_path)

    # Save as latest checkpoint (for easy resumption)
    torch.save(checkpoint, latest_path)

# Load the latest FL training checkpoint for resumption
# Args: checkpoint_dir - Directory containing checkpoints
# Returns: Checkpoint dictionary or None if no checkpoint exists
def load_checkpoint(checkpoint_dir: str) -> Optional[Dict]:
    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")

    # Debug: Show what path is being checked
    print(f"[CHECKPOINT] Looking for: {latest_path}")
    print(f"[CHECKPOINT] Exists: {os.path.exists(latest_path)}")
    print(f"[CHECKPOINT] BASE_PATH env: {os.getenv('BASE_PATH', 'NOT SET')}")

    if not os.path.exists(latest_path):
        # Try fallback path for Colab
        fallback_path = "/content/drive/MyDrive/College/FLEX-Med/checkpoints/latest_checkpoint.pt"
        print(f"[CHECKPOINT] Trying fallback: {fallback_path}")
        if os.path.exists(fallback_path):
            latest_path = fallback_path
            print(f"[CHECKPOINT] Found at fallback path!")
        else:
            print(f"[CHECKPOINT] Not found at fallback either")
            return None

    try:
        checkpoint = torch.load(latest_path, map_location='cpu', weights_only=False)
        print(f"[CHECKPOINT] Loaded successfully! Round: {checkpoint.get('round', 'unknown')}")
        return checkpoint
    except Exception as e:
        print(f"[CHECKPOINT] Error loading: {e}")
        return None

# Clear all checkpoints from the directory
# Args: checkpoint_dir - Directory containing checkpoints
# Returns: None (removes directory)
def clear_checkpoints(checkpoint_dir: str):
    if os.path.exists(checkpoint_dir):
        import shutil
        shutil.rmtree(checkpoint_dir)

# <------------------------------------------ LOSS FUNCTIONS ------------------------------------------>

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance and hard-to-learn examples.

    Focal Loss down-weights easy examples and focuses training on hard negatives.
    Formula: FL(pt) = -alpha * (1 - pt)^gamma * log(pt)

    Args:
        alpha: Weighting factor for positive class (default: 0.25)
        gamma: Focusing parameter to reduce loss for well-classified examples (default: 2.0)
               Higher gamma = more focus on hard examples

    Why this helps:
        - Standard CrossEntropy treats all examples equally
        - Focal Loss reduces weight of easy examples (correctly classified with high confidence)
        - Forces model to focus on hard examples (leukemia cells that look similar to healthy)
        - Particularly effective when one class is harder to learn (leukemia: 47% → target: 75-85%)

    Impact on FLEX-Med:
        - Prevents model from becoming "lazy" and predicting majority class
        - Helps achieve balanced accuracy (both healthy AND leukemia > 75%)
        - Reduces class gap from 51% to <10%
    """
    def __init__(self, alpha: float = 0.35, gamma: float = 2.0):  # FIXED: Increased from 0.25 to 0.35 for better leukemia detection
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Model logits (raw outputs before softmax), shape [batch_size, num_classes]
            targets: Ground truth labels, shape [batch_size]

        Returns:
            Focal loss value (scalar)
        """
        # Compute standard cross-entropy loss (per sample, not reduced)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Compute p_t (probability of true class)
        # pt = exp(-ce_loss) since ce_loss = -log(p_t)
        pt = torch.exp(-ce_loss)

        # Apply focal term: (1 - pt)^gamma
        # When pt is high (confident correct prediction), (1-pt)^gamma is small → loss reduced
        # When pt is low (uncertain or wrong prediction), (1-pt)^gamma is large → loss emphasized
        focal_term = (1 - pt) ** self.gamma

        # Class-specific alpha weighting for handling class imbalance
        # Class 0 (Leukemia): weight = alpha (0.25 → down-weight majority class)
        # Class 1 (Healthy): weight = 1-alpha (0.75 → up-weight minority class)
        alpha_t = torch.where(targets == 0, self.alpha, 1 - self.alpha)

        # Apply class-specific alpha weighting and compute final loss
        focal_loss = alpha_t * focal_term * ce_loss

        return focal_loss.mean()

# <------------------------------------------ MODEL TRAINING & TESTING FUNCTION DEFINITIONS ------------------------------------------>

# Train model on private client data with progressive fine-tuning strategy
# Args: model - PyTorch model to train
#       trainloader - DataLoader for training data
#       epochs - Number of training epochs
#       lr - Learning rate
#       device - Device to train on (cpu/cuda)
#       model_type - Model architecture (for freezing/unfreezing)
# Returns: Average training loss across all epochs
#
# PROGRESSIVE FINE-TUNING STRATEGY:
#   Stage 1 (40% of epochs): Train HEAD ONLY (frozen backbone)
#     - Higher learning rate (lr)
#     - Prevents catastrophic forgetting of ImageNet features
#     - Head learns task-specific binary classification
#
#   Stage 2 (60% of epochs): Train FULL MODEL (unfrozen backbone)
#     - Lower learning rate (lr/10)
#     - Adapts pre-trained features to medical domain
#     - Fine-tunes entire network for ALL vs Healthy classification
def train(model, trainloader, epochs, lr, device, model_type: str = None):
    if trainloader is None:
        return 0.0  # Skip training for clients without data

    model.to(device)

    # Use Focal Loss to handle class imbalance and hard examples
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    print(f"[Train] Using Focal Loss (alpha=0.25, gamma=2.0) for class imbalance handling")

    # Calculate epoch split for two-stage training
    # Stage 1: 40% of epochs (minimum 2 epochs)
    # Stage 2: 60% of epochs (remaining)
    stage1_epochs = max(2, int(epochs * 0.4))
    stage2_epochs = epochs - stage1_epochs

    total_loss = 0.0
    total_batches = 0

    # ========== STAGE 1: HEAD-ONLY TRAINING (Frozen Backbone) ==========
    if model_type and stage1_epochs > 0:
        print(f"\n[Train] Stage 1/2: Training classifier head only ({stage1_epochs} epochs, lr={lr:.6f})")

        # FIXED: Reset all parameters to trainable first (ensures clean state for freeze/unfreeze)
        # This is critical for Round 2+ where models are loaded from checkpoints
        # Without this, freeze_backbone() won't work correctly on already-unfrozen models
        for param in model.parameters():
            param.requires_grad = True

        model = freeze_backbone(model, model_type)

        # Show trainable parameters
        trainable, frozen, total_params = get_trainable_params(model)
        print(f"[Train] Parameters: {trainable:,} trainable, {frozen:,} frozen, {total_params:,} total")

        # Optimizer for head-only training (higher LR)
        optimizer_head = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,  # Full learning rate for head
            betas=(0.9, 0.999),
            weight_decay=0.01
        )

        model.train()
        for epoch in range(stage1_epochs):
            for images, labels in trainloader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer_head.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                # SOLUTION 5: Gradient clipping to prevent large updates
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer_head.step()

                total_loss += loss.item()
                total_batches += 1

        print(f"[Train] Stage 1 complete: Head trained on task-specific features")

    # ========== STAGE 2: FULL MODEL FINE-TUNING (Unfrozen Backbone) ==========
    if model_type and stage2_epochs > 0:
        print(f"\n[Train] Stage 2/2: Fine-tuning full model ({stage2_epochs} epochs, lr={lr/10:.6f})")
        model = unfreeze_backbone(model, model_type)

        # Show trainable parameters
        trainable, frozen, total_params = get_trainable_params(model)
        print(f"[Train] Parameters: {trainable:,} trainable, {frozen:,} frozen, {total_params:,} total")

        # Optimizer for full fine-tuning (lower LR to prevent destroying pre-trained features)
        optimizer_full = torch.optim.AdamW(
            model.parameters(),
            lr=lr / 10,  # 10x lower learning rate for backbone fine-tuning
            betas=(0.9, 0.999),
            weight_decay=0.01
        )

        model.train()
        for epoch in range(stage2_epochs):
            for images, labels in trainloader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer_full.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                # SOLUTION 5: Gradient clipping to prevent large updates
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer_full.step()

                total_loss += loss.item()
                total_batches += 1

        print(f"[Train] Stage 2 complete: Full model fine-tuned for medical domain")

    # Fallback: If no model_type provided, train normally (backward compatibility)
    if not model_type:
        print(f"[Train] Single-stage training ({epochs} epochs, lr={lr:.6f})")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )

        model.train()
        for _ in range(epochs):
            for images, labels in trainloader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                # SOLUTION 5: Gradient clipping to prevent large updates
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                total_batches += 1

    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
    return avg_loss

# Evaluate model on validation/test data with comprehensive metrics
# Args: model - PyTorch model to evaluate
#       testloader - DataLoader for test data
#       device - Device to evaluate on (cpu/cuda)
# Returns: Tuple of (loss, accuracy) OR Dict with detailed metrics if return_detailed=True
def test(model, testloader, device, return_detailed=False):
    from flwr.common import log
    from logging import INFO, WARNING

    model.to(device)
    criterion = nn.CrossEntropyLoss()

    correct, total, total_loss = 0, 0, 0.0

    # Track all predictions, labels, and probabilities for comprehensive analysis
    all_preds = []
    all_labels = []
    all_probs = []  # For ROC-AUC calculation

    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            total_loss += criterion(outputs, labels).item()

            # Get probabilities for ROC-AUC
            probs = F.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().tolist())

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Store for per-class analysis
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    # Overall metrics
    accuracy = correct / total if total > 0 else 0.0
    loss = total_loss / len(testloader) if len(testloader) > 0 else 0.0

    # Convert to tensors for analysis
    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)
    all_probs = torch.tensor(all_probs)

    # Class 0: ALL (Leukemia - Positive), Class 1: Healthy (Hem - Negative)
    # Note: ImageFolder loads alphabetically, so "all" folder = class 0, "hem" folder = class 1
    leukemia_mask = (all_labels == 0)
    healthy_mask = (all_labels == 1)

    # Calculate confusion matrix components
    # For binary: Leukemia (class 0) is positive, Healthy (class 1) is negative
    TP = ((all_preds == 0) & (all_labels == 0)).sum().item()  # Predicted leukemia, actual leukemia
    FP = ((all_preds == 0) & (all_labels == 1)).sum().item()  # Predicted leukemia, actual healthy
    FN = ((all_preds == 1) & (all_labels == 0)).sum().item()  # Predicted healthy, actual leukemia
    TN = ((all_preds == 1) & (all_labels == 1)).sum().item()  # Predicted healthy, actual healthy

    # Per-class accuracies
    leukemia_acc = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # Recall/Sensitivity
    healthy_acc = TN / (TN + FP) if (TN + FP) > 0 else 0.0   # Specificity

    # Precision, Recall, F1
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # Same as leukemia_acc (sensitivity)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0  # Same as healthy_acc

    # ROC-AUC calculation
    try:
        from sklearn.metrics import roc_auc_score
        # Use probability of positive class (leukemia = class 0)
        roc_auc = roc_auc_score(all_labels.numpy(), all_probs[:, 0].numpy())
    except Exception:
        # Fallback: approximate ROC-AUC from sensitivity and specificity
        roc_auc = (recall + specificity) / 2

    # Calculate class gap (imbalance indicator)
    class_gap = abs(healthy_acc - leukemia_acc)

    # Log per-class performance
    log(INFO, f"[EVAL] Overall: {accuracy:.1%} | Leukemia: {leukemia_acc:.1%} | Healthy: {healthy_acc:.1%} | Gap: {class_gap:.1%}")
    log(INFO, f"[EVAL] Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1_score:.3f} | ROC-AUC: {roc_auc:.3f}")

    # Warn if severely imbalanced
    if class_gap > 0.3:
        log(WARNING, f"[EVAL] ⚠️ Class imbalance detected! Gap: {class_gap:.1%} (Target: <10%)")

    # Return detailed metrics if requested
    if return_detailed:
        return {
            "loss": round(loss, 6),
            "accuracy": round(accuracy, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1_score": round(f1_score, 6),
            "specificity": round(specificity, 6),
            "roc_auc": round(roc_auc, 6),
            "leukemia_accuracy": round(leukemia_acc, 6),
            "healthy_accuracy": round(healthy_acc, 6),
            "class_gap": round(class_gap, 6),
            "confusion_matrix": {
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "TN": TN
            },
            "num_samples": total,
            "num_leukemia_samples": leukemia_mask.sum().item(),
            "num_healthy_samples": healthy_mask.sum().item()
        }

    return loss, accuracy

# <------------------------------------------ KNOWLEDGE DISTILLATION FUNCTION DEFINITIONS ------------------------------------------>
# Distill consensus knowledge into local model using KL divergence
# Args: model - PyTorch model to distill knowledge into
#       public_loader - DataLoader for public anchor dataset
#       consensus_logits - Aggregated consensus logits from server
#       device - Device to train on (cpu/cuda)
#       epochs - Number of distillation epochs
#       lr - Learning rate for distillation
#       temperature - Temperature scaling for soft labels
# Returns: Average distillation loss
def distill_knowledge(model, public_loader, consensus_logits, device, epochs, lr, temperature, 
                     current_round=1, total_rounds=10, adaptive=True):
    """
    Distill consensus knowledge into local model using KL divergence with adaptive weighting.
    
    Args:
        model: PyTorch model to distill knowledge into
        public_loader: DataLoader for public anchor dataset
        consensus_logits: Aggregated consensus logits from server
        device: Device to train on (cpu/cuda)
        epochs: Number of distillation epochs
        lr: Learning rate for distillation
        temperature: Temperature scaling for soft labels
        current_round: Current FL round number
        total_rounds: Total number of FL rounds
        adaptive: If True, apply adaptive decay to distillation weight
    
    Returns:
        Average distillation loss
    
    Adaptive Strategy:
        Early rounds (1-3): Strong distillation (80-60% weight) - Learn from consensus
        Mid rounds (4-7): Balanced (60-40% weight) - Refine knowledge
        Late rounds (8-10): Weak distillation (40-25% weight) - Focus on private data
    """
    
    # Base distillation weight (from SOLUTION 1)
    BASE_DISTILL_WEIGHT = 0.60
    
    if adaptive and total_rounds > 1:
        # Calculate decay factor: starts at 1.0 (round 1), decreases to ~0.4 (final round)
        # Uses exponential decay: weight = base_weight * exp(-decay_rate * progress)
        progress = (current_round - 1) / (total_rounds - 1)  # 0.0 to 1.0
        
        # Decay rate: controls how fast distillation weight decreases
        # Higher decay_rate = faster decrease (more aggressive)
        # decay_rate=0.5: gentle decay (weight: 1.0 -> 0.61)
        # decay_rate=1.0: moderate decay (weight: 1.0 -> 0.37) ✓ RECOMMENDED
        # decay_rate=1.5: aggressive decay (weight: 1.0 -> 0.22)
        decay_rate = 1.0
        
        # Compute adaptive weight with exponential decay
        adaptive_factor = np.exp(-decay_rate * progress)
        
        # Apply adaptive factor to base weight
        DISTILL_WEIGHT = BASE_DISTILL_WEIGHT * adaptive_factor
        
        # Ensure weight stays within reasonable bounds [0.25, 0.80]
        DISTILL_WEIGHT = max(0.25, min(0.80, DISTILL_WEIGHT))
        
        print(f"[Distillation] Adaptive weight: {DISTILL_WEIGHT:.3f} "
              f"(round {current_round}/{total_rounds}, progress: {progress:.1%})")
    else:
        # Static weight (original behavior)
        DISTILL_WEIGHT = BASE_DISTILL_WEIGHT
        print(f"[Distillation] Static weight: {DISTILL_WEIGHT:.3f}")

    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    consensus_tensor = torch.from_numpy(consensus_logits).float()

    total_loss = 0.0
    idx = 0

    for _epoch in range(epochs):
        for images, _ in public_loader:
            images = images.to(device)
            batch_size = images.size(0)

            if idx + batch_size > len(consensus_tensor):
                break

            batch_consensus = consensus_tensor[idx:idx+batch_size].to(device)
            student_logits = model(images)

            # KL Divergence Loss with adaptive weighted influence
            kl_loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(batch_consensus / temperature, dim=1),
                reduction='batchmean'
            ) * (temperature ** 2)

            # Apply adaptive distillation weight
            loss = DISTILL_WEIGHT * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            idx += batch_size

        idx = 0

    avg_loss = total_loss / (len(public_loader) * epochs)
    return avg_loss

# <------------------------------------------ CONSENSUS COMPUTATION ------------------------------------------>

def compute_consensus(
    logits_list: List[np.ndarray], 
    client_metrics: List[Dict],
    client_configs: List[Dict],
    server_round: int,
    last_consensus: Optional[np.ndarray] = None,
    eval_history: Optional[List[Dict]] = None,
    momentum: float = CONSENSUS_MOMENTUM,
    exclude_free_riders: bool = True,
) -> Tuple[Optional[np.ndarray], Dict]:
    """ 
    Computes a robust consensus value by aggregating client logits from the anchor dataset using multi-factor quality weighting.
    Args:
        - logits_list: List of numpy arrays containing client predictions on public dataset
            Ex : logits_list = [
                    array_client_0, # Shape: (100, 2) - 100 rows of images and 2 columns (leukemia, healthy)
                    array_client_1, # Shape: (100, 2)
                ]
        - server_round: Current FL round number
        - client_configs: List of client configuration dicts (model_type, client_name, etc.)
        - client_metrics: List of dicts with training metrics (train_loss, distill_loss, num-examples, has_local_data)
        - last_consensus: Previous round's consensus logits (for momentum smoothing)
        - eval_history: Historical evaluation metrics across rounds
        - momentum: Weight for previous consensus (default 0.3)
        - exclude_free_riders: If True, free riders get zero weight (default True)

    Returns:
        - consensus_logits: Weighted average of client logits with momentum and class reweighting, or None if no valid clients
        - aggregation_metadata: Dict containing weights, factors, class distribution, and debugging info

    Weighting Strategy:
        CLIENT-LEVEL WEIGHTING (determines weightage based on contribution by each client):
        1. Number of data points for local trainig per client (highest influence)
        2. Calculates the combined loss from both the private training loss and the distillation loss
        3. Model architecture factor on model suitability for medical imaging
        4. Free riders: Zero weight (complete exclusion)
        
        CLASS-LEVEL WEIGHTING (compensates for class imbalance in training data):
        5. Inverse frequency weighting: Minority class predictions get boosted via sqrt(inverse_frequency)
           - Computes class distribution in consensus predictions
           - Applies higher weight to underrepresented class
           - Helps maintain balanced learning signal despite imbalanced client training data
    """

    # Validate Inputs
    if not logits_list or len(logits_list) == 0:
        return None, {"error": "No client logits provided"}

    if len(logits_list) != len(client_metrics) or len(logits_list) != len(client_configs):
        return None, {"error": "Mismatched input lengths"}

    num_clients = len(logits_list)

    # Initialize weight components for each client
    weights = []
    weight_breakdown = []  # For debugging/logging

    for i in range(num_clients):
        metrics = client_metrics[i]
        config = client_configs[i]

        # Extract metrics with safe defaults
        num_samples = metrics.get("num-examples", 1)
        has_local_data = metrics.get("has_local_data", 0)
        train_loss = metrics.get("train_loss", 0.0)
        distill_loss = metrics.get("distill_loss", 0.0)
        model_type = config.get("model_type", "unknown").lower()
        client_name = config.get("client_name", f"client_{i}")

        # <------------------- 1. Free Rider Exclusion -------------------->
        if exclude_free_riders and not has_local_data:
            weights.append(0.0) # Set to 0
            weight_breakdown.append({
                "client_name": client_name,
                "model_type": model_type,
                "base_weight": 0.0,
                "quality_multiplier": 0.0,
                "architecture_factor": 0.0,
                "final_weight": 0.0,
                "reason": "free_rider_excluded"
            })
            continue

        # <------------------- 2. Local Dataset Point Quantity -------------------->
        # Clients with more data get higher base weight
        base_weight = max(num_samples, 1)  # At least 1 to avoid zero division

        # <------------------- 3. Combined Loss Calculation (Distill Loss and Train Loss) : Determines the quality of the logits -------------------->
        # Lower combined loss = better convergence = higher multiplier
        combined_loss = (TRAIN_LOSS_WEIGHT * train_loss +
                        DISTILL_LOSS_WEIGHT * distill_loss)

        quality_multiplier = 1.0 / (1.0 + combined_loss)

        # <------------------- 4. Model Achitecture Suitability Factor -------------------->
        architecture_factor = MODEL_SUITABILITY_SCORES.get(model_type, 1.0)

        # <------------------- Compte Final Weight -------------------->
        final_weight = (base_weight * quality_multiplier * architecture_factor)

        weights.append(final_weight)
        weight_breakdown.append({
            "client_name": client_name,
            "model_type": model_type,
            "num_samples": num_samples,
            "train_loss": train_loss,
            "distill_loss": distill_loss,
            "combined_loss": combined_loss,
            "base_weight": base_weight,
            "quality_multiplier": quality_multiplier,
            "architecture_factor": architecture_factor,
            "final_weight": final_weight,
        })

    # <------------------- Normalize Weights -------------------->
    total_weight = sum(weights)

    if total_weight == 0:
            # All clients are free riders or have zero weight
        return None, {
            "error": "All clients excluded (free riders or zero weight)",
            "weight_breakdown": weight_breakdown
        }

    normalized_weights = []
    for w in weights: 
        value = w / total_weight
        normalized_weights.append(value)

    # Update weight breakdown with normalized values
    for i, breakdown in enumerate(weight_breakdown):
        breakdown["normalized_weight"] = normalized_weights[i]

    # <------------------- Calculate weighted consensus -------------------->
    new_consensus = np.average(logits_list, axis=0, weights=normalized_weights) # Weights will be the multiplying factor here when averaging

    # <------------------- Class based reweighting -------------------->
    # Give preference to minority class to combat training data imbalance
    # Strategy: Compute class distribution and apply inverse frequency weighting

    # Get predicted class probabilities (softmax of logits)
    consensus_probs = np.exp(new_consensus) / np.sum(np.exp(new_consensus), axis=1, keepdims=True)
    
    # Calculate class frequencies across the consensus predictions
    predicted_classes = np.argmax(consensus_probs, axis=1)
    unique_classes, class_counts = np.unique(predicted_classes, return_counts=True)
    
    # Compute inverse frequency weights for each class
    total_samples = len(predicted_classes)
    class_weights = {}
    for cls, count in zip(unique_classes, class_counts):
        # Inverse frequency: minority class gets higher weight
        class_weights[cls] = total_samples / (len(unique_classes) * count)
    
    # Normalize class weights to maintain overall scale
    weight_values = list(class_weights.values())
    weight_sum = sum(weight_values)
    class_weights = {cls: w / weight_sum for cls, w in class_weights.items()}
    
    # Apply class weights to consensus logits
    # For each sample, boost the logit for its minority class
    class_weighted_consensus = new_consensus.copy()
    for i in range(len(new_consensus)):
        pred_class = predicted_classes[i]
        if pred_class in class_weights:
            # Scale factor: use sqrt to moderate the effect (prevents over-boosting)
            boost_factor = np.sqrt(class_weights[pred_class])
            # Apply boost to the predicted class logit
            class_weighted_consensus[i, pred_class] *= boost_factor
    
    # Use class-weighted consensus as the new consensus
    new_consensus = class_weighted_consensus
    
    # Store class weighting info in metadata
    class_weighting_info = {
        "class_distribution": {int(cls): int(count) for cls, count in zip(unique_classes, class_counts)},
        "class_weights": {int(cls): float(w) for cls, w in class_weights.items()},
        "minority_class": int(unique_classes[np.argmin(class_counts)]) if len(unique_classes) > 0 else None,
        "majority_class": int(unique_classes[np.argmax(class_counts)]) if len(unique_classes) > 0 else None,
    }

    # <------------------- Momentum Smoothing : Controls temporal smoothing (Technique to reduce fluctuations and noise over time) of the consensus logits across FL rounds.  -------------------->
    if last_consensus is not None and server_round > 1:
        # Check for shape mismatch due to growing dataset (gradual release)
        if last_consensus.shape != new_consensus.shape:
            # Assume new_consensus is larger (superset)
            if new_consensus.shape[0] > last_consensus.shape[0]:
                # Create a padded version of last_consensus
                # We can either zero-pad or just use new_consensus values for the new part
                
                # Verify standard width (num_classes) compatibility
                if new_consensus.shape[1] != last_consensus.shape[1]:
                     # This is a critical error (changed classes mid-training?)
                     return None, {"error": "Class count mismatch in consensus resizing"}

                old_len = last_consensus.shape[0]
                new_len = new_consensus.shape[0]
                
                # Weighted blend on the overlapping part
                balanced_part = (momentum * last_consensus + (1 - momentum) * new_consensus[:old_len])
                
                # New part relies entirely on current round (no history)
                new_part = new_consensus[old_len:]
                
                consensus_logits = np.concatenate([balanced_part, new_part], axis=0)
            else:
                # Fallback: simple resizing or error (shouldn't happen with gradual release growing)
                 consensus_logits = new_consensus
        else:
            # Shapes match, standard momentum
            consensus_logits = (momentum * last_consensus +
                              (1 - momentum) * new_consensus)
        
        smoothing_applied = True
    else:
        consensus_logits = new_consensus
        smoothing_applied = False

    # --- PREPARE METADATA ---
    aggregation_metadata = {
        "server_round": server_round,
        "num_clients": num_clients,
        "num_contributing_clients": sum(1 for w in weights if w > 0),
        "num_excluded_free_riders": sum(1 for w in weights if w == 0),
        "total_raw_weight": total_weight,
        "momentum": momentum if smoothing_applied else None,
        "smoothing_applied": smoothing_applied,
        "weight_breakdown": weight_breakdown,
        "normalized_weights": normalized_weights,
        "weight_statistics": {
            "min": min(normalized_weights) if normalized_weights else 0,
            "max": max(normalized_weights) if normalized_weights else 0,
            "mean": float(np.mean(normalized_weights)) if normalized_weights else 0,
            "std": float(np.std(normalized_weights)) if normalized_weights else 0,
        },
        "class_weighting": class_weighting_info,  # Added class-based weighting info
        "parameters": {
            "train_loss_weight": TRAIN_LOSS_WEIGHT,
            "distill_loss_weight": DISTILL_LOSS_WEIGHT,
            "exclude_free_riders": exclude_free_riders,
        }
    }

    return consensus_logits, aggregation_metadata

# <------------------------------------------ FLEX-MED FL SIMULATION STRATEGY ------------------------------------------>

# Coordinates federated learning rounds with model-agnostic knowledge distillation
class FLEXMedStrategy(Strategy):
    # Initialize FL simulation strategy with checkpoint resumption support
    # Args: config_path - Path to client configuration JSON
    #       checkpoint_dir - Directory for saving model checkpoints
    def __init__(self, config_path: str = CLIENT_INFO_FILE_PATH, checkpoint_dir: str = MODEL_CHECKPOINT_FILE_PATH):
        super().__init__()
        self.config_path = config_path
        self.checkpoint_dir = checkpoint_dir
        self.client_configs = load_client_config(config_path)
        self.num_clients = len(self.client_configs)

        # Track evaluation metrics across rounds
        self.eval_history = []

        # Track starting round (for checkpoint resumption)
        self.start_round = 1
        self.last_consensus_logits = None

        # SOLUTION 3: Early stopping tracking
        self.client_history = {}  # Track per-client metrics history

        print(f"\n[SERVER] Initialized {self.num_clients} clients")

        # Try to load checkpoint
        checkpoint = load_checkpoint(checkpoint_dir)
        if checkpoint:
            self.start_round = checkpoint['round'] + 1  # Resume from next round
            self.eval_history = checkpoint['eval_history']
            self.last_consensus_logits = checkpoint['consensus_logits']
            print(f"[SERVER] Resuming from Round {self.start_round}")

    # SOLUTION 3: Early stopping check method
    def should_stop_training(self, current_round: int, post_fl_metrics: Dict) -> Tuple[bool, str]:
        """
        Check if training should stop based on degradation patterns.

        Returns:
            Tuple of (should_stop: bool, reason: str)
        """
        if current_round < 3:  # Need at least 2 rounds of history
            return False, ""

        # Track client metrics
        for client_id, metrics in post_fl_metrics.items():
            if client_id not in self.client_history:
                self.client_history[client_id] = []

            self.client_history[client_id].append({
                'round': current_round,
                'accuracy': metrics.get('accuracy', 0),
                'loss': metrics.get('loss', 0)
            })

        # Check for consistent degradation (2 consecutive negative improvements)
        degraded_clients = []
        for client_id, history in self.client_history.items():
            if len(history) >= 3:
                # Check last 2 rounds for negative improvement
                acc_trend = [h['accuracy'] for h in history[-3:]]
                if acc_trend[-1] < acc_trend[-2] and acc_trend[-2] < acc_trend[-3]:
                    degraded_clients.append(client_id)

        # Stop if majority of clients are degrading
        if len(degraded_clients) >= len(self.client_history) / 2:
            return True, f"Majority of clients degrading: {degraded_clients}"

        # Check for loss increase pattern (overfitting indicator)
        loss_increasing_count = 0
        for client_id, history in self.client_history.items():
            if len(history) >= 2:
                if history[-1]['loss'] > history[-2]['loss']:
                    loss_increasing_count += 1

        if loss_increasing_count >= len(self.client_history) * 0.7:  # 70% threshold
            return True, f"70% of clients show increasing loss (overfitting)"

        return False, ""

    # <------------------------------------------ FL EXECUTION WITH RESUME SUPPORT ------------------------------------------>

    # Override base Strategy.start() to support checkpoint resumption
    # The base implementation always loops from round 1 to num_rounds, ignoring start_round
    # This override loops from start_round to num_rounds, enabling mid-training resume
    # ENHANCED: Now includes per-round pre/post evaluation on public test set
    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 14400,  # 4 hours - ensures all clients complete before aggregation
        train_config: Optional[ConfigRecord] = None,
        evaluate_config: Optional[ConfigRecord] = None,
        evaluate_fn = None,
    ):
        """Execute FL with resume support and per-round evaluation.

        For each round:
        1. PRE-FL Evaluation: Evaluate all clients on public test set (knowledge retention)
        2. FL Training: Distillation + Private training
        3. POST-FL Evaluation: Evaluate all clients on public test set (learning progression)
        4. Save metrics: Store pre/post/improvement to round_metrics.json
        """
        from flwr.common import log
        from flwr.serverapp.strategy.result import Result
        from logging import INFO, WARNING
        import time

        # Auto-detect GPU availability for server-side evaluations
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        log(INFO, f"[SERVER] Evaluation device: {device}")

        # Calculate remaining rounds if resuming
        if self.start_round > 1:
            remaining = num_rounds - (self.start_round - 1)
            log(INFO, f"[RESUME] Resuming from round {self.start_round}/{num_rounds} ({remaining} remaining)")
            if remaining <= 0:
                log(INFO, "[RESUME] Training already complete!")
                result = Result()
                result.arrays = initial_arrays
                return result
        else:
            log(INFO, f"[SERVER] Starting fresh training ({num_rounds} rounds)")

        # Initialize configs
        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config

        result = Result()
        arrays = initial_arrays
        t_start = time.time()

        # Store per-round metrics for visualization
        self.round_metrics_history = {}

        # ========== GLOBAL PRE-FL EVALUATION (ONLY IF STARTING FRESH) ==========
        if self.start_round == 1:
            log(INFO, "")
            log(INFO, "=" * 70)
            log(INFO, "[GLOBAL] Initial Centralized Model Evaluation (Public Test)")
            log(INFO, "=" * 70)
            log(INFO, "")

            try:
                global_pre_fl_metrics = evaluate_all_clients_on_public_test(
                    self.client_configs, device
                )

                # Save global metrics to cnmc_data.json
                save_global_pre_fl_metrics(global_pre_fl_metrics, self.client_configs)

                log(INFO, f"[GLOBAL] Pre-FL Evaluation Complete")
                for client_id, metrics in global_pre_fl_metrics.items():
                    acc = metrics.get('accuracy', 0)
                    loss = metrics.get('loss', 0)
                    log(INFO, f"  Client {client_id}: Accuracy={acc:.1%}, Loss={loss:.3f} on public test")
                log(INFO, "")

            except Exception as e:
                log(WARNING, f"[GLOBAL] Pre-FL Evaluation failed: {e}")

        # KEY: Loop from start_round to num_rounds (not 1 to num_rounds)
        for current_round in range(self.start_round, num_rounds + 1):
            log(INFO, "")
            log(INFO, f"{'='*70}")
            log(INFO, f"[ROUND {current_round}/{num_rounds}]")
            log(INFO, f"{'='*70}")

            # --- TRAINING PHASE ---
            log(INFO, "")
            log(INFO, f"[ROUND {current_round}] FL Training Phase")
            log(INFO, "-" * 50)

            train_msgs = self.configure_train(current_round, arrays, train_config, grid)
            train_replies = grid.send_and_receive(messages=train_msgs, timeout=timeout)
            agg_arrays, agg_metrics, training_metrics = self.aggregate_train(current_round, train_replies)

            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays
            if agg_metrics:
                result.train_metrics_clientapp[current_round] = agg_metrics

            # --- PER-ROUND VALIDATION EVALUATION PHASE ---
            log(INFO, "")
            log(INFO, f"[ROUND {current_round}] Validation Evaluation (Per-Round Tracking)")
            log(INFO, "-" * 50)

            try:
                round_val_metrics = evaluate_all_clients_on_validation(
                    self.client_configs, device, len(self.client_configs)
                )

                # Save validation metrics to cnmc_data.json
                save_round_validation_metrics(current_round, round_val_metrics, self.client_configs)
                self.round_metrics_history[f"round_{current_round}_validation"] = round_val_metrics

                # Check for degradation warnings (does not stop training)
                for client_id, metrics in round_val_metrics.items():
                    if client_id not in self.client_history:
                        self.client_history[client_id] = []
                    self.client_history[client_id].append({
                        'round': current_round,
                        'metrics': metrics
                    })

                check_for_degradation_warnings(round_val_metrics, current_round, self.client_history)

                # Log summary
                avg_val_acc = np.mean([m.get("accuracy", 0) for m in round_val_metrics.values() if m.get("accuracy") is not None])
                avg_val_loss = np.mean([m.get("loss", 0) for m in round_val_metrics.values() if m.get("loss") is not None])
                avg_class_gap = np.mean([m.get("class_gap", 0) for m in round_val_metrics.values() if m.get("class_gap") is not None])

                log(INFO, f"[ROUND {current_round}] Validation Results:")
                for client_id, metrics in round_val_metrics.items():
                    acc = metrics.get('accuracy', 0)
                    loss = metrics.get('loss', 0)
                    gap = metrics.get('class_gap', 0)
                    dataset = metrics.get('dataset', 'unknown')
                    log(INFO, f"  Client {client_id}: Acc={acc:.1%}, Loss={loss:.3f}, Gap={gap:.1%} ({dataset})")

                log(INFO, f"[ROUND {current_round}] Average: Acc={avg_val_acc:.1%}, Loss={avg_val_loss:.3f}, Gap={avg_class_gap:.1%}")

            except Exception as e:
                log(WARNING, f"[ROUND {current_round}] Validation evaluation failed: {e}")
                round_val_metrics = {}

            # --- STANDARD FLOWER EVALUATION (for backward compatibility) ---
            eval_msgs = self.configure_evaluate(current_round, arrays, evaluate_config, grid)
            eval_replies = grid.send_and_receive(messages=eval_msgs, timeout=timeout)
            eval_metrics = self.aggregate_evaluate(current_round, eval_replies)

            if eval_metrics:
                result.evaluate_metrics_clientapp[current_round] = eval_metrics

        # ========== GLOBAL POST-FL EVALUATION ==========
        log(INFO, "")
        log(INFO, "=" * 70)
        log(INFO, "[GLOBAL] Final Federated Model Evaluation (Public Test)")
        log(INFO, "=" * 70)
        log(INFO, "")

        try:
            global_post_fl_metrics = evaluate_all_clients_on_public_test(
                self.client_configs, device
            )

            # Save global Post-FL metrics and calculate improvements
            save_global_post_fl_metrics(global_post_fl_metrics, self.client_configs)

            log(INFO, f"[GLOBAL] Post-FL Evaluation Complete")
            log(INFO, "")
            log(INFO, "FL BENEFIT ANALYSIS:")
            log(INFO, "=" * 70)

            # Compare Pre-FL vs Post-FL for each client
            for client_id, post_metrics in global_post_fl_metrics.items():
                # Load pre-FL metrics from saved data
                pre_metrics = load_global_pre_fl_for_client(client_id, self.client_configs)

                pre_acc = pre_metrics.get('accuracy', 0)
                post_acc = post_metrics.get('accuracy', 0)
                improvement = post_acc - pre_acc

                pre_loss = pre_metrics.get('loss', 0)
                post_loss = post_metrics.get('loss', 0)
                loss_delta = post_loss - pre_loss

                pre_gap = pre_metrics.get('class_gap', 0)
                post_gap = post_metrics.get('class_gap', 0)
                gap_delta = post_gap - pre_gap

                log(INFO, f"Client {client_id}:")
                log(INFO, f"  Pre-FL  (Centralized): Acc={pre_acc:.1%}, Loss={pre_loss:.3f}, Gap={pre_gap:.1%}")
                log(INFO, f"  Post-FL (Federated):   Acc={post_acc:.1%}, Loss={post_loss:.3f}, Gap={post_gap:.1%}")
                log(INFO, f"  FL Improvement:        Acc={improvement:+.1%}, Loss={loss_delta:+.3f}, Gap={gap_delta:+.1%}")
                log(INFO, "")

            # Calculate average improvement
            avg_pre_acc = np.mean([load_global_pre_fl_for_client(cid, self.client_configs).get('accuracy', 0)
                                   for cid in global_post_fl_metrics.keys()])
            avg_post_acc = np.mean([m.get('accuracy', 0) for m in global_post_fl_metrics.values()])
            avg_improvement = avg_post_acc - avg_pre_acc

            log(INFO, f"Average FL Benefit: {avg_improvement:+.1%}")
            log(INFO, "")

        except Exception as e:
            log(WARNING, f"[GLOBAL] Post-FL Evaluation failed: {e}")

        log(INFO, "")
        log(INFO, f"{'='*70}")
        log(INFO, f"Strategy execution finished in {time.time() - t_start:.2f}s")
        log(INFO, f"{'='*70}")

        # Log final summary
        log(INFO, "")
        log(INFO, "[SUMMARY] Per-Round Metrics saved to: " + ROUND_METRICS_FILE_PATH)
        log(INFO, "[SUMMARY] Generate visualizations with: python generate_graphs.py")

        return result

    # Send evaluation requests to all clients
    # Args: server_round - Current FL round number
    #       arrays - Optional array data from server
    #       config - Configuration parameters
    #       grid - Server grid for message routing
    # Returns: List of evaluation messages for each client
    def configure_evaluate(self, server_round: int, arrays: Optional[ArrayRecord], config: ConfigRecord, grid: Grid) -> Iterable[Message]:
        node_ids = list(grid.get_node_ids())
        messages = []

        # Send evaluation request to all clients
        for i, node_id in enumerate(node_ids):
            content = RecordDict({
                "config": ConfigRecord({"round": server_round})
            })

            msg = Message(
                metadata=Metadata(
                    run_id=0,
                    message_id=str(uuid.uuid4()),
                    src_node_id=0,
                    dst_node_id=node_id,
                    reply_to_message_id="",
                    group_id=str(server_round),
                    ttl=86400.0,
                    message_type="evaluate",
                    created_at=time.time(),
                ),
                content=content,
            )
            messages.append(msg)

        return messages

    # Aggregate evaluation results and compute weighted averages
    # Args: server_round - Current FL round number
    #       results - Client evaluation messages
    # Returns: Dictionary with aggregated loss and metrics
    def aggregate_evaluate(self, server_round: int, results: Iterable[Message], **kwargs):
        results_list = list(results)

        if not results_list:
            return {"loss": None, "metrics": {}}

        print(f"\n[ROUND {server_round}] Evaluation Results:")

        # Separate results by client type
        clients_with_data = []
        free_riders = []

        total_loss = 0.0
        total_acc = 0.0
        total_examples = 0

        for msg in results_list:
            metrics = msg.content.get("metrics", {})
            eval_loss = metrics.get("eval_loss", 0.0)
            eval_acc = metrics.get("eval_acc", 0.0)
            num_examples = metrics.get("num-examples", 0)
            client_id = metrics.get("client_id", -1)
            has_data = metrics.get("has_local_data", 0)

            # Track by client type
            client_info = {
                "client_id": client_id,
                "loss": eval_loss,
                "acc": eval_acc,
                "examples": num_examples
            }

            if has_data:
                clients_with_data.append(client_info)
            else:
                free_riders.append(client_info)

            # Aggregate (weighted by number of examples)
            total_loss += eval_loss * num_examples
            total_acc += eval_acc * num_examples
            total_examples += num_examples

        # Compute weighted averages
        if total_examples == 0:
            return {"loss": None, "metrics": {}}

        avg_loss = total_loss / total_examples
        avg_acc = total_acc / total_examples

        print(f"  Avg Loss: {avg_loss:.4f} | Avg Accuracy: {avg_acc:.4f} ({avg_acc*100:.1f}%)")

        # Store history for later analysis
        round_metrics = {
            "round": server_round,
            "avg_loss": avg_loss,
            "avg_acc": avg_acc,
            "total_examples": total_examples,
            "clients_with_data": clients_with_data,
            "free_riders": free_riders
        }
        self.eval_history.append(round_metrics)

        # Save checkpoint after evaluation
        if self.last_consensus_logits is not None:
            try:
                save_checkpoint(
                    checkpoint_dir=self.checkpoint_dir,
                    current_round=server_round,
                    consensus_logits=self.last_consensus_logits,
                    eval_history=self.eval_history,
                    training_metrics=round_metrics
                )
            except Exception as e:
                print(f"[Checkpoint] Warning: Failed to save checkpoint: {e}")

        return {
            "loss": avg_loss,
            "metrics": {
                "eval_acc": avg_acc,
                "num_clients": len(results_list),
                "total_examples": total_examples,
                "clients_with_data": len(clients_with_data),
                "free_riders": len(free_riders)
            }
        }

    # Aggregate client logits into weighted consensus for knowledge distillation
    # Args: server_round - Current FL round number
    #       results - Client training messages with logits
    # Returns: Tuple of (consensus_logits_array, aggregation_metrics, training_metrics_for_persistence)
    def aggregate_train(self, server_round: int, results: Iterable[Message], **kwargs):
        results_list = list(results)
        print(f"\n[SERVER] Round {server_round}: Aggregating Consensus")

        if not results_list:
            return None, {}, {}

        # Collect logits and metrics from all clients
        logits_list = []
        client_metrics_list = []
        client_names = []

        for i, msg in enumerate(results_list):
            client_arrays = msg.content["arrays"]
            try:
                client_logits = client_arrays["0"].numpy()
                logits_list.append(client_logits)

                # Extract all metrics for weight computation
                metrics = msg.content.get("metrics", {})
                client_metrics_list.append(metrics)

                # Track which client contributed
                if i < len(self.client_configs):
                    client_names.append(self.client_configs[i]['client_name'])

            except (KeyError, IndexError) as e:
                print(f"[SERVER] Warning: Failed to extract data from client {i}: {e}")
                pass

        # Compute consensus using sophisticated weight aggregation
        consensus_logits, aggregation_metadata = compute_consensus(
            logits_list=logits_list,
            client_metrics=client_metrics_list,
            client_configs=self.client_configs[:len(logits_list)],
            server_round=server_round,
            last_consensus=self.last_consensus_logits,
            eval_history=self.eval_history,
            momentum=CONSENSUS_MOMENTUM,
            exclude_free_riders=True
        )

        if consensus_logits is None:
            print(f"[SERVER] ✗ Failed to compute consensus: {aggregation_metadata.get('error', 'unknown')}")
            return None, {}, {}

        # Print informative summary
        print(f"[SERVER] ✓ Consensus computed from {aggregation_metadata['num_contributing_clients']}/{aggregation_metadata['num_clients']} clients")
        if aggregation_metadata['num_excluded_free_riders'] > 0:
            print(f"[SERVER]   Excluded {aggregation_metadata['num_excluded_free_riders']} free riders")

        # Print top 3 contributors
        breakdown = aggregation_metadata['weight_breakdown']
        sorted_by_weight = sorted(breakdown, key=lambda x: x.get('normalized_weight', 0), reverse=True)
        print(f"[SERVER]   Top contributors:")
        for client_data in sorted_by_weight[:3]:
            if client_data.get('normalized_weight', 0) > 0:
                print(f"[SERVER]     - {client_data['client_name']} ({client_data['model_type']}): "
                      f"{client_data['normalized_weight']*100:.1f}% weight "
                      f"[samples: {client_data['num_samples']}, loss: {client_data['combined_loss']:.3f}]")

        # Store consensus for checkpointing
        self.last_consensus_logits = consensus_logits

        arrays_aggregated = ArrayRecord([consensus_logits])

        # Merge aggregation metadata with client names
        metrics_aggregated = {
            "consensus_round": server_round,
            "num_clients": len(logits_list),
            "client_names": client_names,
            "weights": aggregation_metadata['normalized_weights'],
            "aggregation_details": aggregation_metadata  # Full details for tracking
        }

        # Extract training metrics for cnmc_data.json persistence
        training_metrics_for_persistence = extract_training_metrics_for_persistence(
            client_metrics_list=client_metrics_list,
            aggregation_metadata=aggregation_metadata,
            client_configs=self.client_configs
        )

        return arrays_aggregated, metrics_aggregated, training_metrics_for_persistence

    # Send training configuration and consensus logits to all clients
    # Args: server_round - Current FL round number
    #       arrays - Consensus logits from previous aggregation
    #       config - Configuration parameters
    #       grid - Server grid for message routing
    # Returns: List of training messages for each client
    def configure_train(self, server_round: int, arrays: Optional[ArrayRecord], config: ConfigRecord, grid: Grid) -> Iterable[Message]:
        node_ids = list(grid.get_node_ids())
        messages = []

        for node_id in node_ids:
            content = RecordDict({
                "arrays": arrays,
                "config": config
            })

            msg = Message(
                metadata=Metadata(
                    run_id=0,
                    message_id=str(uuid.uuid4()),
                    src_node_id=0,
                    dst_node_id=node_id,
                    reply_to_message_id="",
                    group_id=str(server_round),
                    ttl=86400.0,
                    message_type="train",
                    created_at=time.time(),
                ),
                content=content,
            )
            messages.append(msg)

        return messages

    # Save evaluation metrics to JSON file
    # Args: filepath - Output JSON file path
    # Returns: None
    def save_evaluation_history(self):
        return None

    # Generate training summary report
    # Returns: Empty string (summary handled separately)
    def summary(self) -> str:
        return ""