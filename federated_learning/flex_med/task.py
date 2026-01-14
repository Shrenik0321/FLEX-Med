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

# <------------------------------------------ CONFIGURATION ------------------------------------------>

# Base Path
BASE_PATH = os.getenv("BASE_PATH", "/content/drive/MyDrive/College/FLEX-Med")

# Client Information File Path - Acts as the intermediary between the api and fl codebase. Should be connected directly to the database.
CLIENT_INFO_FILE_PATH = os.path.join(BASE_PATH, "flex-med/flex_med/data.json")

# Dataset File Path - Contains the public and private datasets.
DATASET_FILE_PATH = os.path.join(BASE_PATH, "datasets")

PUBLIC_ANCHOR_DATASET_PATH = os.path.join(DATASET_FILE_PATH, "public_anchor")
PUBLIC_TEST_DATASET_PATH = os.path.join(DATASET_FILE_PATH, "public_test")

# Model Checkpoint File Path - Contains the model checkpoints.
MODEL_CHECKPOINT_FILE_PATH = os.path.join(BASE_PATH, "checkpoints")

# Round Metrics File Path - Contains per-round evaluation metrics for visualization.
ROUND_METRICS_FILE_PATH = os.path.join(BASE_PATH, "round_metrics.json")

# Graphs Output Directory - Contains generated visualization graphs.
GRAPHS_OUTPUT_DIR = os.path.join(BASE_PATH, "graphs")

# Model Constraints
NUM_CLASSES = 2   # 0: ALL (Leukemia), 1: Hem (Healthy)  # Fixed: must match ImageFolder alphabetical order (all, hem)
IMG_SIZE = 224    # Resize all inputs to 224X224 for consistency

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

# Factory function for heterogeneous model creation in FL simulation
# Args: model_type - String identifier ('resnet18', 'mobilenet_v2', 'efficientnet_b3')
#       use_pretrained - Whether to load ImageNet pretrained weights
# Returns: PyTorch model initialized for binary classification (NUM_CLASSES=2)
def get_model_by_type(model_type: str, use_pretrained: bool = True):
    model_type = model_type.lower()

    if model_type == 'resnet18':
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        return model

    elif model_type == 'mobilenet_v2':
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if use_pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
        return model

    elif model_type == 'efficientnet_b3':
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if use_pretrained else None
        model = models.efficientnet_b3(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
        return model

    else:
        raise ValueError(f"Unsupported model type: {model_type}. "
                        f"Supported types: resnet18, mobilenet_v2, efficientnet_b3")

# <------------------------------------------ DATA LOADER FUNCTION DEFINITIONS ------------------------------------------>

# Provides shared dataset for generating consensus logits across clients
# Args: batch_size - Batch size for DataLoader
# Returns: DataLoader with shuffle=False (order consistency critical for FL simulation)
def load_public_dataset(batch_size=64):
    if not os.path.exists(PUBLIC_ANCHOR_DATASET_PATH):
        raise FileNotFoundError(f"Public data not found at {PUBLIC_ANCHOR_DATASET_PATH}")

    dataset = datasets.ImageFolder(root=PUBLIC_ANCHOR_DATASET_PATH, transform=COMMON_TRANSFORM)
    
    # Shuffle=False is CRITICAL for FL simulation so all clients see images in the same order
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
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

    # Create a Train/Test split (80% Train, 20% Test)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    # Fixed seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    train_ds, test_ds = torch.utils.data.random_split(
        full_dataset, [train_size, test_size], generator=generator
    )

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
# Returns: Dictionary with accuracy, loss metrics
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

    # Evaluate model
    loss, accuracy = test(model, test_loader, device)

    return {
        "accuracy": accuracy,
        "loss": loss,
        "num_samples": len(test_loader.dataset)
    }

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

# <------------------------------------------ MODEL TRAINING & TESTING FUNCTION DEFINITIONS ------------------------------------------>

# Train model on private client data with AdamW optimizer and class-balanced loss
# Args: model - PyTorch model to train
#       trainloader - DataLoader for training data
#       epochs - Number of training epochs
#       lr - Learning rate
#       device - Device to train on (cpu/cuda)
# Returns: Average training loss across all epochs
def train(model, trainloader, epochs, lr, device):
    if trainloader is None:
        return 0.0  # Skip training for clients without data

    model.to(device)

    # Calculate class weights for balanced training (handles imbalanced datasets)
    try:
        all_labels = []
        for _, labels in trainloader:
            all_labels.extend(labels.tolist())
        class_counts = torch.bincount(torch.tensor(all_labels))
        # Inverse frequency weighting: minority class gets higher weight
        class_weights = 1.0 / (class_counts.float() + 1e-6)
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        class_weights = class_weights.to(device)
        print(f"[Train] Using class weights: {class_weights.cpu().tolist()}")
    except Exception as e:
        class_weights = None
        print(f"[Train] Using unweighted loss (class weights failed: {e})")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=0.01  # L2 regularization
    )

    model.train()
    total_loss = 0.0

    for _ in range(epochs):
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    avg_loss = total_loss / (len(trainloader) * epochs)
    return avg_loss

# Evaluate model on validation/test data
# Args: model - PyTorch model to evaluate
#       testloader - DataLoader for test data
#       device - Device to evaluate on (cpu/cuda)
# Returns: Tuple of (loss, accuracy)
def test(model, testloader, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    correct, total, total_loss = 0, 0, 0.0
    
    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total if total > 0 else 0.0
    loss = total_loss / len(testloader) if len(testloader) > 0 else 0.0
        
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
def distill_knowledge(model, public_loader, consensus_logits, device, epochs, lr, temperature):
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=0.01  # L2 regularization
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
            
            # KL Divergence Loss
            loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=1),
                F.softmax(batch_consensus / temperature, dim=1),
                reduction='batchmean'
            ) * (temperature ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            idx += batch_size
        
        idx = 0

    avg_loss = total_loss / (len(public_loader) * epochs)
    return avg_loss

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

        print(f"\n[SERVER] Initialized {self.num_clients} clients")

        # Try to load checkpoint
        checkpoint = load_checkpoint(checkpoint_dir)
        if checkpoint:
            self.start_round = checkpoint['round'] + 1  # Resume from next round
            self.eval_history = checkpoint['eval_history']
            self.last_consensus_logits = checkpoint['consensus_logits']
            print(f"[SERVER] Resuming from Round {self.start_round}")

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
        timeout: float = 3600,
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
        from logging import INFO
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

        # KEY: Loop from start_round to num_rounds (not 1 to num_rounds)
        for current_round in range(self.start_round, num_rounds + 1):
            log(INFO, "")
            log(INFO, f"{'='*70}")
            log(INFO, f"[ROUND {current_round}/{num_rounds}]")
            log(INFO, f"{'='*70}")

            # --- PRE-FL EVALUATION PHASE ---
            log(INFO, "")
            log(INFO, f"[ROUND {current_round}] PRE-FL Evaluation (Knowledge Retention)")
            log(INFO, "-" * 50)

            try:
                pre_fl_metrics = evaluate_all_clients(self.client_configs, device)
                save_round_metrics(current_round, "pre_fl", pre_fl_metrics)
                self.round_metrics_history[f"round_{current_round}_pre_fl"] = pre_fl_metrics

                # Log summary
                avg_pre_acc = np.mean([m.get("accuracy", 0) for m in pre_fl_metrics.values() if m.get("accuracy") is not None])
                log(INFO, f"[ROUND {current_round}] PRE-FL Average Accuracy: {avg_pre_acc:.4f} ({avg_pre_acc*100:.1f}%)")
            except Exception as e:
                log(INFO, f"[ROUND {current_round}] PRE-FL Evaluation failed: {e}")
                pre_fl_metrics = {}

            # --- TRAINING PHASE ---
            log(INFO, "")
            log(INFO, f"[ROUND {current_round}] FL Training Phase")
            log(INFO, "-" * 50)

            train_msgs = self.configure_train(current_round, arrays, train_config, grid)
            train_replies = grid.send_and_receive(messages=train_msgs, timeout=timeout)
            agg_arrays, agg_metrics = self.aggregate_train(current_round, train_replies)

            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays
            if agg_metrics:
                result.train_metrics_clientapp[current_round] = agg_metrics

            # --- POST-FL EVALUATION PHASE ---
            log(INFO, "")
            log(INFO, f"[ROUND {current_round}] POST-FL Evaluation (Learning Progression)")
            log(INFO, "-" * 50)

            try:
                post_fl_metrics = evaluate_all_clients(self.client_configs, device)
                save_round_metrics(current_round, "post_fl", post_fl_metrics)
                self.round_metrics_history[f"round_{current_round}_post_fl"] = post_fl_metrics

                # Log summary
                avg_post_acc = np.mean([m.get("accuracy", 0) for m in post_fl_metrics.values() if m.get("accuracy") is not None])
                log(INFO, f"[ROUND {current_round}] POST-FL Average Accuracy: {avg_post_acc:.4f} ({avg_post_acc*100:.1f}%)")

                # Calculate and log improvement
                if pre_fl_metrics:
                    avg_pre = np.mean([m.get("accuracy", 0) for m in pre_fl_metrics.values() if m.get("accuracy") is not None])
                    improvement = avg_post_acc - avg_pre
                    log(INFO, f"[ROUND {current_round}] Round Improvement: {improvement:+.4f} ({improvement*100:+.1f}%)")
            except Exception as e:
                log(INFO, f"[ROUND {current_round}] POST-FL Evaluation failed: {e}")
                post_fl_metrics = {}

            # --- STANDARD FLOWER EVALUATION (for backward compatibility) ---
            eval_msgs = self.configure_evaluate(current_round, arrays, evaluate_config, grid)
            eval_replies = grid.send_and_receive(messages=eval_msgs, timeout=timeout)
            eval_metrics = self.aggregate_evaluate(current_round, eval_replies)

            if eval_metrics:
                result.evaluate_metrics_clientapp[current_round] = eval_metrics

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
    # Returns: Tuple of (consensus_logits_array, aggregation_metrics)
    def aggregate_train(self, server_round: int, results: Iterable[Message], **kwargs):
        results_list = list(results)
        print(f"\n[SERVER] Round {server_round}: Aggregating Consensus")

        if not results_list:
            return None, {}

        # Collect logits from all clients
        logits_list = []
        client_info = []
        weights = []

        for i, msg in enumerate(results_list):
            client_arrays = msg.content["arrays"]
            try:
                client_logits = client_arrays["0"].numpy()
                logits_list.append(client_logits)

                # Track which client contributed
                if i < len(self.client_configs):
                    client_info.append(self.client_configs[i]['client_name'])

                # Extract weight from metrics (number of training examples)
                metrics = msg.content.get("metrics", {})
                num_examples = metrics.get("num-examples", 0)
                has_local_data = metrics.get("has_local_data", 0)

                # Weight by training examples - free riders get minimal weight
                # to prevent polluting consensus with untrained predictions
                if has_local_data:
                    # Clients with data: weight by training samples
                    weights.append(max(num_examples, 1))
                else:
                    # Free riders: minimal weight (don't pollute consensus)
                    weights.append(1)

            except (KeyError, IndexError) as e:
                pass

        # Compute consensus with weighted average
        if len(logits_list) > 0:
            # Normalize weights
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]

            # Weighted average of logits
            new_consensus = np.average(logits_list, axis=0, weights=normalized_weights)

            # Apply momentum to smooth consensus updates and reduce oscillation
            momentum = 0.6  # Weight for previous consensus
            if self.last_consensus_logits is not None and server_round > 1:
                consensus_logits = (momentum * self.last_consensus_logits +
                                   (1 - momentum) * new_consensus)
                print(f"[SERVER] ✓ Consensus with momentum (α={momentum}) from {len(logits_list)} clients")
            else:
                consensus_logits = new_consensus
                print(f"[SERVER] ✓ Initial consensus from {len(logits_list)} clients")
        else:
            return None, {}

        arrays_aggregated = ArrayRecord([consensus_logits])

        # Store consensus for checkpointing
        self.last_consensus_logits = consensus_logits

        metrics_aggregated = {
            "consensus_round": server_round,
            "num_clients": len(logits_list),
            "client_names": client_info,
            "weights": normalized_weights,
        }

        return arrays_aggregated, metrics_aggregated

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