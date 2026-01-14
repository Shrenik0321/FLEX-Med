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

# Augmentation for Private Training
PRIVATE_TRAIN_TRANSFORM = Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

    if not os.path.exists(latest_path):
        return None

    try:
        checkpoint = torch.load(latest_path, map_location='cpu')
        return checkpoint
    except Exception as e:
        return None

# Clear all checkpoints from the directory
# Args: checkpoint_dir - Directory containing checkpoints
# Returns: None (removes directory)
def clear_checkpoints(checkpoint_dir: str):
    if os.path.exists(checkpoint_dir):
        import shutil
        shutil.rmtree(checkpoint_dir)

# <------------------------------------------ MODEL TRAINING & TESTING FUNCTION DEFINITIONS ------------------------------------------>

# Train model on private client data with AdamW optimizer
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
    criterion = nn.CrossEntropyLoss()
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
                num_examples = metrics.get("num-examples", 1)
                # Weight by number of training examples (clients with more data have more influence)
                weights.append(max(num_examples, 1))  # Ensure non-zero weight

            except (KeyError, IndexError) as e:
                pass

        # Compute consensus with weighted average
        if len(logits_list) > 0:
            # Normalize weights
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]

            # Weighted average of logits
            consensus_logits = np.average(logits_list, axis=0, weights=normalized_weights)

            print(f"[SERVER] ✓ Consensus generated from {len(logits_list)} clients")
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