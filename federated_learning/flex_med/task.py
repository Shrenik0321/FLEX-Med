"""flex-med: A Flower / PyTorch app with JSON-based client configuration."""

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

# Path to client configuration file - now configurable via environment variable
CONFIG_FILE_PATH = os.getenv(
    "FLEX_MED_CONFIG_FILE",
    "/content/drive/MyDrive/College/FLEX-Med/flex-med/flex_med/data.json"
)

# Public dataset path (remains constant) - now configurable
DATA_ROOT = os.getenv(
    "FLEX_MED_DATA_ROOT",
    "/content/drive/MyDrive/College/FLEX-Med/datasets"
)
PUBLIC_PATH = os.path.join(DATA_ROOT, "public_anchor")

# Checkpoint directory - where FL training state is saved
CHECKPOINT_DIR = os.getenv(
    "FLEX_MED_CHECKPOINT_DIR",
    "/content/drive/MyDrive/College/FLEX-Med/checkpoints"
)

# Model Constraints
NUM_CLASSES = 2   # 0: Hem (Healthy), 1: ALL (Leukemia)
IMG_SIZE = 224    # Resize all inputs to 224X224 for consistency

# <------------------------------------------ CLIENT CONFIGURATION LOADER ------------------------------------------>

def load_client_config(config_path: str = CONFIG_FILE_PATH) -> List[Dict]:
    """
    Load client configuration from JSON file.
    
    Returns:
        List of client configurations with metadata
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        clients = json.load(f)
    
    print(f"[Config] Loaded {len(clients)} client configurations")
    for client in clients:
        print(f"  - Client {client['id']}: {client['client_name']} ({client['model_type']}) - "
              f"Data: {'Yes' if client['has_local_data'] else 'No'}")
    
    return clients

def get_client_by_partition_id(partition_id: int, config_path: str = CONFIG_FILE_PATH) -> Dict:
    """
    Get client configuration by partition ID.
    
    Args:
        partition_id: The partition ID (0, 1, 2, ...)
        config_path: Path to the configuration file
    
    Returns:
        Dictionary containing client configuration
    """
    clients = load_client_config(config_path)
    
    if partition_id >= len(clients):
        raise ValueError(f"Partition ID {partition_id} exceeds number of clients ({len(clients)})")
    
    return clients[partition_id]

def load_model_checkpoint(model, model_path, device):
    """
    Load model from checkpoint, handling both formats:
    - Direct state_dict
    - Checkpoint with metadata
    """
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

# <------------------------------------------ MODEL DEFINITIONS ------------------------------------------>

def get_model_by_type(model_type: str, use_pretrained: bool = True):
    """
    Returns a model based on the model_type string.

    Args:
        model_type: String identifier ('resnet18', 'mobilenet_v2', 'efficientnet_b3')

    Returns:
        PyTorch model initialized for binary classification
    """
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

# Legacy functions for backward compatibility
def get_resnet():
    """Returns a ResNet-18 for Binary Classification."""
    return get_model_by_type('resnet18')

def get_mobilenet():
    """Returns a MobileNetV2 for Binary Classification."""
    return get_model_by_type('mobilenet_v2')

def get_efficientnet():
    """Returns an EfficientNet-B3 for Binary Classification."""
    return get_model_by_type('efficientnet_b3')

# <------------------------------------------ PUBLIC (ANCHOR) DATASET LOADER ------------------------------------------>

def load_public_dataset(batch_size=64):
    """
    Loads the Mixed Public Anchor dataset.
    Used for generating logits and consensus in FedMD.
    """
    if not os.path.exists(PUBLIC_PATH):
        raise FileNotFoundError(f"Public data not found at {PUBLIC_PATH}")

    dataset = datasets.ImageFolder(root=PUBLIC_PATH, transform=COMMON_TRANSFORM)
    
    # Shuffle=False is CRITICAL for FedMD so all clients see images in the same order
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return loader
    
# <------------------------------------------ PRIVATE DATASET LOADING FOR CLIENTS ------------------------------------------>

def load_private_dataset(partition_id: int, num_partitions: int, batch_size=32, 
                        config_path: str = CONFIG_FILE_PATH):
    """
    Loads private medical data based on the Client ID using data.json configuration.
    
    Args:
        partition_id: Client ID (0, 1, 2, ...)
        num_partitions: Total number of partitions (not used but kept for compatibility)
        batch_size: Batch size for data loaders
        config_path: Path to the configuration file
    
    Returns:
        Tuple of (trainloader, testloader) or (None, None) if client has no data
    """
    # Get client configuration
    client_config = get_client_by_partition_id(partition_id, config_path)
    
    # Check if client has local data
    if not client_config['has_local_data'] or client_config['dataset_path'] is None:
        print(f"[Client {partition_id}] {client_config['client_name']}: No private data (Free Rider)")
        return None, None
    
    data_path = client_config['dataset_path']
    
    if not os.path.exists(data_path):
        print(f"[Warning] Client {partition_id} data path does not exist: {data_path}")
        return None, None

    # Load the Full Dataset from folder
    full_dataset = datasets.ImageFolder(root=data_path, transform=PRIVATE_TRAIN_TRANSFORM)
    
    print(f"[Client {partition_id}] Loaded {len(full_dataset)} samples from {data_path}")
    
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

# <------------------------------------------ MODEL TRAINING & TESTING ------------------------------------------>

def train(model, trainloader, epochs, lr, device):
    """Train model on private data."""
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

def test(model, testloader, device):
    """Evaluate model on validation data."""
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

# <------------------------------------------ KNOWLEDGE DISTILLATION ------------------------------------------>

def distill_knowledge(model, public_loader, consensus_logits, device, epochs, lr, temperature):
    """
    Distill consensus knowledge into the local model.
    """
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

def get_public_logits(model, public_loader, device):
    """Generate logits on the public dataset."""
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

# <------------------------------------------ CHECKPOINT UTILITIES ------------------------------------------>

def save_checkpoint(
    checkpoint_dir: str,
    current_round: int,
    consensus_logits: np.ndarray,
    eval_history: List[Dict],
    training_metrics: Dict
):
    """
    Save FL training checkpoint to enable resumption after interruption.

    Args:
        checkpoint_dir: Directory to save checkpoints
        current_round: Current training round number
        consensus_logits: Current consensus matrix
        eval_history: List of evaluation results per round
        training_metrics: Additional training metrics
    """
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

    print(f"[Checkpoint] Saved checkpoint for round {current_round}")
    print(f"  - Path: {checkpoint_path}")
    print(f"  - Size: {os.path.getsize(checkpoint_path) / 1024:.2f} KB")

def load_checkpoint(checkpoint_dir: str) -> Optional[Dict]:
    """
    Load the latest FL training checkpoint.

    Args:
        checkpoint_dir: Directory containing checkpoints

    Returns:
        Checkpoint dictionary or None if no checkpoint exists
    """
    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")

    if not os.path.exists(latest_path):
        print(f"[Checkpoint] No checkpoint found at {latest_path}")
        return None

    try:
        checkpoint = torch.load(latest_path, map_location='cpu')
        print(f"[Checkpoint] Loaded checkpoint from round {checkpoint['round']}")
        print(f"  - Consensus shape: {checkpoint['consensus_logits'].shape}")
        print(f"  - Eval history length: {len(checkpoint['eval_history'])}")
        return checkpoint
    except Exception as e:
        print(f"[Checkpoint] Error loading checkpoint: {e}")
        return None

def clear_checkpoints(checkpoint_dir: str):
    """
    Clear all checkpoints from the directory.

    Args:
        checkpoint_dir: Directory containing checkpoints
    """
    if os.path.exists(checkpoint_dir):
        import shutil
        shutil.rmtree(checkpoint_dir)
        print(f"[Checkpoint] Cleared all checkpoints from {checkpoint_dir}")

# <------------------------------------------ FEDMD STRATEGY ------------------------------------------>

class FedMDStrategy(Strategy):
    """
    Federated Model Distillation (FedMD) Strategy with support for heterogeneous models.
    
    This strategy implements knowledge distillation across different model architectures
    using a shared public dataset as the transfer medium.
    """
    
    def __init__(self, config_path: str = CONFIG_FILE_PATH, checkpoint_dir: str = CHECKPOINT_DIR):
        """
        Initialize FedMD Strategy with client configuration and checkpoint support.

        Args:
            config_path: Path to the client configuration JSON file
            checkpoint_dir: Directory for saving/loading checkpoints
        """
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

        print(f"\n[Strategy] Initialized with {self.num_clients} clients:")
        for i, client in enumerate(self.client_configs):
            print(f"  Client {i}: {client['client_name']} - {client['model_type']} - "
                  f"Data: {'✓' if client['has_local_data'] else '✗'}")

        # Try to load checkpoint
        checkpoint = load_checkpoint(checkpoint_dir)
        if checkpoint:
            self.start_round = checkpoint['round'] + 1  # Resume from next round
            self.eval_history = checkpoint['eval_history']
            self.last_consensus_logits = checkpoint['consensus_logits']
            print(f"\n[Strategy] Resuming from round {self.start_round}")
        else:
            print(f"\n[Strategy] Starting fresh training")

    def configure_evaluate(
        self, 
        server_round: int, 
        arrays: Optional[ArrayRecord],
        config: ConfigRecord,
        grid: Grid
    ) -> Iterable[Message]:
        """
        Configure evaluation for all clients.
        This should be called after each training round.
        """
        print(f"\n[Server] Round {server_round}: Configuring Evaluation")
        
        node_ids = list(grid.get_node_ids())
        messages = []
        
        # Send evaluation request to all clients
        for i, node_id in enumerate(node_ids):
            client_name = self.client_configs[i]['client_name'] if i < len(self.client_configs) else f"Client {i}"
            
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
            print(f"  → Sending eval request to {client_name} (Node {node_id})")
        
        return messages

    def aggregate_evaluate(
        self,
        server_round: int,
        results: Iterable[Message],
        **kwargs
    ):
        """Aggregate evaluation results with detailed tracking."""
        results_list = list(results)

        if not results_list:
            print(f"[Server] Round {server_round}: No evaluation results")
            return {"loss": None, "metrics": {}}

        print(f"\n[Server] Round {server_round}: Aggregating Evaluation Results")
        print("="*60)

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

        # Print detailed results
        print(f"\n[Server] Clients with Private Data ({len(clients_with_data)}):")
        for c in clients_with_data:
            client_name = self.client_configs[c['client_id']]['client_name']
            print(f"  {client_name:15} - Loss: {c['loss']:.4f}, Acc: {c['acc']:.4f} ({c['examples']} samples)")
        
        if free_riders:
            print(f"\n[Server] Free Riders ({len(free_riders)}):")
            for c in free_riders:
                client_name = self.client_configs[c['client_id']]['client_name']
                print(f"  {client_name:15} - Loss: {c['loss']:.4f}, Acc: {c['acc']:.4f} (public data)")
        
        print(f"\n[Server] Overall Metrics:")
        print(f"  - Average Loss: {avg_loss:.4f}")
        print(f"  - Average Accuracy: {avg_acc:.4f} ({avg_acc*100:.2f}%)")
        print(f"  - Total Examples: {total_examples}")
        print("="*60)

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

    def aggregate_train(
        self,
        server_round: int,
        results: Iterable[Message],
        **kwargs
    ) -> Tuple[Optional[ArrayRecord], dict]:
        """
        Aggregate logits from all clients into consensus logits with weighted averaging.
        
        Uses dataset size as weights to give more influence to clients with more data.
        """
        
        results_list = list(results)
        print(f"\n[Server] Round {server_round}: Aggregating Logits")

        if not results_list:
            print("[Server] Warning: No results to aggregate!")
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
                print(f"Error extracting logits from client {i}: {e}")
        
        # Compute consensus with weighted average
        if len(logits_list) > 0:
            # Normalize weights
            total_weight = sum(weights)
            normalized_weights = [w / total_weight for w in weights]
            
            # Weighted average of logits
            consensus_logits = np.average(logits_list, axis=0, weights=normalized_weights)
            
            print(f"[Server] Consensus shape: {consensus_logits.shape}")
            print(f"[Server] Contributors: {', '.join(client_info)}")
            print(f"[Server] Weights: {[f'{w:.3f}' for w in normalized_weights]}")
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

    def configure_train(self, server_round, arrays, config, grid) -> Iterable[Message]:
        """Configure the next round of federated training."""
        print(f"\n{'='*60}")
        print(f"[Server] Round {server_round}: Configuring Training")
        print(f"{'='*60}")
        
        node_ids = list(grid.get_node_ids())
        messages = []
        
        for i, node_id in enumerate(node_ids):
            client_name = self.client_configs[i]['client_name'] if i < len(self.client_configs) else f"Client {i}"
            
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
            print(f"  → Sending to {client_name} (Node {node_id})")
        
        return messages
    
    def save_evaluation_history(self, filepath="evaluation_history.json"):
        """Save evaluation history for analysis."""
        with open(filepath, 'w') as f:
            json.dump(self.eval_history, f, indent=2)
        print(f"[Strategy] Evaluation history saved to {filepath}")

    def summary(self) -> str:
        return f"FLEX-Med: Federated Knowledge Distillation Strategy ({self.num_clients} clients)"