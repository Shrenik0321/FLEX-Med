"""flex-med: A Flower / PyTorch app."""

import torch
import numpy as np
import io
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from datasets import load_dataset
from typing import Tuple, Optional, Iterable
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

# <------------------------------------------ DATA TRANSFORMS ------------------------------------------>

PUBLIC_TRANSFORM = Compose([
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

CLIENT_TRANSFORM = Compose([
    ToTensor(),
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

def hf_apply_transform(batch, transform=PUBLIC_TRANSFORM):
    """Apply transform to HuggingFace dataset batch."""
    batch["img"] = [transform(img) for img in batch["img"]]
    return batch

# <------------------------------------------ MODEL DEFINITIONS ------------------------------------------>

def get_resnet():
    """
    Returns a ResNet-18 modified for CIFAR-10 (32x32 images).
    """
    # Load standard ResNet18, not pretrained (we train from scratch or distill)
    model = models.resnet18(weights=None)
    
    # 1. Modify the first convolution to handle 32x32 images
    # Original: kernel_size=7, stride=2, padding=3 (meant for 224x224)
    # Modified: kernel_size=3, stride=1, padding=1
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # 2. Remove the first MaxPool layer to preserve spatial dimensions
    model.maxpool = nn.Identity()
    
    # 3. Modify the final fully connected layer for 10 classes
    model.fc = nn.Linear(model.fc.in_features, 10)
    
    return model

def get_mobilenet():
    """
    Returns a MobileNetV2 modified for CIFAR-10.
    """
    model = models.mobilenet_v2(weights=None)
    
    # 1. Modify the first convolution layer
    # Access the first layer of the features block
    # Original stride is usually 2, we change to 1 for small images
    model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    
    # 2. Modify the classifier
    # MobileNetV2 classifier is a Sequential block, the last layer is Linear
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
    
    return model

# <------------------------------------------ PUBLIC (ANCHOR) DATASET LOADER ------------------------------------------>

def load_public_dataset(batch_size=64):
    """
    Loads the shared CIFAR-10 public dataset (anchor dataset).
    Used for generating logits and consensus in FedMD.
    """
    ds = load_dataset("cifar10", split="train")
    ds = ds.with_transform(lambda batch: hf_apply_transform(batch, PUBLIC_TRANSFORM))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return loader
    
# <------------------------------------------ PRIVATE DATASET LOADING FOR CLIENTS ------------------------------------------>
_fds_cache = None

def _apply_partition_transform(batch):
    """Apply client-specific transform to private data."""
    batch["img"] = [CLIENT_TRANSFORM(img) for img in batch["img"]]
    return batch

def load_private_dataset(partition_id: int, num_partitions: int, batch_size=32):
    """
    Loads a private CIFAR-10 partition for a specific client.
    Different partition_id => different private dataset (simulates data heterogeneity).
    """
    global _fds_cache
    
    if _fds_cache is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        _fds_cache = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    
    partition = _fds_cache.load_partition(partition_id)
    split = partition.train_test_split(test_size=0.2, seed=42)
    split = split.with_transform(_apply_partition_transform)
    
    trainloader = DataLoader(split["train"], batch_size=batch_size, shuffle=True)
    testloader = DataLoader(split["test"], batch_size=batch_size, shuffle=False)
    
    return trainloader, testloader

# <------------------------------------------ DEFINE MODEL TRAINING ------------------------------------------>

def train(model, trainloader, epochs, lr, device):
    """Train model on private data (standard supervised learning)."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    total_loss = 0.0
    
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
    
    avg_loss = total_loss / (len(trainloader) * epochs)
    return avg_loss

# <------------------------------------------ DEFINE MODEL TESTING ------------------------------------------>

def test(model, testloader, device):
    """Evaluate model on validation data."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    correct, total, total_loss = 0, 0, 0.0
    
    model.eval()
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total if total > 0 else 0.0
    loss = total_loss / len(testloader) if len(testloader) > 0 else 0.0
        
    return loss, accuracy

# <------------------------------------------ STRATEGY HELPERS ------------------------------------------>

def distill_knowledge(model, public_loader,consensus_logits,device, epochs, lr, temperature):
    """
    Distill consensus knowledge into the local model.
    
    This is the CORE of FedMD: instead of averaging weights,
    we teach each model to mimic the consensus predictions.
    
    Args:
        model: Local client model
        public_loader: DataLoader for public dataset
        consensus_logits: Aggregated logits from server (numpy array)
        device: torch device (CPU or CUDA)
        epochs: Number of distillation epochs
        lr: Learning rate for distillation
        temperature: Temperature for softmax (higher = softer targets)
    
    Returns:
        Average distillation loss
    """

    model.to(device) # Move model to device
    model.train() # Set model to training mode

    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # Initialize optimizer - Used for the manipulation of model weights

    # Convert consensus to tensor
    consensus_tensor = torch.from_numpy(consensus_logits).float()

    total_loss = 0.0
    idx = 0

    # Distillation Loop
    for epoch in range(epochs):

        # Sends the public dataset to the model
        for batch in public_loader:
            images = batch["img"].to(device)
            batch_size = images.size(0)
            
            # Get consensus soft targets for this batch
            batch_consensus = consensus_tensor[idx:idx+batch_size].to(device)
            
            # Forward pass
            student_logits = model(images) # Model outputs its own logits for these public images
            
            # KL Divergence Loss (distillation loss)
            # This teaches the model to match the consensus distribution
            # The teacher gives soft probabilities and the student tried to match them

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
        
        idx = 0  # Reset for next epoch

    avg_loss = total_loss / (len(public_loader) * epochs)
    return avg_loss

def get_public_logits(model, public_loader, device):
    """
    Generate logits on the public dataset.
    This is what clients send to the server (NOT model weights).
    """
    if public_loader is None:
      raise ValueError("public_loader is None. Make sure load_public_dataset(...) returns a DataLoader.")

    model.to(device)
    model.eval()
    
    all_logits = []
    with torch.no_grad():
        for batch in public_loader:
            images = batch["img"].to(device)
            outputs = model(images)
            all_logits.append(outputs.cpu().numpy())
    
    return np.concatenate(all_logits)

# <------------------------------------------ DEFINE FEDMD STRATEGY ------------------------------------------>

class FedMDStrategy(Strategy):
    def __init__(self):
        super().__init__()

    def aggregate_evaluate(
        self,
        server_round: int,
        results: Iterable[Message],
        **kwargs
    ):
        """
        Aggregate evaluation results for ServerApp Strategy.

        Returns:
            MetricRecord (dict) or None.
        """
        results_list = list(results)

        if not results_list:
            print(f"[Server] Round {server_round}: No evaluation results")
            return {
                "loss": None,
                "metrics": {}
            }

        print(f"[Server] Round {server_round}: Aggregating evaluation from {len(results_list)} clients")

        total_loss = 0.0
        total_acc = 0.0
        total_examples = 0

        # Aggregate weighted metrics
        for msg in results_list:
            metrics = msg.content.get("metrics", {})

            eval_loss = metrics.get("eval_loss", 0.0)
            eval_acc = metrics.get("eval_acc", 0.0)
            num_examples = metrics.get("num-examples", 0)

            total_loss += eval_loss * num_examples
            total_acc += eval_acc * num_examples
            total_examples += num_examples

        # If no examples provided
        if total_examples == 0:
            return {
                "loss": None,
                "metrics": {}
            }

        # Compute weighted average
        avg_loss = total_loss / total_examples
        avg_acc = total_acc / total_examples

        print(f"[Server] Aggregated Eval - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")

        # This is the required MetricRecord format
        return {
            "loss": avg_loss,
            "metrics": {
                "eval_acc": avg_acc,
                "num_clients": len(results_list),
                "total_examples": total_examples,
            }
        }

    def aggregate_train(
        self,
        server_round: int,
        results: Iterable[Message],
        **kwargs
    ) -> Tuple[Optional[ArrayRecord], dict]:
        """
        Aggregate logits from all clients into consensus logits.
        
        This is the heart of FedMD: instead of averaging weights,
        we average predictions (logits) on the public dataset.
        """

        # Convert iterator to list
        results_list = list(results)

        print(f"\n[Server] Round {server_round}: Aggregating Logits")
        print(f"[Server] Received responses from {len(results_list)} clients")

        if not results_list:
            print("[Server] Warning: No results to aggregate!")
            return None, {}

        # Collect logits from all clients
        logits_list = []

        for msg in results_list:
            client_arrays = msg.content["arrays"]
            
            # Method 1: Direct access (cleaner)
            try:
                client_logits = client_arrays["0"].numpy()
                logits_list.append(client_logits)
            except (KeyError, IndexError):
                # Method 2: Fallback to manual deserialization
                logits_wrapper = client_arrays["0"]
                bytes_io = io.BytesIO(logits_wrapper.data)
                client_logits = np.load(bytes_io, allow_pickle=False)
                logits_list.append(client_logits)
        
        # Compute consensus (simple average)
        consensus_logits = np.mean(logits_list, axis=0)
        
        print(f"[Server] Consensus logits shape: {consensus_logits.shape}")
        print(f"[Server] Consensus created from {len(logits_list)} clients")
        
        # Pack consensus for next round
        arrays_aggregated = ArrayRecord([consensus_logits])
        
        metrics_aggregated = {
            "consensus_round": server_round,
            "num_clients": len(logits_list),
            "consensus_mean": float(consensus_logits.mean()),
            "consensus_std": float(consensus_logits.std())
        }
        
        return arrays_aggregated, metrics_aggregated

    def configure_evaluate(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid
    ) -> Iterable[Message]:
        """Configure evaluation round (optional for FedMD)."""
        # Skip evaluation for now - clients evaluate locally
        return [] 

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid
    ) -> Iterable[Message]:
        """
        Configure the next round of federated training.
        
        Sends consensus logits to all available clients.
        Clients will use these for knowledge distillation.
        """
        print(f"\n{'='*60}")
        print(f"[Server] Round {server_round}: Configuring Training")
        print(f"{'='*60}")
        
        # Get all available node IDs
        node_ids = list(grid.get_node_ids())
        print(f"[Server] Available nodes: {node_ids}")
        
        # Create messages for each client using grid.create_message()
        messages = []
        for node_id in node_ids:
          # Package consensus logits and config
          content = RecordDict({
              "arrays": arrays,  # Consensus logits from previous round
              "config": config   # Training configuration
          })
            
          # Use grid.create_message() which automatically sets run_id and src_node_id
          msg = Message(
            metadata=Metadata(
              run_id=0,                        # <--- FIX: Use 0 for Simulation
              message_id=str(uuid.uuid4()),    # Generate unique ID
              src_node_id=0,                   # Sender ID (Server)
              dst_node_id=node_id,             # Recipient ID (Client)
              reply_to_message_id="",          # No reply needed
              group_id=str(server_round),      # Group by round
              ttl=86400.0,                     # TTL in seconds (1 day)
              message_type="train",            # Action type
              created_at=time.time(),          # Action type
            ),
            content=content,
          )

          messages.append(msg)
        
        print(f"[Server] Sent consensus to {len(messages)} clients")
        return messages

    def summary(self) -> str:
        """Return strategy summary."""
        return "FLEX-Med: Federated Knowledge Distillation Strategy"