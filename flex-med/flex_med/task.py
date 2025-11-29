"""flex-med: A Flower / PyTorch app."""

import torch
import numpy as np
import io
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from datasets import load_dataset
from typing import List, Tuple, Optional, Iterable
import uuid
import time

# Flower imports
from flwr.common import (
    Message,
    Metadata,
    RecordDict,
    ArrayRecord,
    ConfigRecord,
)
from flwr.server import Grid
from flwr.serverapp.strategy import Strategy

# class Net(nn.Module):
#     """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)


# fds = None  # Cache FederatedDataset

# pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


# def apply_transforms(batch):
#     """Apply transforms to the partition from FederatedDataset."""
#     batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
#     return batch


# def load_data(partition_id: int, num_partitions: int):
#     """Load partition CIFAR10 data."""
#     # Only initialize `FederatedDataset` once
#     global fds
#     if fds is None:
#         partitioner = IidPartitioner(num_partitions=num_partitions)
#         fds = FederatedDataset(
#             dataset="uoft-cs/cifar10",
#             partitioners={"train": partitioner},
#         )
#     partition = fds.load_partition(partition_id)
#     # Divide data on each node: 80% train, 20% test
#     partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
#     # Construct dataloaders
#     partition_train_test = partition_train_test.with_transform(apply_transforms)
#     trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
#     testloader = DataLoader(partition_train_test["test"], batch_size=32)
#     return trainloader, testloader


# def train(net, trainloader, epochs, lr, device):
#     """Train the model on the training set."""
#     net.to(device)  # move model to GPU if available
#     criterion = torch.nn.CrossEntropyLoss().to(device)
#     optimizer = torch.optim.Adam(net.parameters(), lr=lr)
#     net.train()
#     running_loss = 0.0
#     for _ in range(epochs):
#         for batch in trainloader:
#             images = batch["img"].to(device)
#             labels = batch["label"].to(device)
#             optimizer.zero_grad()
#             loss = criterion(net(images), labels)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#     avg_trainloss = running_loss / len(trainloader)
#     return avg_trainloss


# def test(net, testloader, device):
#     """Validate the model on the test set."""
#     net.to(device)
#     criterion = torch.nn.CrossEntropyLoss()
#     correct, loss = 0, 0.0
#     with torch.no_grad():
#         for batch in testloader:
#             images = batch["img"].to(device)
#             labels = batch["label"].to(device)
#             outputs = net(images)
#             loss += criterion(outputs, labels).item()
#             correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
#     accuracy = correct / len(testloader.dataset)
#     loss = loss / len(testloader)
#     return loss, accuracy


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
                client_logits = client_arrays[0]
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


def load_public_dataset(batch_size=64):
    """
    Loads the shared CIFAR-10 public dataset (anchor dataset).
    Used for generating logits and consensus in FedMD.
    """
    ds = load_dataset("cifar10", split="train")
    ds = ds.with_transform(lambda batch: hf_apply_transform(batch, PUBLIC_TRANSFORM))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return loader

def load_CNMC_dataset():

def load_ALLIDB2_dataset():

def get_public_logits(model, public_loader, device):
    """
    Generate logits on the public dataset.
    This is what clients send to the server (NOT model weights).
    """
    model.to(device)
    model.eval()
    
    all_logits = []
    with torch.no_grad():
        for batch in public_loader:
            images = batch["img"].to(device)
            outputs = model(images)
            all_logits.append(outputs.cpu().numpy())
    
    return np.concatenate(all_logits)


def distill_knowledge():
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