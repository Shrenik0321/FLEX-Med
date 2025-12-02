"""flex-med: A Flower / PyTorch app."""

import torch
import numpy as np
import os
import torch.nn as nn
from torchvision import models, datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
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

# <------------------------------------------ CONFIGURATION ------------------------------------------>

# Paths to the data created by the stratification script
DATA_ROOT = "/content/drive/MyDrive/College/Datasets/fed_data"
PUBLIC_PATH = os.path.join(DATA_ROOT, "public_anchor")
CLIENT_0_PATH = os.path.join(DATA_ROOT, "client_allidb") # ResNet (ALL-IDB2)
CLIENT_1_PATH = os.path.join(DATA_ROOT, "client_cnmc")   # MobileNet (CNMC)

# Model Constraints
NUM_CLASSES = 2   # 0: Hem (Healthy), 1: ALL (Leukemia)
IMG_SIZE = 128    # Resize all inputs to 128x128 for consistency

# <------------------------------------------ DATA TRANSFORMS ------------------------------------------>

# Standardize inputs: Resize -> Tensor -> ImageNet Normalization
# We use this for Public Data and Validation
COMMON_TRANSFORM = Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Augmentation for Private Training
# Helps models learn robust features from small medical datasets
PRIVATE_TRAIN_TRANSFORM = Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def hf_apply_transform(batch, transform=COMMON_TRANSFORM):
    """(Legacy Helper) Not used with ImageFolder, but kept for compatibility if needed."""
    batch["img"] = [transform(img) for img in batch["img"]]
    return batch

# <------------------------------------------ MODEL DEFINITIONS ------------------------------------------>

def get_resnet():
    """
    Returns a ResNet-18 for Binary Classification (ALL vs Healthy).
    """
    # Load standard ResNet18
    model = models.resnet18(weights=None)
    
    # Reset the final fully connected layer for 2 classes
    # ResNet18 fc in_features is usually 512
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    
    return model

def get_mobilenet():
    """
    Returns a MobileNetV2 for Binary Classification.
    """
    model = models.mobilenet_v2(weights=None)
    
    # MobileNetV2 classifier is a Sequential block. 
    # Index [1] is the Linear layer.
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    
    return model

def get_densenet():
    """
    Returns a DenseNet-121 for Binary Classification.
    """
    model = models.densenet121(weights=None)
    
    # DenseNet classifier is a single Linear layer
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    
    return model

# <------------------------------------------ PUBLIC (ANCHOR) DATASET LOADER ------------------------------------------>

def load_public_dataset(batch_size=64):
    """
    Loads the Mixed Public Anchor dataset (CNMC + ALL-IDB subset).
    Used for generating logits and consensus in FedMD.
    """
    if not os.path.exists(PUBLIC_PATH):
        raise FileNotFoundError(f"Public data not found at {PUBLIC_PATH}. Run the data setup script first.")

    # ImageFolder automatically uses subfolders 'all' and 'hem' as labels 0 and 1
    dataset = datasets.ImageFolder(root=PUBLIC_PATH, transform=COMMON_TRANSFORM)
    
    # Shuffle=False is CRITICAL for FedMD so all clients see images in the same order
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return loader
    
# <------------------------------------------ PRIVATE DATASET LOADING FOR CLIENTS ------------------------------------------>

def load_private_dataset(partition_id: int, num_partitions: int, batch_size=32):
    """
    Loads private medical data based on the Client ID.
    
    ID 0: ALL-IDB2 (ResNet)
    ID 1: CNMC (MobileNet)
    ID 2: None (DenseNet Free-Rider)
    """
    data_path = None
    
    if partition_id == 0:
        data_path = CLIENT_0_PATH # ALL-IDB2
    elif partition_id == 1:
        data_path = CLIENT_1_PATH # CNMC
    elif partition_id == 2:
        # DenseNet (Free Rider) has NO private training data
        return None, None

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Client {partition_id} data not found at {data_path}")

    # Load the Full Dataset from folder
    full_dataset = datasets.ImageFolder(root=data_path, transform=PRIVATE_TRAIN_TRANSFORM)
    
    # Create a Train/Test split (e.g., 80% Train, 20% Test)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    # We use a fixed seed generator for reproducibility of splits
    generator = torch.Generator().manual_seed(42)
    train_ds, test_ds = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=generator)
    
    # Note: Ideally, test_ds should use COMMON_TRANSFORM (no augmentation), 
    # but for simplicity in this script, we use the same transform.
    
    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader

# <------------------------------------------ DEFINE MODEL TRAINING ------------------------------------------>

def train(model, trainloader, epochs, lr, device):
    """Train model on private data."""
    if trainloader is None:
        return 0.0  # Skip training for Free Rider
        
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
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

# <------------------------------------------ DEFINE MODEL TESTING ------------------------------------------>

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

# <------------------------------------------ STRATEGY HELPERS ------------------------------------------>

def distill_knowledge(model, public_loader, consensus_logits, device, epochs, lr, temperature):
    """
    Distill consensus knowledge into the local model.
    """
    model.to(device) # Move model to device
    model.train() # Set model to training mode

    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 

    # Convert consensus to tensor
    consensus_tensor = torch.from_numpy(consensus_logits).float()

    total_loss = 0.0
    idx = 0

    # Distillation Loop
    for epoch in range(epochs):
        # Sends the public dataset to the model
        for images, _ in public_loader: # Ignore the folder labels, we use Consensus!
            images = images.to(device)
            batch_size = images.size(0)
            
            # Safe indexing in case loader drops last incomplete batch
            if idx + batch_size > len(consensus_tensor):
                break

            # Get consensus soft targets for this batch
            batch_consensus = consensus_tensor[idx:idx+batch_size].to(device)
            
            # Forward pass
            student_logits = model(images) 
            
            # KL Divergence Loss (distillation loss)
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
    """
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
        """Aggregate evaluation results."""
        results_list = list(results)

        if not results_list:
            print(f"[Server] Round {server_round}: No evaluation results")
            return {"loss": None, "metrics": {}}

        print(f"[Server] Round {server_round}: Aggregating evaluation from {len(results_list)} clients")

        total_loss = 0.0
        total_acc = 0.0
        total_examples = 0

        for msg in results_list:
            metrics = msg.content.get("metrics", {})
            eval_loss = metrics.get("eval_loss", 0.0)
            eval_acc = metrics.get("eval_acc", 0.0)
            num_examples = metrics.get("num-examples", 0)

            total_loss += eval_loss * num_examples
            total_acc += eval_acc * num_examples
            total_examples += num_examples

        if total_examples == 0:
            return {"loss": None, "metrics": {}}

        avg_loss = total_loss / total_examples
        avg_acc = total_acc / total_examples

        print(f"[Server] Aggregated Eval - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}")

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
        """Aggregate logits from all clients into consensus logits."""
        
        results_list = list(results)
        print(f"\n[Server] Round {server_round}: Aggregating Logits")

        if not results_list:
            print("[Server] Warning: No results to aggregate!")
            return None, {}

        # Collect logits from all clients
        logits_list = []

        for msg in results_list:
            client_arrays = msg.content["arrays"]
            try:
                # Extract logits from ArrayRecord (key "0")
                client_logits = client_arrays["0"].numpy()
                logits_list.append(client_logits)
            except (KeyError, IndexError) as e:
                print(f"Error extracting logits: {e}")
        
        # Compute consensus (simple average)
        if len(logits_list) > 0:
            consensus_logits = np.mean(logits_list, axis=0)
            print(f"[Server] Consensus logits shape: {consensus_logits.shape}")
        else:
            return None, {}
        
        # Pack consensus for next round
        arrays_aggregated = ArrayRecord([consensus_logits])
        
        metrics_aggregated = {
            "consensus_round": server_round,
            "num_clients": len(logits_list),
        }
        
        return arrays_aggregated, metrics_aggregated

    def configure_evaluate(self, server_round, arrays, config, grid) -> Iterable[Message]:
        return [] 

    def configure_train(self, server_round, arrays, config, grid) -> Iterable[Message]:
        """Configure the next round of federated training."""
        print(f"\n{'='*60}")
        print(f"[Server] Round {server_round}: Configuring Training")
        print(f"{'='*60}")
        
        node_ids = list(grid.get_node_ids())
        messages = []
        for node_id in node_ids:
            content = RecordDict({
                "arrays": arrays,  # Consensus logits
                "config": config   # Training config
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
        
        print(f"[Server] Sent consensus to {len(messages)} clients")
        return messages

    def summary(self) -> str:
        return "FLEX-Med: Federated Knowledge Distillation Strategy"