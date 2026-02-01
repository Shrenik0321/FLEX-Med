# This file contains the task definition for the federated learning application.

import torch
import numpy as np
import os
import json
import torch.nn as nn
from torchvision import models, datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
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
from flwr_datasets.partitioner import DirichletPartitioner
from datasets import Dataset
from flex_med.utils.config import (
    BASE_PATH,
    CLIENT_INFO_FILE_PATH,
    DATASET_FILE_PATH,
    PUBLIC_ANCHOR_DATASET_PATH,
    PUBLIC_TEST_DATASET_PATH,
    LOCAL_TRAIN_DATASET_PATH,
    MODEL_CHECKPOINT_FILE_PATH,
    GRAPHS_OUTPUT_DIR,
    NUM_CLASSES,
    IMG_SIZE,
    CLIENT_INFO_FILE_PATH,
    MODEL_SUITABILITY_SCORES,
    TRAIN_LOSS_WEIGHT,
    DISTILL_LOSS_WEIGHT,
    CONSENSUS_MOMENTUM,
    DIRICHLET_ALPHA,
    DIRICHLET_SEED,
    DIRICHLET_MIN_PARTITION_SIZE,
)
from datetime import datetime

# ============================================
# SUPABASE DATABASE CLIENT (for metrics storage)
# ============================================
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    print("[Database] Warning: supabase-py not installed. Run: pip install supabase")
    SUPABASE_AVAILABLE = False


def get_supabase_client() -> Optional[Client]:
    """
    Initialize Supabase client from environment variables.

    Returns:
        Supabase client or None if credentials not available
    """
    if not SUPABASE_AVAILABLE:
        return None

    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')

    if not supabase_url or not supabase_key:
        print("[Database] Warning: SUPABASE_URL or SUPABASE_KEY not set")
        print("[Database] Metrics will not be saved to database")
        return None

    try:
        client = create_client(supabase_url, supabase_key)
        print(f"[Database] ✓ Connected to Supabase")
        return client
    except Exception as e:
        print(f"[Database] Error connecting to Supabase: {e}")
        return None


# Initialize global Supabase client (reused across calls)
SUPABASE_CLIENT = get_supabase_client()

# Get simulation ID from environment (passed from backend API)
SIMULATION_ID = None
env_sim_id = os.getenv('FLEX_MED_SIMULATION_ID')
if env_sim_id:
    try:
        SIMULATION_ID = int(env_sim_id)
    except ValueError:
        pass

# Fallback: Try to read from config file if environment variable is missing
if SIMULATION_ID is None:
    try:
        env_config_path = os.getenv('FLEX_MED_CONFIG_FILE')
        if env_config_path and os.path.exists(env_config_path):
            with open(env_config_path, 'r') as f:
                config_data = json.load(f)
                if isinstance(config_data, dict) and "simulation_id" in config_data:
                    SIMULATION_ID = config_data["simulation_id"]
                    print(f"[Database] Simulation ID loaded from config file: {SIMULATION_ID}")
    except Exception as e:
        print(f"[Database] Error reading simulation_id from config: {e}")

if SIMULATION_ID is not None:
    print(f"[Database] Final Simulation ID: {SIMULATION_ID}")
else:
    print("[Database] Warning: No FLEX_MED_SIMULATION_ID set")

# <------------------------------------------ DATA TRANSFORMS ------------------------------------------>

# Standardize inputs: Resize -> Tensor -> ImageNet Normalization
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

# <------------------------------------------ UTILITY FUNCTIONS ------------------------------------------>

# Retrieves all client metadata (model type, dataset paths.)
# Args: config_path - Path to client configuration JSON file
# Returns: List of client configuration dictionaries
def load_client_config(config_path: str = CLIENT_INFO_FILE_PATH) -> List[Dict]:
    """
    Load client configuration with priority:
    1. Database fetch using SUPABASE_CLIENT and SIMULATION_ID - Preferred
    2. FLEX_MED_CONFIG_FILE environment variable (file path) - Fallback 1 (deprecated)
    3. Default config_path parameter - Fallback 2 (deprecated)
    """

    # Priority 1: Database fetch if SUPABASE is available
    if SUPABASE_CLIENT is not None and SIMULATION_ID is not None:
        try:
            print(f"[Config] Attempting to load clients from database for simulation {SIMULATION_ID}")

            # Fetch client_simulation_metrics joined with clients table
            response = SUPABASE_CLIENT.from_('client_simulation_metrics') \
                .select('client_id, clients(*)') \
                .eq('simulation_id', SIMULATION_ID) \
                .execute()

            if response.data and len(response.data) > 0:
                # Extract client data from joined response
                clients = []
                for record in response.data:
                    client_data = record.get('clients')
                    if client_data:
                        clients.append(client_data)

                if clients:
                    # Sanitize paths for current environment
                    clients = sanitize_client_paths(clients)
                    print(f"[Config] ✓ Loaded {len(clients)} clients from database")
                    return clients

            print("[Config] No clients found in database, falling back to file-based config")

        except Exception as e:
            print(f"[Config] Error fetching clients from database: {e}")
            print("[Config] Falling back to file-based config")

    # Priority 2: Check for file path from environment variable (deprecated)
    env_config_path = os.getenv('FLEX_MED_CONFIG_FILE')
    if env_config_path:
        config_path = env_config_path
        print(f"[Config] Using file-based config from: {config_path} (DEPRECATED)")

    # Priority 3: Use default config_path parameter (deprecated)
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found at {config_path}. "
            f"Please ensure database is configured or config file exists."
        )

    print(f"[Config] Loading from file: {config_path} (DEPRECATED - use database instead)")
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    # Handle both original (List) and new (Dict with 'clients' key) formats
    if isinstance(config_data, dict) and "clients" in config_data:
        clients = config_data["clients"]
    else:
        clients = config_data

    # Sanitize paths for current environment
    clients = sanitize_client_paths(clients)

    print(f"[Config] ✓ Loaded {len(clients)} clients from file")
    return clients


def sanitize_client_paths(clients: List[Dict]) -> List[Dict]:
    """
    Sanitize model paths to work in current environment.
    Converts environment-specific paths (e.g., Colab, WSL) to local paths.

    Args:
        clients: List of client configuration dictionaries

    Returns:
        List of client configurations with sanitized paths
    """
    from pathlib import Path

    # Get backend root (parent of federated_learning directory)
    backend_root = Path(__file__).parent.parent.parent

    for client in clients:
        # Sanitize model_path
        if 'model_path' in client and client['model_path']:
            orig_path = client['model_path']

            if 'models' in orig_path:
                relative_part = orig_path.split('models')[-1].lstrip('/')
                sanitized_path = str(backend_root / "models" / relative_part)
                if sanitized_path != orig_path:
                    print(f"[Config] Sanitized model_path for client {client.get('client_name', client.get('id'))}: {sanitized_path}")
                client['model_path'] = sanitized_path

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
def get_model_by_type(model_type: str, use_pretrained: bool = True, dropout_rate: float = None):
    """
    Create a model instance by type with dropout-enhanced classifier.

    Args:
        model_type: Architecture identifier (resnet50, mobilenet_v2, densenet121, efficientnet_b0)
        use_pretrained: Whether to use pretrained weights (currently unused, kept for compatibility)
        dropout_rate: Optional dropout rate override. If None, uses architecture-specific default.

    Returns:
        PyTorch model with dropout layer before final classifier
    """
    model_type = model_type.lower()

    # Determine dropout rate (use provided or architecture default)
    if dropout_rate is None:
        dropout_rate = get_initial_dropout_rate(model_type)

    if model_type == 'resnet50':
        # ResNet-50: The Industry Standard (Standard residual network)
        # Higher capacity for complex morphological features in pathology
        model = models.resnet50(weights=None)
        model = add_dropout_to_classifier(model, model_type, dropout_rate)
        return model

    elif model_type == 'mobilenet_v2':
        # MobileNet-V2: Mobile Optimized (Lightweight architecture)
        # Represents resource-constrained clients or point-of-care devices
        model = models.mobilenet_v2(weights=None)
        model = add_dropout_to_classifier(model, model_type, dropout_rate)
        return model

    elif model_type == 'densenet121':
        # DenseNet-121: High Dense Connections (Efficient feature reuse)
        # Excellent at preserving subtle textural patterns in blood smears
        model = models.densenet121(weights=None)
        model = add_dropout_to_classifier(model, model_type, dropout_rate)
        return model

    elif model_type == 'efficientnet_b0':
        # EfficientNet-B0: Modern Efficiency Optimizer (Compound scaling)
        # SOTA balance between parameter count and feature extraction quality
        model = models.efficientnet_b0(weights=None)
        model = add_dropout_to_classifier(model, model_type, dropout_rate)
        return model

    elif model_type == 'resnet18':
        # ResNet-18: Lightweight ResNet variant
        # Faster training with lower memory requirements
        model = models.resnet18(weights=None)
        model = add_dropout_to_classifier(model, model_type, dropout_rate)
        return model

    else:
        raise ValueError(f"Unsupported model type: {model_type}. "
                        f"Supported types: resnet50, resnet18, mobilenet_v2, densenet121, efficientnet_b0")

# <------------------------------------------ DROPOUT MANAGEMENT UTILITIES ------------------------------------------>

def get_initial_dropout_rate(model_type: str) -> float:
    """
    Get default initial dropout rate for model architecture.

    Different architectures require different dropout rates based on their capacity and tendency to overfit.
    Larger models (ResNet50, DenseNet121) use moderate dropout, while lightweight models (MobileNetV2)
    use higher dropout to prevent overfitting.

    Args:
        model_type: Architecture identifier (e.g., 'resnet50', 'mobilenet_v2')

    Returns:
        Initial dropout probability (float between 0.0 and 1.0)
    """
    DEFAULT_DROPOUT_RATES = {
        'resnet50': 0.35,        # Higher capacity model, moderate dropout
        'mobilenet_v2': 0.45,    # Lightweight model, higher dropout needed
        'densenet121': 0.35,     # Dense connections provide natural regularization
        'efficientnet_b0': 0.40, # Modern efficient architecture, moderate dropout
        'resnet18': 0.35,        # Similar to ResNet50
    }
    return DEFAULT_DROPOUT_RATES.get(model_type.lower(), 0.3)


def add_dropout_to_classifier(model, model_type: str, dropout_rate: float = 0.3):
    """
    Add dropout layer before the final classifier head.

    This function modifies the classifier to include dropout for regularization.
    Dropout rates are adaptive and will be adjusted based on overfitting signals.

    This function is idempotent - if dropout has already been added, it will only
    update the dropout rate without modifying the structure.

    Args:
        model: PyTorch model instance
        model_type: Architecture identifier
        dropout_rate: Initial dropout probability (default: 0.3)

    Returns:
        Modified model with dropout layer
    """
    model_type = model_type.lower()

    if model_type == 'resnet50' or model_type == 'resnet18':
        # ResNet: fc = nn.Linear(in_features, num_classes)
        # Replace with Sequential: Dropout → Linear
        # Check if dropout already exists (idempotent behavior)
        if isinstance(model.fc, nn.Sequential):
            # Dropout already added - just update the rate
            if len(model.fc) > 0 and isinstance(model.fc[0], nn.Dropout):
                model.fc[0].p = dropout_rate
        else:
            # First time - add dropout
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, NUM_CLASSES)
            )

    elif model_type == 'mobilenet_v2':
        # MobileNetV2: classifier[1] = nn.Linear(1280, num_classes)
        # Replace classifier[1] with Sequential: Dropout → Linear
        # Check if dropout already exists (idempotent behavior)
        if isinstance(model.classifier[1], nn.Sequential):
            # Dropout already added - just update the rate
            if len(model.classifier[1]) > 0 and isinstance(model.classifier[1][0], nn.Dropout):
                model.classifier[1][0].p = dropout_rate
        else:
            # First time - add dropout
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, NUM_CLASSES)
            )

    elif model_type == 'densenet121':
        # DenseNet121: classifier = nn.Linear(1024, num_classes)
        # Replace with Sequential: Dropout → Linear
        # Check if dropout already exists (idempotent behavior)
        if isinstance(model.classifier, nn.Sequential):
            # Dropout already added - just update the rate
            if len(model.classifier) > 0 and isinstance(model.classifier[0], nn.Dropout):
                model.classifier[0].p = dropout_rate
        else:
            # First time - add dropout
            in_features = model.classifier.in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, NUM_CLASSES)
            )

    elif model_type == 'efficientnet_b0':
        # EfficientNetB0: classifier[1] = nn.Linear(1280, num_classes)
        # Replace classifier[1] with Sequential: Dropout → Linear
        # Check if dropout already exists (idempotent behavior)
        if isinstance(model.classifier[1], nn.Sequential):
            # Dropout already added - just update the rate
            if len(model.classifier[1]) > 0 and isinstance(model.classifier[1][0], nn.Dropout):
                model.classifier[1][0].p = dropout_rate
        else:
            # First time - add dropout
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(in_features, NUM_CLASSES)
            )

    return model


def get_dropout_rate_from_model(model, model_type: str) -> float:
    """
    Extract current dropout rate from model architecture.

    Args:
        model: PyTorch model instance
        model_type: Architecture identifier

    Returns:
        Current dropout rate (0.0 if no dropout found)
    """
    model_type = model_type.lower()

    try:
        if model_type == 'resnet50' or model_type == 'resnet18':
            if isinstance(model.fc, nn.Sequential) and len(model.fc) > 0:
                if isinstance(model.fc[0], nn.Dropout):
                    return model.fc[0].p
        elif model_type in ['mobilenet_v2', 'efficientnet_b0']:
            if isinstance(model.classifier[1], nn.Sequential) and len(model.classifier[1]) > 0:
                if isinstance(model.classifier[1][0], nn.Dropout):
                    return model.classifier[1][0].p
        elif model_type == 'densenet121':
            if isinstance(model.classifier, nn.Sequential) and len(model.classifier) > 0:
                if isinstance(model.classifier[0], nn.Dropout):
                    return model.classifier[0].p
    except Exception as e:
        print(f"[Dropout] Error extracting dropout rate: {e}")

    return 0.0  # Default if no dropout found


def update_dropout_rate(model, model_type: str, new_dropout_rate: float):
    """
    Update dropout rate in existing model architecture.

    This function dynamically adjusts the dropout probability without recreating the model.
    Used for adaptive dropout adjustment between federated learning rounds.

    Args:
        model: PyTorch model instance
        model_type: Architecture identifier
        new_dropout_rate: New dropout probability

    Returns:
        Modified model with updated dropout rate
    """
    model_type = model_type.lower()

    try:
        if model_type == 'resnet50' or model_type == 'resnet18':
            if isinstance(model.fc, nn.Sequential) and len(model.fc) > 0:
                if isinstance(model.fc[0], nn.Dropout):
                    model.fc[0].p = new_dropout_rate

        elif model_type in ['mobilenet_v2', 'efficientnet_b0']:
            if isinstance(model.classifier[1], nn.Sequential) and len(model.classifier[1]) > 0:
                if isinstance(model.classifier[1][0], nn.Dropout):
                    model.classifier[1][0].p = new_dropout_rate

        elif model_type == 'densenet121':
            if isinstance(model.classifier, nn.Sequential) and len(model.classifier) > 0:
                if isinstance(model.classifier[0], nn.Dropout):
                    model.classifier[0].p = new_dropout_rate
    except Exception as e:
        print(f"[Dropout] Error updating dropout rate: {e}")

    return model

# <------------------------------------------ TRANSFER LEARNING UTILITIES ------------------------------------------>

def freeze_backbone(model, model_type: str):
    """
    Freeze all layers except the classifier head for initial training.
    """
    model_type = model_type.lower()

    if model_type == 'resnet50':
        # Freeze all layers except 'fc'
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
        print(f"[Transfer Learning] ResNet50 backbone frozen, training head only")

    elif model_type in ['mobilenet_v2', 'efficientnet_b0', 'densenet121']:
        # Freeze all layers except 'classifier'
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        print(f"[Transfer Learning] {model_type.upper()} backbone frozen, training head only")

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

# Computes a WeightedRandomSampler to handle class imbalance
# Args: targets - List or array of class labels for the dataset
# Returns: WeightedRandomSampler instance
def get_weighted_sampler(targets):
    # Calculate class counts
    class_counts = Counter(targets)
    
    # Calculate weight for each class (inverse frequency)
    # Strategy: Weight = 1.0 / count
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    
    # Assign weight to each sample
    sample_weights = [class_weights[t] for t in targets]
    
    # Create sampler
    # replacement=True is standard for oversampling minority classes
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler

# <------------------------------------------ ADAPTIVE TRAINING STATE MANAGEMENT ------------------------------------------>

class AdaptiveTrainingState:
    """
    Manages per-client adaptive training state for overfitting control.

    This class tracks validation loss history and adaptively adjusts dropout rates
    based on overfitting/stability signals. State persists across FL rounds via
    checkpoint serialization.

    Features:
    - Bidirectional dropout adaptation (can increase or decrease)
    - Conservative strategy (requires 2 consecutive rounds for adjustment)
    - Per-client state (each client has independent adaptation)
    - Checkpoint serialization for persistence

    Attributes:
        client_id: Numeric client identifier
        model_type: Architecture name (for dropout management)
        dropout_rate: Current dropout probability
        max_dropout: Maximum allowed dropout (default 0.6)
        dropout_increment: Adjustment step size (default 0.1)
        patience_rounds: Rounds to wait before adjustment (default 2)
        val_loss_history: List of validation losses from recent rounds
        scheduler_state: Serialized scheduler state (optional)
    """

    def __init__(
        self,
        client_id: int,
        model_type: str,
        initial_dropout: float = 0.3,
        max_dropout: float = 0.6,
        min_dropout: float = 0.2,
        dropout_increment: float = 0.1,
        patience_rounds: int = 2
    ):
        """
        Initialize adaptive training state.

        Args:
            client_id: Numeric client identifier
            model_type: Architecture name
            initial_dropout: Starting dropout rate (default: 0.3)
            max_dropout: Maximum dropout ceiling (default: 0.6)
            min_dropout: Minimum dropout floor (default: 0.2)
            dropout_increment: Adjustment step size (default: 0.1)
            patience_rounds: Rounds needed for trend detection (default: 2)
        """
        self.client_id = client_id
        self.model_type = model_type
        self.dropout_rate = initial_dropout
        self.max_dropout = max_dropout
        self.min_dropout = min_dropout
        self.dropout_increment = dropout_increment
        self.patience_rounds = patience_rounds

        # Validation loss history (stores last N rounds)
        self.val_loss_history = []

        # Scheduler state (will be populated when scheduler is created)
        self.scheduler_state = None

    def should_increase_dropout(self) -> bool:
        """
        Check if dropout should be increased based on validation loss trend.

        Detects overfitting by looking for 2 consecutive rounds of increasing
        validation loss.

        Returns:
            True if overfitting detected (dropout should increase)
        """
        if len(self.val_loss_history) < self.patience_rounds + 1:
            return False

        # Check last 3 rounds (need 3 points to detect 2 consecutive increases)
        recent_losses = self.val_loss_history[-3:]

        # Increasing trend: loss[i] > loss[i-1] for 2 consecutive rounds
        if recent_losses[1] > recent_losses[0] and recent_losses[2] > recent_losses[1]:
            return True

        return False

    def should_decrease_dropout(self) -> bool:
        """
        Check if dropout should be decreased based on validation loss improvement.

        Detects stable learning by looking for 2 consecutive rounds of decreasing
        validation loss.

        Returns:
            True if learning is stable (dropout can safely decrease)
        """
        if len(self.val_loss_history) < self.patience_rounds + 1:
            return False

        # Check last 3 rounds (need 3 points to detect 2 consecutive decreases)
        recent_losses = self.val_loss_history[-3:]

        # Decreasing trend: loss[i] < loss[i-1] for 2 consecutive rounds
        if recent_losses[1] < recent_losses[0] and recent_losses[2] < recent_losses[1]:
            return True

        return False

    def update_dropout(self, model) -> float:
        """
        Adaptively adjust dropout rate based on validation loss trends.

        Strategy:
        - Increase dropout (+0.1) if overfitting detected (2 consecutive increases)
        - Decrease dropout (-0.1) if learning is stable (2 consecutive decreases)
        - Conservative: Max dropout 0.6, min dropout 0.2
        - Bidirectional: Can both increase and decrease

        Args:
            model: PyTorch model to update

        Returns:
            New dropout rate after adjustment
        """
        old_dropout = self.dropout_rate

        if self.should_increase_dropout() and self.dropout_rate < self.max_dropout:
            # Overfitting detected - increase regularization
            self.dropout_rate = min(self.max_dropout, self.dropout_rate + self.dropout_increment)
            print(f"[Client {self.client_id}] Dropout INCREASED: {old_dropout:.2f} → {self.dropout_rate:.2f} (overfitting detected)")

        elif self.should_decrease_dropout() and self.dropout_rate > self.min_dropout:
            # Stable learning - can reduce regularization
            self.dropout_rate = max(self.min_dropout, self.dropout_rate - self.dropout_increment)
            print(f"[Client {self.client_id}] Dropout DECREASED: {old_dropout:.2f} → {self.dropout_rate:.2f} (stable learning)")

        else:
            # No change needed
            if len(self.val_loss_history) >= 3:
                print(f"[Client {self.client_id}] Dropout UNCHANGED: {self.dropout_rate:.2f} (no clear trend)")

        # Update model dropout rate if changed
        if self.dropout_rate != old_dropout:
            update_dropout_rate(model, self.model_type, self.dropout_rate)

        return self.dropout_rate

    def add_val_loss(self, val_loss: float):
        """
        Add validation loss to history.

        Args:
            val_loss: Validation loss from current round
        """
        self.val_loss_history.append(val_loss)
        # Keep only last 5 rounds for memory efficiency
        if len(self.val_loss_history) > 5:
            self.val_loss_history.pop(0)

    def to_dict(self) -> dict:
        """
        Serialize state for checkpoint saving.

        Returns:
            Dictionary containing all state for serialization
        """
        return {
            'client_id': self.client_id,
            'model_type': self.model_type,
            'dropout_rate': self.dropout_rate,
            'max_dropout': self.max_dropout,
            'min_dropout': self.min_dropout,
            'dropout_increment': self.dropout_increment,
            'patience_rounds': self.patience_rounds,
            'val_loss_history': self.val_loss_history,
            'scheduler_state': self.scheduler_state
        }

    @classmethod
    def from_dict(cls, state_dict: dict) -> 'AdaptiveTrainingState':
        """
        Deserialize state from checkpoint.

        Args:
            state_dict: Dictionary from checkpoint

        Returns:
            AdaptiveTrainingState instance with restored state
        """
        state = cls(
            client_id=state_dict['client_id'],
            model_type=state_dict['model_type'],
            initial_dropout=state_dict['dropout_rate'],
            max_dropout=state_dict.get('max_dropout', 0.6),
            min_dropout=state_dict.get('min_dropout', 0.2),
            dropout_increment=state_dict.get('dropout_increment', 0.1),
            patience_rounds=state_dict.get('patience_rounds', 2)
        )
        state.val_loss_history = state_dict.get('val_loss_history', [])
        state.scheduler_state = state_dict.get('scheduler_state')
        return state

# <------------------------------------------ DIRICHLET PARTITIONER FOR RUNTIME DATA HETEROGENEITY ------------------------------------------>

# Cache for the partitioner (created once per configuration, reused across clients)
_PARTITIONER_CACHE = {}

def create_dirichlet_partitioner(
    dataset_path: str,
    num_partitions: int,
    alpha: float = DIRICHLET_ALPHA,
    seed: int = DIRICHLET_SEED,
    min_partition_size: int = DIRICHLET_MIN_PARTITION_SIZE
) -> Tuple[DirichletPartitioner, datasets.ImageFolder]:
    """
    Create a Dirichlet partitioner for the local training dataset.

    This partitioner will be used to split the combined training data
    across multiple clients with configurable heterogeneity (alpha).

    The Dirichlet distribution is used to sample class proportions for each client,
    resulting in heterogeneous data distributions. Lower alpha values create more
    extreme heterogeneity, while higher values approach IID distributions.

    Args:
        dataset_path: Path to the combined local training dataset
        num_partitions: Number of clients (partitions)
        alpha: Dirichlet concentration parameter
               - Lower values (0.1-0.5) = high heterogeneity
               - Higher values (1.0-10.0) = low heterogeneity
        seed: Random seed for reproducibility
        min_partition_size: Minimum samples each client must receive

    Returns:
        Tuple of (DirichletPartitioner instance, full ImageFolder dataset)
    """

    # Load the full dataset to get image paths and labels
    full_dataset = datasets.ImageFolder(
        root=dataset_path,
        transform=None  # No transforms needed for partitioning
    )

    # Extract image paths and labels
    image_paths = [path for path, _ in full_dataset.samples]
    labels = [label for _, label in full_dataset.samples]

    print(f"[Partitioner] Loaded {len(image_paths)} samples from {dataset_path}")
    print(f"[Partitioner] Class distribution: ALL={sum(l == 0 for l in labels)}, "
          f"Healthy={sum(l == 1 for l in labels)}")

    # Create HuggingFace Dataset for DirichletPartitioner
    # DirichletPartitioner requires a Dataset object with a "label" column
    hf_dataset = Dataset.from_dict({
        "image_path": image_paths,
        "label": labels
    })

    # Initialize Dirichlet Partitioner
    partitioner = DirichletPartitioner(
        num_partitions=num_partitions,
        partition_by="label",  # Partition based on class labels
        alpha=alpha,
        min_partition_size=min_partition_size,
        self_balancing=False,  # Allow natural heterogeneity
        shuffle=True,
        seed=seed
    )

    # Assign dataset to partitioner
    partitioner.dataset = hf_dataset

    print(f"[Partitioner] Created Dirichlet partitioner:")
    print(f"  - Num partitions: {num_partitions}")
    print(f"  - Alpha: {alpha}")
    print(f"  - Min partition size: {min_partition_size}")
    print(f"  - Seed: {seed}")

    return partitioner, full_dataset

# <------------------------------------------ DATA LOADER FUNCTION DEFINITIONS ------------------------------------------>

# Provides shared dataset for generating consensus logits across clients
# Args: batch_size - Batch size for DataLoader
# Returns: DataLoader with shuffle=False (order consistency critical for FL simulation)
def load_public_dataset(batch_size=64, round_num=1, total_rounds=10):    
    if not os.path.exists(PUBLIC_ANCHOR_DATASET_PATH):
        raise FileNotFoundError(f"Public data not found at {PUBLIC_ANCHOR_DATASET_PATH}")

    # Load full dataset
    # Note: round_num and total_rounds are ignored as we now use the full dataset every round
    # This change was requested to improve training stability by using the full anchor
    full_dataset = datasets.ImageFolder(root=PUBLIC_ANCHOR_DATASET_PATH, transform=COMMON_TRANSFORM)
    
    # Deterministically shuffle all indices to ensure:
    # 1. Consistent order across all clients and server (critical for consensus alignment)
    # 2. Mixed classes in each batch (better for training/distillation)
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(full_dataset), generator=generator).tolist()
    
    subset = torch.utils.data.Subset(full_dataset, indices)
    
    # Shuffle=False because we already shuffled indices deterministically
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)
    return loader

# Handles heterogeneous client datasets using runtime Dirichlet partitioning
# Args: partition_id - Client ID (0, 1, 2, ...)
#       num_partitions - Total number of clients participating in FL
#       batch_size - Batch size for data loaders
#       config_path - Path to the configuration file
# Returns: Tuple of (trainloader, testloader) or (None, None) if client has no data
def load_private_dataset(partition_id: int, num_partitions: int, batch_size=32, config_path: str = CLIENT_INFO_FILE_PATH):
    """
    Load private dataset for a specific client using Dirichlet partitioning.

    This function loads the shared local training dataset and uses
    DirichletPartitioner to extract the heterogeneous subset for this client.
    The partitioning happens at runtime based on the alpha parameter.

    All clients are required to have data from the shared LOCAL_TRAIN_DATASET_PATH.

    Args:
        partition_id: Client ID (0, 1, 2, ...)
        num_partitions: Total number of clients
        batch_size: Batch size for DataLoaders
        config_path: Path to client configuration

    Returns:
        Tuple of (trainloader, testloader)
    """

    # Check if shared training dataset exists
    if not os.path.exists(LOCAL_TRAIN_DATASET_PATH):
        raise FileNotFoundError(
            f"Shared training dataset not found at {LOCAL_TRAIN_DATASET_PATH}"
        )

    # Create or retrieve cached partitioner
    cache_key = (LOCAL_TRAIN_DATASET_PATH, num_partitions, DIRICHLET_ALPHA, DIRICHLET_SEED)

    if cache_key not in _PARTITIONER_CACHE:
        print(f"[Data] Initializing Dirichlet partitioner for {num_partitions} clients")
        partitioner, full_dataset = create_dirichlet_partitioner(
            dataset_path=LOCAL_TRAIN_DATASET_PATH,
            num_partitions=num_partitions,
            alpha=DIRICHLET_ALPHA,
            seed=DIRICHLET_SEED
        )
        _PARTITIONER_CACHE[cache_key] = (partitioner, full_dataset)
    else:
        print(f"[Data] Using cached partitioner for {num_partitions} clients")
        partitioner, full_dataset = _PARTITIONER_CACHE[cache_key]

    # Load this client's partition
    partition_dataset = partitioner.load_partition(partition_id)

    # Extract image paths and labels for this partition
    partition_paths = partition_dataset["image_path"]
    partition_labels = partition_dataset["label"]

    print(f"[Data] Client {partition_id} partition:")
    print(f"  - Total samples: {len(partition_paths)}")
    print(f"  - ALL: {sum(l == 0 for l in partition_labels)} "
          f"({sum(l == 0 for l in partition_labels) / len(partition_labels) * 100:.1f}%)")
    print(f"  - Healthy: {sum(l == 1 for l in partition_labels)} "
          f"({sum(l == 1 for l in partition_labels) / len(partition_labels) * 100:.1f}%)")

    # Create indices mapping for the client's subset
    # We need to map partition_paths back to indices in full_dataset
    path_to_idx = {path: idx for idx, (path, _) in enumerate(full_dataset.samples)}
    client_indices = [path_to_idx[path] for path in partition_paths]

    # Create subset for this client with transforms
    client_dataset = torch.utils.data.Subset(
        datasets.ImageFolder(
            root=LOCAL_TRAIN_DATASET_PATH,
            transform=PRIVATE_TRAIN_TRANSFORM
        ),
        client_indices
    )

    # Create stratified train/validation split (85/15) on the client's partition
    # Group indices by class
    class_indices = {0: [], 1: []}
    for subset_idx, global_idx in enumerate(client_indices):
        label = full_dataset.targets[global_idx]
        class_indices[label].append(subset_idx)

    # Split each class separately
    train_indices = []
    val_indices = []
    generator = torch.Generator().manual_seed(42)

    for label, indices in class_indices.items():
        if len(indices) == 0:
            continue

        # Shuffle indices for this class
        indices_tensor = torch.tensor(indices)
        perm = torch.randperm(len(indices), generator=generator)
        shuffled_indices = indices_tensor[perm].tolist()

        # Split 85/15
        split_point = int(0.85 * len(shuffled_indices))
        train_indices.extend(shuffled_indices[:split_point])
        val_indices.extend(shuffled_indices[split_point:])

    print(f"[Data] Client {partition_id} split: Train={len(train_indices)}, Val={len(val_indices)}")

    # Create train/validation subsets
    train_ds = torch.utils.data.Subset(client_dataset, train_indices)
    test_ds = torch.utils.data.Subset(client_dataset, val_indices)

    # Apply weighted sampler for class imbalance
    train_targets = [full_dataset.targets[client_indices[i]] for i in train_indices]
    train_sampler = get_weighted_sampler(train_targets)
    print(f"[Data] Client {partition_id}: Activated WeightedRandomSampler")

    # Create DataLoaders
    trainloader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=False,  # Must be False with sampler
        num_workers=2
    )
    testloader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

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
    Evaluate all clients on their private validation sets (15% of private partition).
    Used for per-round progress tracking without exposing the test set.
    All clients use runtime Dirichlet partitioning (v3.0).
    """
    client_metrics = {}

    for i, client in enumerate(client_configs):
        client_id = str(i)
        model_path = client['model_path']
        model_type = client['model_type']
        client_name = client['client_name']

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

            # All clients use runtime Dirichlet partitioning (v3.0)
            # Load validation set (15% of private partition)
            _, valloader = load_private_dataset(i, num_partitions, batch_size=64)

            if valloader is not None:
                dataset_type = 'validation'
                num_samples = len(valloader.dataset)
            else:
                # Fallback if data loading fails
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
            leukemia_mask = (all_labels == 0)
            healthy_mask = (all_labels == 1)

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

# Helper function to extract training metrics from aggregate results (for persistence)
def extract_training_metrics_for_persistence(
    client_metrics_list: List[Dict],
    aggregation_metadata: Dict,
    client_configs: List[Dict]
) -> Dict:
    """
    Extract training metrics from aggregate_train results for client_data.json persistence.

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
            "val_loss": metrics.get("val_loss"),  # Added for per-round tracking
            "num-examples": metrics.get("num-examples"),
            "training_time": metrics.get("training_time"),
            "consensus_weight": consensus_weight
        }

    return training_metrics


# Note: save_global_pre_fl_metrics removed - Pre-FL evaluation on untrained models
# gives meaningless ~50% accuracy for binary classification and has been eliminated.


def save_global_post_fl_metrics(
    metrics: Dict,
    client_configs: List[Dict]
):
    """
    Save Post-FL metrics to client_simulation_metrics table.
    Also computes and saves aggregate metrics to fl_simulations.

    Args:
        metrics: Dict mapping client_id (str) to metrics dict
        client_configs: List of client configuration dicts with 'id' and 'client_name' fields
    """
    if SUPABASE_CLIENT is None or SIMULATION_ID is None:
        print("[Post-FL] Database not available, skipping save")
        return

    try:
        print(f"[Post-FL] Saving metrics for {len(metrics)} clients to database...")

        # Update individual client metrics
        for client_id_str, client_metrics in metrics.items():
            client_idx = int(client_id_str)
            if client_idx >= len(client_configs):
                continue

            client_config = client_configs[client_idx]
            db_client_id = client_config.get('id')

            if db_client_id is None:
                continue

            # Fetch current metrics
            response = SUPABASE_CLIENT.from_('client_simulation_metrics') \
                .select('metrics') \
                .eq('simulation_id', SIMULATION_ID) \
                .eq('client_id', db_client_id) \
                .execute()

            if not response.data:
                print(f"[Post-FL] Warning: No metrics record for client {db_client_id}")
                continue

            existing_metrics = response.data[0].get('metrics', {})

            if 'global' not in existing_metrics:
                existing_metrics['global'] = {}

            # Save post_fl metrics
            existing_metrics['global']['post_fl'] = client_metrics

            # Note: Improvement calculation removed since we no longer store pre_fl metrics
            # (pre_fl on untrained models gives meaningless ~50% baseline)
            # Improvement is now tracked via round-over-round progression in 'rounds' array

            # Save back to database
            SUPABASE_CLIENT.from_('client_simulation_metrics').update({
                'metrics': existing_metrics,
                'status': 'completed',
                'completed_at': datetime.now().isoformat()
            }).eq('simulation_id', SIMULATION_ID) \
              .eq('client_id', db_client_id) \
              .execute()

            print(f"[Post-FL] ✓ Client {db_client_id} ({client_config.get('client_name')})")

        # Compute aggregate metrics
        print(f"[Post-FL] Computing aggregate metrics for simulation {SIMULATION_ID}...")

        # Fetch all client metrics for this simulation
        response = SUPABASE_CLIENT.from_('client_simulation_metrics') \
            .select('metrics') \
            .eq('simulation_id', SIMULATION_ID) \
            .execute()

        if response.data:
            all_metrics = [row['metrics'] for row in response.data]
            aggregate_metrics = compute_aggregate_metrics_local(all_metrics)

            # Fetch simulation start time for duration calculation
            sim_response = SUPABASE_CLIENT.from_('fl_simulations') \
                .select('started_at') \
                .eq('id', SIMULATION_ID) \
                .execute()

            # Calculate duration if started_at is available
            duration = None
            if sim_response.data and sim_response.data[0].get('started_at'):
                from datetime import timezone
                started_at_str = sim_response.data[0]['started_at']
                # Parse ISO string with timezone
                started_at = datetime.fromisoformat(started_at_str.replace('Z', '+00:00'))
                completed_at = datetime.now(timezone.utc)
                duration = int((completed_at - started_at).total_seconds())
                print(f"[Post-FL] Calculated duration: {duration} seconds")

            # Update fl_simulations with aggregate metrics, status, and duration
            update_data = {
                'aggregate_metrics': aggregate_metrics,
                'status': 'completed',
                'completed_at': datetime.now(timezone.utc).isoformat()
            }
            if duration is not None:
                update_data['duration'] = duration

            SUPABASE_CLIENT.from_('fl_simulations').update(update_data).eq('id', SIMULATION_ID).execute()

            print(f"[Post-FL] ✓ Saved aggregate metrics for simulation {SIMULATION_ID}")
            print(f"  - Total rounds: {aggregate_metrics.get('total_rounds_completed', 0)}")
            best_acc = aggregate_metrics.get('best_round', {}).get('avg_accuracy', 0)
            if best_acc > 0:
                print(f"  - Best accuracy: {best_acc:.2%}")

        print("[Post-FL] Successfully saved all metrics to database")

    except Exception as e:
        print(f"[Post-FL] Error: {e}")
        import traceback
        traceback.print_exc()


def compute_aggregate_metrics_local(all_metrics: List[dict]) -> dict:
    """
    Compute aggregate metrics from list of client metrics.

    This is a local copy of compute_aggregate_metrics from backend schema
    to avoid import dependencies.

    Args:
        all_metrics: List of metrics dicts from all clients

    Returns:
        Aggregated metrics with averages, std dev, best round, etc.
    """
    if not all_metrics:
        return {}

    # Calculate aggregates for post_fl only (pre_fl removed as it was meaningless)
    def calc_avg(metric_name: str, stage: str) -> float:
        """Calculate average of a metric across all clients"""
        values = []
        for m in all_metrics:
            if stage in m.get('global', {}):
                val = m['global'][stage].get(metric_name)
                if val is not None:
                    values.append(val)
        return sum(values) / len(values) if values else 0.0

    def calc_std(metric_name: str, stage: str) -> float:
        """Calculate standard deviation of a metric across all clients"""
        values = []
        for m in all_metrics:
            if stage in m.get('global', {}):
                val = m['global'][stage].get(metric_name)
                if val is not None:
                    values.append(val)
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5

    # Aggregate round-by-round metrics
    rounds_aggregate = []
    num_rounds = max(len(m.get('rounds', [])) for m in all_metrics) if all_metrics else 0

    for round_num in range(1, num_rounds + 1):
        round_metrics = {
            'round': round_num,
            'avg_accuracy': 0.0,
            'avg_loss': 0.0,
            'avg_f1': 0.0,
            'avg_precision': 0.0,
            'avg_recall': 0.0,
            'avg_train_loss': 0.0,  # Added training losses
            'avg_val_loss': 0.0,     # Added validation loss from training
            'num_clients_trained': 0,
            'timestamp': None
        }

        values_acc = []
        values_loss = []
        values_f1 = []
        values_precision = []
        values_recall = []
        values_train_loss = []
        values_val_loss = []

        for m in all_metrics:
            rounds = m.get('rounds', [])
            for r in rounds:
                if r.get('round') == round_num:
                    # Validation evaluation metrics
                    if 'validation' in r:
                        val = r['validation']
                        values_acc.append(val.get('accuracy', 0))
                        values_loss.append(val.get('loss', 0))
                        values_f1.append(val.get('f1_score', 0))
                        values_precision.append(val.get('precision', 0))
                        values_recall.append(val.get('recall', 0))
                        if not round_metrics['timestamp']:
                            round_metrics['timestamp'] = val.get('evaluated_at')
                    
                    # Training metrics (train_loss, val_loss from local training)
                    if 'training' in r:
                        train_data = r['training']
                        if train_data.get('train_loss') is not None:
                            values_train_loss.append(train_data['train_loss'])
                        if train_data.get('val_loss') is not None:
                            values_val_loss.append(train_data['val_loss'])

        if values_acc:
            round_metrics['avg_accuracy'] = sum(values_acc) / len(values_acc)
            round_metrics['avg_loss'] = sum(values_loss) / len(values_loss)
            round_metrics['avg_f1'] = sum(values_f1) / len(values_f1)
            round_metrics['avg_precision'] = sum(values_precision) / len(values_precision)
            round_metrics['avg_recall'] = sum(values_recall) / len(values_recall)
            round_metrics['num_clients_trained'] = len(values_acc)
        
        if values_train_loss:
            round_metrics['avg_train_loss'] = sum(values_train_loss) / len(values_train_loss)
        if values_val_loss:
            round_metrics['avg_val_loss'] = sum(values_val_loss) / len(values_val_loss)
        
        # Only add rounds with actual data
        if values_acc or values_train_loss:
            rounds_aggregate.append(round_metrics)

    # Find best round
    best_round = max(rounds_aggregate, key=lambda r: r['avg_accuracy']) if rounds_aggregate else {}
    
    # Calculate improvement: Round 1 vs Final Round (more meaningful than untrained baseline)
    round_1_metrics = next((r for r in rounds_aggregate if r['round'] == 1), {})
    final_round_metrics = rounds_aggregate[-1] if rounds_aggregate else {}
    
    improvement = {}
    if round_1_metrics and final_round_metrics:
        for metric in ['avg_accuracy', 'avg_loss', 'avg_f1', 'avg_precision', 'avg_recall']:
            r1_val = round_1_metrics.get(metric, 0)
            final_val = final_round_metrics.get(metric, 0)
            if r1_val is not None and final_val is not None:
                improvement[metric] = round(final_val - r1_val, 6)

    return {
        "aggregate": {
            "post_fl": {
                "avg_accuracy": calc_avg('accuracy', 'post_fl'),
                "avg_loss": calc_avg('loss', 'post_fl'),
                "avg_precision": calc_avg('precision', 'post_fl'),
                "avg_recall": calc_avg('recall', 'post_fl'),
                "avg_f1": calc_avg('f1_score', 'post_fl'),
                "std_accuracy": calc_std('accuracy', 'post_fl'),
                "num_clients": len(all_metrics)
            },
            "improvement": improvement  # Now based on Round 1 vs Final Round
        },
        "rounds": rounds_aggregate,
        "best_round": {
            "round": best_round.get('round'),
            "avg_accuracy": best_round.get('avg_accuracy')
        } if best_round else {},
        "total_rounds_completed": len(rounds_aggregate),
        "total_clients": len(all_metrics)
    }


def save_round_training_metrics(
    round_num: int,
    training_metrics: Dict,
    client_configs: List[Dict]
):
    """
    Save per-round training metrics (train_loss, val_loss from local training) to database.

    This stores the training loss and validation loss from the client's local training phase,
    which is different from the validation evaluation on test data.

    Args:
        round_num: Current round number
        training_metrics: Dict mapping client_id (str) to training metrics dict with:
            - train_loss: Average training loss during local training
            - val_loss: Average validation loss during local training
            - distill_loss: Knowledge distillation loss
            - training_time: Time spent training
        client_configs: List of client configuration dicts with 'id' field
    """
    if SUPABASE_CLIENT is None or SIMULATION_ID is None:
        print(f"[Round {round_num}] Database not available, skipping training metrics save")
        return

    try:
        print(f"[Round {round_num}] Saving training metrics for {len(training_metrics)} clients...")

        for client_id_str, client_metrics in training_metrics.items():
            client_idx = int(client_id_str)
            if client_idx >= len(client_configs):
                continue

            client_config = client_configs[client_idx]
            db_client_id = client_config.get('id')

            if db_client_id is None:
                continue

            # Fetch current metrics from client_simulation_metrics
            response = SUPABASE_CLIENT.from_('client_simulation_metrics') \
                .select('metrics') \
                .eq('simulation_id', SIMULATION_ID) \
                .eq('client_id', db_client_id) \
                .execute()

            if not response.data:
                print(f"[Round {round_num}] Warning: No metrics record for client {db_client_id}")
                continue

            # Update metrics
            existing_metrics = response.data[0].get('metrics', {})

            # Initialize rounds array if needed
            if 'rounds' not in existing_metrics:
                existing_metrics['rounds'] = []

            # Find or create round entry
            round_entry = None
            for r in existing_metrics['rounds']:
                if r.get('round') == round_num:
                    round_entry = r
                    break

            if round_entry is None:
                round_entry = {'round': round_num}
                existing_metrics['rounds'].append(round_entry)

            # Add training metrics (from local client training)
            round_entry['training'] = {
                'train_loss': client_metrics.get('train_loss'),
                'val_loss': client_metrics.get('val_loss'),
                'distill_loss': client_metrics.get('distill_loss'),
                'training_time': client_metrics.get('training_time'),
                'num_examples': client_metrics.get('num-examples'),
                'consensus_weight': client_metrics.get('consensus_weight')
            }

            # Save back to database
            SUPABASE_CLIENT.from_('client_simulation_metrics').update({
                'metrics': existing_metrics
            }).eq('simulation_id', SIMULATION_ID) \
              .eq('client_id', db_client_id) \
              .execute()

        print(f"[Round {round_num}] ✓ Saved training metrics (train_loss, val_loss)")

    except Exception as e:
        print(f"[Round {round_num}] Error saving training metrics: {e}")
        import traceback
        traceback.print_exc()


def save_round_validation_metrics(
    round_num: int,
    metrics: Dict,
    client_configs: List[Dict]
):
    """
    Save per-round validation metrics to client_simulation_metrics table.

    Appends validation metrics for the current round to each client's rounds array.

    Args:
        round_num: Current round number
        metrics: Dict mapping client_id (str) to metrics dict
        client_configs: List of client configuration dicts with 'id' field
    """
    if SUPABASE_CLIENT is None or SIMULATION_ID is None:
        print(f"[Round {round_num}] Database not available, skipping save")
        return

    try:
        print(f"[Round {round_num}] Saving metrics for {len(metrics)} clients to database...")

        for client_id_str, client_metrics in metrics.items():
            client_idx = int(client_id_str)
            if client_idx >= len(client_configs):
                continue

            client_config = client_configs[client_idx]
            db_client_id = client_config.get('id')

            if db_client_id is None:
                continue

            # Fetch current metrics from client_simulation_metrics
            response = SUPABASE_CLIENT.from_('client_simulation_metrics') \
                .select('metrics') \
                .eq('simulation_id', SIMULATION_ID) \
                .eq('client_id', db_client_id) \
                .execute()

            if not response.data:
                print(f"[Round {round_num}] Warning: No metrics record for client {db_client_id}")
                continue

            # Update metrics
            existing_metrics = response.data[0].get('metrics', {})

            # Initialize rounds array if needed
            if 'rounds' not in existing_metrics:
                existing_metrics['rounds'] = []

            # Find or create round entry
            round_entry = None
            for r in existing_metrics['rounds']:
                if r.get('round') == round_num:
                    round_entry = r
                    break

            if round_entry is None:
                round_entry = {'round': round_num}
                existing_metrics['rounds'].append(round_entry)

            # Add validation metrics
            round_entry['validation'] = client_metrics

            # Save back to database
            SUPABASE_CLIENT.from_('client_simulation_metrics').update({
                'metrics': existing_metrics
            }).eq('simulation_id', SIMULATION_ID) \
              .eq('client_id', db_client_id) \
              .execute()

            print(f"[Round {round_num}] ✓ Client {db_client_id}")

        print(f"[Round {round_num}] Successfully saved all validation metrics to database")

    except Exception as e:
        print(f"[Round {round_num}] Error: {e}")
        import traceback
        traceback.print_exc()


# Note: load_global_pre_fl_for_client removed - no longer needed since we don't store pre_fl metrics


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
    """
    Load existing model from checkpoint with metadata and adaptive training state.

    Handles both legacy (direct state_dict) and new (checkpoint with metadata) formats.
    Provides backward compatibility for checkpoints created before adaptive training was added.

    Args:
        model: PyTorch model instance to load weights into
        model_path: Path to checkpoint file
        device: Device to map model tensors to

    Returns:
        Tuple of (model, metadata) where metadata contains:
        - model_type: Architecture identifier
        - num_classes: Number of output classes
        - round: FL round number
        - adaptive_state: Adaptive training state dict (NEW)
        - scheduler_state: Scheduler state dict (NEW)
    """
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        # Load from checkpoint with metadata
        model.load_state_dict(checkpoint['state_dict'])

        # Return metadata if available (including new adaptive state)
        metadata = {
            'model_type': checkpoint.get('model_type'),
            'num_classes': checkpoint.get('num_classes'),
            'round': checkpoint.get('round'),
            'adaptive_state': checkpoint.get('adaptive_state'),  # NEW
            'scheduler_state': checkpoint.get('scheduler_state')  # NEW
        }
        return model, metadata
    else:
        # Direct state_dict (legacy format)
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
    print(f"[CHECKPOINT] CHECKPOINTS_PATH env: {os.getenv('CHECKPOINTS_PATH', 'NOT SET')}")

    if not os.path.exists(latest_path):
        print(f"[CHECKPOINT] Checkpoint not found at: {latest_path}")
        print(f"[CHECKPOINT] Please ensure BASE_PATH or CHECKPOINTS_PATH is set correctly")
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

def _validate_epoch(model, valloader, criterion, device):
    """
    Perform validation for a single epoch.

    This helper function evaluates the model on the validation set and returns
    the average loss. It temporarily switches the model to eval mode and back.

    Args:
        model: PyTorch model
        valloader: Validation DataLoader
        criterion: Loss function
        device: Device to run on

    Returns:
        Average validation loss for the epoch
    """
    model.eval()
    val_loss = 0.0
    val_batches = 0

    with torch.no_grad():
        for images, labels in valloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            val_batches += 1

    model.train()  # Switch back to training mode

    return val_loss / val_batches if val_batches > 0 else 0.0


def train(model, trainloader, epochs, lr, device, model_type: str = None,
          valloader=None, adaptive_state=None):
    """
    Train model with two-stage transfer learning, per-epoch validation, and adaptive scheduling.

    NEW FEATURES (Adaptive Training):
    - Per-epoch validation loop during training
    - ReduceLROnPlateau scheduler for both stages
    - Returns validation loss and scheduler state

    Args:
        model: PyTorch model to train
        trainloader: DataLoader for training data
        epochs: Total number of epochs
        lr: Initial learning rate
        device: Device to train on
        model_type: Architecture identifier
        valloader: DataLoader for validation data (REQUIRED for adaptive training)
        adaptive_state: AdaptiveTrainingState instance for dropout management

    Returns:
        Tuple of (avg_train_loss, avg_val_loss, scheduler_state_dict)
    """
    if trainloader is None:
        return 0.0, 0.0, None  # Skip training for clients without data

    model.to(device)

    # Use Focal Loss to handle class imbalance and hard examples
    criterion = FocalLoss(alpha=0.60, gamma=2.0)
    print(f"[Train] Using Focal Loss (alpha=0.60, gamma=2.0) for class imbalance handling")

    # Calculate epoch split for two-stage training
    # Stage 1: 40% of epochs (minimum 2 epochs)
    # Stage 2: 60% of epochs (remaining)
    stage1_epochs = max(2, int(epochs * 0.4))
    stage2_epochs = epochs - stage1_epochs

    total_train_loss = 0.0
    total_val_loss = 0.0
    total_train_batches = 0
    total_val_epochs = 0

    scheduler_state = {}  # Store both schedulers' states

    # Check if validation is available
    has_validation = valloader is not None

    # ========== STAGE 1: HEAD-ONLY TRAINING (Frozen Backbone) ==========
    if model_type and stage1_epochs > 0:
        print(f"\n[Train] Stage 1/2: Training classifier head only ({stage1_epochs} epochs, lr={lr:.6f})")

        # Reset all parameters to trainable first (ensures clean state)
        for param in model.parameters():
            param.requires_grad = True

        model = freeze_backbone(model, model_type)

        # Show trainable parameters
        trainable, frozen, total_params = get_trainable_params(model)
        print(f"[Train] Parameters: {trainable:,} trainable, {frozen:,} frozen, {total_params:,} total")

        # Optimizer for head-only training
        optimizer_head = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )

        # ReduceLROnPlateau scheduler for Stage 1
        if has_validation:
            scheduler_head = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_head,
                mode='min',
                factor=0.5,
                patience=2,
                min_lr=1e-6
            )

        # Training loop with per-epoch validation
        model.train()
        for epoch in range(stage1_epochs):
            epoch_train_loss = 0.0
            epoch_train_batches = 0

            # Training phase
            for images, labels in trainloader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer_head.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer_head.step()

                epoch_train_loss += loss.item()
                epoch_train_batches += 1

            avg_epoch_train_loss = epoch_train_loss / epoch_train_batches if epoch_train_batches > 0 else 0.0
            total_train_loss += epoch_train_loss
            total_train_batches += epoch_train_batches

            # Validation phase (if available)
            if has_validation:
                epoch_val_loss = _validate_epoch(model, valloader, criterion, device)
                total_val_loss += epoch_val_loss
                total_val_epochs += 1

                # Step scheduler based on validation loss
                scheduler_head.step(epoch_val_loss)

                # Get current LR
                current_lr = optimizer_head.param_groups[0]['lr']

                print(f"[Train] Stage 1 Epoch {epoch+1}/{stage1_epochs}: "
                      f"Train Loss={avg_epoch_train_loss:.4f}, Val Loss={epoch_val_loss:.4f}, LR={current_lr:.6f}")
            else:
                print(f"[Train] Stage 1 Epoch {epoch+1}/{stage1_epochs}: "
                      f"Train Loss={avg_epoch_train_loss:.4f}")

        # Save scheduler state
        if has_validation:
            scheduler_state['stage1'] = scheduler_head.state_dict()

        print(f"[Train] Stage 1 complete: Head trained on task-specific features")

    # ========== STAGE 2: FULL MODEL FINE-TUNING (Unfrozen Backbone) ==========
    if model_type and stage2_epochs > 0:
        print(f"\n[Train] Stage 2/2: Fine-tuning full model ({stage2_epochs} epochs, lr={lr/10:.6f})")
        model = unfreeze_backbone(model, model_type)

        # Show trainable parameters
        trainable, frozen, total_params = get_trainable_params(model)
        print(f"[Train] Parameters: {trainable:,} trainable, {frozen:,} frozen, {total_params:,} total")

        # Optimizer for full fine-tuning (lower LR)
        optimizer_full = torch.optim.AdamW(
            model.parameters(),
            lr=lr / 10,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )

        # ReduceLROnPlateau scheduler for Stage 2
        if has_validation:
            scheduler_full = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer_full,
                mode='min',
                factor=0.5,
                patience=2,
                min_lr=1e-7  # Lower min_lr for fine-tuning
            )

        # Training loop with per-epoch validation
        model.train()
        for epoch in range(stage2_epochs):
            epoch_train_loss = 0.0
            epoch_train_batches = 0

            # Training phase
            for images, labels in trainloader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer_full.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer_full.step()

                epoch_train_loss += loss.item()
                epoch_train_batches += 1

            avg_epoch_train_loss = epoch_train_loss / epoch_train_batches if epoch_train_batches > 0 else 0.0
            total_train_loss += epoch_train_loss
            total_train_batches += epoch_train_batches

            # Validation phase (if available)
            if has_validation:
                epoch_val_loss = _validate_epoch(model, valloader, criterion, device)
                total_val_loss += epoch_val_loss
                total_val_epochs += 1

                # Step scheduler based on validation loss
                scheduler_full.step(epoch_val_loss)

                # Get current LR
                current_lr = optimizer_full.param_groups[0]['lr']

                print(f"[Train] Stage 2 Epoch {epoch+1}/{stage2_epochs}: "
                      f"Train Loss={avg_epoch_train_loss:.4f}, Val Loss={epoch_val_loss:.4f}, LR={current_lr:.6f}")
            else:
                print(f"[Train] Stage 2 Epoch {epoch+1}/{stage2_epochs}: "
                      f"Train Loss={avg_epoch_train_loss:.4f}")

        # Save scheduler state
        if has_validation:
            scheduler_state['stage2'] = scheduler_full.state_dict()

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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_train_loss += loss.item()
                total_train_batches += 1

    # Calculate overall averages
    avg_train_loss = total_train_loss / total_train_batches if total_train_batches > 0 else 0.0
    avg_val_loss = total_val_loss / total_val_epochs if total_val_epochs > 0 else 0.0

    return avg_train_loss, avg_val_loss, scheduler_state

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
        decay_rate = 0.3
        
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
        - client_metrics: List of dicts with training metrics (train_loss, distill_loss, num-examples)
        - last_consensus: Previous round's consensus logits (for momentum smoothing)
        - eval_history: Historical evaluation metrics across rounds
        - momentum: Weight for previous consensus (default 0.3)

    Returns:
        - consensus_logits: Weighted average of client logits with momentum and class reweighting, or None if no valid clients
        - aggregation_metadata: Dict containing weights, factors, class distribution, and debugging info

    Weighting Strategy:
        CLIENT-LEVEL WEIGHTING (determines weightage based on contribution by each client):
        1. Number of data points for local trainig per client (highest influence)
        2. Calculates the combined loss from both the private training loss and the distillation loss
        3. Model architecture factor on model suitability for medical imaging

        CLASS-LEVEL WEIGHTING (compensates for class imbalance in training data):
        4. Inverse frequency weighting: Minority class predictions get boosted via sqrt(inverse_frequency)
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
        train_loss = metrics.get("train_loss", 0.0)
        distill_loss = metrics.get("distill_loss", 0.0)
        model_type = config.get("model_type", "unknown").lower()
        client_name = config.get("client_name", f"client_{i}")

        # <------------------- 1. Local Dataset Point Quantity -------------------->
        # Clients with more data get higher base weight
        base_weight = max(num_samples, 1)  # At least 1 to avoid zero division

        # <------------------- 2. Combined Loss Calculation (Distill Loss and Train Loss) : Determines the quality of the logits -------------------->
        # Lower combined loss = better convergence = higher multiplier
        combined_loss = (TRAIN_LOSS_WEIGHT * train_loss +
                        DISTILL_LOSS_WEIGHT * distill_loss)

        quality_multiplier = 1.0 / (1.0 + combined_loss)

        # <------------------- 3. Model Achitecture Suitability Factor -------------------->
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
            # All clients have zero weight (should not happen in normal operation)
        return None, {
            "error": "All clients have zero weight",
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

        # Note: Pre-FL evaluation removed as it evaluates untrained models which gives
        # meaningless ~50% accuracy results for binary classification.
        # Post-FL evaluation now only runs after FL training completes.

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

            # Save training metrics (train_loss, val_loss from local training)
            if training_metrics:
                save_round_training_metrics(current_round, training_metrics, self.client_configs)

            # --- PER-ROUND VALIDATION EVALUATION PHASE ---
            log(INFO, "")
            log(INFO, f"[ROUND {current_round}] Validation Evaluation (Per-Round Tracking)")
            log(INFO, "-" * 50)

            try:
                round_val_metrics = evaluate_all_clients_on_validation(
                    self.client_configs, device, len(self.client_configs)
                )

                # Save validation metrics to client_data.json
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

            # Save global Post-FL metrics
            save_global_post_fl_metrics(global_post_fl_metrics, self.client_configs)

            log(INFO, f"[GLOBAL] Post-FL Evaluation Complete")
            log(INFO, "")
            log(INFO, "FINAL FL RESULTS (Public Test Evaluation):")
            log(INFO, "=" * 70)

            # Show final results for each client
            for client_id, post_metrics in global_post_fl_metrics.items():
                post_acc = post_metrics.get('accuracy', 0)
                post_loss = post_metrics.get('loss', 0)
                post_gap = post_metrics.get('class_gap', 0)
                f1 = post_metrics.get('f1_score', 0)

                log(INFO, f"Client {client_id}:")
                log(INFO, f"  Accuracy: {post_acc:.1%}, Loss: {post_loss:.3f}")
                log(INFO, f"  F1 Score: {f1:.3f}, Class Gap: {post_gap:.1%}")
                log(INFO, "")

            # Calculate averages
            avg_post_acc = np.mean([m.get('accuracy', 0) for m in global_post_fl_metrics.values()])
            avg_post_loss = np.mean([m.get('loss', 0) for m in global_post_fl_metrics.values()])
            avg_f1 = np.mean([m.get('f1_score', 0) for m in global_post_fl_metrics.values()])

            log(INFO, f"Average Results: Accuracy={avg_post_acc:.1%}, Loss={avg_post_loss:.3f}, F1={avg_f1:.3f}")
            log(INFO, "")

        except Exception as e:
            log(WARNING, f"[GLOBAL] Post-FL Evaluation failed: {e}")

        log(INFO, "")
        log(INFO, f"{'='*70}")
        log(INFO, f"Strategy execution finished in {time.time() - t_start:.2f}s")
        log(INFO, f"{'='*70}")

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

        # Track client evaluation results
        clients_with_data = []

        total_loss = 0.0
        total_acc = 0.0
        total_examples = 0

        for i, msg in enumerate(results_list):
            # Check if message has content before accessing
            if not msg.has_content():
                client_name = self.client_configs[i]['client_name'] if i < len(self.client_configs) else f"Client {i}"
                print(f"[SERVER] ✗ {client_name} returned empty message during evaluation")
                continue

            metrics = msg.content.get("metrics", {})
            eval_loss = metrics.get("eval_loss", 0.0)
            eval_acc = metrics.get("eval_acc", 0.0)
            num_examples = metrics.get("num-examples", 0)
            client_id = metrics.get("client_id", -1)

            # Track by client type
            client_info = {
                "client_id": client_id,
                "loss": eval_loss,
                "acc": eval_acc,
                "examples": num_examples
            }

            # All clients have data (v3.0 runtime Dirichlet partitioning)
            clients_with_data.append(client_info)

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
            "clients_with_data": clients_with_data
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
                "clients_with_data": len(clients_with_data)
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
            # Check if message has content before accessing
            if not msg.has_content():
                client_name = self.client_configs[i]['client_name'] if i < len(self.client_configs) else f"Client {i}"
                print(f"[SERVER] ✗ {client_name} returned empty message (likely crashed during training)")
                continue

            try:
                client_arrays = msg.content["arrays"]
                client_logits = client_arrays["0"].numpy()
                logits_list.append(client_logits)

                # Extract all metrics for weight computation
                metrics = msg.content.get("metrics", {})
                client_metrics_list.append(metrics)

                # Track which client contributed
                if i < len(self.client_configs):
                    client_names.append(self.client_configs[i]['client_name'])

            except (KeyError, IndexError) as e:
                client_name = self.client_configs[i]['client_name'] if i < len(self.client_configs) else f"Client {i}"
                print(f"[SERVER] ✗ {client_name} failed to extract data: {e}")
                pass

        # Compute consensus using sophisticated weight aggregation
        consensus_logits, aggregation_metadata = compute_consensus(
            logits_list=logits_list,
            client_metrics=client_metrics_list,
            client_configs=self.client_configs[:len(logits_list)],
            server_round=server_round,
            last_consensus=self.last_consensus_logits,
            eval_history=self.eval_history,
            momentum=CONSENSUS_MOMENTUM
        )

        if consensus_logits is None:
            print(f"[SERVER] ✗ Failed to compute consensus: {aggregation_metadata.get('error', 'unknown')}")
            return None, {}, {}

        # Print informative summary
        print(f"[SERVER] ✓ Consensus computed from {aggregation_metadata['num_contributing_clients']}/{aggregation_metadata['num_clients']} clients")

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

        # Extract training metrics for client_data.json persistence
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