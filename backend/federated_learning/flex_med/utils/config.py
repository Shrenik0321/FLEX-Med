import os
import sys
from pathlib import Path

# Try to import from app.config (assuming running from backend root)
try:
    from app.config import get_settings
except ImportError:
    # If app is not found, maybe we are running inside federated_learning or backend is top level
    # Add backend root to path
    current_file = Path(__file__).resolve()
    # utils -> flex_med -> federated_learning -> backend
    backend_root = current_file.parent.parent.parent.parent
    sys.path.append(str(backend_root))
    try:
        from app.config import get_settings
    except ImportError:
        # Last resort: try importing backend.app.config
        try:
             from backend.app.config import get_settings
        except ImportError:
             raise ImportError("Could not import app.config. Ensure python path is set correctly.")

_settings = get_settings()

# <------------------------------------------ CONFIGURATION ------------------------------------------>
BASE_PATH = str(_settings.base_path)
PROJECT_DIR = str(_settings.project_dir)

# Logs
SIMULATION_LOG_PATH = str(_settings.simulation_log_path)

# Client Information
CLIENT_INFO_FILE_PATH = str(_settings.client_info_file_path)

# Pyproject.toml
PYPROJECT_PATH = str(_settings.pyproject_path)

# Dataset paths
DATASET_FILE_PATH = str(_settings.dataset_file_path)
PUBLIC_ANCHOR_DATASET_PATH = str(_settings.public_anchor_dataset_path)
PUBLIC_TEST_DATASET_PATH = str(_settings.public_test_dataset_path)
LOCAL_TRAIN_DATASET_PATH = str(_settings.local_train_dataset_path)

# Model checkpoints
MODEL_CHECKPOINT_FILE_PATH = str(_settings.model_checkpoint_file_path)

# Metrics & graphs
GRAPHS_OUTPUT_DIR = str(_settings.graphs_output_dir)

# Model constraints
NUM_CLASSES = _settings.num_classes
IMG_SIZE = _settings.img_size

# Shell runner
RUN_SIMULATION_SHELL_FILE_PATH = str(_settings.run_simulation_shell_file_path)



# Loss weighting
TRAIN_LOSS_WEIGHT = _settings.train_loss_weight
DISTILL_LOSS_WEIGHT = _settings.distill_loss_weight

CONSENSUS_MOMENTUM = _settings.consensus_momentum

# Dirichlet partitioning (runtime data heterogeneity)
DIRICHLET_ALPHA = _settings.dirichlet_alpha
DIRICHLET_SEED = _settings.dirichlet_seed
DIRICHLET_MIN_PARTITION_SIZE = _settings.dirichlet_min_partition_size