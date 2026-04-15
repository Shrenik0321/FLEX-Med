"""Standalone configuration for FLEX-Med FL simulation (Colab-compatible)."""
import os
from pathlib import Path

# ============================================================================
# PATH RESOLUTION
# ============================================================================
_CURRENT_FILE = Path(__file__).resolve()
_FL_DIR = _CURRENT_FILE.parent.parent.parent  # flex_med/utils/config.py -> federated_learning/
_PROJECT_ROOT = _FL_DIR.parent  # federated_learning/ -> FLEX-Med/

BASE_PATH = os.getenv("BASE_PATH", str(_PROJECT_ROOT))
PROJECT_DIR = str(_FL_DIR)

# ============================================================================
# LOGS
# ============================================================================
SIMULATION_LOG_PATH = os.getenv(
    "SIMULATION_LOG_PATH",
    str(_FL_DIR / "flex_med" / "utils" / "simulation_run.log")
)

# ============================================================================
# CLIENT CONFIGURATION
# ============================================================================
CLIENT_INFO_FILE_PATH = os.getenv(
    "FLEX_MED_CONFIG_FILE",
    str(_FL_DIR / "flex_med" / "utils" / "client_data.json")
)

# ============================================================================
# PYPROJECT.TOML
# ============================================================================
PYPROJECT_PATH = str(_FL_DIR / "pyproject.toml")

# ============================================================================
# DATASETS
# ============================================================================
DATASET_FILE_PATH = os.getenv(
    "DATASET_FILE_PATH",
    str(Path(BASE_PATH) / "datasets" / "cnmc")
)
PUBLIC_ANCHOR_DATASET_PATH = os.getenv(
    "PUBLIC_ANCHOR_DATASET_PATH",
    str(Path(DATASET_FILE_PATH) / "cnmc_public_anchor")
)
PUBLIC_TEST_DATASET_PATH = os.getenv(
    "PUBLIC_TEST_DATASET_PATH",
    str(Path(DATASET_FILE_PATH) / "cnmc_public_test")
)
LOCAL_TRAIN_DATASET_PATH = os.getenv(
    "LOCAL_TRAIN_DATASET_PATH",
    str(Path(DATASET_FILE_PATH) / "cnmc_local_train")
)

# ============================================================================
# MODEL CHECKPOINTS
# ============================================================================
MODEL_CHECKPOINT_FILE_PATH = os.getenv(
    "CHECKPOINTS_PATH",
    str(Path(BASE_PATH) / "checkpoints")
)

# ============================================================================
# METRICS & GRAPHS
# ============================================================================
GRAPHS_OUTPUT_DIR = os.getenv(
    "GRAPHS_OUTPUT_DIR",
    str(Path(BASE_PATH) / "graphical_visualisation")
)

# ============================================================================
# MODEL CONSTRAINTS
# ============================================================================
NUM_CLASSES = 2
IMG_SIZE = 256

# ============================================================================
# SIMULATION RUNNER
# ============================================================================
RUN_SIMULATION_SHELL_FILE_PATH = str(_FL_DIR / "flex_med" / "utils" / "run_simulation.sh")

# ============================================================================
# LOSS WEIGHTING
# ============================================================================
TRAIN_LOSS_WEIGHT = float(os.getenv("FLEX_MED_TRAIN_LOSS_WEIGHT", "0.72"))
DISTILL_LOSS_WEIGHT = float(os.getenv("FLEX_MED_DISTILL_LOSS_WEIGHT", "0.28"))

# ============================================================================
# DIRICHLET PARTITIONING
# ============================================================================
DIRICHLET_ALPHA = float(os.getenv("FLEX_MED_DIRICHLET_ALPHA", "1.5"))
DIRICHLET_SEED = int(os.getenv("FLEX_MED_DIRICHLET_SEED", "42"))
DIRICHLET_MIN_PARTITION_SIZE = int(os.getenv("FLEX_MED_DIRICHLET_MIN_PARTITION_SIZE", "400"))

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================)
MINORITY_BOOST = float(os.getenv("FLEX_MED_MINORITY_BOOST", "0.90"))

DISTILL_WEIGHT_BASE = float(os.getenv("FLEX_MED_DISTILL_WEIGHT", "0.80"))
DISTILL_DECAY_RATE = float(os.getenv("FLEX_MED_DISTILL_DECAY", "0.10"))

WEIGHT_DECAY = float(os.getenv("FLEX_MED_WEIGHT_DECAY", "0.02"))