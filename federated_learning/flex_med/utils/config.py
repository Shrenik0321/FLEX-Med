import os

# <------------------------------------------ CONFIGURATION ------------------------------------------>
BASE_PATH = os.getenv("BASE_PATH", "/content/drive/MyDrive/College/FLEX-Med")

# Project root (flex-med)
PROJECT_DIR = os.getenv(
    "FLEX_MED_PROJECT_DIR",
    os.path.join(BASE_PATH, "flex-med")
)

# Logs
LOG_FILE_PATH = os.path.join(PROJECT_DIR, "flex_med/utils/server_debug.log")
SIMULATION_LOG_PATH = os.path.join(PROJECT_DIR, "flex_med/utils/simulation_run.log")

# Client Information (API ↔ FL bridge)
CLIENT_INFO_FILE_PATH = os.path.join(PROJECT_DIR, "flex_med/data.json")

# Data JSON used by orchestrator
DATA_JSON_PATH = os.path.join(PROJECT_DIR, "cnmc_data.json")

# Pyproject.toml (Flower config)
PYPROJECT_PATH = os.path.join(PROJECT_DIR, "pyproject.toml")

# Dataset paths
DATASET_FILE_PATH = os.path.join(BASE_PATH, "cnmc_datasets")
PUBLIC_ANCHOR_DATASET_PATH = os.path.join(DATASET_FILE_PATH, "cnmc_public_anchor")
PUBLIC_TEST_DATASET_PATH = os.path.join(DATASET_FILE_PATH, "cnmc_public_test")

# Model checkpoints
MODEL_CHECKPOINT_FILE_PATH = os.path.join(BASE_PATH, "checkpoints")

# Metrics & graphs
ROUND_METRICS_FILE_PATH = os.path.join(BASE_PATH, "round_metrics.json")
GRAPHS_OUTPUT_DIR = os.path.join(BASE_PATH, "graphical_visualisation")

# Model constraints
NUM_CLASSES = 2
IMG_SIZE = 224

# Shell runner
RUN_SIMULATION_SHELL_FILE_PATH = os.path.join(
    PROJECT_DIR, "flex_med/utils/run_simulation.sh"
)

SIMULATION_STATUS_JSON_FILE_PATH = os.path.join(
    PROJECT_DIR, "flex_med/utils/simulation_status.json"
)

# Model suitability scores for medical imaging tasks
MODEL_SUITABILITY_SCORES = {
    # Tier 1: Excellent (ResNet, DenseNet - skip connections, dense features)
    'resnet18': 1.10, 'resnet34': 1.10, 'resnet50': 1.10, 'resnet101': 1.10, 'resnet152': 1.10,
    'densenet121': 1.10, 'densenet161': 1.10, 'densenet169': 1.10, 'densenet201': 1.10,

    # Tier 2: Good (EfficientNet, Inception - balanced efficiency/accuracy)
    'efficientnet_b0': 1.05, 'efficientnet_b1': 1.05, 'efficientnet_b2': 1.05,
    'efficientnet_b3': 1.05, 'efficientnet_b4': 1.05, 'efficientnet_b5': 1.05,
    'efficientnet_b6': 1.05, 'efficientnet_b7': 1.05,
    'inception_v3': 1.05, 'googlenet': 1.05,

    # Tier 3: Standard (VGG - proven baseline)
    'vgg16': 1.00, 'vgg19': 1.00,

    # Tier 4: Mobile-optimized (may sacrifice accuracy for efficiency)
    'mobilenet_v2': 0.95, 'mobilenet_v3_small': 0.90, 'mobilenet_v3_large': 0.95,
    'squeezenet': 0.90,

    # Tier 5: Older/less suitable
    'alexnet': 0.90, 'vgg11': 0.95, 'vgg13': 0.95,
}

# Combined Loss weighting parameters (for quality multiplier)
TRAIN_LOSS_WEIGHT = 0.7  # Private training more important
DISTILL_LOSS_WEIGHT = 0.3  # Distillation secondary

CONSENSUS_MOMENTUM=0.5