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