#!/bin/bash
# Flexible FL Simulation Runner
# Usage: ./run_simulation.sh [path_to_federated_learning_dir]
#
# If no path provided, uses environment variable FLEX_MED_PROJECT_DIR
# or defaults to the script's parent directory structure

# Determine the FL project directory
if [ -n "$1" ]; then
    # Use provided argument
    FL_DIR="$1"
elif [ -n "$FLEX_MED_PROJECT_DIR" ]; then
    # Use environment variable
    FL_DIR="$FLEX_MED_PROJECT_DIR"
else
    # Default: go up from utils -> flex_med -> federated_learning
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    FL_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi

echo "[Script] FL Project Directory: $FL_DIR"
echo "[Script] Starting FLEX-Med Simulation at $(date)..."

# Change to FL directory
cd "$FL_DIR" || {
    echo "[Error] Failed to change to directory: $FL_DIR"
    exit 1
}

# Only install the local package in editable mode (Fast & Lightweight)
echo "[Script] Installing flex_med package in editable mode..."
pip install -e . -q

# Run Flower CLI directly
echo "[Script] Running Flower simulation..."
flwr run .

echo "[Script] Simulation completed at $(date)"
