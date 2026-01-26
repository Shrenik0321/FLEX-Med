#!/bin/bash
cd "/content/drive/MyDrive/College/FLEX-Med/flex-med"

echo "[Script] Starting Flex-Med Simulation at $(date)..."

# Only install the local package in editable mode (Fast & Lightweight)
pip install -e . -q

# Run Flower CLI directly
flwr run .
