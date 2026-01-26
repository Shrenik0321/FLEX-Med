#!/bin/bash
# Simple script to tail FL simulation logs

LOG_FILE="/home/sshre/Code-WSL/FLEX-Med/federated_learning/flex_med/utils/simulation_run.log"

echo "========================================="
echo "FL Simulation Logs Monitor"
echo "========================================="
echo "Log file: $LOG_FILE"
echo ""

if [ ! -f "$LOG_FILE" ]; then
    echo "⚠️  Log file not found. Simulation may not have started yet."
    echo "Run the simulation first via: POST /api/start_fl_simulation"
    exit 1
fi

echo "Showing last 50 lines and following new logs..."
echo "Press Ctrl+C to stop"
echo ""
echo "========================================="
echo ""

tail -f -n 50 "$LOG_FILE"
