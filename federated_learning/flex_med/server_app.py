"""flex-med: Server Application with JSON-based configuration."""

import numpy as np
import json
import os
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import ServerApp
from flwr.server import Grid
from flex_med.task import FedMDStrategy, load_public_dataset, CONFIG_FILE_PATH

# Create ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    """
    Main entry point for the ServerApp.
    
    This version uses data.json for client configuration.
    Note: num-supernodes is configured in the FastAPI endpoint before this runs.
    """
    
    print("\n" + "="*70)
    print("FLEX-MED: Federated Learning for Medical Image Classification")
    print("="*70)

    # ===== STEP 1: LOAD CONFIGURATION =====
    print("\n[Server] Step 1: Loading Configuration...")
    
    # Load run configuration
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    local_epochs: int = context.run_config.get("local-epochs", 1) 
    
    print(f"  - Number of rounds: {num_rounds}")
    print(f"  - Learning rate: {lr}")
    
    # Load client configuration from JSON
    if os.path.exists(CONFIG_FILE_PATH):
        with open(CONFIG_FILE_PATH, 'r') as f:
            client_configs = json.load(f)
        
        num_clients = len(client_configs)
        print(f"  - Number of clients: {num_clients}")
        
        print("\n[Server] Client Overview:")
        for i, client in enumerate(client_configs):
            data_status = "✓ Has Data" if client['has_local_data'] else "✗ Free Rider"
            print(f"  Client {i}: {client['client_name']:15} | {client['model_type']:15} | {data_status}")
    else:
        print(f"  ⚠ Warning: Configuration file not found at {CONFIG_FILE_PATH}")
        print("  Continuing with default settings...")
    
    # ===== STEP 2: INITIALIZE PUBLIC DATASET & CONSENSUS =====
    print("\n[Server] Step 2: Initializing Consensus Matrix...")
    
    # Load public dataset metadata
    public_loader = load_public_dataset(batch_size=1)
    num_samples = len(public_loader.dataset)
    num_classes = 2  # Binary classification: Healthy vs Leukemia
    
    print(f"  - Public dataset samples: {num_samples}")
    print(f"  - Number of classes: {num_classes}")
    print(f"  - Consensus matrix shape: ({num_samples}, {num_classes})")
    
    # Create zero consensus matrix for Round 1
    zero_consensus = np.zeros((num_samples, num_classes), dtype=np.float32)
    initial_arrays = ArrayRecord([zero_consensus])
    
    print(f"  ✓ Zero consensus matrix initialized")

    # ===== STEP 3: INITIALIZE STRATEGY =====
    print("\n[Server] Step 3: Initializing FedMD Strategy...")
    
    strategy = FedMDStrategy(config_path=CONFIG_FILE_PATH)
    
    print(f"  ✓ Strategy initialized with {strategy.num_clients} clients")

    # ===== STEP 4: START FEDERATED LEARNING =====
    print("\n[Server] Step 4: Starting Federated Learning...")
    print("="*70)
    
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=ConfigRecord({"lr": lr, "local_epochs": local_epochs}),
        num_rounds=num_rounds,
    )

    # ===== STEP 5: SAVE RESULTS =====
    print("\n" + "="*70)
    print("[Server] Federated Learning Complete!")
    print("="*70)
    
    print("\n[Server] Step 5: Saving Results...")
    
    # Save final consensus logits
    consensus_save_path = "final_consensus.npy"
    try:
        final_logits = result.arrays["0"].numpy()
        np.save(consensus_save_path, final_logits)
        print(f"  ✓ Final consensus saved to: {consensus_save_path}")
        print(f"    - Shape: {final_logits.shape}")
        print(f"    - Size: {os.path.getsize(consensus_save_path) / 1024:.2f} KB")
    except Exception as e:
        print(f"  ✗ Error saving consensus: {e}")
    
    # Save metrics summary
    metrics_save_path = "training_summary.json"
    try:
        summary = {
            "num_rounds": num_rounds,
            "num_clients": strategy.num_clients,
            "learning_rate": lr,
            "public_dataset_size": num_samples,
            "num_classes": num_classes,
            "clients": [
                {
                    "id": i,
                    "name": client['client_name'],
                    "model_type": client['model_type'],
                    "has_local_data": client['has_local_data'],
                    "model_path": client['model_path']
                }
                for i, client in enumerate(client_configs)
            ]
        }
        
        with open(metrics_save_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"  ✓ Training summary saved to: {metrics_save_path}")
    except Exception as e:
        print(f"  ⚠ Could not save training summary: {e}")
    
    # ===== STEP 6: UPDATE CLIENT METRICS (OPTIONAL) =====
    print("\n[Server] Step 6: Updating Client Metrics...")
    
    # This is where you could update the data.json file with post-FL metrics
    # For now, we'll just print what would be updated
    print("  Note: To update data.json with post-FL metrics, implement metric tracking")
    print("        during the evaluation phase.")
    
    print("\n" + "="*70)
    print("FLEX-MED: Process Complete")
    print("="*70)
    print("\nNext Steps:")
    print("  1. Check final_consensus.npy for the distilled knowledge")
    print("  2. Review individual client models in their respective paths")
    print("  3. Analyze training_summary.json for experiment details")
    print("  4. Use the trained models for inference on new data")
    print("\n" + "="*70 + "\n")