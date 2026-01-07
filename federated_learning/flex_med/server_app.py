import numpy as np
import json
import os
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import ServerApp
from flwr.server import Grid
from flex_med.task import FLEXMedStrategy, load_public_dataset, CLIENT_INFO_FILE_PATH, NUM_CLASSES, MODEL_CHECKPOINT_FILE_PATH, load_checkpoint

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:

    # <------------------------------------------ FEDERATED LEARNING CONFIGURATION ------------------------------------------>

    # Extract FL hyperparameters from run configuration (defined in pyproject.toml)
    # Args: num-server-rounds - Total number of FL training rounds
    #       lr - Learning rate for client-side training
    #       local-epochs - Number of epochs per client per round
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    local_epochs: int = context.run_config.get("local-epochs", 1)

    print(f"\n[SERVER] Starting Federated Learning ({num_rounds} rounds, lr={lr})")

    # <------------------------------------------ CLIENT CONFIGURATION LOADING ------------------------------------------>

    # Load client metadata from data.json containing model types, dataset paths, etc.
    # Used to display client overview and verify configuration before training starts
    if os.path.exists(CLIENT_INFO_FILE_PATH):
        with open(CLIENT_INFO_FILE_PATH, 'r') as f:
            client_configs = json.load(f)

        num_clients = len(client_configs)
        print(f"[SERVER] Loaded {num_clients} clients:")
        for i, client in enumerate(client_configs):
            data_status = "✓" if client['has_local_data'] else "✗"
            print(f"  [{i}] {client['client_name']:15} | {client['model_type']:15} | {data_status}")
    else:
        print(f"[SERVER] ⚠ Configuration file not found: {CLIENT_INFO_FILE_PATH}")

    # <------------------------------------------ CONSENSUS MATRIX INITIALIZATION ------------------------------------------>

    # Initialize consensus matrix for FedMD knowledge distillation
    # Matrix shape: (num_public_samples, num_classes) stores soft predictions
    # Round 1: Zero matrix (no consensus yet), Round 2+: Aggregated logits from previous round
    public_loader = load_public_dataset(batch_size=1)
    num_samples = len(public_loader.dataset)
    num_classes = NUM_CLASSES

    print(f"[SERVER] Consensus matrix: ({num_samples}, {num_classes})")

    # Check for checkpoint to enable resumption from interruptions
    checkpoint = load_checkpoint(MODEL_CHECKPOINT_FILE_PATH)

    if checkpoint:
        initial_consensus = checkpoint['consensus_logits']
        print(f"[SERVER] ✓ Resuming from checkpoint (Round {checkpoint['round']})")
    else:
        initial_consensus = np.zeros((num_samples, num_classes), dtype=np.float32)
        print(f"[SERVER] ✓ Starting fresh training")

    initial_arrays = ArrayRecord([initial_consensus])

    # <------------------------------------------ STRATEGY INITIALIZATION ------------------------------------------>

    # Initialize FedMD strategy with weighted consensus aggregation
    # Strategy handles: client selection, consensus aggregation, evaluation coordination
    strategy = FLEXMedStrategy(config_path=CLIENT_INFO_FILE_PATH, checkpoint_dir=MODEL_CHECKPOINT_FILE_PATH)

    # <------------------------------------------ FEDERATED LEARNING EXECUTION ------------------------------------------>

    print(f"\n{'='*70}")
    print(f"[SERVER] Executing Federated Learning")
    print(f"{'='*70}\n")

    # Start FL rounds with FedMD two-phase training:
    # Phase 1 (Distillation): Clients learn from consensus on public dataset
    # Phase 2 (Private Training): Clients train on their local private data
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=ConfigRecord({"lr": lr, "local_epochs": local_epochs}),
        num_rounds=num_rounds,
    )

    # <------------------------------------------ RESULTS PERSISTENCE ------------------------------------------>

    print(f"\n{'='*70}")
    print(f"[SERVER] Federated Learning Complete")
    print(f"{'='*70}\n")

    print(f"[SERVER] Saving results...")

    # Save final consensus logits (aggregated soft predictions on public dataset)
    # Used for: model interpretation, future rounds, checkpoint resumption
    consensus_save_path = "final_consensus.npy"
    final_logits = None
    try:
        final_logits = result.arrays["0"].numpy()
        np.save(consensus_save_path, final_logits)
        print(f"  ✓ Consensus saved: {consensus_save_path}")
    except Exception as e:
        print(f"  ✗ Error saving consensus: {e}")

    # Save training summary with configuration and client details
    # Includes: hyperparameters, client list, model types, data availability
    metrics_save_path = "training_summary.json"
    try:
        summary = {
            "num_rounds": num_rounds,
            "num_clients": strategy.num_clients,
            "learning_rate": lr,
            "local_epochs": local_epochs,
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

        print(f"  ✓ Summary saved: {metrics_save_path}")
    except Exception as e:
        print(f"  ✗ Error saving summary: {e}")

    # Save final checkpoint for resumption or analysis
    # Contains: consensus logits, evaluation history, training metrics
    if final_logits is not None and strategy.eval_history:
        try:
            from flex_med.task import save_checkpoint
            save_checkpoint(
                checkpoint_dir=MODEL_CHECKPOINT_FILE_PATH,
                current_round=num_rounds,
                consensus_logits=final_logits,
                eval_history=strategy.eval_history,
                training_metrics=summary
            )
            print(f"  ✓ Checkpoint saved: {MODEL_CHECKPOINT_FILE_PATH}")
        except Exception as e:
            print(f"  ✗ Error saving checkpoint: {e}")

    print(f"\n[SERVER] Training complete!\n")
