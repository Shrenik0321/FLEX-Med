import numpy as np
import json
import os
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import ServerApp
from flwr.server import Grid
from flex_med.utils.config import CLIENT_INFO_FILE_PATH, MODEL_CHECKPOINT_FILE_PATH
from flex_med.task import FLEXMedStrategy, load_public_dataset, NUM_CLASSES, load_checkpoint, load_client_config

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

    # Check for resume mode
    resume_mode = os.getenv("FLEX_MED_RESUME", "auto").lower()
    checkpoint = load_checkpoint(MODEL_CHECKPOINT_FILE_PATH)

    if checkpoint and resume_mode != "false":
        start_round = checkpoint['round'] + 1
        remaining = num_rounds - checkpoint['round']
        print(f"\n[SERVER] Resuming Federated Learning from round {start_round}/{num_rounds}")
        print(f"[SERVER] {remaining} rounds remaining (lr={lr})")
    else:
        print(f"\n[SERVER] Starting Federated Learning ({num_rounds} rounds, lr={lr})")

    # Identify the current FL round from server message
    try:
        server_round = int(context.run_config["round"])
    except:
        server_round = 1

    # <------------------------------------------ CLIENT CONFIGURATION LOADING ------------------------------------------>

    # Load client metadata containing model types, dataset paths, etc.
    # Uses load_client_config() which prioritizes:
    # 1. FLEX_MED_CLIENT_CONFIGS environment variable (in-memory JSON)
    # 2. Database fetch using SUPABASE_CLIENT
    # 3. File-based config (deprecated fallback)
    try:
        client_configs = load_client_config()
        num_clients = len(client_configs)
        print(f"[SERVER] Loaded {num_clients} clients:")
        for i, client in enumerate(client_configs):
            client_name = client.get('client_name', f'Client_{i}')
            model_type = client.get('model_type', 'unknown')
            print(f"  [{i}] {client_name:15} | {model_type:15}")
    except Exception as e:
        print(f"[SERVER] ⚠ Error loading client configuration: {e}")
        raise

    # <------------------------------------------ CONSENSUS MATRIX INITIALIZATION ------------------------------------------>

    # Initialize consensus matrix for FedMD knowledge distillation
    # Matrix shape: (num_public_samples, num_classes) stores soft predictions
    # Round 1: Zero matrix (no consensus yet), Round 2+: Aggregated logits from previous round
    public_loader = load_public_dataset(batch_size=1, round_num=server_round, total_rounds=num_rounds)
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

    final_logits = None
    try:
        final_logits = result.arrays["0"].numpy()
        np.save(consensus_save_path, final_logits)
        print(f"  ✓ Consensus saved: {consensus_save_path}")
    except Exception as e:
        print(f"  ✗ Error saving consensus: {e}")
        
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
    