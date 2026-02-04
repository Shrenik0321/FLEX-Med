import numpy as np
import json
import os
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import ServerApp
from flwr.server import Grid
from flex_med.utils.config import CLIENT_INFO_FILE_PATH, MODEL_CHECKPOINT_FILE_PATH
from flex_med.task import (
    FLEXMedStrategy, load_public_dataset, NUM_CLASSES,
    load_checkpoint, save_checkpoint, load_client_config
)

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    # FL hyperparameters from pyproject.toml
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    local_epochs: int = context.run_config.get("local-epochs", 1)
    server_round: int = context.run_config.get("round", 1)

    # Load checkpoint for potential resume
    resume_mode = os.getenv("FLEX_MED_RESUME", "auto").lower()
    checkpoint = load_checkpoint(MODEL_CHECKPOINT_FILE_PATH)

    if checkpoint and resume_mode != "false":
        start_round = checkpoint['round'] + 1
        print(f"\n[SERVER] Resuming from round {start_round}/{num_rounds} (lr={lr})")
    else:
        checkpoint = None  # Force fresh start if resume disabled
        print(f"\n[SERVER] Starting FL ({num_rounds} rounds, lr={lr})")

    # Load client configurations
    client_configs = load_client_config()
    print(f"[SERVER] {len(client_configs)} clients loaded")

    # Initialize consensus matrix for FedMD knowledge distillation
    public_loader = load_public_dataset(batch_size=1, round_num=server_round, total_rounds=num_rounds)
    num_samples = len(public_loader.dataset)

    if checkpoint:
        initial_consensus = checkpoint['consensus_logits']
    else:
        initial_consensus = np.zeros((num_samples, NUM_CLASSES), dtype=np.float32)

    # Initialize strategy and execute FL
    strategy = FLEXMedStrategy(config_path=CLIENT_INFO_FILE_PATH, checkpoint_dir=MODEL_CHECKPOINT_FILE_PATH)

    print(f"\n{'='*60}")
    print(f"[SERVER] Executing Federated Learning")
    print(f"{'='*60}\n")

    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord([initial_consensus]),
        train_config=ConfigRecord({"lr": lr, "local_epochs": local_epochs}),
        num_rounds=num_rounds,
    )

    # Save results
    print(f"\n{'='*60}")
    print(f"[SERVER] Federated Learning Complete")
    print(f"{'='*60}\n")

    checkpoint_dir = os.path.dirname(MODEL_CHECKPOINT_FILE_PATH)
    consensus_save_path = os.path.join(checkpoint_dir, "final_consensus.npy")
    metrics_save_path = os.path.join(checkpoint_dir, "training_summary.json")

    final_logits = None
    try:
        final_logits = result.arrays["0"].numpy()
        np.save(consensus_save_path, final_logits)
        print(f"  Consensus saved: {consensus_save_path}")
    except Exception as e:
        print(f"  Error saving consensus: {e}")

    summary = {
        "num_rounds": num_rounds,
        "num_clients": strategy.num_clients,
        "learning_rate": lr,
        "local_epochs": local_epochs,
        "public_dataset_size": num_samples,
        "num_classes": NUM_CLASSES,
        "clients": [
            {"id": i, "name": c['client_name'], "model_type": c['model_type'], "model_path": c['model_path']}
            for i, c in enumerate(client_configs)
        ]
    }

    try:
        with open(metrics_save_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary saved: {metrics_save_path}")
    except Exception as e:
        print(f"  Error saving summary: {e}")

    if final_logits is not None and strategy.eval_history:
        try:
            save_checkpoint(
                checkpoint_dir=MODEL_CHECKPOINT_FILE_PATH,
                current_round=num_rounds,
                consensus_logits=final_logits,
                eval_history=strategy.eval_history,
                training_metrics=summary
            )
            print(f"  Checkpoint saved: {MODEL_CHECKPOINT_FILE_PATH}")
        except Exception as e:
            print(f"  Error saving checkpoint: {e}")

    print(f"\n[SERVER] Training complete!\n")
    