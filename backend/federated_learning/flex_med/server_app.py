import numpy as np
import json
import os
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import ServerApp
from flwr.server import Grid
from flex_med.utils.config import CLIENT_INFO_FILE_PATH, MODEL_CHECKPOINT_FILE_PATH
from flex_med.task import (
    FLEXMedStrategy, load_public_dataset, NUM_CLASSES,
    load_client_config
)

app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:

    # <-------------------------------------- LOAD CONFIGURATIONS -------------------------------------->
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    local_epochs: int = context.run_config.get("local-epochs", 1)
    server_round: int = context.run_config.get("round", 1)
    batch_size: int = context.run_config.get("batch-size", 32)

    print(f"\n[SERVER] Starting FL ({num_rounds} rounds, lr={lr})")

    # <-------------------------------------- LOAD CLIENT CONFIGURATIONS -------------------------------------->
    client_configs = load_client_config()
    print(f"[SERVER] {len(client_configs)} clients loaded")

    # <-------------------------------------- LOAD PUBLIC ANCHOR DATASET -------------------------------------->
    public_loader = load_public_dataset(batch_size=1, round_num=server_round, total_rounds=num_rounds)
    num_samples = len(public_loader.dataset)

    initial_consensus = np.zeros((num_samples, NUM_CLASSES), dtype=np.float32)

    # <-------------------------------------- LOAD FL STRATEGY -------------------------------------->
    strategy = FLEXMedStrategy(
        config_path=CLIENT_INFO_FILE_PATH,
        batch_size=batch_size
    )

    # <-------------------------------------- EXECUTE FL SIMULATION -------------------------------------->
    print(f"\n{'='*60}")
    print(f"[SERVER] Executing Federated Learning")
    print(f"{'='*60}\n")

    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord([initial_consensus]),
        train_config=ConfigRecord({"lr": lr, "local_epochs": local_epochs}),
        num_rounds=num_rounds,
    )

    # <-------------------------------------- SAVE RESULTS -------------------------------------->
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

    print(f"\n[SERVER] Training complete!\n")