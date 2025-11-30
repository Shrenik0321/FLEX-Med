"""flex-med: A Flower / PyTorch app."""

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import ServerApp
from flwr.server import Grid
from flex_med.task import FedMDStrategy, load_public_dataset

# Create ServerApp
app = ServerApp()

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]

    # <--- DECOUPLING LOGIC STARTS HERE --->
    
    # 1. Load just the data loader to get the dataset size
    # We do NOT load any model here.
    public_loader = load_public_dataset(batch_size=64)
    
    # 2. Determine dimensions mathematically
    # Rows = Number of samples in the public dataset
    num_samples = len(public_loader.dataset)
    
    # Columns = Number of classes (CIFAR-10 has 10 classes)
    # You could also put this in pyproject.toml/context.run_config
    num_classes = 10 
    
    print(f"[Server] Initializing consensus for {num_samples} samples and {num_classes} classes.")

    # 3. Create the zero consensus matrix directly
    # Shape: (5000, 10) if using the subset defined in task.py
    zero_consensus = np.zeros((num_samples, num_classes), dtype=np.float32)
    
    # <--- DECOUPLING LOGIC ENDS HERE --->

    # Pack initial consensus
    initial_arrays = ArrayRecord([zero_consensus])

    strategy = FedMDStrategy()

    # Start strategy
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    # Save final result (Consensus Logits) to disk
    # Note: We save the *logits*, not a model, because the server has no model.
    print("\n[Server] FedMD Simulation Complete.")
    print("[Server] Saving final consensus logits to 'final_consensus.npy'...")
    
    final_logits = result.arrays["0"].numpy() # Extract numpy array
    np.save("final_consensus.npy", final_logits)