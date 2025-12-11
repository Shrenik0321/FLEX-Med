"""flex-med: Server Application."""

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
    # We set batch_size=1 because we only need the total length (len(dataset))
    # This reads from the /content/fed_data/public_anchor folder we created
    print("[Server] Loading Public Anchor metadata...")
    public_loader = load_public_dataset(batch_size=1)
    
    # 2. Determine dimensions mathematically
    # Rows = Number of samples in the public dataset
    num_samples = len(public_loader.dataset)
    
    # Columns = Number of classes (0: Healthy, 1: ALL)
    num_classes = 2 
    
    print(f"[Server] Initializing consensus for {num_samples} samples and {num_classes} classes.")

    # 3. Create the zero consensus matrix directly
    # Shape: (Total_Public_Images, 2)
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
    # This file represents the "Universal Knowledge" distilled from all hospitals
    print("\n[Server] FedMD Simulation Complete.")
    print("[Server] Saving final consensus logits to 'final_consensus.npy'...")
    
    try:
        final_logits = result.arrays["0"].numpy() # Extract numpy array
        np.save("final_consensus.npy", final_logits)
        print("[Server] Save successful.")
    except Exception as e:
        print(f"[Server] Error saving consensus: {e}")