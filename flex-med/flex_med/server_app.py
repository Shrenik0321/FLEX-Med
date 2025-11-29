"""flex-med: A Flower / PyTorch app."""

import torch
import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import ServerApp
from flwr.server import Grid
from fedmd_impl.task import FedMD, load_public_dataset, Net, get_public_logits

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    # fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    # local_epochs: int = context.run_config.get("local-epochs", 1)

    # # Load global model
    # global_model = Net()
    # arrays = ArrayRecord(global_model.state_dict())

    # # Initialize FedAvg strategy
    # strategy = FedAvg(fraction_train=fraction_train)

    # Initialise consensus
    public_loader = load_public_dataset(batch_size=64)
    
    dummy_model = Net()
    device = torch.device("cpu")
    
    # Generate logits to get shape
    initial_logits_np = get_public_logits(dummy_model, public_loader, device)
    
    # Initialize with zeros (Round 1 will skip distillation)
    zero_consensus = np.zeros_like(initial_logits_np)
    
    # Pack initial consensus (using list format)
    initial_arrays = ArrayRecord([zero_consensus])

    strategy = FedMD()

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    print("FedMD SERVER COMPLETED")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
