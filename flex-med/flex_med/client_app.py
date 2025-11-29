"""flex-med: A Flower / PyTorch app."""

import torch
import os
import time
import numpy as np
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

# Import helper functions
from fedmd_impl.task import (
    Net,
    load_private_dataset,
    load_public_dataset,
    get_public_logits,
    distill_knowledge,
    train as train_fn,
    test as test_fn
)

# Flower ClientApp
app = ClientApp()

def get_model_path(partition_id):
    """Get unique save path for each client's model."""
    return f"model_client_{partition_id}.pt"

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net()
    model_path = get_model_path(partition_id)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"[Client {partition_id}] Loaded model from: {model_path}")
    else:
        print(f"[Client {partition_id}] Warning: No saved model. Using random weights.")

    model.to(device)
    distill_loss = 0.0

    # Knowledge distillation
    if "arrays" in msg.content and msg.content["arrays"]:
        try:
            # Extract consensus logits from server
            consensus_data = msg.content["arrays"]
            consensus_logits = consensus_data[0]  # List format
            
            # Check if consensus is non-zero (skip Round 1 which has zero consensus)
            if np.any(consensus_logits != 0):
                print(f"[Client {partition_id}] Consensus received! Starting distillation...")
                
                # Load public dataset for distillation
                public_loader = load_public_dataset(batch_size=64)
                
                # Distill consensus knowledge into local model
                distill_loss = distill_knowledge(
                    model=model,
                    public_loader=public_loader,
                    consensus_logits=consensus_logits,
                    device=device,
                    epochs=3,        # Distillation epochs
                    lr=0.001,        # Distillation learning rate
                    temperature=3.0  # Temperature for soft targets
                )
                
                print(f"[Client {partition_id}] ✓ Distillation complete. Loss: {distill_loss:.4f}")
                
                # Save model after distillation
                torch.save(model.state_dict(), model_path)
            else:
                print(f"[Client {partition_id}] Zero consensus (Round 1). Skipping distillation.")
        
        except Exception as e:
            print(f"[Client {partition_id}] Error during distillation: {e}")

    print(f"[Client {partition_id}] Starting private training...")

    trainloader, _ = load_private_dataset(partition_id, num_partitions)

    start_time = time.time()

    train_loss = train_fn(
        model=model,
        trainloader=trainloader,
        epochs=context.run_config["local-epochs"],
        lr=msg.content["config"]["lr"],
        device=device
    )
    
    training_time = time.time() - start_time

    # Save model after private training
    torch.save(model.state_dict(), model_path)

    # Generate Logits - Public Dataset

    public_loader = load_public_dataset(batch_size=64)
    public_logits = get_public_logits(model, public_loader, device)

    logits_record = ArrayRecord([public_logits])

    # # Load the model and initialize it with the received weights
    # model = Net()
    # model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    # # Load the data
    # partition_id = context.node_config["partition-id"]
    # num_partitions = context.node_config["num-partitions"]
    # trainloader, _ = load_data(partition_id, num_partitions)

    # # Call the training function
    # train_loss = train_fn(
    #     model,
    #     trainloader,
    #     context.run_config["local-epochs"],
    #     msg.content["config"]["lr"],
    #     device,
    # )

    # # Construct and return reply Message
    # model_record = ArrayRecord(model.state_dict())
    # metrics = {
    #     "train_loss": train_loss,
    #     "num-examples": len(trainloader.dataset),
    # }
    metrics = {
        "train_loss": train_loss,
        "distill_loss": distill_loss,
        "num-examples": len(trainloader.dataset),
        "training_time": training_time
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": logits_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    # model = Net()
    # model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model.to(device)

    model = Net()
    model_path = get_model_path(partition_id)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"[Client {partition_id}] Loaded model from: {model_path}")
    else:
        print(f"[Client {partition_id}] Warning: No saved model. Using random weights.")
    
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    # _, valloader = load_data(partition_id, num_partitions)
    _, valloader = load_private_dataset(partition_id, num_partitions)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
