"""flex-med: A Flower / PyTorch app."""

import torch
import os
import time
import numpy as np
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flex_med.task import (
    get_resnet,   
    get_mobilenet,   
    load_private_dataset,
    load_public_dataset,
    get_public_logits,
    distill_knowledge,
    train as train_fn,
    test as test_fn
)

app = ClientApp()

def get_model_path(partition_id):
    return f"model_client_{partition_id}.pt"

def load_model_for_client(partition_id):
    """
    Selects the architecture based on partition ID.
    Even IDs -> ResNet
    Odd IDs  -> MobileNet
    """
    if partition_id % 2 == 0:
        print(f"[Client {partition_id}] Initializing ResNet-18")
        return get_resnet()
    else:
        print(f"[Client {partition_id}] Initializing MobileNetV2")
        return get_mobilenet()

@app.train()
def train(msg: Message, context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Instantiate the specific model for this client
    model = load_model_for_client(partition_id)
    
    model_path = get_model_path(partition_id)

    if os.path.exists(model_path):
        # We must map location to device to avoid GPU/CPU mismatch on load
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[Client {partition_id}] Loaded saved model state.")
    else:
        print(f"[Client {partition_id}] No saved model. Starting fresh.")

    model.to(device)
    distill_loss = 0.0

    # 2. Knowledge Distillation (FedMD Core)
    if "arrays" in msg.content and msg.content["arrays"]:
        try:
            consensus_data = msg.content["arrays"]
            consensus_logits = consensus_data[0]
            
            # Only distill if consensus is not all zeros (Round > 1)
            if np.any(consensus_logits != 0):
                print(f"[Client {partition_id}] Consensus received. Distilling...")
                public_loader = load_public_dataset(batch_size=64)
                
                distill_loss = distill_knowledge(
                    model=model,
                    public_loader=public_loader,
                    consensus_logits=consensus_logits,
                    device=device,
                    epochs=1,        # Keep epochs low for demo
                    lr=0.001,
                    temperature=2.0 
                )
                print(f"[Client {partition_id}] Distillation Done. Loss: {distill_loss:.4f}")
            else:
                print(f"[Client {partition_id}] Round 1: Skipping distillation.")
        except Exception as e:
            print(f"[Client {partition_id}] Distillation Error: {e}")

    # 3. Private Training
    print(f"[Client {partition_id}] Training on private data...")
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

    # Save model state
    torch.save(model.state_dict(), model_path)

    # 4. Generate Logits on Public Dataset to send back
    public_loader = load_public_dataset(batch_size=64)
    public_logits = get_public_logits(model, public_loader, device)

    logits_record = ArrayRecord([public_logits])
    metrics = {
        "train_loss": train_loss,
        "distill_loss": distill_loss,
        "num-examples": len(trainloader.dataset),
        "training_time": training_time
    }
    
    return Message(content=RecordDict({"arrays": logits_record, "metrics": MetricRecord(metrics)}), reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Instantiate correct architecture
    model = load_model_for_client(partition_id)
    
    model_path = get_model_path(partition_id)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.to(device)

    _, valloader = load_private_dataset(partition_id, num_partitions)

    eval_loss, eval_acc = test_fn(model, valloader, device)

    metrics = {"eval_loss": eval_loss, "eval_acc": eval_acc, "num-examples": len(valloader.dataset)}
    return Message(content=RecordDict({"metrics": MetricRecord(metrics)}), reply_to=msg)