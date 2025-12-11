"""flex-med: Client Application."""

import torch
import os
import time
import numpy as np
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flex_med.task import (
    get_resnet,    
    get_mobilenet,    
    get_densenet,
    load_private_dataset,
    load_public_dataset,
    get_public_logits,
    distill_knowledge,
    train as train_fn,
    test as test_fn
)

app = ClientApp()

def get_model_path(partition_id):
    """Unique path for saving each client's model."""
    return f"/content/drive/MyDrive/College/models/model_client_{partition_id}.pt"

def load_model_for_client(partition_id):
    """
    Selects the architecture based on partition ID.
    0 -> ResNet-18 (ALL-IDB2)
    1 -> MobileNetV2 (CNMC)
    2 -> DenseNet-121 (Free-Rider / Zero-Shot)
    """
    mod = partition_id % 3
    if mod == 0:
        print(f"[Client {partition_id}] Initializing ResNet-18 (ALL-IDB2)")
        return get_resnet()
    elif mod == 1:
        print(f"[Client {partition_id}] Initializing MobileNetV2 (CNMC)")
        return get_mobilenet()
    else:
        print(f"[Client {partition_id}] Initializing DenseNet-121 (Free-Rider)")
        return get_densenet()

@app.train()
def train(msg: Message, context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    # Identify the current server round
    try:
        server_round = int(msg.metadata.group_id)
    except:
        server_round = 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Load Model (Architecture + Previous State)
    model = load_model_for_client(partition_id)
    model_path = get_model_path(partition_id)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[Client {partition_id}] Loaded saved model state.")
    else:
        print(f"[Client {partition_id}] Starting fresh (Round 0 state).")

    model.to(device)
    
    # <--- PHASE 1: KNOWLEDGE DISTILLATION (FedMD Core) --->
    distill_loss = 0.0
    
    # Check if we received consensus logits from the server
    if "arrays" in msg.content and msg.content["arrays"]:
        try:
            # Extract consensus logits
            consensus_logits = msg.content["arrays"]["0"].numpy()
            
            # Only distill if the consensus is valid (not all zeros from Round 0)
            if np.any(consensus_logits != 0):
                print(f"[Client {partition_id}] Consensus received. Distilling...")
                
                # Load public anchor data (images only, labels ignored)
                public_loader = load_public_dataset(batch_size=32)
                
                distill_loss = distill_knowledge(
                    model=model,
                    public_loader=public_loader,
                    consensus_logits=consensus_logits,
                    device=device,
                    epochs=1,       # Distillation epochs
                    lr=0.001,       # Distillation LR
                    temperature=2.0 
                )
                print(f"[Client {partition_id}] Distillation Done. Loss: {distill_loss:.4f}")
            else:
                print(f"[Client {partition_id}] Round 1: No consensus yet. Skipping distillation.")
        except Exception as e:
            print(f"[Client {partition_id}] Distillation Error: {e}")

    # <--- PHASE 2: PRIVATE TRAINING --->
    train_loss = 0.0
    training_time = 0.0
    
    # Attempt to load private data
    # Note: Client 2 (Free Rider) will return None here based on task.py logic
    trainloader, _ = load_private_dataset(partition_id, num_partitions, batch_size=32)

    if trainloader is not None:
        print(f"[Client {partition_id}] Training on private data...")
        dataset_len = len(trainloader.dataset)
        
        start_time = time.time()
        train_loss = train_fn(
            model=model,
            trainloader=trainloader,
            epochs=context.run_config["local-epochs"],
            lr=msg.content["config"]["lr"], # Learning rate from server config
            device=device
        )
        training_time = time.time() - start_time
    else:
        print(f"[Client {partition_id}] Free-Rider (No Private Data). Skipping Private Training.")
        dataset_len = 0 

    # <--- PHASE 3: SAVE STATE & GENERATE PUBLIC LOGITS --->
    
    # Save the model state (Critical: overwrites previous round)
    torch.save(model.state_dict(), model_path)
    print(f"[Client {partition_id}] Saved model state: {model_path}")

    # Generate logits on Public Dataset to send back to Server
    # These logits represent what this client "knows" after Distillation + Private Training
    public_loader = load_public_dataset(batch_size=32)
    public_logits = get_public_logits(model, public_loader, device)

    # Pack results
    logits_record = ArrayRecord([public_logits])
    metrics = {
        "train_loss": train_loss,
        "distill_loss": distill_loss,
        "num-examples": dataset_len,
        "training_time": training_time
    }
    
    return Message(content=RecordDict({"arrays": logits_record, "metrics": MetricRecord(metrics)}), reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. Load Model
    model = load_model_for_client(partition_id)
    model_path = get_model_path(partition_id)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.to(device)

    # 2. Load Evaluation Data
    _, valloader = load_private_dataset(partition_id, num_partitions, batch_size=32)

    # <--- CRITICAL: Handle Free Rider Evaluation --->
    # If Client 2 has no private data, valloader will be None.
    # We evaluate it on the Public Dataset (or a subset) to see if it learned anything.
    if valloader is None:
        print(f"[Client {partition_id}] No private test set. Evaluating on Public Dataset (Proxy).")
        valloader = load_public_dataset(batch_size=32)

    # 3. Run Evaluation
    eval_loss, eval_acc = test_fn(model, valloader, device)

    metrics = {
        "eval_loss": eval_loss, 
        "eval_acc": eval_acc, 
        "num-examples": len(valloader.dataset)
    }
    return Message(content=RecordDict({"metrics": MetricRecord(metrics)}), reply_to=msg)