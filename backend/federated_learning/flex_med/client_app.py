import torch
import os
import time
import numpy as np
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flex_med.utils.config import CLIENT_INFO_FILE_PATH
from flex_med.task import (
    get_model_by_type,
    get_client_by_partition_id,
    load_private_dataset,
    load_public_dataset,
    get_public_logits,
    distill_knowledge,
    train as train_fn,
    test as test_fn,
    load_existing_model,
    NUM_CLASSES
)

app = ClientApp()

# <------------------------------------------ MODEL CHECKPOINT UTILITIES ------------------------------------------>

# Saves model with metadata for tracking training progress and configuration
# Args: model - PyTorch model to save
#       model_path - File path for saving checkpoint
#       model_type - Architecture identifier (e.g., 'resnet18', 'mobilenet_v2')
#       client_id - Numeric client identifier
#       round_num - Optional FL round number for tracking training progress
# Returns: None (saves checkpoint to disk)
def save_model_checkpoint(model, model_path, model_type, client_id, round_num=None):
    checkpoint = {
        'model_type': model_type,
        'num_classes': NUM_CLASSES,
        'state_dict': model.state_dict(),
        'client_id': client_id,
    }

    if round_num is not None:
        checkpoint['round'] = round_num

    torch.save(checkpoint, model_path)

# Initializes model architecture and loads existing weights from data.json configuration
# Args: partition_id - Client ID used to fetch configuration from data.json
#       config_path - Path to client configuration JSON file
# Returns: Tuple of (model, model_path, client_config) where model is initialized architecture
def load_model_for_client(partition_id: int, config_path: str = CLIENT_INFO_FILE_PATH):
    client_config = get_client_by_partition_id(partition_id, config_path)

    client_name = client_config['client_name']
    model_type = client_config['model_type']
    model_path = client_config['model_path']

    print(f"[Client {partition_id}] {client_name} | {model_type}")

    model = get_model_by_type(model_type)

    return model, model_path, client_config

# <------------------------------------------ FEDERATED TRAINING LOGIC ------------------------------------------>

@app.train()
def train(msg: Message, context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    total_rounds = context.run_config["num-server-rounds"]

    # Identify the current FL round from server message
    try:
        server_round = int(msg.metadata.group_id)
    except:
        server_round = 1

    # Auto-detect GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"[ROUND {server_round}] Client {partition_id} - Training Phase")
    print(f"{'='*60}")

    # Load client model architecture and configuration from data.json
    model, model_path, client_config = load_model_for_client(partition_id)

    # Ensure model storage directory exists for checkpoint saving
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    # Resume from existing checkpoint if available
    if os.path.exists(model_path):
        try:
            model, metadata = load_existing_model(model, model_path, device)
            round_info = f"(from Round {metadata['round']})" if metadata.get('round') else ""
            print(f"[Client {partition_id}] Loaded model {round_info}")
        except Exception as e:
            print(f"[Client {partition_id}] Starting fresh model")

    model.to(device)

    # Phase 1: Adaptive Knowledge Distillation from Server Consensus
    distill_loss = 0.0
    has_local_data = client_config['has_local_data']

    if "arrays" in msg.content and msg.content["arrays"]:
        try:
            consensus_logits = msg.content["arrays"]["0"].numpy()

            # Skip distillation in Round 1 (no consensus yet)
            if np.any(consensus_logits != 0):
                # Free riders need more distillation epochs
                distill_epochs = 2 if has_local_data else 8
                distill_lr = 0.001 if has_local_data else 0.002
                temperature = 3.0

                print(f"[Client {partition_id}] Phase 1: Adaptive Knowledge Distillation ({distill_epochs} epochs)")

                public_loader = load_public_dataset(
                    batch_size=32, 
                    round_num=server_round, 
                    total_rounds=total_rounds
                )

                # KEY CHANGE: Pass current_round and total_rounds for adaptive weighting
                distill_loss = distill_knowledge(
                    model=model,
                    public_loader=public_loader,
                    consensus_logits=consensus_logits,
                    device=device,
                    epochs=distill_epochs,
                    lr=distill_lr,
                    temperature=temperature,
                    current_round=server_round,  # NEW
                    total_rounds=total_rounds,   # NEW
                    adaptive=True                # NEW: Enable adaptive decay
                )
                print(f"[Client {partition_id}] ✓ Distillation Loss: {distill_loss:.4f}")
        except Exception as e:
            print(f"[Client {partition_id}] ✗ Distillation failed: {e}")

    # Phase 2: Private Training on Client's Local Dataset
    print(f"[Client {partition_id}] Phase 2: Private Training")

    train_loss = 0.0
    training_time = 0.0
    dataset_len = 0

    model_type = client_config['model_type']

    # Learning Rate Decay
    lr_decay_factor = context.run_config.get("lr-decay", 0.90)
    base_lr = msg.content["config"]["lr"]
    decayed_lr = base_lr * (lr_decay_factor ** (server_round - 1))
    print(f"[Client {partition_id}] Learning Rate: {decayed_lr:.6f}")

    trainloader, _ = load_private_dataset(partition_id, num_partitions, batch_size=32)

    if trainloader is not None:
        dataset_len = len(trainloader.dataset)

        start_time = time.time()
        train_loss = train_fn(
            model=model,
            trainloader=trainloader,
            epochs=context.run_config["local-epochs"],
            lr=decayed_lr,
            device=device,
            model_type=model_type
        )
        training_time = time.time() - start_time

        print(f"[Client {partition_id}] ✓ Training Loss: {train_loss:.4f} ({dataset_len} samples, {training_time:.1f}s)")
    else:
        # Free riders: Public dataset training
        print(f"[Client {partition_id}] Phase 2b: Public Dataset Training (Free Rider)")

        public_supervised_loader = load_public_dataset(
            batch_size=32,
            round_num=server_round, 
            total_rounds=total_rounds
        )
        dataset_len = len(public_supervised_loader.dataset)

        start_time = time.time()
        train_loss = train_fn(
            model=model,
            trainloader=public_supervised_loader,
            epochs=3,
            lr=0.0005,
            device=device,
            model_type=model_type
        )
        training_time = time.time() - start_time

        print(f"[Client {partition_id}] ✓ Public Training Loss: {train_loss:.4f}")

    # Save updated model checkpoint
    try:
        save_model_checkpoint(
            model=model,
            model_path=model_path,
            model_type=client_config['model_type'],
            client_id=partition_id,
            round_num=server_round
        )
    except Exception as e:
        print(f"[Client {partition_id}] ✗ Save failed: {e}")

    # Generate public logits for aggregation
    print(f"[Client {partition_id}] Generating public logits for aggregation...")
    public_loader = load_public_dataset(
        batch_size=32,
        round_num=server_round, 
        total_rounds=total_rounds
    )
    public_logits = get_public_logits(model, public_loader, device)

    print(f"[Client {partition_id}] Round {server_round} Complete\n")

    # Package results
    logits_record = ArrayRecord([public_logits])

    metrics = {
        "train_loss": train_loss,
        "distill_loss": distill_loss,
        "num-examples": dataset_len,
        "training_time": training_time,
        "client_id": partition_id,
        "has_local_data": int(client_config['has_local_data'])
    }

    return Message(
        content=RecordDict({
            "arrays": logits_record,
            "metrics": MetricRecord(metrics)
        }),
        reply_to=msg
    )

# <------------------------------------------ FEDERATED EVALUATION LOGIC ------------------------------------------>

@app.evaluate()
def evaluate(msg: Message, context: Context):
    # Evaluates model performance on client's test data or public dataset
    # Free riders (no local data) evaluate on public dataset as generalization proxy
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    total_rounds = context.run_config["num-server-rounds"]

    # Identify the current FL round from server message
    try:
        server_round = int(msg.metadata.group_id)
    except:
        server_round = 1

    print(f"\n[Client {partition_id}] Evaluation Phase")

    # Load model architecture and checkpoint
    model, model_path, client_config = load_model_for_client(partition_id)

    if os.path.exists(model_path):
        try:
            model, metadata = load_existing_model(model, model_path, device)
        except Exception as e:
            print(f"[Client {partition_id}] ⚠ Using untrained model")

    model.to(device)

    # Load test dataset (private if available, otherwise public)
    _, valloader = load_private_dataset(partition_id, num_partitions, batch_size=32)

    if valloader is None:
        print(f"[Client {partition_id}] Using public dataset for evaluation")
        valloader = load_public_dataset(batch_size=32,round_num=server_round, total_rounds=total_rounds)

    # Run evaluation
    eval_loss, eval_acc = test_fn(model, valloader, device)

    print(f"[Client {partition_id}] ✓ Loss: {eval_loss:.4f} | Accuracy: {eval_acc:.4f} ({eval_acc*100:.1f}%)")

    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
        "client_id": partition_id,
        "has_local_data": int(client_config['has_local_data'])
    }

    return Message(
        content=RecordDict({"metrics": MetricRecord(metrics)}),
        reply_to=msg
    )