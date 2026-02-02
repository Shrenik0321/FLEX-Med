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
    get_initial_dropout_rate,
    add_dropout_to_classifier,
    AdaptiveTrainingState,
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
def save_model_checkpoint(model, model_path, model_type, client_id, round_num=None,
                          adaptive_state: dict = None, scheduler_state: dict = None):
    """
    Save model checkpoint with metadata and adaptive training state.

    Args:
        model: PyTorch model to save
        model_path: File path for checkpoint
        model_type: Architecture identifier
        client_id: Numeric client identifier
        round_num: Optional FL round number
        adaptive_state: Adaptive training state dict (dropout, val_loss_history)
        scheduler_state: ReduceLROnPlateau scheduler state dict

    Returns: None (saves checkpoint to disk)
    """
    checkpoint = {
        'model_type': model_type,
        'num_classes': NUM_CLASSES,
        'state_dict': model.state_dict(),
        'client_id': client_id,
    }

    if round_num is not None:
        checkpoint['round'] = round_num

    # NEW: Save adaptive training state for dropout management
    if adaptive_state is not None:
        checkpoint['adaptive_state'] = adaptive_state

    # NEW: Save scheduler state for ReduceLROnPlateau
    if scheduler_state is not None:
        checkpoint['scheduler_state'] = scheduler_state

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

    # ========== ADAPTIVE TRAINING STATE INITIALIZATION ==========
    # Load or create adaptive training state for this client
    adaptive_state = None

    if os.path.exists(model_path):
        try:
            _, metadata = load_existing_model(model, model_path, device)

            # Load existing adaptive state
            if metadata.get('adaptive_state'):
                adaptive_state = AdaptiveTrainingState.from_dict(metadata['adaptive_state'])
                print(f"[Client {partition_id}] Loaded adaptive state: "
                      f"Dropout={adaptive_state.dropout_rate:.2f}, "
                      f"Val history={len(adaptive_state.val_loss_history)} rounds")
            else:
                # First time after upgrade - create new state
                initial_dropout = get_initial_dropout_rate(client_config['model_type'])
                adaptive_state = AdaptiveTrainingState(
                    client_id=partition_id,
                    model_type=client_config['model_type'],
                    initial_dropout=initial_dropout
                )
                print(f"[Client {partition_id}] Created new adaptive state: "
                      f"Dropout={adaptive_state.dropout_rate:.2f}")
        except Exception as e:
            print(f"[Client {partition_id}] Error loading adaptive state: {e}")
            # Create fresh state
            initial_dropout = get_initial_dropout_rate(client_config['model_type'])
            adaptive_state = AdaptiveTrainingState(
                client_id=partition_id,
                model_type=client_config['model_type'],
                initial_dropout=initial_dropout
            )
    else:
        # Fresh model - create new adaptive state
        initial_dropout = get_initial_dropout_rate(client_config['model_type'])
        adaptive_state = AdaptiveTrainingState(
            client_id=partition_id,
            model_type=client_config['model_type'],
            initial_dropout=initial_dropout
        )
        print(f"[Client {partition_id}] Fresh model with adaptive state: "
              f"Dropout={adaptive_state.dropout_rate:.2f}")

    # Ensure model has dropout layers with correct rate
    model = add_dropout_to_classifier(model, client_config['model_type'], adaptive_state.dropout_rate)
    model.to(device)

    # Phase 1: Adaptive Knowledge Distillation from Server Consensus
    distill_loss = 0.0

    if "arrays" in msg.content and msg.content["arrays"]:
        try:
            consensus_logits = msg.content["arrays"]["0"].numpy()

            # Skip distillation in Round 1 (no consensus yet)
            if np.any(consensus_logits != 0):
                # All clients use standardized distillation parameters
                distill_epochs = 2
                distill_lr = 0.001
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

    # Load training and validation data (keep both loaders for adaptive training)
    trainloader, valloader = load_private_dataset(partition_id, num_partitions, batch_size=32)

    if trainloader is None:
        raise ValueError(
            f"[Client {partition_id}] No training data available. "
            f"All clients must have a data partition from LOCAL_TRAIN_DATASET_PATH"
        )

    dataset_len = len(trainloader.dataset)

    start_time = time.time()
    # Call training with validation loader and adaptive state
    train_loss, val_loss, train_accuracy, scheduler_state = train_fn(
        model=model,
        trainloader=trainloader,
        epochs=context.run_config["local-epochs"],
        lr=decayed_lr,
        device=device,
        model_type=model_type,
        valloader=valloader,  # NEW: Pass validation loader
        adaptive_state=adaptive_state  # NEW: Pass adaptive state
    )
    training_time = time.time() - start_time

    print(f"[Client {partition_id}] ✓ Training Loss: {train_loss:.4f}, "
          f"Train Accuracy: {train_accuracy:.2%}, "
          f"Validation Loss: {val_loss:.4f} ({dataset_len} samples, {training_time:.1f}s)")

    # Update adaptive state with validation loss
    adaptive_state.add_val_loss(val_loss)

    # Adjust dropout for next round (requires 2+ rounds of history)
    if server_round >= 2:
        new_dropout = adaptive_state.update_dropout(model)

    # Save updated model checkpoint with adaptive state
    try:
        save_model_checkpoint(
            model=model,
            model_path=model_path,
            model_type=client_config['model_type'],
            client_id=partition_id,
            round_num=server_round,
            adaptive_state=adaptive_state.to_dict(),  # NEW
            scheduler_state=scheduler_state  # NEW
        )
        print(f"[Client {partition_id}] ✓ Saved checkpoint with adaptive state "
              f"(dropout={adaptive_state.dropout_rate:.2f})")
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
        "train_accuracy": train_accuracy,  # Training accuracy for combined charts
        "val_loss": val_loss,  # Added for per-round tracking in database
        "distill_loss": distill_loss,
        "num-examples": dataset_len,
        "training_time": training_time,
        "client_id": partition_id
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
    # Evaluates model performance on client's validation data
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

    # Load test dataset
    _, valloader = load_private_dataset(partition_id, num_partitions, batch_size=32)

    if valloader is None:
        raise ValueError(
            f"[Client {partition_id}] No validation data available. "
            f"All clients must have a data partition."
        )

    # Run evaluation
    eval_loss, eval_acc = test_fn(model, valloader, device)

    print(f"[Client {partition_id}] ✓ Loss: {eval_loss:.4f} | Accuracy: {eval_acc:.4f} ({eval_acc*100:.1f}%)")

    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
        "client_id": partition_id
    }

    return Message(
        content=RecordDict({"metrics": MetricRecord(metrics)}),
        reply_to=msg
    )