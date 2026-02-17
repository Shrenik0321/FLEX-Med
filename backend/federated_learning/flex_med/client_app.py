import torch
import os
import time
import numpy as np
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
import warnings

# Suppress the specific Pillow deprecation warning heavily spamming the logs
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torchvision.transforms._functional_pil")
warnings.filterwarnings("ignore", message=".*'mode' parameter is deprecated.*")
from flex_med.utils.config import CLIENT_INFO_FILE_PATH, FOCAL_ALPHA_DEFAULT
from flex_med.task import (
    get_model_by_type, get_client_by_partition_id, load_private_dataset,
    load_public_dataset, get_public_logits, distill_knowledge,
    train as train_fn, test as test_fn, load_existing_model,
    get_initial_dropout_rate, add_dropout_to_classifier,
    AdaptiveTrainingState, NUM_CLASSES
)
from flex_med.utils.helpers import apply_freeze_strategy

app = ClientApp()

def get_display_id(partition_id, client_config):
    """Get display ID (DB ID) for logging."""
    return client_config.get('client_name') or f"Client {client_config.get('id', partition_id + 1)}"

# <----------------------------- MODEL CHECKPOINT UTILITIES ----------------------------->

def save_model_checkpoint(model, model_path, model_type, client_id, round_num=None,
                          adaptive_state: dict = None, scheduler_state: dict = None):
    """Save model checkpoint with metadata and adaptive training state."""
    checkpoint = {
        'model_type': model_type,
        'num_classes': NUM_CLASSES,
        'state_dict': model.state_dict(),
        'client_id': client_id,
    }
    if round_num is not None:
        checkpoint['round'] = round_num
    if adaptive_state is not None:
        checkpoint['adaptive_state'] = adaptive_state
    if scheduler_state is not None:
        checkpoint['scheduler_state'] = scheduler_state
    torch.save(checkpoint, model_path)

def load_model_for_client(partition_id: int, config_path: str = CLIENT_INFO_FILE_PATH):
    """Load client config and create model architecture."""
    client_config = get_client_by_partition_id(partition_id, config_path)
    model = get_model_by_type(client_config['model_type'])
    # print(f"[Client {partition_id}] {client_config['client_name']} | {client_config['model_type']}") # Removed to avoid clutter
    return model, client_config['model_path'], client_config


def _load_or_create_adaptive_state(partition_id, model_path, model_type, device):
    """Load existing adaptive state from checkpoint or create new one."""
    initial_dropout = get_initial_dropout_rate(model_type)

    if not os.path.exists(model_path):
        return AdaptiveTrainingState(
            client_id=partition_id, model_type=model_type, initial_dropout=initial_dropout
        )

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and checkpoint.get('adaptive_state'):
            return AdaptiveTrainingState.from_dict(checkpoint['adaptive_state'])
    except Exception:
        pass

    return AdaptiveTrainingState(
        client_id=partition_id, model_type=model_type, initial_dropout=initial_dropout
    )


# <----------------------------- FEDERATED TRAINING LOGIC ----------------------------->

@app.train()
def train(msg: Message, context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    total_rounds = context.run_config["num-server-rounds"]

    try:
        server_round = int(msg.metadata.group_id)
    except:
        server_round = 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    
    # Load model and config early for logging
    model, model_path, client_config = load_model_for_client(partition_id)
    display_id = get_display_id(partition_id, client_config)

    print(f"[{display_id} (Partition {partition_id})] ROUND {server_round} - Training Phase")
    print(f"{'='*60}")
    model_type = client_config['model_type']

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Resume from checkpoint if available
    if os.path.exists(model_path):
        try:
            model, metadata = load_existing_model(model, model_path, device)
            round_info = f"(from Round {metadata['round']})" if metadata.get('round') else ""
            print(f"[{display_id}] Loaded model {round_info}")
        except Exception:
            print(f"[{display_id}] Starting fresh model")

    model.to(device)

    # Load or create adaptive training state
    adaptive_state = _load_or_create_adaptive_state(partition_id, model_path, model_type, device)
    print(f"[{display_id}] Adaptive state: Dropout={adaptive_state.dropout_rate:.2f}")

    model = add_dropout_to_classifier(model, model_type, adaptive_state.dropout_rate)
    model.to(device)

    # Apply freeze strategy before any training (distillation + private training both respect it)
    # Apply freeze strategy before any training (distillation + private training both respect it)
    if model_type:
        model, _ = apply_freeze_strategy(model, model_type, server_round, total_rounds)
        # Log freeze status with client name
        freeze_backbone = (server_round <= total_rounds // 2) # Rough approximation for logging, logic is in helper
        phase = "Phase 1 - Backbone FROZEN" if freeze_backbone else "Phase 2 - Fine-tuning"
        print(f"[{display_id}] [Freeze] Round {server_round}/{total_rounds}: {phase}")


    # Phase 1: Knowledge Distillation from Server Consensus
    distill_loss = 0.0
    if "arrays" in msg.content and msg.content["arrays"]:
        try:
            consensus_logits = msg.content["arrays"]["0"].numpy()
            if np.any(consensus_logits != 0):
                print(f"[{display_id}] Phase 1: Knowledge Distillation")
                public_loader = load_public_dataset(batch_size=context.run_config["batch-size"], round_num=server_round, total_rounds=total_rounds)
                distill_loss = distill_knowledge(
                    model=model, public_loader=public_loader, consensus_logits=consensus_logits,
                    device=device, epochs=context.run_config["distill-epochs"], 
                    lr=context.run_config["distill-lr"], temperature=context.run_config["temperature"],
                    current_round=server_round, total_rounds=total_rounds, adaptive=True
                )
                print(f"[{display_id}] Distillation Loss: {distill_loss:.4f}")
        except Exception as e:
            print(f"[Client {partition_id}] Distillation failed: {e}")

    # Phase 2: Private Training
    print(f"[{display_id}] Phase 2: Private Training")

    lr_decay_factor = context.run_config.get("lr-decay", 0.90)
    base_lr = msg.content["config"]["lr"]
    decayed_lr = base_lr * (lr_decay_factor ** (server_round - 1))

    trainloader, valloader, class_counts = load_private_dataset(partition_id, num_partitions, batch_size=context.run_config["batch-size"])
    if trainloader is None:
        raise ValueError(f"[{display_id}] No training data available")

    # Get client name for better logging
    client_name = f"Client_{partition_id}"
    try:
        if 'client_config' in locals() and 'client_name' in client_config:
             client_name = client_config['client_name']
    except Exception:
         pass

    # Focal alpha = 0.5 (neutral). Class balance is handled by WeightedRandomSampler.
    # Gamma=2.0 still focuses on hard examples, which is complementary to the sampler.
    n_leukemia = class_counts.get(0, 0)
    n_healthy = class_counts.get(1, 0)
    focal_alpha = FOCAL_ALPHA_DEFAULT

    print(f"[{display_id} | {client_name}] Focal Alpha: {focal_alpha:.4f} (L:{n_leukemia}, H:{n_healthy})")

    dataset_len = len(trainloader.dataset)

    start_time = time.time()
    train_loss, val_loss, train_accuracy, scheduler_state = train_fn(
        model=model, trainloader=trainloader, epochs=context.run_config["local-epochs"],
        lr=decayed_lr, device=device, model_type=model_type,
        valloader=valloader, adaptive_state=adaptive_state,
        server_round=server_round, total_rounds=total_rounds,
        focal_alpha=focal_alpha
    )
    training_time = time.time() - start_time

    print(f"[{display_id}] Train Loss: {train_loss:.4f}, Acc: {train_accuracy:.2%}, "
          f"Val Loss: {val_loss:.4f} ({dataset_len} samples, {training_time:.1f}s)")

    # Update adaptive state
    adaptive_state.add_val_loss(val_loss)
    if server_round >= 2:
        adaptive_state.update_dropout(model)

    # Save checkpoint
    try:
        save_model_checkpoint(
            model=model, model_path=model_path, model_type=model_type,
            client_id=partition_id, round_num=server_round,
            adaptive_state=adaptive_state.to_dict(), scheduler_state=scheduler_state
        )
        print(f"[{display_id}] Saved checkpoint (dropout={adaptive_state.dropout_rate:.2f})")
    except Exception as e:
        print(f"[{display_id}] Save failed: {e}")

    # Generate public logits for aggregation
    public_loader = load_public_dataset(batch_size=context.run_config["batch-size"], round_num=server_round, total_rounds=total_rounds)
    public_logits = get_public_logits(model, public_loader, device)

    print(f"[{display_id}] Round {server_round} Complete\n")

    return Message(
        content=RecordDict({
            "arrays": ArrayRecord([public_logits]),
            "metrics": MetricRecord({
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "distill_loss": distill_loss,
                "num-examples": dataset_len,
                "training_time": training_time,
                "client_id": partition_id
            })
        }),
        reply_to=msg
    )


# <----------------------------- FEDERATED EVALUATION LOGIC ----------------------------->

@app.evaluate()
def evaluate(msg: Message, context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    try:
        server_round = int(msg.metadata.group_id)
    except:
        server_round = 1

    model, model_path, client_config = load_model_for_client(partition_id)
    display_id = get_display_id(partition_id, client_config)

    print(f"\n[{display_id}] Evaluation Phase")

    if os.path.exists(model_path):
        try:
            model, _ = load_existing_model(model, model_path, device)
        except Exception:
            print(f"[{display_id}] Using untrained model")

    model.to(device)

    _, valloader, _ = load_private_dataset(partition_id, num_partitions, batch_size=context.run_config["batch-size"])
    if valloader is None:
        raise ValueError(f"[{display_id}] No validation data available")

    eval_loss, eval_acc = test_fn(model, valloader, device)
    print(f"[{display_id}] Loss: {eval_loss:.4f} | Accuracy: {eval_acc:.2%}")

    return Message(
        content=RecordDict({
            "metrics": MetricRecord({
                "eval_loss": eval_loss,
                "eval_acc": eval_acc,
                "num-examples": len(valloader.dataset),
                "client_id": partition_id
            })
        }),
        reply_to=msg
    )
