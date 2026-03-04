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
import random
from flex_med.utils.config import CLIENT_INFO_FILE_PATH, DIRICHLET_SEED
from flex_med.task import (
    load_private_dataset,
    load_public_dataset, get_public_logits, distill_knowledge,
    train as train_fn, NUM_CLASSES
)
from flex_med.utils.helpers import (
    apply_freeze_strategy,
    get_model_by_type, get_client_by_partition_id, load_model_weights,
    get_display_id, save_model, load_model_for_client,
    compute_dynamic_focal_alpha, compute_per_class_accuracy
)

app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    # <-------------------------------------- LOAD CONFIGURATIONS -------------------------------------->
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    total_rounds = context.run_config["num-server-rounds"]

    try:
        server_round = int(msg.metadata.group_id)
    except (ValueError, TypeError, AttributeError):
        server_round = 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Deterministic seeding for reproducibility
    random.seed(DIRICHLET_SEED)
    np.random.seed(DIRICHLET_SEED)
    torch.manual_seed(DIRICHLET_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(DIRICHLET_SEED)

    print(f"\n{'='*60}")
    
    # Load model and config early for logging
    model, model_path, client_config = load_model_for_client(partition_id)
    display_id = get_display_id(partition_id, client_config)

    print(f"[{display_id} (Partition {partition_id})] ROUND {server_round} - Training Phase")
    print(f"{'='*60}")
    model_type = client_config['model_type']

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if os.path.exists(model_path):
        model = load_model_weights(model, model_path, device)
        print(f"[{display_id}] Loaded model weights")

    model.to(device)

    # <-------------------------------------- APPLY FREEZE STRATEGY -------------------------------------->
    if model_type:
        model, _ = apply_freeze_strategy(model, model_type, server_round, total_rounds)
        freeze_backbone = (server_round <= total_rounds // 2)
        phase = "Phase 1 - Backbone FROZEN" if freeze_backbone else "Phase 2 - Fine-tuning"
        print(f"[{display_id}] [Freeze] Round {server_round}/{total_rounds}: {phase}")

    # <-------------------------------------- KNOWLEDGE DISTILLATION -------------------------------------->
    distill_loss = 0.0
    if "arrays" in msg.content and msg.content["arrays"]:
        try:
            consensus_logits = msg.content["arrays"]["0"].numpy()
            if consensus_logits.shape[0] > 0:
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

    # <-------------------------------------- PRIVATE TRAINING -------------------------------------->
    print(f"[{display_id}] Phase 2: Private Training")

    lr_decay_factor = context.run_config.get("lr-decay", 0.90)
    base_lr = msg.content["config"]["lr"]
    decayed_lr = base_lr * (lr_decay_factor ** (server_round - 1))

    trainloader, valloader, class_counts = load_private_dataset(partition_id, num_partitions, batch_size=context.run_config["batch-size"])
    if trainloader is None:
        raise ValueError(f"[{display_id}] No training data available")

    client_name = client_config.get('client_name', f"Client_{partition_id}")

    n_leukemia = class_counts.get(0, 0)
    n_healthy = class_counts.get(1, 0)
    focal_alpha = compute_dynamic_focal_alpha(class_counts)
    print(f"[{display_id} | {client_name}] Focal Alpha: {focal_alpha:.4f} (L:{n_leukemia}, H:{n_healthy})")

    dataset_len = len(trainloader.dataset)

    start_time = time.time()
    # Local training strategy
    train_loss, val_loss, train_accuracy, val_accuracy = train_fn(
        model=model, trainloader=trainloader, epochs=context.run_config["local-epochs"],
        lr=decayed_lr, device=device,
        valloader=valloader, server_round=server_round, total_rounds=total_rounds,
        focal_alpha=focal_alpha
    )
    training_time = time.time() - start_time

    print(f"[{display_id}] Train Loss: {train_loss:.4f}, Acc: {train_accuracy:.2%}, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2%} ({dataset_len} samples, {training_time:.1f}s)")

    # <-------------------------------------- COMPUTE BALANCED ACCURACY -------------------------------------->
    balanced_accuracy = compute_per_class_accuracy(model, valloader, device)
    print(f"[{display_id}] Balanced Accuracy: {balanced_accuracy:.2%}")

    # <-------------------------------------- SAVE MODEL -------------------------------------->
    try:
        save_model(model, model_path, model_type)
        print(f"[{display_id}] Model saved")
    except Exception as e:
        print(f"[{display_id}] Save failed: {e}")

    # <-------------------------------------- GENERATE PUBLIC LOGITS -------------------------------------->
    public_loader = load_public_dataset(batch_size=context.run_config["batch-size"], round_num=server_round, total_rounds=total_rounds)
    public_logits = get_public_logits(model, public_loader, device)

    print(f"[{display_id}] Round {server_round} Complete\n")

    # <-------------------------------------- RETURN METRICS -------------------------------------->
    return Message(
        content=RecordDict({
            "arrays": ArrayRecord([public_logits]),
            "metrics": MetricRecord({
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "balanced_accuracy": balanced_accuracy,
                "distill_loss": distill_loss,
                "num-examples": dataset_len,
                "training_time": training_time,
                "client_id": partition_id,
            })
        }),
        reply_to=msg
    )