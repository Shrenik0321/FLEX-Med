import torch
import os
import time
import numpy as np
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
import warnings
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
    apply_freeze_strategy, load_model_weights,
    get_display_id, save_model, load_model_for_client,
    compute_per_class_accuracy, freeze_backbone, unfreeze_fraction_of_last_block
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
    
    model, model_path, client_config = load_model_for_client(partition_id) # load model with the partition assigned to it
    display_id = get_display_id(partition_id, client_config)

    print(f"[{display_id} (Partition {partition_id})] ROUND {server_round} - Training Phase")
    print(f"{'='*60}")
    model_type = client_config['model_type']

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if os.path.exists(model_path):
        model = load_model_weights(model, model_path, device)
        print(f"[{display_id}] Loaded model weights")

    model.to(device)

    # <-------------------------------------- FREEZE/UNFREEZE STRATEGY -------------------------------------->
    base_lr = msg.content["config"]["lr"]
    progress = server_round / total_rounds

    # Applying the 2-stage unfreeze strategy
    model = apply_freeze_strategy(model, model_type, server_round, total_rounds)

    # Discriminative learning rates — Phase transitions
    from flex_med.utils.helpers import PHASE2_START_ROUND

    phase = "INITIALIZING"

    if server_round < PHASE2_START_ROUND:
        # Phase 1: Classifier Head Only (R1-4)
        classifier_lr = base_lr
        backbone_lr = 0.0
        phase = "PHASE 1 - Classifier Head Only (R1-4)"
    else:
        # Phase 2: 50% Backbone Refinement (R5+)
        classifier_lr = base_lr * 0.6
        backbone_lr = base_lr * 0.03
        phase = "PHASE 2 - 50% Backbone Refinement (R5+)"

    print(f"[{display_id}] [Strategy] Round {server_round}/{total_rounds}: {phase}")
    print(f"[{display_id}] Round {server_round}/{total_rounds} ({progress:.0%})")
    print(f"[{display_id}] Classifier LR: {classifier_lr:.6f} | Backbone LR: {backbone_lr:.6f} (dynamic)")

    # <-------------------------------------- KNOWLEDGE DISTILLATION -------------------------------------->
    distill_loss = 0.0
    if "arrays" in msg.content and msg.content["arrays"]:
        try:
            consensus_logits = msg.content["arrays"]["0"].numpy()
            if consensus_logits.shape[0] > 0:
                print(f"[{display_id}] Phase 1: Knowledge Distillation")

                #  Load public dataset for logit generation
                public_loader = load_public_dataset(batch_size=context.run_config["batch-size"])

                #  Perform knowledge distillation
                distill_loss = distill_knowledge(
                    model=model, public_loader=public_loader, consensus_logits=consensus_logits,
                    device=device, epochs=context.run_config["distill-epochs"], 
                    classifier_lr=classifier_lr, backbone_lr=backbone_lr, temperature=context.run_config["temperature"],
                    current_round=server_round, total_rounds=total_rounds, adaptive=True
                )
                print(f"[{display_id}] Distillation Loss: {distill_loss:.4f}")
        except Exception as e:
            print(f"[Client {partition_id}] Distillation failed: {e}")

    # <-------------------------------------- PRIVATE TRAINING -------------------------------------->
    print(f"[{display_id}] Private Training")

    trainloader, valloader, _ = load_private_dataset(partition_id, num_partitions, batch_size=context.run_config["batch-size"])
    if trainloader is None:
        raise ValueError(f"[{display_id}] No training data available")

    dataset_len = len(trainloader.dataset)

    start_time = time.time()
    train_loss, val_loss, train_accuracy, val_accuracy = train_fn(
        model=model, trainloader=trainloader, epochs=context.run_config["local-epochs"],
        classifier_lr=classifier_lr, backbone_lr=backbone_lr, device=device,
        valloader=valloader, server_round=server_round, total_rounds=total_rounds
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
    public_loader = load_public_dataset(batch_size=context.run_config["batch-size"])
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