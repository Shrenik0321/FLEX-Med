"""flex-med: Client Application with improved model saving."""

import torch
import os
import time
import numpy as np
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flex_med.task import (
    get_model_by_type,
    get_client_by_partition_id,
    load_private_dataset,
    load_public_dataset,
    get_public_logits,
    distill_knowledge,
    train as train_fn,
    test as test_fn,
    load_model_checkpoint,
    CONFIG_FILE_PATH,
    NUM_CLASSES
)

app = ClientApp()

def save_model_checkpoint(model, model_path, model_type, client_id, round_num=None):
    """
    Save model with metadata for better tracking and debugging.
    
    Args:
        model: The PyTorch model
        model_path: Where to save
        model_type: Architecture type (e.g., 'resnet18')
        client_id: Client ID
        round_num: Optional training round number
    """
    checkpoint = {
        'model_type': model_type,
        'num_classes': NUM_CLASSES,
        'state_dict': model.state_dict(),
        'client_id': client_id,
    }
    
    if round_num is not None:
        checkpoint['round'] = round_num
    
    torch.save(checkpoint, model_path)



def load_model_for_client(partition_id: int, config_path: str = CONFIG_FILE_PATH):
    """
    Load model architecture and state based on client configuration from data.json.
    
    Args:
        partition_id: The partition/client ID
        config_path: Path to the configuration file
    
    Returns:
        Tuple of (model, model_path, client_config)
    """
    # Get client configuration
    client_config = get_client_by_partition_id(partition_id, config_path)
    
    # Extract info
    client_name = client_config['client_name']
    model_type = client_config['model_type']
    model_path = client_config['model_path']
    
    print(f"\n[Client {partition_id}] === {client_name} ===")
    print(f"[Client {partition_id}] Model Type: {model_type}")
    print(f"[Client {partition_id}] Model Path: {model_path}")
    print(f"[Client {partition_id}] Has Local Data: {client_config['has_local_data']}")
    
    # Initialize model based on type
    model = get_model_by_type(model_type)
    
    return model, model_path, client_config

@app.train()
def train(msg: Message, context: Context):
    """
    Train function that uses data.json for client configuration.
    """
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    # Identify the current server round
    try:
        server_round = int(msg.metadata.group_id)
    except:
        server_round = 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"[Client {partition_id}] ROUND {server_round} - TRAINING")
    print(f"{'='*60}")

    # ===== STEP 1: LOAD MODEL FROM CONFIGURATION =====
    model, model_path, client_config = load_model_for_client(partition_id)
    
    # Ensure the model directory exists
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
        print(f"[Client {partition_id}] Created model directory: {model_dir}")
    
    # Load existing model state if it exists
    if os.path.exists(model_path):
        try:
            model, metadata = load_model_checkpoint(model, model_path, device)
            print(f"[Client {partition_id}] ✓ Loaded existing model from: {model_path}")
            if metadata.get('round'):
                print(f"[Client {partition_id}]   (Last trained: Round {metadata['round']})")
        except Exception as e:
            print(f"[Client {partition_id}] ⚠ Error loading model: {e}")
            print(f"[Client {partition_id}] Starting with fresh model")
    else:
        print(f"[Client {partition_id}] Starting with fresh model (will be saved at: {model_path})")

    model.to(device)
    
    # ===== PHASE 1: KNOWLEDGE DISTILLATION =====
    print(f"\n[Client {partition_id}] --- Phase 1: Knowledge Distillation ---")
    distill_loss = 0.0
    
    if "arrays" in msg.content and msg.content["arrays"]:
        try:
            consensus_logits = msg.content["arrays"]["0"].numpy()
            
            # Only distill if consensus is valid (not all zeros from Round 1)
            if np.any(consensus_logits != 0):
                print(f"[Client {partition_id}] Consensus received (shape: {consensus_logits.shape})")
                print(f"[Client {partition_id}] Starting distillation...")
                
                public_loader = load_public_dataset(batch_size=32)
                
                distill_loss = distill_knowledge(
                    model=model,
                    public_loader=public_loader,
                    consensus_logits=consensus_logits,
                    device=device,
                    epochs=1,
                    lr=0.001,
                    temperature=2.0
                )
                print(f"[Client {partition_id}] ✓ Distillation complete. Loss: {distill_loss:.4f}")
            else:
                print(f"[Client {partition_id}] Round 1: Zero consensus. Skipping distillation.")
        except Exception as e:
            print(f"[Client {partition_id}] ✗ Distillation error: {e}")

    # ===== PHASE 2: PRIVATE TRAINING =====
    print(f"\n[Client {partition_id}] --- Phase 2: Private Training ---")
    train_loss = 0.0
    training_time = 0.0
    dataset_len = 0
    
    trainloader, _ = load_private_dataset(partition_id, num_partitions, batch_size=32)

    if trainloader is not None:
        dataset_len = len(trainloader.dataset)
        print(f"[Client {partition_id}] Training on {dataset_len} samples...")
        
        start_time = time.time()
        train_loss = train_fn(
            model=model,
            trainloader=trainloader,
            epochs=context.run_config["local-epochs"],
            lr=msg.content["config"]["lr"],
            device=device
        )
        training_time = time.time() - start_time
        
        print(f"[Client {partition_id}] ✓ Training complete.")
        print(f"[Client {partition_id}]   - Loss: {train_loss:.4f}")
        print(f"[Client {partition_id}]   - Time: {training_time:.2f}s")
    else:
        print(f"[Client {partition_id}] ⚠ Free Rider: No private data. Skipping private training.")
        print(f"[Client {partition_id}]   (Will learn solely from knowledge distillation)")

    # ===== PHASE 3: SAVE STATE & GENERATE PUBLIC LOGITS =====
    print(f"\n[Client {partition_id}] --- Phase 3: Save & Generate Logits ---")
    
    # Save the model checkpoint with metadata
    try:
        save_model_checkpoint(
            model=model,
            model_path=model_path,
            model_type=client_config['model_type'],
            client_id=partition_id,
            round_num=server_round
        )
        print(f"[Client {partition_id}] ✓ Model saved: {model_path}")
    except Exception as e:
        print(f"[Client {partition_id}] ✗ Error saving model: {e}")

    # Generate logits on public dataset
    print(f"[Client {partition_id}] Generating predictions on public dataset...")
    public_loader = load_public_dataset(batch_size=32)
    public_logits = get_public_logits(model, public_loader, device)
    print(f"[Client {partition_id}] ✓ Generated logits (shape: {public_logits.shape})")

    # Pack results
    logits_record = ArrayRecord([public_logits])
    
    # MetricRecord only accepts numeric types (int, float, list[int], list[float])
    metrics = {
        "train_loss": train_loss,
        "distill_loss": distill_loss,
        "num-examples": dataset_len,
        "training_time": training_time,
        "client_id": partition_id,  # Use numeric ID instead of string name
        "has_local_data": int(client_config['has_local_data'])  # Convert bool to int
    }
    
    print(f"[Client {partition_id}] Round {server_round} complete!")
    print(f"  Summary: train_loss={train_loss:.4f}, distill_loss={distill_loss:.4f}")
    print(f"{'='*60}\n")
    
    return Message(
        content=RecordDict({
            "arrays": logits_record, 
            "metrics": MetricRecord(metrics)
        }), 
        reply_to=msg
    )

@app.evaluate()
def evaluate(msg: Message, context: Context):
    """
    Evaluation function that uses data.json for client configuration.
    """
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"\n[Client {partition_id}] === EVALUATION ===")

    # Load model from configuration
    model, model_path, client_config = load_model_for_client(partition_id)
    
    # Load existing model state
    if os.path.exists(model_path):
        try:
            model, metadata = load_model_checkpoint(model, model_path, device)
            print(f"[Client {partition_id}] Loaded model from: {model_path}")
            if metadata.get('round'):
                print(f"[Client {partition_id}]   (Trained through Round {metadata['round']})")
        except Exception as e:
            print(f"[Client {partition_id}] ⚠ Error loading model: {e}")
            print(f"[Client {partition_id}] Using untrained model")
    else:
        print(f"[Client {partition_id}] ⚠ Warning: No saved model found. Using untrained model.")
    
    model.to(device)

    # Load evaluation data
    _, valloader = load_private_dataset(partition_id, num_partitions, batch_size=32)

    # Handle Free Rider evaluation
    if valloader is None:
        print(f"[Client {partition_id}] No private test set.")
        print(f"[Client {partition_id}] Evaluating on Public Dataset (proxy for generalization)...")
        valloader = load_public_dataset(batch_size=32)

    # Run evaluation
    eval_loss, eval_acc = test_fn(model, valloader, device)

    print(f"[Client {partition_id}] Evaluation Results:")
    print(f"  - Loss: {eval_loss:.4f}")
    print(f"  - Accuracy: {eval_acc:.4f} ({eval_acc*100:.2f}%)")
    print(f"  - Samples: {len(valloader.dataset)}")

    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
        "client_id": partition_id,  # Use numeric ID
        "has_local_data": int(client_config['has_local_data'])  # Convert bool to int
    }
    
    return Message(
        content=RecordDict({"metrics": MetricRecord(metrics)}), 
        reply_to=msg
    )