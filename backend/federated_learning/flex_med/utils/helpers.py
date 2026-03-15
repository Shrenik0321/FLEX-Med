import os
import json
from datetime import datetime
from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights
from pathlib import Path
from flex_med.utils.config import (
    LOCAL_TRAIN_DATASET_PATH, DIRICHLET_ALPHA, DIRICHLET_SEED, CLIENT_INFO_FILE_PATH, NUM_CLASSES)

# <----------------------------- CONSTANTS & GLOBAL VARIABLES ----------------------------->
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

def get_supabase_client() -> Optional['Client']:
    """Initialize Supabase client from environment variables."""
    if not SUPABASE_AVAILABLE:
        return None
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')
    if not supabase_url or not supabase_key:
        return None
    try:
        return create_client(supabase_url, supabase_key)
    except Exception:
        return None

def get_simulation_id() -> Optional[int]:
    """Get simulation ID from environment or config file."""
    env_sim_id = os.getenv('FLEX_MED_SIMULATION_ID')
    if env_sim_id:
        try:
            return int(env_sim_id)
        except ValueError:
            pass

    try:
        env_config_path = os.getenv('FLEX_MED_CONFIG_FILE')
        if env_config_path and os.path.exists(env_config_path):
            with open(env_config_path, 'r') as f:
                config_data = json.load(f)
                if isinstance(config_data, dict) and "simulation_id" in config_data:
                    return config_data["simulation_id"]
    except Exception:
        pass
    return None

SUPABASE_CLIENT = get_supabase_client()
SIMULATION_ID = get_simulation_id()

# <----------------------------- PATH UTILITIES ----------------------------->
def sanitize_client_paths(clients: List[Dict]) -> List[Dict]:
    """Sanitize model paths to work in current environment."""
    backend_root = Path(__file__).parent.parent.parent
    for client in clients:
        if 'model_path' in client and client['model_path']:
            orig_path = client['model_path']
            if 'models' in orig_path:
                relative_part = orig_path.split('models')[-1].lstrip('/')
                client['model_path'] = str(backend_root / "models" / relative_part)
    return clients

# <----------------------------- CLIENT CONFIGURATION ----------------------------->
# Load client configuration from database.
def load_client_config(config_path: str = CLIENT_INFO_FILE_PATH) -> List[Dict]:
    if SUPABASE_CLIENT is not None and SIMULATION_ID is not None:
        try:
            # Loads the clients only relevant to this FL simulation
            response = SUPABASE_CLIENT.from_('client_simulation_metrics') \
                .select('client_id, clients(*)') \
                .eq('simulation_id', SIMULATION_ID) \
                .order('client_id') \
                .execute()
            if response.data:
                clients = [r.get('clients') for r in response.data if r.get('clients')]
                if clients:
                    return sanitize_client_paths(clients)
        except Exception:
            pass

    env_config_path = os.getenv('FLEX_MED_CONFIG_FILE')
    if env_config_path:
        config_path = env_config_path

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, 'r') as f:
        config_data = json.load(f)

    clients = config_data.get("clients", config_data) if isinstance(config_data, dict) else config_data
    return sanitize_client_paths(clients)

# Retrieve individual client config by partition ID.
def get_client_by_partition_id(partition_id: int, config_path: str = CLIENT_INFO_FILE_PATH) -> Dict:
    from flwr.common import log
    from logging import WARNING

    clients = load_client_config(config_path)
    if partition_id >= len(clients):
        # Fallback: Cycle through available clients using modulo
        effective_id = partition_id % len(clients)
        log(WARNING, f"[WARNING] Partition ID {partition_id} exceeds clients ({len(clients)}). "
              f"Using client {effective_id} (Modulo fallback).")
        return clients[effective_id]
    return clients[partition_id]

def get_display_id(partition_id: int, client_config: Dict) -> str:
    """Get display ID for logging."""
    return client_config.get('client_name') or f"Client {client_config.get('id', partition_id + 1)}"

# <----------------------------- DATA HETEROGENEITY UTILITIES ----------------------------->
def get_partition_stats(partition_id: int, num_partitions: int, partitioner_cache: dict) -> Dict:
    """Get data heterogeneity statistics for a client's partition."""
    from flwr.common import log
    from logging import INFO, WARNING
    
    cache_key = (LOCAL_TRAIN_DATASET_PATH, num_partitions, DIRICHLET_ALPHA, DIRICHLET_SEED)

    # Try to find the partitioner in cache
    partitioner = None
    if cache_key in partitioner_cache:
        partitioner, _ = partitioner_cache[cache_key]
    else:
        # Try to find any matching cache entry (path might differ slightly)
        log(INFO, f"[DATA_HET] Exact cache key not found. Looking for alternatives...")
        log(INFO, f"[DATA_HET] Expected key: {cache_key}")
        log(INFO, f"[DATA_HET] Available keys: {list(partitioner_cache.keys())}")
        for key in partitioner_cache:
            if key[1] == num_partitions:  # Match by num_partitions
                partitioner, _ = partitioner_cache[key]
                log(INFO, f"[DATA_HET] Using fallback cache key: {key}")
                break
    
    if partitioner is None:
        log(WARNING, f"[DATA_HET] Partitioner not found in cache for {num_partitions} partitions")
        return {"error": "Partitioner not initialized"}

    try:
        partition_dataset = partitioner.load_partition(partition_id)
        partition_labels = partition_dataset["label"]

        total_samples = len(partition_labels)
        leukemia_count = sum(l == 0 for l in partition_labels)
        healthy_count = sum(l == 1 for l in partition_labels)

        leukemia_pct = leukemia_count / total_samples * 100 if total_samples > 0 else 0
        healthy_pct = healthy_count / total_samples * 100 if total_samples > 0 else 0

        max_class, min_class = max(leukemia_count, healthy_count), min(leukemia_count, healthy_count)
        imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')

        log(INFO, f"[DATA_HET] Client {partition_id}: {total_samples} samples (L:{leukemia_count}, H:{healthy_count})")
        
        return {
            "total_samples": total_samples,
            "train_samples": int(total_samples * 0.85),
            "val_samples": total_samples - int(total_samples * 0.85),
            "class_distribution": {
                "leukemia": leukemia_count, "healthy": healthy_count,
                "leukemia_pct": round(leukemia_pct, 1), "healthy_pct": round(healthy_pct, 1)
            },
            "imbalance_ratio": round(imbalance_ratio, 2),
            "partition_id": partition_id
        }
    except Exception as e:
        log(WARNING, f"[DATA_HET] Error loading partition {partition_id}: {e}")
        return {"error": str(e)}

def save_all_clients_data_heterogeneity(client_configs: List[Dict], partitioner_cache: dict):
    """
    Save data heterogeneity statistics for all clients to the database.
    
    This function should be called at FL start to ensure data partitioning
    information is captured for visualization in the frontend.
    """
    from flwr.common import log
    from logging import INFO, WARNING
    
    if SUPABASE_CLIENT is None or SIMULATION_ID is None:
        log(WARNING, "[DATA_HET] Cannot save data heterogeneity: Supabase not initialized")
        return
    
    log(INFO, f"[DATA_HET] Saving data heterogeneity for {len(client_configs)} clients...")
    
    saved_count = 0
    for client_idx, config in enumerate(client_configs):
        db_client_id = config.get('id')
        if db_client_id is None:
            log(WARNING, f"[DATA_HET] Client {client_idx} has no database ID")
            continue
        
        try:
            # Get partition stats for this client
            stats = get_partition_stats(client_idx, len(client_configs), partitioner_cache)
            
            if "error" in stats:
                log(WARNING, f"[DATA_HET] Failed to get stats for client {client_idx}: {stats['error']}")
                continue
            
            # Fetch existing metrics
            response = SUPABASE_CLIENT.from_('client_simulation_metrics').select('metrics') \
                .eq('simulation_id', SIMULATION_ID).eq('client_id', db_client_id).execute()
            
            if not response.data:
                log(WARNING, f"[DATA_HET] No metrics record found for client {db_client_id}")
                continue
            
            existing_metrics = response.data[0].get('metrics', {})
            
            # Only update if data_heterogeneity not already present
            if 'data_heterogeneity' not in existing_metrics:
                existing_metrics['data_heterogeneity'] = stats
                
                SUPABASE_CLIENT.from_('client_simulation_metrics').update({'metrics': existing_metrics}) \
                    .eq('simulation_id', SIMULATION_ID).eq('client_id', db_client_id).execute()
                
                saved_count += 1
                log(INFO, f"[DATA_HET] Saved for client {client_idx}: {stats['total_samples']} samples "
                          f"(L:{stats['class_distribution']['leukemia']}, H:{stats['class_distribution']['healthy']})")
            else:
                log(INFO, f"[DATA_HET] Client {client_idx} already has data_heterogeneity")
                
        except Exception as e:
            log(WARNING, f"[DATA_HET] Error saving data heterogeneity for client {client_idx}: {e}")
    
    log(INFO, f"[DATA_HET] Saved data heterogeneity for {saved_count}/{len(client_configs)} clients")

def print_client_data_distribution_summary(client_configs: List[Dict], partitioner_cache: dict):
    """Print data distribution summary across all clients."""
    from flwr.common import log
    from logging import INFO

    log(INFO, "")
    log(INFO, "=" * 90)
    log(INFO, "CLIENT DATA DISTRIBUTION SUMMARY")
    log(INFO, "=" * 90)

    cache_key = (LOCAL_TRAIN_DATASET_PATH, len(client_configs), DIRICHLET_ALPHA, DIRICHLET_SEED)

    if cache_key in partitioner_cache:
        partitioner, _ = partitioner_cache[cache_key]
        total_all, total_healthy = 0, 0

        for i, config in enumerate(client_configs):
            client_name = config.get('client_name', f'Client {i}')
            partition_labels = partitioner.load_partition(i)["label"]
            all_count = sum(l == 0 for l in partition_labels)
            healthy_count = sum(l == 1 for l in partition_labels)
            total_samples = len(partition_labels)
            all_pct = all_count / total_samples * 100 if total_samples > 0 else 0
            healthy_pct = healthy_count / total_samples * 100 if total_samples > 0 else 0
            ratio = all_count / healthy_count if healthy_count > 0 else float('inf')

            total_all += all_count
            total_healthy += healthy_count

            log(INFO, f"Client {i} ({client_name}): Total={total_samples:4d} | "
                      f"ALL={all_count:4d} ({all_pct:5.1f}%) | Healthy={healthy_count:4d} ({healthy_pct:5.1f}%) | "
                      f"Ratio={ratio:5.1f}:1")

        overall_total = total_all + total_healthy
        overall_ratio = total_all / total_healthy if total_healthy > 0 else float('inf')
        log(INFO, "-" * 90)
        log(INFO, f"OVERALL: Total={overall_total:4d} | Ratio={overall_ratio:5.1f}:1 | "
                  f"Dirichlet alpha={DIRICHLET_ALPHA}, seed={DIRICHLET_SEED}")
        log(INFO, "=" * 90)
    else:
        log(INFO, "Partitioner not initialized yet")

# <----------------------------- MODEL UTILITIES ----------------------------->
def get_model_by_type(model_type: str, use_pretrained: bool = True, dropout_rate: float = None):
    """Create a model instance by type with dropout-enhanced classifier."""
    model_type = model_type.lower()
    if dropout_rate is None:
        dropout_rate = get_initial_dropout_rate(model_type)

    model_map = {
        'efficientnet_b0': (models.efficientnet_b0, EfficientNet_B0_Weights.DEFAULT),
        'efficientnet_b1': (models.efficientnet_b1, EfficientNet_B1_Weights.DEFAULT),
        'efficientnet_b2': (models.efficientnet_b2, EfficientNet_B2_Weights.DEFAULT),
    }

    if model_type not in model_map:
        raise ValueError(f"Unsupported model type: {model_type}. Supported: {list(model_map.keys())}")

    model_fn, default_weights = model_map[model_type]
    weights = default_weights if use_pretrained else None

    model = model_fn(weights=weights)
    return add_dropout_to_classifier(model, model_type, dropout_rate)

def get_initial_dropout_rate(model_type: str) -> float:
    """Get default dropout rate for model architecture."""
    rates = {'efficientnet_b0': 0.40, 'efficientnet_b1': 0.40, 'efficientnet_b2': 0.35}
    return rates.get(model_type.lower(), 0.3)

def add_dropout_to_classifier(model, model_type: str, dropout_rate: float = 0.3):
    """Add/update dropout layer before the final classifier head (idempotent)."""
    model_type = model_type.lower()

    match model_type:
        case 'efficientnet_b0' | 'efficientnet_b1' | 'efficientnet_b2':
            if isinstance(model.classifier[1], nn.Sequential) and isinstance(model.classifier[1][0], nn.Dropout):
                model.classifier[1][0].p = dropout_rate
            else:
                in_features = model.classifier[1].in_features
                model.classifier[1] = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(in_features, NUM_CLASSES))

    return model

def load_model_weights(model, model_path, device):
    """Load model weights from file (handles legacy and new formats)."""
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return model

def save_model(model: torch.nn.Module, model_path: str, model_type: str):
    """Save model weights with minimal metadata."""
    torch.save({
        'model_type': model_type,
        'num_classes': NUM_CLASSES,
        'state_dict': model.state_dict(),
    }, model_path)

def load_model_for_client(partition_id: int, config_path: str = CLIENT_INFO_FILE_PATH):
    """Load client config and create model architecture."""
    client_config = get_client_by_partition_id(partition_id, config_path)
    model = get_model_by_type(client_config['model_type'])
    return model, client_config['model_path'], client_config

# <----------------------------- MODEL EVALUATION & TRAINING UTILITIES ----------------------------->
def compute_per_class_accuracy(model, valloader, device):
    """
    Compute per-class accuracies for balanced accuracy metric.
    """
    model.eval()
    class_correct = {0: 0, 1: 0}
    class_total = {0: 0, 1: 0}

    with torch.no_grad():
        for images, labels in valloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1

    # Handle edge case: if one class has zero samples in validation set
    if class_total[0] == 0 or class_total[1] == 0:
        # Fall back to total accuracy
        total_correct = class_correct[0] + class_correct[1]
        total_samples = class_total[0] + class_total[1]
        return total_correct / total_samples if total_samples > 0 else 0.5

    # Compute per-class accuracies
    leukemia_acc = class_correct[0] / class_total[0]
    healthy_acc = class_correct[1] / class_total[1]

    # Return only balanced accuracy (privacy-preserving)
    balanced_accuracy = (leukemia_acc + healthy_acc) / 2.0

    return balanced_accuracy

# Freeze all backbone layers, keeping only the classifier trainable.
# Use in early FL rounds to prevent biased gradients from corrupting pretrained features.
def freeze_backbone(model, model_type: str):
    model_type = model_type.lower()

    # First, freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Then unfreeze only the classifier head
    if model_type.startswith('efficientnet'):
        for param in model.classifier.parameters():
            param.requires_grad = True

    return model

def set_bn_eval(m):
    """Freeze running statistics and parameters for ALL types of BatchNorm layers."""
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        m.eval()
        if m.weight is not None:
            m.weight.requires_grad = False
        if m.bias is not None:
            m.bias.requires_grad = False

# Progressive unfreezing: unfreeze fractions of the final backbone block (25% -> 50% -> 100%).
def unfreeze_fraction_of_last_block(model, model_type: str, fraction: float):
    """Unfreeze a specific fraction of the final block of the architecture."""
    model_type = model_type.lower()
    last_block = None

    if model_type.startswith('efficientnet'):
        # Unfreeze the final 1x1 conv + BN layer (Bridge to classifier)
        # This layer is critical for adapting ImageNet features to medical images.
        for param in model.features[-1].parameters():
            param.requires_grad = True
        last_block = model.features[-2]  # Stage 7 (last MBConv stage)

    if last_block is not None:
        children = list(last_block.children())
        num_children = len(children)

        num_unfreeze = int(num_children * fraction)
        if fraction > 0.0 and num_unfreeze == 0:
            num_unfreeze = 1

        unfreeze_start_idx = num_children - num_unfreeze

        for i, child in enumerate(children):
            if i >= unfreeze_start_idx:
                for param in child.parameters(recurse=True):  # recurse=True (default): all params in block
                    param.requires_grad = True

    return model

def apply_freeze_strategy(model, model_type: str, server_round: int, total_rounds: int = 10):
    """2-stage progressive unfreezing strategy.
    Rounds 1-6: Head only (Stabilization)
    Rounds 7-10: 25% last backbone block (Soft Refinement)
    """
    from flwr.common import log
    from logging import INFO

    # Always start by freezing everything and unfreezing the classifier head
    model = freeze_backbone(model, model_type)

    if server_round <= 6:
        # Rounds 1-6: Head Only Warm-up (Stabilization)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(INFO, f"[Freeze] Round {server_round}/{total_rounds}: Stage 1 - Classifier only ({trainable_params:,} params)")
    else:
        # Rounds 7-10: Gradual 25% Backbone Refinement
        model = unfreeze_fraction_of_last_block(model, model_type, fraction=0.25)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(INFO, f"[Freeze] Round {server_round}/{total_rounds}: Stage 2 - 25% backbone block ({trainable_params:,} params)")

    return model

# <----------------------------- DATABASE PERSISTENCE & MONITORING UTILITIES ----------------------------->
def extract_training_metrics_for_persistence(client_metrics_list: List[Dict], aggregation_metadata: Dict, client_configs: List[Dict]) -> Dict:
    """Extract training metrics for persistence."""
    training_metrics = {}
    weight_breakdown = aggregation_metadata.get("weight_breakdown", [])
    for i, metrics in enumerate(client_metrics_list):
        consensus_weight = weight_breakdown[i].get("normalized_weight") if i < len(weight_breakdown) else None
        training_metrics[str(i)] = {
            "distill_loss": metrics.get("distill_loss"), "train_loss": metrics.get("train_loss"),
            "train_accuracy": metrics.get("train_accuracy"), "val_loss": metrics.get("val_loss"),
            "num-examples": metrics.get("num-examples"), "training_time": metrics.get("training_time"),
            "consensus_weight": consensus_weight
        }
    return training_metrics

def save_global_post_fl_metrics(metrics: Dict, client_configs: List[Dict], best_round: int = 0):
    """Save Post-FL metrics to database and compute aggregates."""
    if SUPABASE_CLIENT is None or SIMULATION_ID is None:
        return
    try:
        BASELINE = {'accuracy': 0.5, 'precision': 0.5, 'recall': 0.0, 'f1_score': 0.0, 'loss': 0.693,
                    'specificity': 0.5, 'roc_auc': 0.5, 'healthy_accuracy': 0.5, 'leukemia_accuracy': 0.5, 'class_gap': 1.0}

        for client_id_str, client_metrics in metrics.items():
            client_idx = int(client_id_str)
            if client_idx >= len(client_configs):
                continue
            db_client_id = client_configs[client_idx].get('id')
            if db_client_id is None:
                continue

            response = SUPABASE_CLIENT.from_('client_simulation_metrics').select('metrics') \
                .eq('simulation_id', SIMULATION_ID).eq('client_id', db_client_id).execute()
            if not response.data:
                continue

            existing_metrics = response.data[0].get('metrics', {})
            if 'global' not in existing_metrics:
                existing_metrics['global'] = {}
            existing_metrics['global']['post_fl'] = client_metrics
            existing_metrics['global']['improvement'] = {m: client_metrics[m] - b for m, b in BASELINE.items() if m in client_metrics}

            SUPABASE_CLIENT.from_('client_simulation_metrics').update({
                'metrics': existing_metrics, 'status': 'completed', 'completed_at': datetime.now().isoformat()
            }).eq('simulation_id', SIMULATION_ID).eq('client_id', db_client_id).execute()

        response = SUPABASE_CLIENT.from_('client_simulation_metrics').select('metrics').eq('simulation_id', SIMULATION_ID).execute()
        if response.data:
            aggregate_metrics = compute_aggregate_metrics_local([row['metrics'] for row in response.data])

            # Persist the best model round into aggregate metrics
            if best_round > 0:
                aggregate_metrics['best_model_round'] = best_round

            sim_response = SUPABASE_CLIENT.from_('fl_simulations').select('started_at').eq('id', SIMULATION_ID).execute()
            duration = None
            if sim_response.data and sim_response.data[0].get('started_at'):
                from datetime import timezone
                started_at = datetime.fromisoformat(sim_response.data[0]['started_at'].replace('Z', '+00:00'))
                duration = int((datetime.now(timezone.utc) - started_at).total_seconds())

            update_data = {'aggregate_metrics': aggregate_metrics, 'status': 'completed', 'completed_at': datetime.now().isoformat()}
            if duration:
                update_data['duration'] = duration
            SUPABASE_CLIENT.from_('fl_simulations').update(update_data).eq('id', SIMULATION_ID).execute()
    except Exception:
        pass

def compute_aggregate_metrics_local(all_metrics: List[dict]) -> dict:
    """Compute aggregate metrics from all client metrics."""
    if not all_metrics:
        return {}

    def calc_avg(metric_name: str, stage: str) -> float:
        values = [m['global'][stage].get(metric_name) for m in all_metrics
                  if stage in m.get('global', {}) and m['global'][stage].get(metric_name) is not None]
        return sum(values) / len(values) if values else 0.0

    def calc_std(metric_name: str, stage: str) -> float:
        values = [m['global'][stage].get(metric_name) for m in all_metrics
                  if stage in m.get('global', {}) and m['global'][stage].get(metric_name) is not None]
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5

    rounds_aggregate = []
    num_rounds = max((len(m.get('rounds', [])) for m in all_metrics), default=0)

    for round_num in range(1, num_rounds + 1):
        round_metrics = {'round': round_num, 'avg_accuracy': 0.0, 'avg_loss': 0.0, 'avg_f1': 0.0,
                         'avg_precision': 0.0, 'avg_recall': 0.0, 'avg_train_loss': 0.0, 'avg_val_loss': 0.0,
                         'num_clients_trained': 0, 'timestamp': None}

        val_data, train_data = [], []
        for m in all_metrics:
            for r in m.get('rounds', []):
                if r.get('round') == round_num:
                    if 'validation' in r:
                        val_data.append(r['validation'])
                        if not round_metrics['timestamp']:
                            round_metrics['timestamp'] = r['validation'].get('evaluated_at')
                    if 'training' in r:
                        train_data.append(r['training'])

        if val_data:
            round_metrics.update({
                'avg_accuracy': sum(v.get('accuracy', 0) for v in val_data) / len(val_data),
                'avg_loss': sum(v.get('loss', 0) for v in val_data) / len(val_data),
                'avg_f1': sum(v.get('f1_score', 0) for v in val_data) / len(val_data),
                'avg_precision': sum(v.get('precision', 0) for v in val_data) / len(val_data),
                'avg_recall': sum(v.get('recall', 0) for v in val_data) / len(val_data),
                'num_clients_trained': len(val_data)
            })

        train_losses = [t['train_loss'] for t in train_data if t.get('train_loss') is not None]
        val_losses = [t['val_loss'] for t in train_data if t.get('val_loss') is not None]
        if train_losses:
            round_metrics['avg_train_loss'] = sum(train_losses) / len(train_losses)
        if val_losses:
            round_metrics['avg_val_loss'] = sum(val_losses) / len(val_losses)

        if val_data or train_data:
            rounds_aggregate.append(round_metrics)

    best_round = max(rounds_aggregate, key=lambda r: r['avg_accuracy']) if rounds_aggregate else {}
    round_1 = next((r for r in rounds_aggregate if r['round'] == 1), {})
    final = rounds_aggregate[-1] if rounds_aggregate else {}

    improvement = {}
    if round_1 and final:
        for m in ['avg_accuracy', 'avg_loss', 'avg_f1', 'avg_precision', 'avg_recall']:
            if round_1.get(m) is not None and final.get(m) is not None:
                improvement[m] = round(final[m] - round_1[m], 6)

    return {
        "aggregate": {
            "post_fl": {"avg_accuracy": calc_avg('accuracy', 'post_fl'), "avg_loss": calc_avg('loss', 'post_fl'),
                        "avg_precision": calc_avg('precision', 'post_fl'), "avg_recall": calc_avg('recall', 'post_fl'),
                        "avg_f1": calc_avg('f1_score', 'post_fl'), "std_accuracy": calc_std('accuracy', 'post_fl'),
                        "num_clients": len(all_metrics)},
            "improvement": improvement
        },
        "rounds": rounds_aggregate,
        "best_round": {"round": best_round.get('round'), "avg_accuracy": best_round.get('avg_accuracy')} if best_round else {},
        "total_rounds_completed": len(rounds_aggregate), "total_clients": len(all_metrics)
    }

def save_round_training_metrics(round_num: int, training_metrics: Dict, client_configs: List[Dict], partitioner_cache: dict):
    """Save per-round training metrics to database."""
    if SUPABASE_CLIENT is None or SIMULATION_ID is None:
        return
    try:
        for client_id_str, client_metrics in training_metrics.items():
            client_idx = int(client_id_str)
            if client_idx >= len(client_configs):
                continue
            db_client_id = client_configs[client_idx].get('id')
            if db_client_id is None:
                continue

            response = SUPABASE_CLIENT.from_('client_simulation_metrics').select('metrics') \
                .eq('simulation_id', SIMULATION_ID).eq('client_id', db_client_id).execute()
            if not response.data:
                continue

            existing_metrics = response.data[0].get('metrics', {})
            if 'rounds' not in existing_metrics:
                existing_metrics['rounds'] = []

            round_entry = next((r for r in existing_metrics['rounds'] if r.get('round') == round_num), None)
            if round_entry is None:
                round_entry = {'round': round_num}
                existing_metrics['rounds'].append(round_entry)

            round_entry['training'] = {
                'train_loss': client_metrics.get('train_loss'), 'train_accuracy': client_metrics.get('train_accuracy'),
                'val_loss': client_metrics.get('val_loss'), 'distill_loss': client_metrics.get('distill_loss'),
                'training_time': client_metrics.get('training_time'), 'num_examples': client_metrics.get('num-examples'),
                'consensus_weight': client_metrics.get('consensus_weight')
            }

            if round_num == 1 and 'data_heterogeneity' not in existing_metrics:
                from flwr.common import log
                from logging import INFO, WARNING
                log(INFO, f"[DATA_HET] Attempting to save data_heterogeneity for client_idx={client_idx}")
                try:
                    stats = get_partition_stats(client_idx, len(client_configs), partitioner_cache)
                    if "error" not in stats:
                        existing_metrics['data_heterogeneity'] = stats
                        log(INFO, f"[DATA_HET] Successfully added data_heterogeneity for client {client_idx}")
                    else:
                        log(WARNING, f"[DATA_HET] get_partition_stats returned error: {stats}")
                except Exception as e:
                    log(WARNING, f"[DATA_HET] Exception while getting partition stats: {e}")

            SUPABASE_CLIENT.from_('client_simulation_metrics').update({'metrics': existing_metrics}) \
                .eq('simulation_id', SIMULATION_ID).eq('client_id', db_client_id).execute()
    except Exception as e:
        from flwr.common import log
        from logging import WARNING
        log(WARNING, f"[METRICS] Error saving round training metrics: {e}")

def save_round_validation_metrics(round_num: int, metrics: Dict, client_configs: List[Dict]):
    """Save per-round validation metrics to database."""
    if SUPABASE_CLIENT is None or SIMULATION_ID is None:
        return
    try:
        for client_id_str, client_metrics in metrics.items():
            client_idx = int(client_id_str)
            if client_idx >= len(client_configs):
                continue
            db_client_id = client_configs[client_idx].get('id')
            if db_client_id is None:
                continue

            response = SUPABASE_CLIENT.from_('client_simulation_metrics').select('metrics') \
                .eq('simulation_id', SIMULATION_ID).eq('client_id', db_client_id).execute()
            if not response.data:
                continue

            existing_metrics = response.data[0].get('metrics', {})
            if 'rounds' not in existing_metrics:
                existing_metrics['rounds'] = []

            round_entry = next((r for r in existing_metrics['rounds'] if r.get('round') == round_num), None)
            if round_entry is None:
                round_entry = {'round': round_num}
                existing_metrics['rounds'].append(round_entry)

            round_entry['validation'] = client_metrics

            SUPABASE_CLIENT.from_('client_simulation_metrics').update({'metrics': existing_metrics}) \
                .eq('simulation_id', SIMULATION_ID).eq('client_id', db_client_id).execute()
    except Exception:
        pass

def check_for_degradation_warnings(current_metrics: Dict, round_num: int, client_history: Dict):
    """Check validation metrics and log warnings if degradation detected."""
    from flwr.common import log
    from logging import WARNING

    if round_num < 3:
        return

    for client_id, history in client_history.items():
        if len(history) >= 3:
            last_3 = history[-3:]

            accs = []
            for r in last_3:
                val = r['metrics'].get('accuracy')
                accs.append(float(val) if val is not None and isinstance(val, (int, float)) else 0.0)

            if all(a is not None for a in accs):
                try:
                    if accs[-1] < accs[-2] < accs[-3]:
                        log(WARNING, f"Client {client_id}: Accuracy declining for 2 consecutive rounds")
                except TypeError:
                    pass

            losses = []
            for r in last_3:
                val = r['metrics'].get('loss')
                losses.append(float(val) if val is not None and isinstance(val, (int, float)) else 999.0)

            if all(l is not None for l in losses):
                try:
                    if losses[-1] > losses[-2] > losses[-3]:
                        log(WARNING, f"Client {client_id}: Loss increasing for 2 consecutive rounds (overfitting)")
                except TypeError:
                    pass
