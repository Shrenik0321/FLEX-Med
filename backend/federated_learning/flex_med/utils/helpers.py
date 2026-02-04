"""
Helper utilities for FLEX-Med federated learning.
Contains data utilities, evaluation functions, and database persistence helpers.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from torch.utils.data import WeightedRandomSampler

from flex_med.utils.config import (
    LOCAL_TRAIN_DATASET_PATH, DIRICHLET_ALPHA, DIRICHLET_SEED, DIRICHLET_MIN_PARTITION_SIZE
)

# ============================================================================
# SUPABASE CLIENT
# ============================================================================

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

# ============================================================================
# PATH UTILITIES
# ============================================================================

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


# ============================================================================
# DATA UTILITIES
# ============================================================================

def get_weighted_sampler(targets):
    """Create WeightedRandomSampler to handle class imbalance."""
    class_counts = Counter(targets)
    minority_boost = 0.6 # Lenient towards minority classes slightly

    class_weights = {cls: 1.0 / (count ** minority_boost) for cls, count in class_counts.items()}
    sample_weights = [class_weights[t] for t in targets]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


def get_partition_stats(partition_id: int, num_partitions: int, partitioner_cache: dict) -> Dict:
    """Get data heterogeneity statistics for a client's partition."""
    cache_key = (LOCAL_TRAIN_DATASET_PATH, num_partitions, DIRICHLET_ALPHA, DIRICHLET_SEED)

    if cache_key not in partitioner_cache:
        return {"error": "Partitioner not initialized"}

    partitioner, _ = partitioner_cache[cache_key]
    partition_dataset = partitioner.load_partition(partition_id)
    partition_labels = partition_dataset["label"]

    total_samples = len(partition_labels)
    leukemia_count = sum(l == 0 for l in partition_labels)
    healthy_count = sum(l == 1 for l in partition_labels)

    leukemia_pct = leukemia_count / total_samples * 100 if total_samples > 0 else 0
    healthy_pct = healthy_count / total_samples * 100 if total_samples > 0 else 0

    max_class, min_class = max(leukemia_count, healthy_count), min(leukemia_count, healthy_count)
    imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')

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


# ============================================================================
# MODEL UTILITIES - Freeze/Unfreeze for Gradual Training
# ============================================================================

def freeze_backbone(model, model_type: str):
    """
    Freeze all backbone layers, keeping only the classifier trainable.

    Use in early FL rounds (1-2) to prevent biased gradients from corrupting
    pretrained ImageNet features. Only the classifier head will be updated.

    Args:
        model: PyTorch model instance
        model_type: Model architecture name (resnet50, mobilenet_v2, etc.)

    Returns:
        model with frozen backbone
    """
    model_type = model_type.lower()

    # First, freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Then unfreeze only the classifier head
    if model_type in ('resnet50', 'resnet18'):
        for param in model.fc.parameters():
            param.requires_grad = True

    elif model_type == 'mobilenet_v2':
        for param in model.classifier.parameters():
            param.requires_grad = True

    elif model_type == 'densenet121':
        for param in model.classifier.parameters():
            param.requires_grad = True

    elif model_type == 'efficientnet_b0':
        for param in model.classifier.parameters():
            param.requires_grad = True

    return model


def unfreeze_last_block(model, model_type: str):
    """
    Unfreeze the last backbone block in addition to the classifier.

    Use in mid FL rounds (3-4) after classifier has stabilized.
    Allows fine-tuning of high-level features while preserving lower layers.

    Args:
        model: PyTorch model instance
        model_type: Model architecture name

    Returns:
        model with last block unfrozen
    """
    model_type = model_type.lower()

    if model_type in ('resnet50', 'resnet18'):
        # Unfreeze layer4 (last residual block)
        for param in model.layer4.parameters():
            param.requires_grad = True

    elif model_type == 'mobilenet_v2':
        # Unfreeze last 3 inverted residual blocks
        features_list = list(model.features.children())
        for block in features_list[-3:]:
            for param in block.parameters():
                param.requires_grad = True

    elif model_type == 'densenet121':
        # Unfreeze denseblock4 and transition3
        for param in model.features.denseblock4.parameters():
            param.requires_grad = True
        if hasattr(model.features, 'transition3'):
            for param in model.features.transition3.parameters():
                param.requires_grad = True

    elif model_type == 'efficientnet_b0':
        # Unfreeze last 3 blocks
        features_list = list(model.features.children())
        for block in features_list[-3:]:
            for param in block.parameters():
                param.requires_grad = True

    return model


def unfreeze_all(model):
    """
    Unfreeze all model parameters for full fine-tuning.

    Use in later FL rounds (5+) after model has stabilized.

    Args:
        model: PyTorch model instance

    Returns:
        model with all parameters trainable
    """
    for param in model.parameters():
        param.requires_grad = True
    return model


def apply_freeze_strategy(model, model_type: str, server_round: int, total_rounds: int = 10):
    """
    Apply appropriate freeze/unfreeze strategy based on current FL round.

    Strategy:
        - Rounds 1-2: Freeze backbone, train only classifier (prevent feature corruption)
        - Rounds 3-4: Unfreeze last block + classifier (gradual fine-tuning)
        - Rounds 5+: Unfreeze all layers (full fine-tuning)

    Args:
        model: PyTorch model instance
        model_type: Model architecture name
        server_round: Current FL round number (1-indexed)
        total_rounds: Total number of FL rounds

    Returns:
        model with appropriate layers frozen/unfrozen
    """
    from flwr.common import log
    from logging import INFO

    # Calculate phase thresholds (roughly 20%, 40% of training)
    phase1_end = max(2, int(total_rounds * 0.2))
    phase2_end = max(4, int(total_rounds * 0.4))

    if server_round <= phase1_end:
        # Phase 1: Classifier only
        model = freeze_backbone(model, model_type)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(INFO, f"[Freeze] Round {server_round}: Backbone FROZEN, classifier only ({trainable:,} params)")

    elif server_round <= phase2_end:
        # Phase 2: Last block + classifier
        model = freeze_backbone(model, model_type)
        model = unfreeze_last_block(model, model_type)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(INFO, f"[Freeze] Round {server_round}: Last block + classifier ({trainable:,} params)")

    else:
        # Phase 3: Full fine-tuning
        model = unfreeze_all(model)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(INFO, f"[Freeze] Round {server_round}: All layers UNFROZEN ({trainable:,} params)")

    return model


def get_trainable_params_count(model) -> int:
    """Get count of trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_frozen_params_count(model) -> int:
    """Get count of frozen parameters in model."""
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)


# ============================================================================
# EVALUATION UTILITIES
# ============================================================================

def evaluate_client_on_public_test(client_id: int, model_path: str, model_type: str, device: torch.device,
                                    get_model_fn, load_model_fn, test_fn, load_test_dataset_fn) -> Dict:
    """Evaluate a client model on the public test dataset."""
    model = get_model_fn(model_type)
    if os.path.exists(model_path):
        try:
            model, _ = load_model_fn(model, model_path, device)
        except Exception:
            pass
    model.to(device)
    test_loader = load_test_dataset_fn(batch_size=64)
    metrics = test_fn(model, test_loader, device, return_detailed=True)
    metrics["evaluated_at"] = datetime.now().isoformat()
    return metrics


def evaluate_all_clients(client_configs: List[Dict], device: torch.device,
                          get_model_fn, load_model_fn, test_fn, load_test_dataset_fn) -> Dict:
    """Evaluate all clients on the public test dataset."""
    client_metrics = {}
    for i, client in enumerate(client_configs):
        try:
            metrics = evaluate_client_on_public_test(
                i, client['model_path'], client['model_type'], device,
                get_model_fn, load_model_fn, test_fn, load_test_dataset_fn
            )
            client_metrics[str(i)] = metrics
        except Exception as e:
            client_metrics[str(i)] = {"accuracy": None, "loss": None, "error": str(e)}
    return client_metrics


def evaluate_all_clients_on_public_test(client_configs: List[Dict], device: torch.device,
                                         get_model_fn, load_model_fn, test_fn, load_test_dataset_fn) -> Dict:
    """Evaluate all clients on public test dataset (for Pre/Post-FL comparison)."""
    client_metrics = {}
    for i, client in enumerate(client_configs):
        try:
            model = get_model_fn(client['model_type'])
            if os.path.exists(client['model_path']):
                try:
                    model, _ = load_model_fn(model, client['model_path'], device)
                except Exception:
                    pass
            model.to(device)
            test_loader = load_test_dataset_fn(batch_size=64)
            metrics = test_fn(model, test_loader, device, return_detailed=True)
            metrics.update({
                'dataset': 'public_test', 'evaluation_type': 'global',
                'num_samples': len(test_loader.dataset), 'evaluated_at': datetime.now().isoformat()
            })
            client_metrics[str(i)] = metrics
        except Exception as e:
            client_metrics[str(i)] = {"accuracy": None, "loss": None, "error": str(e)}
    return client_metrics


def evaluate_all_clients_on_validation(client_configs: List[Dict], device: torch.device, num_partitions: int,
                                        get_model_fn, load_model_fn, load_private_dataset_fn, load_test_dataset_fn) -> Dict:
    """Evaluate all clients on their private validation sets (per-round tracking)."""
    from sklearn.metrics import precision_recall_fscore_support

    client_metrics = {}
    for i, client in enumerate(client_configs):
        try:
            model = get_model_fn(client['model_type'])
            if os.path.exists(client['model_path']):
                try:
                    model, _ = load_model_fn(model, client['model_path'], device)
                except Exception:
                    pass
            model.to(device)
            _, valloader = load_private_dataset_fn(i, num_partitions, batch_size=64)

            if valloader is not None:
                dataset_type = 'validation'
                num_samples = len(valloader.dataset)
            else:
                valloader = load_test_dataset_fn(batch_size=64)
                dataset_type = 'public_test_proxy'
                num_samples = len(valloader.dataset)

            model.eval()
            all_preds, all_labels = [], []
            total_loss = 0.0
            criterion = nn.CrossEntropyLoss()

            with torch.no_grad():
                for images, labels in valloader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    total_loss += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            all_preds, all_labels = np.array(all_preds), np.array(all_labels)
            accuracy = (all_preds == all_labels).mean()
            loss = total_loss / len(all_labels)
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)

            leukemia_mask = (all_labels == 0)
            healthy_mask = (all_labels == 1)
            leukemia_acc = (all_preds[leukemia_mask] == all_labels[leukemia_mask]).mean() if leukemia_mask.any() else 0
            healthy_acc = (all_preds[healthy_mask] == all_labels[healthy_mask]).mean() if healthy_mask.any() else 0
            class_gap = abs(leukemia_acc - healthy_acc)

            metrics = {
                'loss': float(loss), 'accuracy': float(accuracy),
                'precision': float(precision), 'recall': float(recall), 'f1_score': float(f1),
                'class_gap': float(class_gap),
                'leukemia_accuracy': float(leukemia_acc), 'healthy_accuracy': float(healthy_acc),
                'num_samples': num_samples, 'dataset': dataset_type,
                'evaluation_type': 'per_round', 'evaluated_at': datetime.now().isoformat()
            }
            client_metrics[str(i)] = metrics
        except Exception as e:
            client_metrics[str(i)] = {"accuracy": None, "loss": None, "error": str(e)}
    return client_metrics


# ============================================================================
# DATABASE PERSISTENCE UTILITIES
# ============================================================================

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


def save_global_post_fl_metrics(metrics: Dict, client_configs: List[Dict]):
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
                try:
                    stats = get_partition_stats(client_idx, len(client_configs), partitioner_cache)
                    if "error" not in stats:
                        existing_metrics['data_heterogeneity'] = stats
                except Exception:
                    pass

            SUPABASE_CLIENT.from_('client_simulation_metrics').update({'metrics': existing_metrics}) \
                .eq('simulation_id', SIMULATION_ID).eq('client_id', db_client_id).execute()
    except Exception:
        pass


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
            accs = [r['metrics'].get('accuracy', 0) for r in last_3]
            if accs[-1] < accs[-2] < accs[-3]:
                log(WARNING, f"Client {client_id}: Accuracy declining for 2 consecutive rounds")

            losses = [r['metrics'].get('loss', 999) for r in last_3]
            if losses[-1] > losses[-2] > losses[-3]:
                log(WARNING, f"Client {client_id}: Loss increasing for 2 consecutive rounds (overfitting)")
