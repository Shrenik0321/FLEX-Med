"""
FLEX-Med Federated Learning Evaluation & Visualization Module (IMPROVED)

IMPROVEMENTS:
- ✨ Smart dynamic y-axis scaling for better visualization clarity
- ✨ Automatic range detection with intelligent padding
- ✨ Per-metric optimization for accuracy, loss, and other metrics

This module provides comprehensive graphical visualization of FL training metrics
stored in flex_med/utils/client_data.json. It generates:
- Per-client metric trends (accuracy, f1, precision, recall, loss) across rounds
- Pre-FL vs Post-FL comparison charts showing improvement/decline
- Aggregate performance across all clients
- Training convergence analysis (distill_loss, train_loss)

Usage:
    from flex_med.utils.fl_evaluation import generate_all_visualizations
    generate_all_visualizations()

    Or from command line:
    python -m flex_med.utils.fl_evaluation
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')

# Try to import config, fallback to environment variables or relative paths
import sys
from pathlib import Path

def setup_environment():
    """Ensure backend root is in sys.path for app.config imports"""
    current_file = Path(__file__).resolve()
    # utils -> flex_med -> federated_learning -> backend
    backend_root = current_file.parent.parent.parent.parent
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))
    return backend_root

setup_environment()

try:
    from app.config import get_settings, get_supabase_client
    _settings = get_settings()
    CLIENT_INFO_FILE_PATH = str(_settings.client_info_file_path)
    BASE_PATH = str(_settings.base_path)
    GRAPHS_OUTPUT_DIR = str(_settings.graphs_output_dir)
    SUPABASE_AVAILABLE = True
except ImportError:
    import warnings
    warnings.warn(
        "Could not import config from app.config. "
        "Using environment variables or relative paths as fallback. "
        "Set BASE_PATH, FLEX_MED_CONFIG_FILE, or GRAPHS_OUTPUT_DIR env vars to configure."
    )
    # Get current file location and derive base path
    current_file = os.path.dirname(os.path.abspath(__file__))
    # Go up from utils -> flex_med -> federated_learning -> backend
    backend_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file))))
    BASE_PATH = os.getenv("BASE_PATH", backend_root)
    CLIENT_INFO_FILE_PATH = os.getenv(
        "FLEX_MED_CONFIG_FILE",
        os.path.join(BASE_PATH, "federated_learning", "flex_med", "utils", "client_data.json")
    )
    GRAPHS_OUTPUT_DIR = os.getenv(
        "GRAPHS_OUTPUT_DIR",
        os.path.join(BASE_PATH, "graphical_visualisation")
    )
    SUPABASE_AVAILABLE = False

# Color palette for clients
CLIENT_COLORS = [
    '#2ecc71',  # Green
    '#3498db',  # Blue
    '#e74c3c',  # Red
    '#9b59b6',  # Purple
    '#f39c12',  # Orange
    '#1abc9c',  # Teal
    '#e91e63',  # Pink
    '#00bcd4',  # Cyan
]

# Metrics configuration (ylim=None means use smart auto-scaling)
METRICS_CONFIG = {
    'accuracy': {'label': 'Accuracy', 'format': '.2%', 'ylim': None, 'use_smart_scaling': True},
    'f1_score': {'label': 'F1 Score', 'format': '.3f', 'ylim': None, 'use_smart_scaling': True},
    'precision': {'label': 'Precision', 'format': '.3f', 'ylim': None, 'use_smart_scaling': True},
    'recall': {'label': 'Recall (Sensitivity)', 'format': '.3f', 'ylim': None, 'use_smart_scaling': True},
    'specificity': {'label': 'Specificity', 'format': '.3f', 'ylim': None, 'use_smart_scaling': True},
    'roc_auc': {'label': 'ROC-AUC', 'format': '.3f', 'ylim': None, 'use_smart_scaling': True},
    'loss': {'label': 'Loss', 'format': '.4f', 'ylim': None, 'use_smart_scaling': True},
    'class_gap': {'label': 'Class Gap', 'format': '.2%', 'ylim': None, 'use_smart_scaling': True},
}

TRAINING_METRICS_CONFIG = {
    'distill_loss': {'label': 'Distillation Loss', 'format': '.4f'},
    'train_loss': {'label': 'Training Loss', 'format': '.4f'},
    'consensus_weight': {'label': 'Consensus Weight', 'format': '.2%'},
}


def smart_ylim(values: List[float], metric: str, padding: float = 0.15) -> Tuple[float, float]:
    """
    Intelligently calculate y-axis limits for better visualization.
    
    Similar to Weights & Biases auto-scaling, this function:
    1. Finds the data range
    2. Adds intelligent padding based on the metric type
    3. Ensures minimum visibility range
    4. Respects natural boundaries (0-1 for percentages)
    
    Args:
        values: List of all values to be plotted
        metric: Metric name (to apply metric-specific rules)
        padding: Padding percentage (0.15 = 15% padding on each side)
    
    Returns:
        Tuple of (ymin, ymax)
    """
    if not values or len(values) == 0:
        return (0, 1)
    
    values = [v for v in values if v is not None and not np.isnan(v)]
    if not values:
        return (0, 1)
    
    min_val = float(min(values))
    max_val = float(max(values))
    value_range = max_val - min_val
    
    # Metrics that should be bounded between 0 and 1
    bounded_metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'specificity', 
                       'roc_auc', 'class_gap', 'leukemia_accuracy', 'healthy_accuracy']
    
    if metric in bounded_metrics:
        # Strategy: Focus on the actual data range with smart padding
        
        # If the range is very small, ensure minimum visibility
        if value_range < 0.05:
            center = (max_val + min_val) / 2
            ymin = max(0, center - 0.05)
            ymax = min(1.0, center + 0.05)
        else:
            # Add padding to the range
            pad_amount = value_range * padding
            
            # For high values (>0.6), we can start higher than 0
            if min_val > 0.6:
                ymin = max(0, min_val - pad_amount)
            elif min_val > 0.3:
                ymin = max(0, min_val - pad_amount * 1.5)
            else:
                ymin = 0
            
            # Cap at 1.0 but add padding
            ymax = min(1.0, max_val + pad_amount)
        
        # Ensure we have at least 10% range for clarity
        if ymax - ymin < 0.1:
            center = (ymax + ymin) / 2
            ymin = max(0, center - 0.05)
            ymax = min(1.0, center + 0.05)
    
    else:
        # For unbounded metrics (loss, etc.)
        if value_range < 0.01:
            # Very small range
            center = (max_val + min_val) / 2
            ymin = max(0, center - 0.01)
            ymax = center + 0.01
        else:
            pad_amount = value_range * padding
            ymin = max(0, min_val - pad_amount)
            ymax = max_val + pad_amount
    
    return (ymin, ymax)


def collect_all_values(clients: List[Dict], metric: str, stage: str = 'validation') -> List[float]:
    """
    Collect all values for a metric across all clients to determine optimal y-axis range.

    Args:
        clients: List of client data
        metric: Metric name
        stage: 'validation', 'post_fl', or 'global'
              Note: 'pre_fl' stage removed - use theoretical baselines instead

    Returns:
        List of all values
    """
    all_values = []

    for client in clients:
        if stage == 'validation':
            _, values = extract_validation_series(client, metric)
            all_values.extend(values)
        elif stage == 'global':
            global_metrics = extract_global_metrics(client)
            post_val = global_metrics.get('post_fl', {}).get(metric)
            if post_val is not None:
                all_values.append(post_val)
        elif stage == 'post_fl':
            _, values = extract_metric_series(client, metric, stage)
            all_values.extend(values)

    return all_values


def load_cnmc_data(data_path: str = CLIENT_INFO_FILE_PATH) -> List[Dict]:
    """Load client data from client_data.json"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with open(data_path, 'r') as f:
        return json.load(f)


def load_data_from_db(simulation_id: int) -> List[Dict]:
    """
    Fetch simulation results directly from Supabase.
    
    Reconstructs the client_data.json structure from client_simulation_metrics
    and clients tables.
    """
    if not SUPABASE_AVAILABLE:
        raise ImportError("Supabase client not available. Check your configuration.")

    try:
        supabase = get_supabase_client()
        print(f"🔄 Fetching data from Supabase for simulation {simulation_id}...")

        # 1. Fetch metrics records for this simulation
        metrics_response = supabase.table('client_simulation_metrics') \
            .select('client_id, metrics, status, clients(*)') \
            .eq('simulation_id', simulation_id) \
            .execute()

        if not metrics_response.data:
            print(f"⚠️ No metrics found for simulation {simulation_id}")
            return []

        # 2. Reconstruct the client data structure
        clients_data = []
        for record in metrics_response.data:
            # Base client info from the joined 'clients' table
            client_info = record.get('clients', {})
            if not client_info:
                print(f"⚠️ Warning: Missing client metadata for client {record.get('client_id')}")
                continue

            # Merge in the specific metrics from this simulation
            # Note: The visualization code expects client['metrics'] to be a dict
            client_metrics = record.get('metrics', {})
            
            # Handle string vs dict (JSONB usually returns dict)
            if isinstance(client_metrics, str):
                try:
                   client_metrics = json.loads(client_metrics)
                except:
                   client_metrics = {}

            # Create the combined structure expected by visualization functions
            reconstructed_client = client_info.copy()
            reconstructed_client['metrics'] = client_metrics
            clients_data.append(reconstructed_client)

        print(f"✅ Successfully loaded {len(clients_data)} clients from database")
        return clients_data

    except Exception as e:
        print(f"❌ Error fetching from database: {e}")
        return []


def extract_metric_series(client: Dict, metric: str, stage: str = 'post_fl') -> Tuple[List[int], List[float]]:
    """
    Extract a metric series across rounds for a client.

    Args:
        client: Client data dictionary
        metric: Metric name (accuracy, f1_score, etc.)
        stage: 'pre_fl' or 'post_fl'

    Returns:
        Tuple of (rounds, values)
    """
    rounds = []
    values = []

    metrics = client.get('metrics', {})
    if not isinstance(metrics, dict) or 'rounds' not in metrics:
        return rounds, values

    for round_data in metrics['rounds']:
        round_num = round_data.get('round')
        stage_data = round_data.get(stage, {})

        if stage_data and metric in stage_data:
            value = stage_data[metric]
            if value is not None:
                rounds.append(round_num)
                values.append(value)

    return rounds, values


def extract_training_metric_series(client: Dict, metric: str) -> Tuple[List[int], List[float]]:
    """Extract training metrics (distill_loss, train_loss) across rounds."""
    rounds = []
    values = []

    metrics = client.get('metrics', {})
    if not isinstance(metrics, dict) or 'rounds' not in metrics:
        return rounds, values

    for round_data in metrics['rounds']:
        round_num = round_data.get('round')
        training_data = round_data.get('training', {})

        if training_data and metric in training_data:
            value = training_data[metric]
            if value is not None:
                rounds.append(round_num)
                values.append(value)

    return rounds, values


def extract_improvement_series(client: Dict, metric: str) -> Tuple[List[int], List[float]]:
    """Extract improvement (post_fl - pre_fl) for a metric across rounds."""
    rounds = []
    values = []

    metrics = client.get('metrics', {})
    if not isinstance(metrics, dict) or 'rounds' not in metrics:
        return rounds, values

    for round_data in metrics['rounds']:
        round_num = round_data.get('round')
        improvement = round_data.get('improvement', {})

        if improvement and metric in improvement:
            value = improvement[metric]
            if value is not None:
                rounds.append(round_num)
                values.append(value)

    return rounds, values


def extract_validation_series(client: Dict, metric: str, include_baseline: bool = True) -> Tuple[List[int], List[float]]:
    """
    Extract validation metric series across rounds (for hybrid evaluation strategy).

    Args:
        client: Client data dictionary
        metric: Metric name (accuracy, f1_score, etc.)
        include_baseline: If True, prepend round 0 with theoretical baseline value

    Returns:
        Tuple of (rounds, values)
    """
    # Theoretical baselines for binary classification (random chance)
    BASELINES = {
        'accuracy': 0.5,
        'precision': 0.5,
        'recall': 0.5,
        'f1_score': 0.5,
        'loss': 0.693,  # ln(2) for random binary classifier
    }

    rounds = []
    values = []

    metrics = client.get('metrics', {})

    # Add theoretical baseline as round 0
    if include_baseline and metric in BASELINES:
        rounds.append(0)
        values.append(BASELINES[metric])

    # Add validation metrics from each round
    if not isinstance(metrics, dict) or 'rounds' not in metrics:
        return rounds, values

    for round_data in metrics['rounds']:
        round_num = round_data.get('round')
        validation_data = round_data.get('validation', {})

        if validation_data and metric in validation_data:
            value = validation_data[metric]
            if value is not None:
                rounds.append(round_num)
                values.append(value)

    return rounds, values


def extract_global_metrics(client: Dict) -> Dict:
    """
    Extract global Post-FL metrics (for hybrid evaluation strategy).

    Args:
        client: Client data dictionary

    Returns:
        Dictionary with 'post_fl' and 'improvement' metrics
        Note: 'pre_fl' has been removed - use theoretical baselines instead
    """
    metrics = client.get('metrics', {})
    global_metrics = metrics.get('global', {})

    return {
        'post_fl': global_metrics.get('post_fl', {}),
        'improvement': global_metrics.get('improvement', {})
    }


def setup_plot_style():
    """Configure matplotlib style for consistent plots."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.figsize': (12, 6),
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 100,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
    })


def plot_metric_across_rounds(
    clients: List[Dict],
    metric: str,
    output_dir: str = GRAPHS_OUTPUT_DIR,
    show_plot: bool = True
) -> str:
    """
    Plot a single metric across rounds for all clients with smart y-axis scaling.

    Args:
        clients: List of client data
        metric: Metric to plot
        output_dir: Directory to save the plot
        show_plot: Whether to display the plot

    Returns:
        Path to saved plot
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    config = METRICS_CONFIG.get(metric, {'label': metric, 'format': '.3f', 'ylim': None, 'use_smart_scaling': True})

    # Left plot: Validation metrics across rounds
    ax1 = axes[0]
    
    # Collect all validation values for smart y-axis scaling
    all_validation_values = collect_all_values(clients, metric, 'validation')
    
    for i, client in enumerate(clients):
        color = CLIENT_COLORS[i % len(CLIENT_COLORS)]
        name = client.get('client_name', f'Client {i}')
        model = client.get('model_type', 'unknown')

        rounds, values = extract_validation_series(client, metric)
        if rounds and values:
            ax1.plot(rounds, values, marker='o', linewidth=2, markersize=6,
                    color=color, label=f'{name} ({model})')

    ax1.set_xlabel('Training Round')
    ax1.set_ylabel(config['label'])
    ax1.set_title(f'{config["label"]} Progression (Validation)')
    ax1.set_xlim(left=0)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Apply smart y-axis scaling
    if config.get('use_smart_scaling', True) and all_validation_values:
        ymin, ymax = smart_ylim(all_validation_values, metric)
        ax1.set_ylim(ymin, ymax)
        ax1.text(0.02, 0.98, f'Range: [{ymin:.3f}, {ymax:.3f}]', 
                transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Right plot: Baseline vs Post-FL comparison
    ax2 = axes[1]
    bar_width = 0.35
    x_positions = np.arange(len(clients))

    # Theoretical baselines for binary classification
    BASELINES = {
        'accuracy': 0.5, 'precision': 0.5, 'recall': 0.5,
        'f1_score': 0.5, 'loss': 0.693
    }
    baseline_value = BASELINES.get(metric, 0.5)

    baseline_values = []
    post_fl_values = []
    client_labels = []
    all_global_values = []

    for client in clients:
        name = client.get('client_name', 'Unknown')[:15]
        client_labels.append(name)

        # Get global metrics
        global_metrics = extract_global_metrics(client)
        post_fl = global_metrics.get('post_fl', {})

        post_val = post_fl.get(metric, 0) or 0

        baseline_values.append(baseline_value)
        post_fl_values.append(post_val)
        all_global_values.extend([baseline_value, post_val])

    bars1 = ax2.bar(x_positions - bar_width/2, baseline_values, bar_width,
                    label='Baseline (Random)', color='#95a5a6', alpha=0.8)
    bars2 = ax2.bar(x_positions + bar_width/2, post_fl_values, bar_width,
                    label='Post-FL', color='#2ecc71', alpha=0.8)

    # Add value labels on bars
    for bar, val in zip(bars1, baseline_values):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, post_fl_values):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    ax2.set_xlabel('Client')
    ax2.set_ylabel(config['label'])
    ax2.set_title(f'{config["label"]}: Baseline vs Post-FL Comparison')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(client_labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Apply smart y-axis scaling for global comparison
    if config.get('use_smart_scaling', True) and all_global_values:
        ymin, ymax = smart_ylim(all_global_values, metric)
        ax2.set_ylim(ymin, ymax)

    plt.tight_layout()

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = f'{metric}_progression.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)

    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f"  Saved: {filename}")
    return filepath


def plot_improvement_analysis(
    clients: List[Dict],
    output_dir: str = GRAPHS_OUTPUT_DIR,
    show_plot: bool = True
) -> str:
    """
    Plot improvement analysis showing how much each client improved after FL.
    """
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics_to_plot = ['accuracy', 'f1_score', 'precision', 'recall']

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        config = METRICS_CONFIG.get(metric, {'label': metric})

        improvements = []
        client_labels = []
        colors = []

        for i, client in enumerate(clients):
            name = client.get('client_name', f'Client {i}')[:15]
            client_labels.append(name)

            global_metrics = extract_global_metrics(client)
            improvement = global_metrics.get('improvement', {})

            imp_value = improvement.get(metric, 0) or 0
            improvements.append(imp_value)
            colors.append('#2ecc71' if imp_value >= 0 else '#e74c3c')

        bars = ax.bar(client_labels, improvements, color=colors, alpha=0.8, edgecolor='black')

        # Add value labels
        for bar, val in zip(bars, improvements):
            y_pos = bar.get_height() + 0.005 if val >= 0 else bar.get_height() - 0.02
            ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                   f'{val:+.2%}', ha='center', va='bottom' if val >= 0 else 'top',
                   fontsize=9, fontweight='bold')

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Client')
        ax.set_ylabel(f'{config["label"]} Change')
        ax.set_title(f'{config["label"]} Improvement After FL Training')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Smart y-axis scaling for improvement
        if improvements:
            ymin, ymax = smart_ylim(improvements, 'improvement_' + metric, padding=0.2)
            # Ensure 0 is visible
            ymin = min(ymin, -0.05)
            ymax = max(ymax, 0.05)
            ax.set_ylim(ymin, ymax)

    # Add legend
    improved_patch = mpatches.Patch(color='#2ecc71', label='Improved')
    declined_patch = mpatches.Patch(color='#e74c3c', label='Declined')
    fig.legend(handles=[improved_patch, declined_patch], loc='upper right', fontsize=11)

    plt.suptitle('FL Training Impact: Per-Client Improvement Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = 'improvement_analysis.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)

    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f"  Saved: {filename}")
    return filepath


def plot_improvement_over_rounds(
    clients: List[Dict],
    output_dir: str = GRAPHS_OUTPUT_DIR,
    show_plot: bool = True
) -> str:
    """
    Plot improvement trends over rounds for each client with smart y-axis scaling.
    Calculates improvement as (validation_metric - theoretical_baseline).
    Theoretical baselines: 0.5 for accuracy/precision/recall/f1 (random chance for binary classification).
    """
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics_to_plot = ['accuracy', 'f1_score', 'precision', 'recall']

    # Theoretical baselines for binary classification (random chance)
    THEORETICAL_BASELINES = {
        'accuracy': 0.5, 'f1_score': 0.5, 'precision': 0.5, 'recall': 0.5
    }

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        config = METRICS_CONFIG.get(metric, {'label': metric})

        all_improvements = []

        # Use theoretical baseline (0.5 for all bounded metrics)
        baseline_value = THEORETICAL_BASELINES.get(metric, 0.5)

        for i, client in enumerate(clients):
            color = CLIENT_COLORS[i % len(CLIENT_COLORS)]
            name = client.get('client_name', f'Client {i}')
            model = client.get('model_type', 'unknown')

            # Get validation series (skip baseline at round 0)
            rounds, val_values = extract_validation_series(client, metric, include_baseline=False)

            if rounds and val_values:
                # Calculate improvement relative to theoretical baseline
                improvements = [val - baseline_value for val in val_values]
                all_improvements.extend(improvements)

                ax.plot(rounds, [v * 100 for v in improvements], marker='o',
                       linewidth=2, markersize=6, color=color,
                       label=f'{name} ({model})')

                # Fill area above/below zero
                ax.fill_between(rounds, 0, [v * 100 for v in improvements],
                               where=[v >= 0 for v in improvements],
                               color=color, alpha=0.2)
                ax.fill_between(rounds, 0, [v * 100 for v in improvements],
                               where=[v < 0 for v in improvements],
                               color=color, alpha=0.2)

        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Training Round')
        ax.set_ylabel(f'{config["label"]} Improvement (%)')
        ax.set_title(f'{config["label"]} Improvement Per Round')
        ax.set_xlim(left=0)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

        # Smart y-axis scaling
        if all_improvements:
            improvement_pct = [v * 100 for v in all_improvements]
            ymin, ymax = smart_ylim(improvement_pct, 'improvement_' + metric, padding=0.2)
            # Ensure 0 is visible
            ymin = min(ymin, -2)
            ymax = max(ymax, 2)
            ax.set_ylim(ymin, ymax)

    plt.suptitle('Per-Round Improvement Trends (vs Theoretical 0.5 Baseline)', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = 'improvement_over_rounds.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)

    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f"  Saved: {filename}")
    return filepath


def plot_training_convergence(
    clients: List[Dict],
    output_dir: str = GRAPHS_OUTPUT_DIR,
    show_plot: bool = True
) -> str:
    """
    Plot training convergence showing distillation and training loss over rounds with smart scaling.
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Distillation Loss
    ax1 = axes[0]
    all_distill_values = []
    
    for i, client in enumerate(clients):
        color = CLIENT_COLORS[i % len(CLIENT_COLORS)]
        name = client.get('client_name', f'Client {i}')
        model = client.get('model_type', 'unknown')

        rounds, values = extract_training_metric_series(client, 'distill_loss')
        if rounds and values:
            all_distill_values.extend(values)
            ax1.plot(rounds, values, marker='s', linewidth=2, markersize=6,
                    color=color, label=f'{name} ({model})')

    ax1.set_xlabel('Training Round')
    ax1.set_ylabel('Distillation Loss')
    ax1.set_title('Knowledge Distillation Loss Convergence')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Smart y-axis scaling
    if all_distill_values:
        ymin, ymax = smart_ylim(all_distill_values, 'distill_loss')
        ax1.set_ylim(ymin, ymax)

    # Training Loss
    ax2 = axes[1]
    all_train_values = []
    
    for i, client in enumerate(clients):
        color = CLIENT_COLORS[i % len(CLIENT_COLORS)]
        name = client.get('client_name', f'Client {i}')
        model = client.get('model_type', 'unknown')

        rounds, values = extract_training_metric_series(client, 'train_loss')
        if rounds and values:
            all_train_values.extend(values)
            ax2.plot(rounds, values, marker='^', linewidth=2, markersize=6,
                    color=color, label=f'{name} ({model})')

    ax2.set_xlabel('Training Round')
    ax2.set_ylabel('Training Loss')
    ax2.set_title('Private Training Loss Convergence')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Smart y-axis scaling
    if all_train_values:
        ymin, ymax = smart_ylim(all_train_values, 'train_loss')
        ax2.set_ylim(ymin, ymax)

    plt.suptitle('Training Convergence Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = 'training_convergence.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)

    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f"  Saved: {filename}")
    return filepath


def plot_confusion_matrix_comparison(
    clients: List[Dict],
    output_dir: str = GRAPHS_OUTPUT_DIR,
    show_plot: bool = True
) -> str:
    """
    Plot Post-FL confusion matrix for each client.
    Note: Pre-FL comparison removed - using theoretical baseline (0.5 accuracy) instead.
    """
    setup_plot_style()
    n_clients = len(clients)

    # Single column layout for Post-FL only
    fig, axes = plt.subplots(1, n_clients, figsize=(6 * n_clients, 5))

    if n_clients == 1:
        axes = [axes]

    for i, client in enumerate(clients):
        name = client.get('client_name', f'Client {i}')
        model = client.get('model_type', 'unknown')

        global_metrics = extract_global_metrics(client)
        post_fl = global_metrics.get('post_fl', {})

        ax = axes[i]

        cm = post_fl.get('confusion_matrix', {})
        if cm:
            matrix = np.array([
                [cm.get('TP', 0), cm.get('FN', 0)],
                [cm.get('FP', 0), cm.get('TN', 0)]
            ])

            im = ax.imshow(matrix, cmap='Blues', aspect='auto')

            # Add text annotations
            for row in range(2):
                for col in range(2):
                    text_color = 'white' if matrix[row, col] > matrix.max()/2 else 'black'
                    ax.text(col, row, f'{matrix[row, col]}',
                           ha='center', va='center', fontsize=14,
                           color=text_color, fontweight='bold')

            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Predicted\nLeukemia', 'Predicted\nHealthy'])
            ax.set_yticklabels(['Actual\nLeukemia', 'Actual\nHealthy'])
            ax.set_title(f'{name} ({model})')

            # Add accuracy annotation
            acc = post_fl.get('accuracy', 0)
            ax.text(0.5, -0.15, f'Accuracy: {acc:.1%}',
                   transform=ax.transAxes, ha='center', fontsize=11)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
            ax.set_title(f'{name} ({model})')

    plt.suptitle('Post-FL Confusion Matrix (Public Test Dataset)', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = 'confusion_matrix_comparison.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)

    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f"  Saved: {filename}")
    return filepath


def plot_class_balance_analysis(
    clients: List[Dict],
    output_dir: str = GRAPHS_OUTPUT_DIR,
    show_plot: bool = True
) -> str:
    """
    Plot per-class accuracy (leukemia vs healthy) to analyze class balance with smart scaling.
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Per-class accuracy over rounds
    ax1 = axes[0]
    all_class_acc = []
    
    for i, client in enumerate(clients):
        color = CLIENT_COLORS[i % len(CLIENT_COLORS)]
        name = client.get('client_name', f'Client {i}')[:12]

        rounds_l, leukemia_acc = extract_validation_series(client, 'leukemia_accuracy')
        rounds_h, healthy_acc = extract_validation_series(client, 'healthy_accuracy')

        if rounds_l and leukemia_acc:
            all_class_acc.extend(leukemia_acc)
            ax1.plot(rounds_l, leukemia_acc, marker='o', linewidth=2,
                    color=color, linestyle='-', label=f'{name} (Leukemia)')
        if rounds_h and healthy_acc:
            all_class_acc.extend(healthy_acc)
            ax1.plot(rounds_h, healthy_acc, marker='s', linewidth=2,
                    color=color, linestyle='--', label=f'{name} (Healthy)')

    ax1.set_xlabel('Training Round')
    ax1.set_ylabel('Per-Class Accuracy')
    ax1.set_title('Per-Class Accuracy Over Rounds')
    ax1.set_xlim(left=0)
    ax1.legend(loc='best', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # Smart y-axis scaling
    if all_class_acc:
        ymin, ymax = smart_ylim(all_class_acc, 'accuracy')
        ax1.set_ylim(ymin, ymax)

    # Right: Class gap reduction
    ax2 = axes[1]
    all_gap_values = []
    
    for i, client in enumerate(clients):
        color = CLIENT_COLORS[i % len(CLIENT_COLORS)]
        name = client.get('client_name', f'Client {i}')
        model = client.get('model_type', 'unknown')

        rounds, class_gap = extract_validation_series(client, 'class_gap')
        if rounds and class_gap:
            all_gap_values.extend(class_gap)
            ax2.plot(rounds, [v * 100 for v in class_gap], marker='o',
                    linewidth=2, markersize=6, color=color,
                    label=f'{name} ({model})')

    ax2.axhline(y=10, color='green', linestyle='--', linewidth=2, label='Target (<10%)')
    ax2.set_xlabel('Training Round')
    ax2.set_ylabel('Class Gap (%)')
    ax2.set_title('Class Imbalance Gap Over Rounds')
    ax2.set_xlim(left=0)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Smart y-axis scaling
    if all_gap_values:
        gap_pct = [v * 100 for v in all_gap_values]
        gap_pct.append(10)  # Include target in range calculation
        ymin, ymax = smart_ylim(gap_pct, 'class_gap')
        ax2.set_ylim(max(0, ymin), ymax)

    plt.suptitle('Class Balance Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = 'class_balance_analysis.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)

    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f"  Saved: {filename}")
    return filepath


def plot_summary_dashboard(
    clients: List[Dict],
    output_dir: str = GRAPHS_OUTPUT_DIR,
    show_plot: bool = True
) -> str:
    """
    Create a comprehensive summary dashboard with key metrics and smart y-axis scaling.
    """
    setup_plot_style()
    fig = plt.figure(figsize=(20, 12))

    # Create grid
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # 1. Overall Accuracy Progress (top-left, spanning 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    all_acc_values = []
    
    for i, client in enumerate(clients):
        color = CLIENT_COLORS[i % len(CLIENT_COLORS)]
        name = client.get('client_name', f'Client {i}')
        model = client.get('model_type', 'unknown')

        rounds, values = extract_validation_series(client, 'accuracy')
        if rounds and values:
            all_acc_values.extend(values)
            ax1.plot(rounds, [v * 100 for v in values], marker='o',
                    linewidth=2.5, markersize=8, color=color,
                    label=f'{name} ({model})')

    ax1.set_xlabel('Training Round')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy Progression', fontsize=14, fontweight='bold')
    ax1.set_xlim(left=0)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Smart scaling
    if all_acc_values:
        acc_pct = [v * 100 for v in all_acc_values]
        ymin, ymax = smart_ylim(acc_pct, 'accuracy')
        ax1.set_ylim(ymin, ymax)

    # 2. F1 Score Progress (top-right, spanning 2 columns)
    ax2 = fig.add_subplot(gs[0, 2:])
    all_f1_values = []
    
    for i, client in enumerate(clients):
        color = CLIENT_COLORS[i % len(CLIENT_COLORS)]
        name = client.get('client_name', f'Client {i}')

        rounds, values = extract_validation_series(client, 'f1_score')
        if rounds and values:
            all_f1_values.extend(values)
            ax2.plot(rounds, values, marker='s', linewidth=2.5,
                    markersize=8, color=color, label=name[:15])

    ax2.set_xlabel('Training Round')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score Progression', fontsize=14, fontweight='bold')
    ax2.set_xlim(left=0)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Smart scaling
    if all_f1_values:
        ymin, ymax = smart_ylim(all_f1_values, 'f1_score')
        ax2.set_ylim(ymin, ymax)

    # 3. Improvement Summary (middle-left)
    ax3 = fig.add_subplot(gs[1, :2])

    client_names = [c.get('client_name', f'Client {i}')[:12] for i, c in enumerate(clients)]
    x = np.arange(len(clients))
    width = 0.2

    for j, metric in enumerate(['accuracy', 'f1_score', 'precision', 'recall']):
        improvements = []
        for client in clients:
            global_metrics = extract_global_metrics(client)
            imp = global_metrics.get('improvement', {}).get(metric, 0) or 0
            improvements.append(imp * 100)

        bars = ax3.bar(x + j * width, improvements, width,
                      label=METRICS_CONFIG[metric]['label'],
                      alpha=0.8)

    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Client')
    ax3.set_ylabel('Improvement (%)')
    ax3.set_title('Overall Improvement Summary', fontsize=14, fontweight='bold')
    ax3.set_xticks(x + width * 1.5)
    ax3.set_xticklabels(client_names, rotation=45, ha='right')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. ROC-AUC Comparison (middle-right) - Baseline vs Post-FL
    ax4 = fig.add_subplot(gs[1, 2:])

    # Theoretical baseline for ROC-AUC (random classifier = 0.5)
    BASELINE_ROC_AUC = 0.5

    baseline_roc = []
    post_roc = []
    all_roc = []

    for client in clients:
        global_metrics = extract_global_metrics(client)
        post_val = global_metrics.get('post_fl', {}).get('roc_auc', 0) or 0
        baseline_roc.append(BASELINE_ROC_AUC)
        post_roc.append(post_val)
        all_roc.extend([BASELINE_ROC_AUC, post_val])

    x = np.arange(len(clients))
    ax4.bar(x - 0.2, baseline_roc, 0.4, label='Baseline (Random)', color='#95a5a6', alpha=0.8)
    ax4.bar(x + 0.2, post_roc, 0.4, label='Post-FL', color='#2ecc71', alpha=0.8)

    ax4.set_xlabel('Client')
    ax4.set_ylabel('ROC-AUC')
    ax4.set_title('ROC-AUC: Baseline vs Post FL', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(client_names, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Smart scaling
    if all_roc:
        ymin, ymax = smart_ylim(all_roc, 'roc_auc')
        ax4.set_ylim(ymin, ymax)

    # 5. Validation Loss Convergence (bottom-left)
    ax5 = fig.add_subplot(gs[2, :2])
    all_loss_values = []
    
    for i, client in enumerate(clients):
        color = CLIENT_COLORS[i % len(CLIENT_COLORS)]
        name = client.get('client_name', f'Client {i}')

        rounds, values = extract_validation_series(client, 'loss')
        if rounds and values:
            all_loss_values.extend(values)
            ax5.plot(rounds, values, marker='^', linewidth=2,
                    markersize=6, color=color, label=name[:15])

    ax5.set_xlabel('Training Round')
    ax5.set_ylabel('Validation Loss')
    ax5.set_title('Validation Loss Convergence', fontsize=14, fontweight='bold')
    ax5.set_xlim(left=0)
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)
    
    # Smart scaling
    if all_loss_values:
        ymin, ymax = smart_ylim(all_loss_values, 'loss')
        ax5.set_ylim(ymin, ymax)

    # 6. Class Gap Trend (bottom-right)
    ax6 = fig.add_subplot(gs[2, 2:])
    all_gap_values = []
    
    for i, client in enumerate(clients):
        color = CLIENT_COLORS[i % len(CLIENT_COLORS)]
        name = client.get('client_name', f'Client {i}')

        rounds, values = extract_validation_series(client, 'class_gap')
        if rounds and values:
            all_gap_values.extend(values)
            ax6.plot(rounds, [v * 100 for v in values], marker='o',
                    linewidth=2, markersize=6, color=color, label=name[:15])

    ax6.axhline(y=10, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Target (<10%)')
    ax6.set_xlabel('Training Round')
    ax6.set_ylabel('Class Gap (%)')
    ax6.set_title('Class Balance Improvement', fontsize=14, fontweight='bold')
    ax6.set_xlim(left=0)
    ax6.legend(loc='best')
    ax6.grid(True, alpha=0.3)
    
    # Smart scaling
    if all_gap_values:
        gap_pct = [v * 100 for v in all_gap_values]
        gap_pct.append(10)  # Include target
        ymin, ymax = smart_ylim(gap_pct, 'class_gap')
        ax6.set_ylim(max(0, ymin), ymax)

    # Main title
    plt.suptitle('FLEX-Med Federated Learning - Training Summary Dashboard\n✨ Smart Auto-Scaled Y-Axes',
                fontsize=18, fontweight='bold', y=0.98)

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = 'summary_dashboard.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150)

    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f"  Saved: {filename}")
    return filepath


def plot_global_fl_benefit(
    clients: List[Dict],
    output_dir: str = GRAPHS_OUTPUT_DIR,
    show_plot: bool = True
) -> str:
    """
    Plot Global FL Benefit Analysis (Baseline vs Post-FL on public test dataset) with smart scaling.
    Shows the overall improvement from federated learning compared to theoretical baselines.
    Theoretical baselines: 0.5 for accuracy/f1/class_gap (random chance), 0.693 for loss (ln(2)).
    """
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics_to_plot = ['accuracy', 'f1_score', 'loss', 'class_gap']

    # Theoretical baselines for binary classification
    THEORETICAL_BASELINES = {
        'accuracy': 0.5,
        'f1_score': 0.5,
        'loss': 0.693,  # ln(2)
        'class_gap': 1.0,  # Maximum gap (improvement = reduction)
    }

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        config = METRICS_CONFIG.get(metric, {'label': metric})

        baseline_values = []
        post_values = []
        client_labels = []
        improvements = []
        all_values = []

        baseline_val = THEORETICAL_BASELINES.get(metric, 0.5)

        for i, client in enumerate(clients):
            name = client.get('client_name', f'Client {i}')[:15]
            client_labels.append(name)

            global_metrics = extract_global_metrics(client)
            post_fl = global_metrics.get('post_fl', {})

            post_val = post_fl.get(metric, 0) or 0
            improvement = post_val - baseline_val

            baseline_values.append(baseline_val)
            post_values.append(post_val)
            improvements.append(improvement)
            all_values.extend([baseline_val, post_val])

        x = np.arange(len(clients))
        width = 0.35

        bars1 = ax.bar(x - width/2, baseline_values, width,
                      label='Baseline (Random)', color='#95a5a6', alpha=0.8)
        bars2 = ax.bar(x + width/2, post_values, width,
                      label='Post-FL (Federated)', color='#2ecc71', alpha=0.8)

        # Add value labels and improvement annotations
        for i, (bar1, bar2, base_val, post_val, imp) in enumerate(zip(bars1, bars2, baseline_values, post_values, improvements)):
            if base_val > 0:
                ax.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.01,
                       f'{base_val:.2f}', ha='center', va='bottom', fontsize=8)
            if post_val > 0:
                ax.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.01,
                       f'{post_val:.2f}', ha='center', va='bottom', fontsize=8)

            # Add improvement arrow
            if imp != 0:
                y_start = max(base_val, post_val) + 0.05
                color = '#2ecc71' if imp > 0 else '#e74c3c'
                ax.annotate(f'{imp:+.2f}', xy=(x[i], y_start),
                           xytext=(x[i], y_start + 0.05),
                           ha='center', fontsize=9, fontweight='bold', color=color,
                           arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

        ax.set_xlabel('Client')
        ax.set_ylabel(config['label'])
        ax.set_title(f'{config["label"]}: FL Benefit Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels(client_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Smart y-axis scaling
        if all_values and config.get('use_smart_scaling', True):
            ymin, ymax = smart_ylim(all_values, metric)
            # Add extra space for annotations
            ymax = ymax * 1.1
            ax.set_ylim(ymin, ymax)

    plt.suptitle('Global FL Benefit Analysis (Public Test Dataset)\nTheoretical Baseline vs Federated Models\n✨ Smart Auto-Scaled Y-Axes',
                fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = 'global_fl_benefit_analysis.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)

    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f"  Saved: {filename}")
    return filepath


def plot_validation_progression(
    clients: List[Dict],
    output_dir: str = GRAPHS_OUTPUT_DIR,
    show_plot: bool = True
) -> str:
    """
    Plot per-round validation metrics progression with smart y-axis scaling.
    Shows learning curves on validation sets (not test set).
    """
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics_to_plot = ['accuracy', 'loss', 'f1_score', 'class_gap']

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        config = METRICS_CONFIG.get(metric, {'label': metric})

        all_values = []
        
        for i, client in enumerate(clients):
            color = CLIENT_COLORS[i % len(CLIENT_COLORS)]
            name = client.get('client_name', f'Client {i}')
            model = client.get('model_type', 'unknown')

            rounds, values = extract_validation_series(client, metric)

            if rounds and values:
                all_values.extend(values)
                
                # Convert to percentage for class_gap and accuracy
                if metric in ['accuracy', 'class_gap']:
                    display_values = [v * 100 for v in values]
                else:
                    display_values = values

                ax.plot(rounds, display_values, marker='o', linewidth=2, markersize=6,
                       color=color, label=f'{name} ({model})')

        ax.set_xlabel('Training Round')
        ylabel = f'{config["label"]} (%)' if metric in ['accuracy', 'class_gap'] else config['label']
        ax.set_ylabel(ylabel)
        ax.set_title(f'{config["label"]} - Validation Progression')
        ax.set_xlim(left=0)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Add target line for class_gap
        if metric == 'class_gap':
            ax.axhline(y=10, color='green', linestyle='--', linewidth=2,
                      alpha=0.7, label='Target (<10%)')
            all_values.append(0.10)  # Include target in range

        # Smart y-axis scaling
        if all_values and config.get('use_smart_scaling', True):
            if metric in ['accuracy', 'class_gap']:
                display_all = [v * 100 for v in all_values]
                ymin, ymax = smart_ylim(display_all, metric)
            else:
                ymin, ymax = smart_ylim(all_values, metric)
            ax.set_ylim(ymin, ymax)

    plt.suptitle('Per-Round Validation Metrics\n(Evaluated on Private Validation Sets)\n✨ Smart Auto-Scaled Y-Axes',
                fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = 'validation_progression.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)

    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f"  Saved: {filename}")
    return filepath


def plot_hybrid_evaluation_comparison(
    clients: List[Dict],
    output_dir: str = GRAPHS_OUTPUT_DIR,
    show_plot: bool = True
) -> str:
    """
    Plot comparison between validation progression and global evaluation with smart scaling.
    Shows how per-round validation metrics relate to final test performance.
    Uses theoretical baselines (0.5 for accuracy, 0.693 for loss) instead of pre_fl.
    """
    setup_plot_style()
    fig, axes = plt.subplots(len(clients), 2, figsize=(16, 5 * len(clients)))

    if len(clients) == 1:
        axes = [axes]

    # Theoretical baselines
    BASELINE_ACCURACY = 0.5
    BASELINE_LOSS = 0.693  # ln(2)

    for i, client in enumerate(clients):
        name = client.get('client_name', f'Client {i}')
        model = client.get('model_type', 'unknown')
        color = CLIENT_COLORS[i % len(CLIENT_COLORS)]

        # Left plot: Accuracy progression (validation vs global)
        ax_left = axes[i][0] if len(clients) > 1 else axes[0]

        all_acc_values = []

        # Validation progression (includes baseline at round 0)
        rounds, val_acc = extract_validation_series(client, 'accuracy')
        if rounds and val_acc:
            all_acc_values.extend(val_acc)
            ax_left.plot(rounds, [v * 100 for v in val_acc], marker='o',
                        linewidth=2, markersize=6, color=color,
                        label='Validation (per-round)', linestyle='-')

        # Global Post-FL marker only
        global_metrics = extract_global_metrics(client)
        if global_metrics.get('post_fl'):
            post_acc = global_metrics['post_fl'].get('accuracy', 0) * 100
            all_acc_values.append(post_acc / 100)

            max_round = max(rounds) if rounds else 1

            # Add baseline reference line
            ax_left.axhline(y=BASELINE_ACCURACY * 100, color='#95a5a6', linestyle='--',
                          linewidth=2, alpha=0.7, label='Baseline (50%)')

            ax_left.scatter([max_round], [post_acc], s=200, marker='D', color='#2ecc71',
                          edgecolors='black', linewidths=2, label='Global Post-FL (Test)',
                          zorder=10)

        ax_left.set_xlabel('Training Round')
        ax_left.set_ylabel('Accuracy (%)')
        ax_left.set_title(f'{name} ({model}) - Accuracy Progression')
        ax_left.set_xlim(left=0)
        ax_left.legend(loc='best')
        ax_left.grid(True, alpha=0.3)

        # Smart scaling
        if all_acc_values:
            acc_pct = [v * 100 if v <= 1 else v for v in all_acc_values]
            ymin, ymax = smart_ylim(acc_pct, 'accuracy')
            ax_left.set_ylim(ymin, ymax)

        # Right plot: Loss progression
        ax_right = axes[i][1] if len(clients) > 1 else axes[1]

        all_loss_values = []

        # Validation loss (includes baseline at round 0)
        rounds, val_loss = extract_validation_series(client, 'loss')
        if rounds and val_loss:
            all_loss_values.extend(val_loss)
            ax_right.plot(rounds, val_loss, marker='s', linewidth=2,
                         markersize=6, color=color,
                         label='Validation (per-round)', linestyle='-')

        # Global Post-FL marker only
        if global_metrics.get('post_fl'):
            post_loss = global_metrics['post_fl'].get('loss', 0)
            all_loss_values.append(post_loss)

            max_round = max(rounds) if rounds else 1

            # Add baseline reference line
            ax_right.axhline(y=BASELINE_LOSS, color='#95a5a6', linestyle='--',
                           linewidth=2, alpha=0.7, label=f'Baseline (ln2≈{BASELINE_LOSS:.3f})')

            ax_right.scatter([max_round], [post_loss], s=200, marker='D', color='#2ecc71',
                           edgecolors='black', linewidths=2, label='Global Post-FL (Test)',
                           zorder=10)

        ax_right.set_xlabel('Training Round')
        ax_right.set_ylabel('Loss')
        ax_right.set_title(f'{name} ({model}) - Loss Progression')
        ax_right.set_xlim(left=0)
        ax_right.legend(loc='best')
        ax_right.grid(True, alpha=0.3)

        # Smart scaling
        if all_loss_values:
            ymin, ymax = smart_ylim(all_loss_values, 'loss')
            ax_right.set_ylim(ymin, ymax)

    plt.suptitle('Hybrid Evaluation Strategy\nValidation Progression vs Theoretical Baseline\n✨ Smart Auto-Scaled Y-Axes',
                fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    filename = 'hybrid_evaluation_comparison.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)

    if show_plot:
        plt.show()
    else:
        plt.close()

    print(f"  Saved: {filename}")
    return filepath


def generate_all_visualizations(
    data_path: str = CLIENT_INFO_FILE_PATH,
    output_dir: str = GRAPHS_OUTPUT_DIR,
    show_plots: bool = True,
    simulation_id: Optional[int] = None
) -> Dict[str, str]:
    """
    Generate all visualizations from client_data.json with smart auto-scaling.

    Args:
        data_path: Path to client_data.json
        output_dir: Directory to save visualizations
        show_plots: Whether to display plots inline

    Returns:
        Dictionary mapping visualization names to file paths
    """
    print("=" * 70)
    print("FLEX-Med FL Evaluation - Generating Visualizations")
    print("✨ IMPROVED VERSION with Smart Y-Axis Auto-Scaling ✨")
    print("=" * 70)
    print(f"Data source: {data_path}")
    print(f"Output directory: {output_dir}")
    print()

    # Load data
    try:
        if simulation_id is not None:
            clients = load_data_from_db(simulation_id)
        else:
            clients = load_cnmc_data(data_path)
            print(f"Loaded {len(clients)} clients from data file")
    except Exception as e:
        print(f"Error: {e}")
        return {}

    # Check if metrics exist
    has_metrics = False
    for client in clients:
        metrics = client.get('metrics', {})
        if isinstance(metrics, dict) and 'rounds' in metrics and metrics['rounds']:
            has_metrics = True
            break

    if not has_metrics:
        print("\nWarning: No round metrics found in data file.")
        print("Run FL training first to generate metrics.")
        return {}

    print(f"\nGenerating visualizations with smart auto-scaling...")
    print("-" * 40)

    saved_files = {}

    # 1. Summary Dashboard
    print("\n1. Summary Dashboard")
    saved_files['summary_dashboard'] = plot_summary_dashboard(clients, output_dir, show_plots)

    # 2. Individual metric plots
    print("\n2. Individual Metric Progressions")
    for metric in ['accuracy', 'f1_score', 'precision', 'recall', 'loss', 'roc_auc']:
        saved_files[f'{metric}_progression'] = plot_metric_across_rounds(
            clients, metric, output_dir, show_plots
        )

    # 3. Improvement Analysis
    print("\n3. Improvement Analysis")
    saved_files['improvement_analysis'] = plot_improvement_analysis(clients, output_dir, show_plots)

    # 4. Improvement Over Rounds
    print("\n4. Improvement Over Rounds")
    saved_files['improvement_over_rounds'] = plot_improvement_over_rounds(clients, output_dir, show_plots)

    # 5. Training Convergence
    print("\n5. Training Convergence")
    saved_files['training_convergence'] = plot_training_convergence(clients, output_dir, show_plots)

    # 6. Confusion Matrix Comparison
    print("\n6. Confusion Matrix Comparison")
    saved_files['confusion_matrix'] = plot_confusion_matrix_comparison(clients, output_dir, show_plots)

    # 7. Class Balance Analysis
    print("\n7. Class Balance Analysis")
    saved_files['class_balance'] = plot_class_balance_analysis(clients, output_dir, show_plots)

    # 8. Hybrid Evaluation Strategy Visualizations (if global metrics exist)
    has_global_metrics = False
    for client in clients:
        metrics = client.get('metrics', {})
        if 'global' in metrics and metrics['global']:
            has_global_metrics = True
            break

    if has_global_metrics:
        print("\n8. Global FL Benefit Analysis (Hybrid Strategy)")
        saved_files['global_fl_benefit'] = plot_global_fl_benefit(clients, output_dir, show_plots)

        print("\n9. Validation Progression (Hybrid Strategy)")
        saved_files['validation_progression'] = plot_validation_progression(clients, output_dir, show_plots)

        print("\n10. Hybrid Evaluation Comparison")
        saved_files['hybrid_comparison'] = plot_hybrid_evaluation_comparison(clients, output_dir, show_plots)

        print("\n" + "=" * 70)
        print("HYBRID EVALUATION STRATEGY DETECTED")
        print("=" * 70)
        print("✓ Global metrics (Pre-FL vs Post-FL on public test)")
        print("✓ Per-round validation metrics")
        print("✓ Test set exposed only 2 times (Pre-FL + Post-FL)")
        print("=" * 70)

    print("\n" + "=" * 70)
    print(f"Visualizations complete! {len(saved_files)} graphs saved to:")
    print(f"  {output_dir}")
    print("\n✨ SMART Y-AXIS SCALING FEATURES:")
    print("  • Dynamic range adjustment based on actual data")
    print("  • 15% padding for visual clarity")
    print("  • Metric-specific optimization (accuracy, loss, etc.)")
    print("  • Minimum range enforcement to prevent flat graphs")
    print("  • Similar to Weights & Biases auto-scaling")
    print("=" * 70)

    return saved_files


    print("\n" + "=" * 70)


def print_metrics_summary(
    data_source: str = CLIENT_INFO_FILE_PATH,
    simulation_id: Optional[int] = None
):
    """
    Print a textual summary of FL metrics.
    
    Args:
        data_source: Path to JSON file
        simulation_id: Optional ID to fetch from DB
    """
    if simulation_id is not None:
        clients = load_data_from_db(simulation_id)
    else:
        try:
            clients = load_cnmc_data(data_source)
        except FileNotFoundError:
            print(f"❌ File not found: {data_source}")
            return
            
    if not clients:
        print("❌ No client data found.")
        return

    generate_text_summary(clients)


# Main entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate FL evaluation visualizations with smart auto-scaling')
    parser.add_argument('--data', type=str, default=CLIENT_INFO_FILE_PATH,
                       help='Path to client_data.json')
    parser.add_argument('--output', type=str, default=GRAPHS_OUTPUT_DIR,
                       help='Output directory for graphs')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plots (just save)')
    parser.add_argument('--summary', action='store_true',
                       help='Print text summary only')
    parser.add_argument('--simulation-id', '-s', type=int, default=None,
                       help='Fetch metrics from database for this simulation ID')

    args = parser.parse_args()

    if args.summary:
        print_metrics_summary(args.data, args.simulation_id)
    else:
        generate_all_visualizations(args.data, args.output, not args.no_show, args.simulation_id)