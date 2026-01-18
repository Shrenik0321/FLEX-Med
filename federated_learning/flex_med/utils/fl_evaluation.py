"""
FLEX-Med Federated Learning Evaluation & Visualization Module

This module provides comprehensive graphical visualization of FL training metrics
stored in cnmc_data.json. It generates:
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

# Try to import config, fallback to defaults
try:
    from flex_med.utils.config import DATA_JSON_PATH, BASE_PATH, GRAPHS_OUTPUT_DIR
except ImportError:
    BASE_PATH = "/content/drive/MyDrive/College/FLEX-Med"
    DATA_JSON_PATH = os.path.join(BASE_PATH, "flex-med/cnmc_data.json")
    GRAPHS_OUTPUT_DIR = os.path.join(BASE_PATH, "graphical_visualisation")

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

# Metrics configuration
METRICS_CONFIG = {
    'accuracy': {'label': 'Accuracy', 'format': '.2%', 'ylim': (0, 1)},
    'f1_score': {'label': 'F1 Score', 'format': '.3f', 'ylim': (0, 1)},
    'precision': {'label': 'Precision', 'format': '.3f', 'ylim': (0, 1)},
    'recall': {'label': 'Recall (Sensitivity)', 'format': '.3f', 'ylim': (0, 1)},
    'specificity': {'label': 'Specificity', 'format': '.3f', 'ylim': (0, 1)},
    'roc_auc': {'label': 'ROC-AUC', 'format': '.3f', 'ylim': (0, 1)},
    'loss': {'label': 'Loss', 'format': '.4f', 'ylim': None},
    'class_gap': {'label': 'Class Gap', 'format': '.2%', 'ylim': (0, 1)},
}

TRAINING_METRICS_CONFIG = {
    'distill_loss': {'label': 'Distillation Loss', 'format': '.4f'},
    'train_loss': {'label': 'Training Loss', 'format': '.4f'},
    'consensus_weight': {'label': 'Consensus Weight', 'format': '.2%'},
}


def load_cnmc_data(data_path: str = DATA_JSON_PATH) -> List[Dict]:
    """Load client data from cnmc_data.json"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with open(data_path, 'r') as f:
        return json.load(f)


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
    Plot a single metric across rounds for all clients.

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

    config = METRICS_CONFIG.get(metric, {'label': metric, 'format': '.3f', 'ylim': None})

    # Left plot: Post-FL metrics across rounds
    ax1 = axes[0]
    for i, client in enumerate(clients):
        color = CLIENT_COLORS[i % len(CLIENT_COLORS)]
        name = client.get('client_name', f'Client {i}')
        model = client.get('model_type', 'unknown')

        rounds, values = extract_metric_series(client, metric, 'post_fl')
        if rounds and values:
            ax1.plot(rounds, values, marker='o', linewidth=2, markersize=6,
                    color=color, label=f'{name} ({model})')

    ax1.set_xlabel('Training Round')
    ax1.set_ylabel(config['label'])
    ax1.set_title(f'{config["label"]} Progression (Post-FL)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    if config['ylim']:
        ax1.set_ylim(config['ylim'])

    # Right plot: Pre-FL vs Post-FL comparison
    ax2 = axes[1]
    bar_width = 0.35
    x_positions = np.arange(len(clients))

    pre_fl_values = []
    post_fl_values = []
    client_labels = []

    for client in clients:
        name = client.get('client_name', 'Unknown')[:15]
        client_labels.append(name)

        # Get latest round metrics
        metrics = client.get('metrics', {})
        current = metrics.get('current', {})

        pre_fl = current.get('pre_fl', {})
        post_fl = current.get('post_fl', {})

        pre_fl_values.append(pre_fl.get(metric, 0) or 0)
        post_fl_values.append(post_fl.get(metric, 0) or 0)

    bars1 = ax2.bar(x_positions - bar_width/2, pre_fl_values, bar_width,
                    label='Pre-FL', color='#e74c3c', alpha=0.8)
    bars2 = ax2.bar(x_positions + bar_width/2, post_fl_values, bar_width,
                    label='Post-FL', color='#2ecc71', alpha=0.8)

    # Add value labels on bars
    for bar, val in zip(bars1, pre_fl_values):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, post_fl_values):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    ax2.set_xlabel('Client')
    ax2.set_ylabel(config['label'])
    ax2.set_title(f'{config["label"]}: Pre-FL vs Post-FL Comparison')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(client_labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    if config['ylim']:
        ax2.set_ylim(config['ylim'])

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

            metrics = client.get('metrics', {})
            current = metrics.get('current', {})
            improvement = current.get('improvement', {})

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
    Plot improvement trends over rounds for each client.
    """
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    metrics_to_plot = ['accuracy', 'f1_score', 'precision', 'recall']

    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        config = METRICS_CONFIG.get(metric, {'label': metric})

        for i, client in enumerate(clients):
            color = CLIENT_COLORS[i % len(CLIENT_COLORS)]
            name = client.get('client_name', f'Client {i}')
            model = client.get('model_type', 'unknown')

            rounds, improvements = extract_improvement_series(client, metric)

            if rounds and improvements:
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
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Per-Round Improvement Trends (Post-FL - Pre-FL)', fontsize=16, fontweight='bold')
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
    Plot training convergence showing distillation and training loss over rounds.
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Distillation Loss
    ax1 = axes[0]
    for i, client in enumerate(clients):
        color = CLIENT_COLORS[i % len(CLIENT_COLORS)]
        name = client.get('client_name', f'Client {i}')
        model = client.get('model_type', 'unknown')

        rounds, values = extract_training_metric_series(client, 'distill_loss')
        if rounds and values:
            ax1.plot(rounds, values, marker='s', linewidth=2, markersize=6,
                    color=color, label=f'{name} ({model})')

    ax1.set_xlabel('Training Round')
    ax1.set_ylabel('Distillation Loss')
    ax1.set_title('Knowledge Distillation Loss Convergence')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Training Loss
    ax2 = axes[1]
    for i, client in enumerate(clients):
        color = CLIENT_COLORS[i % len(CLIENT_COLORS)]
        name = client.get('client_name', f'Client {i}')
        model = client.get('model_type', 'unknown')

        rounds, values = extract_training_metric_series(client, 'train_loss')
        if rounds and values:
            ax2.plot(rounds, values, marker='^', linewidth=2, markersize=6,
                    color=color, label=f'{name} ({model})')

    ax2.set_xlabel('Training Round')
    ax2.set_ylabel('Training Loss')
    ax2.set_title('Private Training Loss Convergence')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

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
    Plot confusion matrix comparison (Pre-FL vs Post-FL) for each client.
    """
    setup_plot_style()
    n_clients = len(clients)
    fig, axes = plt.subplots(n_clients, 2, figsize=(12, 5 * n_clients))

    if n_clients == 1:
        axes = [axes]

    for i, client in enumerate(clients):
        name = client.get('client_name', f'Client {i}')

        metrics = client.get('metrics', {})
        current = metrics.get('current', {})

        pre_fl = current.get('pre_fl', {})
        post_fl = current.get('post_fl', {})

        for j, (stage, stage_data, title) in enumerate([
            ('pre_fl', pre_fl, f'{name} - Pre-FL'),
            ('post_fl', post_fl, f'{name} - Post-FL')
        ]):
            ax = axes[i][j] if n_clients > 1 else axes[j]

            cm = stage_data.get('confusion_matrix', {})
            if cm:
                matrix = np.array([
                    [cm.get('TP', 0), cm.get('FP', 0)],
                    [cm.get('FN', 0), cm.get('TN', 0)]
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
                ax.set_title(title)

                # Add accuracy annotation
                acc = stage_data.get('accuracy', 0)
                ax.text(0.5, -0.15, f'Accuracy: {acc:.1%}',
                       transform=ax.transAxes, ha='center', fontsize=11)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)
                ax.set_title(title)

    plt.suptitle('Confusion Matrix: Pre-FL vs Post-FL Comparison', fontsize=16, fontweight='bold')
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
    Plot per-class accuracy (leukemia vs healthy) to analyze class balance.
    """
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Per-class accuracy over rounds
    ax1 = axes[0]
    for i, client in enumerate(clients):
        color = CLIENT_COLORS[i % len(CLIENT_COLORS)]
        name = client.get('client_name', f'Client {i}')[:12]

        rounds_l, leukemia_acc = extract_metric_series(client, 'leukemia_accuracy', 'post_fl')
        rounds_h, healthy_acc = extract_metric_series(client, 'healthy_accuracy', 'post_fl')

        if rounds_l and leukemia_acc:
            ax1.plot(rounds_l, leukemia_acc, marker='o', linewidth=2,
                    color=color, linestyle='-', label=f'{name} (Leukemia)')
        if rounds_h and healthy_acc:
            ax1.plot(rounds_h, healthy_acc, marker='s', linewidth=2,
                    color=color, linestyle='--', label=f'{name} (Healthy)')

    ax1.set_xlabel('Training Round')
    ax1.set_ylabel('Per-Class Accuracy')
    ax1.set_title('Per-Class Accuracy Over Rounds')
    ax1.legend(loc='best', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Right: Class gap reduction
    ax2 = axes[1]
    for i, client in enumerate(clients):
        color = CLIENT_COLORS[i % len(CLIENT_COLORS)]
        name = client.get('client_name', f'Client {i}')
        model = client.get('model_type', 'unknown')

        rounds, class_gap = extract_metric_series(client, 'class_gap', 'post_fl')
        if rounds and class_gap:
            ax2.plot(rounds, [v * 100 for v in class_gap], marker='o',
                    linewidth=2, markersize=6, color=color,
                    label=f'{name} ({model})')

    ax2.axhline(y=10, color='green', linestyle='--', linewidth=2, label='Target (<10%)')
    ax2.set_xlabel('Training Round')
    ax2.set_ylabel('Class Gap (%)')
    ax2.set_title('Class Imbalance Gap Over Rounds')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

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
    Create a comprehensive summary dashboard with key metrics.
    """
    setup_plot_style()
    fig = plt.figure(figsize=(20, 12))

    # Create grid
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # 1. Overall Accuracy Progress (top-left, spanning 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    for i, client in enumerate(clients):
        color = CLIENT_COLORS[i % len(CLIENT_COLORS)]
        name = client.get('client_name', f'Client {i}')
        model = client.get('model_type', 'unknown')

        rounds, values = extract_metric_series(client, 'accuracy', 'post_fl')
        if rounds and values:
            ax1.plot(rounds, [v * 100 for v in values], marker='o',
                    linewidth=2.5, markersize=8, color=color,
                    label=f'{name} ({model})')

    ax1.set_xlabel('Training Round')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Accuracy Progression', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)

    # 2. F1 Score Progress (top-right, spanning 2 columns)
    ax2 = fig.add_subplot(gs[0, 2:])
    for i, client in enumerate(clients):
        color = CLIENT_COLORS[i % len(CLIENT_COLORS)]
        name = client.get('client_name', f'Client {i}')

        rounds, values = extract_metric_series(client, 'f1_score', 'post_fl')
        if rounds and values:
            ax2.plot(rounds, values, marker='s', linewidth=2.5,
                    markersize=8, color=color, label=name[:15])

    ax2.set_xlabel('Training Round')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score Progression', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    # 3. Improvement Summary (middle-left)
    ax3 = fig.add_subplot(gs[1, :2])

    client_names = [c.get('client_name', f'Client {i}')[:12] for i, c in enumerate(clients)]
    x = np.arange(len(clients))
    width = 0.2

    for j, metric in enumerate(['accuracy', 'f1_score', 'precision', 'recall']):
        improvements = []
        for client in clients:
            metrics = client.get('metrics', {})
            current = metrics.get('current', {})
            imp = current.get('improvement', {}).get(metric, 0) or 0
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

    # 4. ROC-AUC Comparison (middle-right)
    ax4 = fig.add_subplot(gs[1, 2:])

    pre_roc = []
    post_roc = []
    for client in clients:
        metrics = client.get('metrics', {})
        current = metrics.get('current', {})
        pre_roc.append(current.get('pre_fl', {}).get('roc_auc', 0) or 0)
        post_roc.append(current.get('post_fl', {}).get('roc_auc', 0) or 0)

    x = np.arange(len(clients))
    ax4.bar(x - 0.2, pre_roc, 0.4, label='Pre-FL', color='#e74c3c', alpha=0.8)
    ax4.bar(x + 0.2, post_roc, 0.4, label='Post-FL', color='#2ecc71', alpha=0.8)

    ax4.set_xlabel('Client')
    ax4.set_ylabel('ROC-AUC')
    ax4.set_title('ROC-AUC: Pre vs Post FL', fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(client_names, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0, 1)

    # 5. Training Loss Convergence (bottom-left)
    ax5 = fig.add_subplot(gs[2, :2])
    for i, client in enumerate(clients):
        color = CLIENT_COLORS[i % len(CLIENT_COLORS)]
        name = client.get('client_name', f'Client {i}')

        rounds, values = extract_training_metric_series(client, 'train_loss')
        if rounds and values:
            ax5.plot(rounds, values, marker='^', linewidth=2,
                    markersize=6, color=color, label=name[:15])

    ax5.set_xlabel('Training Round')
    ax5.set_ylabel('Training Loss')
    ax5.set_title('Training Loss Convergence', fontsize=14, fontweight='bold')
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)

    # 6. Class Gap Trend (bottom-right)
    ax6 = fig.add_subplot(gs[2, 2:])
    for i, client in enumerate(clients):
        color = CLIENT_COLORS[i % len(CLIENT_COLORS)]
        name = client.get('client_name', f'Client {i}')

        rounds, values = extract_metric_series(client, 'class_gap', 'post_fl')
        if rounds and values:
            ax6.plot(rounds, [v * 100 for v in values], marker='o',
                    linewidth=2, markersize=6, color=color, label=name[:15])

    ax6.axhline(y=10, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Target (<10%)')
    ax6.set_xlabel('Training Round')
    ax6.set_ylabel('Class Gap (%)')
    ax6.set_title('Class Balance Improvement', fontsize=14, fontweight='bold')
    ax6.legend(loc='best')
    ax6.grid(True, alpha=0.3)

    # Main title
    plt.suptitle('FLEX-Med Federated Learning - Training Summary Dashboard',
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


def generate_all_visualizations(
    data_path: str = DATA_JSON_PATH,
    output_dir: str = GRAPHS_OUTPUT_DIR,
    show_plots: bool = True
) -> Dict[str, str]:
    """
    Generate all visualizations from cnmc_data.json.

    Args:
        data_path: Path to cnmc_data.json
        output_dir: Directory to save visualizations
        show_plots: Whether to display plots inline

    Returns:
        Dictionary mapping visualization names to file paths
    """
    print("=" * 70)
    print("FLEX-Med FL Evaluation - Generating Visualizations")
    print("=" * 70)
    print(f"Data source: {data_path}")
    print(f"Output directory: {output_dir}")
    print()

    # Load data
    try:
        clients = load_cnmc_data(data_path)
        print(f"Loaded {len(clients)} clients from data file")
    except FileNotFoundError as e:
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

    print(f"\nGenerating visualizations...")
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

    print("\n" + "=" * 70)
    print(f"Visualizations complete! {len(saved_files)} graphs saved to:")
    print(f"  {output_dir}")
    print("=" * 70)

    return saved_files


def print_metrics_summary(data_path: str = DATA_JSON_PATH):
    """Print a text summary of the metrics."""
    clients = load_cnmc_data(data_path)

    print("\n" + "=" * 70)
    print("FLEX-Med FL Training Summary")
    print("=" * 70)

    for i, client in enumerate(clients):
        name = client.get('client_name', f'Client {i}')
        model = client.get('model_type', 'unknown')

        metrics = client.get('metrics', {})
        current = metrics.get('current', {})
        rounds = metrics.get('rounds', [])

        print(f"\n[{i}] {name} ({model})")
        print("-" * 50)

        if current:
            pre = current.get('pre_fl', {})
            post = current.get('post_fl', {})
            imp = current.get('improvement', {})

            print(f"  Rounds completed: {current.get('last_round', 'N/A')}")
            print(f"  Pre-FL Accuracy:  {pre.get('accuracy', 0):.2%}")
            print(f"  Post-FL Accuracy: {post.get('accuracy', 0):.2%}")
            print(f"  Improvement:      {imp.get('accuracy', 0):+.2%}")
            print(f"  Post-FL F1:       {post.get('f1_score', 0):.3f}")
            print(f"  Post-FL ROC-AUC:  {post.get('roc_auc', 0):.3f}")
        else:
            print("  No metrics available")

    print("\n" + "=" * 70)


# Main entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate FL evaluation visualizations')
    parser.add_argument('--data', type=str, default=DATA_JSON_PATH,
                       help='Path to cnmc_data.json')
    parser.add_argument('--output', type=str, default=GRAPHS_OUTPUT_DIR,
                       help='Output directory for graphs')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plots (just save)')
    parser.add_argument('--summary', action='store_true',
                       help='Print text summary only')

    args = parser.parse_args()

    if args.summary:
        print_metrics_summary(args.data)
    else:
        generate_all_visualizations(args.data, args.output, not args.no_show)
