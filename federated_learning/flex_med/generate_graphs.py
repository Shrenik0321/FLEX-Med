#!/usr/bin/env python3
"""
Generate visualization graphs for FLEX-Med federated learning round metrics.

This script reads round_metrics.json and generates separate graphs for each client:
1. Per-Client Graph 1: Pre-FL Accuracy Over Rounds (knowledge retention)
2. Per-Client Graph 2: Post-FL Accuracy Over Rounds (learning progression)
3. Per-Client Graph 3: Per-Round Improvement (diminishing returns analysis)
4. Per-Client Combined view with all three graphs
5. Text summary report (all clients)

Each client's graphs are saved in separate subdirectories (client_0/, client_1/, etc.)

Usage:
    python generate_graphs.py [--metrics-path PATH] [--output-dir DIR] [--config-path PATH]

Author: FLEX-Med Team
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

# Default paths (can be overridden via environment variables or CLI args)
BASE_PATH = os.getenv("BASE_PATH", "/content/drive/MyDrive/College/FLEX-Med")
ROUND_METRICS_FILE_PATH = os.path.join(BASE_PATH, "round_metrics.json")
GRAPHS_OUTPUT_DIR = os.path.join(BASE_PATH, "graphs")
CLIENT_INFO_FILE_PATH = os.path.join(BASE_PATH, "flex-med/flex_med/data.json")

# Visualization constants
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
FIGURE_DPI = 300
LINE_WIDTH = 2
MARKER_SIZE = 8


def load_round_metrics(metrics_path: str = ROUND_METRICS_FILE_PATH) -> Dict:
    """Load round metrics from JSON file."""
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    print(f"[Graphs] Loaded metrics from: {metrics_path}")
    print(f"[Graphs] Found {len(metrics)} rounds of data")

    return metrics


def load_client_names(config_path: str = CLIENT_INFO_FILE_PATH) -> Dict[str, str]:
    """Load client names from data.json configuration file."""
    client_names = {}

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                clients = json.load(f)

            for i, client in enumerate(clients):
                client_names[str(i)] = client.get('client_name', f'Client {i}')

            print(f"[Graphs] Loaded {len(client_names)} client names from config")
        except Exception as e:
            print(f"[Graphs] Warning: Could not load client names: {e}")

    return client_names


def extract_round_data(metrics: Dict) -> Tuple[List[int], Dict[str, List], Dict[str, List], Dict[str, List]]:
    """Extract round numbers and per-client metrics from the metrics dictionary.

    Returns:
        rounds: Sorted list of round numbers
        pre_fl_data: Dict mapping client_id to list of pre-FL accuracies per round
        post_fl_data: Dict mapping client_id to list of post-FL accuracies per round
        improvement_data: Dict mapping client_id to list of improvements per round
    """
    # Extract and sort round numbers
    round_keys = [k for k in metrics.keys() if k.startswith("round_")]
    rounds = sorted([int(k.split("_")[1]) for k in round_keys])

    # Get all unique client IDs
    all_client_ids = set()
    for round_key in round_keys:
        round_data = metrics[round_key]
        if "pre_fl" in round_data and "clients" in round_data["pre_fl"]:
            all_client_ids.update(round_data["pre_fl"]["clients"].keys())
        if "post_fl" in round_data and "clients" in round_data["post_fl"]:
            all_client_ids.update(round_data["post_fl"]["clients"].keys())

    client_ids = sorted(all_client_ids, key=lambda x: int(x))

    # Initialize data structures
    pre_fl_data = {cid: [] for cid in client_ids}
    post_fl_data = {cid: [] for cid in client_ids}
    improvement_data = {cid: [] for cid in client_ids}

    # Extract data for each round
    for round_num in rounds:
        round_key = f"round_{round_num}"
        round_data = metrics.get(round_key, {})

        for client_id in client_ids:
            # Pre-FL accuracy
            pre_acc = None
            if "pre_fl" in round_data and "clients" in round_data["pre_fl"]:
                client_metrics = round_data["pre_fl"]["clients"].get(client_id, {})
                pre_acc = client_metrics.get("accuracy")
            pre_fl_data[client_id].append(pre_acc)

            # Post-FL accuracy
            post_acc = None
            if "post_fl" in round_data and "clients" in round_data["post_fl"]:
                client_metrics = round_data["post_fl"]["clients"].get(client_id, {})
                post_acc = client_metrics.get("accuracy")
            post_fl_data[client_id].append(post_acc)

            # Improvement (post - pre)
            improvement = None
            if "improvement" in round_data:
                improvement = round_data["improvement"].get(client_id)
            elif pre_acc is not None and post_acc is not None:
                improvement = post_acc - pre_acc
            improvement_data[client_id].append(improvement)

    return rounds, pre_fl_data, post_fl_data, improvement_data


def filter_none_values(rounds: List[int], data: List) -> Tuple[List[int], List]:
    """Filter out None values from data and corresponding rounds."""
    filtered_rounds = []
    filtered_data = []

    for r, d in zip(rounds, data):
        if d is not None:
            filtered_rounds.append(r)
            filtered_data.append(d)

    return filtered_rounds, filtered_data


def plot_pre_fl_accuracy(
    ax: plt.Axes,
    rounds: List[int],
    accuracies: List,
    client_name: str,
    client_id: str,
    annotate: bool = True
) -> None:
    """Plot Pre-FL Accuracy Over Rounds for a single client (Knowledge Retention)."""
    ax.set_title(f'{client_name} - Pre-FL Accuracy Over Rounds\n(Knowledge Retention)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)

    # Filter None values
    valid_rounds, valid_acc = filter_none_values(rounds, accuracies)

    if not valid_rounds:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        return

    # Use first color/marker for consistency per client
    color = COLORS[0]
    marker = MARKERS[0]

    ax.plot(valid_rounds, valid_acc, label=client_name, color=color, marker=marker,
            linewidth=LINE_WIDTH, markersize=MARKER_SIZE, linestyle='-')

    # Add annotations at each data point
    if annotate:
        for r, acc in zip(valid_rounds, valid_acc):
            ax.annotate(
                f'{acc*100:.1f}%',
                xy=(r, acc),
                xytext=(0, 8),
                textcoords='offset points',
                fontsize=9,
                color=color,
                ha='center',
                fontweight='bold'
            )

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(0, 1.15)  # Increased to accommodate annotations

    # Set x-axis to show only integer rounds
    ax.set_xticks(valid_rounds)

    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)


def plot_post_fl_accuracy(
    ax: plt.Axes,
    rounds: List[int],
    accuracies: List,
    client_name: str,
    client_id: str,
    annotate: bool = True
) -> None:
    """Plot Post-FL Accuracy Over Rounds for a single client (Learning Progression)."""
    ax.set_title(f'{client_name} - Post-FL Accuracy Over Rounds\n(Learning Progression)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)

    # Filter None values
    valid_rounds, valid_acc = filter_none_values(rounds, accuracies)

    if not valid_rounds:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        return

    # Use first color/marker for consistency per client
    color = COLORS[0]
    marker = MARKERS[0]

    ax.plot(valid_rounds, valid_acc, label=client_name, color=color, marker=marker,
            linewidth=LINE_WIDTH, markersize=MARKER_SIZE, linestyle='-')

    # Add annotations at each data point
    if annotate:
        for r, acc in zip(valid_rounds, valid_acc):
            ax.annotate(
                f'{acc*100:.1f}%',
                xy=(r, acc),
                xytext=(0, 8),
                textcoords='offset points',
                fontsize=9,
                color=color,
                ha='center',
                fontweight='bold'
            )

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylim(0, 1.15)  # Increased to accommodate annotations

    # Set x-axis to show only integer rounds
    ax.set_xticks(valid_rounds)

    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)


def plot_improvement(
    ax: plt.Axes,
    rounds: List[int],
    improvements: List,
    client_name: str,
    client_id: str,
    annotate: bool = True
) -> None:
    """Plot Per-Round Improvement for a single client (Diminishing Returns)."""
    ax.set_title(f'{client_name} - Per-Round Improvement\n(Post-FL - Pre-FL Accuracy)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Improvement', fontsize=12)

    # Filter None values
    valid_rounds, valid_imp = filter_none_values(rounds, improvements)

    if not valid_rounds:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        return

    # Use first color/marker for consistency per client
    color = COLORS[0]
    marker = MARKERS[0]

    ax.plot(valid_rounds, valid_imp, label=client_name, color=color, marker=marker,
            linewidth=LINE_WIDTH, markersize=MARKER_SIZE, linestyle='-')

    # Add annotations at each data point
    if annotate:
        for r, imp in zip(valid_rounds, valid_imp):
            # Position annotation above for positive, below for negative
            y_offset = 8 if imp >= 0 else -12
            ax.annotate(
                f'{imp*100:+.1f}%',
                xy=(r, imp),
                xytext=(0, y_offset),
                textcoords='offset points',
                fontsize=9,
                color=color,
                ha='center',
                fontweight='bold'
            )

    # Format y-axis as percentage with +/- sign
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))

    # Add horizontal line at y=0
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Set x-axis to show only integer rounds
    ax.set_xticks(valid_rounds)

    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)


def generate_individual_graphs(
    rounds: List[int],
    pre_fl_data: Dict[str, List],
    post_fl_data: Dict[str, List],
    improvement_data: Dict[str, List],
    client_names: Dict[str, str],
    output_dir: str,
    annotate: bool = True
) -> List[str]:
    """Generate separate PNG graphs for each client (pre_fl, post_fl, improvement)."""
    saved_paths = []

    # Generate graphs for each client separately
    for client_id in sorted(pre_fl_data.keys(), key=lambda x: int(x)):
        client_name = client_names.get(client_id, f"Client {client_id}")

        # Create client-specific subdirectory
        client_dir = os.path.join(output_dir, f"client_{client_id}")
        os.makedirs(client_dir, exist_ok=True)

        # Graph 1: Pre-FL Accuracy for this client
        fig1, ax1 = plt.subplots(figsize=(12, 7))
        plot_pre_fl_accuracy(ax1, rounds, pre_fl_data[client_id], client_name, client_id, annotate=annotate)
        plt.tight_layout()
        path1 = os.path.join(client_dir, f"client_{client_id}_pre_fl_accuracy.png")
        fig1.savefig(path1, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig1)
        saved_paths.append(path1)
        print(f"[Graphs] Saved: {path1}")

        # Graph 2: Post-FL Accuracy for this client
        fig2, ax2 = plt.subplots(figsize=(12, 7))
        plot_post_fl_accuracy(ax2, rounds, post_fl_data[client_id], client_name, client_id, annotate=annotate)
        plt.tight_layout()
        path2 = os.path.join(client_dir, f"client_{client_id}_post_fl_accuracy.png")
        fig2.savefig(path2, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig2)
        saved_paths.append(path2)
        print(f"[Graphs] Saved: {path2}")

        # Graph 3: Per-Round Improvement for this client
        fig3, ax3 = plt.subplots(figsize=(12, 7))
        plot_improvement(ax3, rounds, improvement_data[client_id], client_name, client_id, annotate=annotate)
        plt.tight_layout()
        path3 = os.path.join(client_dir, f"client_{client_id}_improvement.png")
        fig3.savefig(path3, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig3)
        saved_paths.append(path3)
        print(f"[Graphs] Saved: {path3}")

    return saved_paths


def generate_combined_graph_per_client(
    rounds: List[int],
    pre_fl_data: Dict[str, List],
    post_fl_data: Dict[str, List],
    improvement_data: Dict[str, List],
    client_names: Dict[str, str],
    output_dir: str,
    annotate: bool = False  # Disable annotations by default for combined view (too cluttered)
) -> List[str]:
    """Generate combined view with all three graphs for each client separately."""
    saved_paths = []

    # Generate combined graph for each client
    for client_id in sorted(pre_fl_data.keys(), key=lambda x: int(x)):
        client_name = client_names.get(client_id, f"Client {client_id}")

        # Create client-specific subdirectory
        client_dir = os.path.join(output_dir, f"client_{client_id}")
        os.makedirs(client_dir, exist_ok=True)

        # Create combined figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))

        plot_pre_fl_accuracy(axes[0], rounds, pre_fl_data[client_id], client_name, client_id, annotate=annotate)
        plot_post_fl_accuracy(axes[1], rounds, post_fl_data[client_id], client_name, client_id, annotate=annotate)
        plot_improvement(axes[2], rounds, improvement_data[client_id], client_name, client_id, annotate=annotate)

        fig.suptitle(f'{client_name} - FLEX-Med Federated Learning: Per-Round Evaluation Metrics',
                     fontsize=16, fontweight='bold', y=1.02)

        plt.tight_layout()

        path = os.path.join(client_dir, f"client_{client_id}_combined_metrics.png")
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches='tight')
        plt.close(fig)

        saved_paths.append(path)
        print(f"[Graphs] Saved combined view for {client_name}: {path}")

    return saved_paths


def generate_text_summary(
    rounds: List[int],
    pre_fl_data: Dict[str, List],
    post_fl_data: Dict[str, List],
    improvement_data: Dict[str, List],
    client_names: Dict[str, str],
    output_dir: str
) -> str:
    """Generate a text summary report of the training metrics."""
    lines = []
    lines.append("=" * 70)
    lines.append("FLEX-Med Federated Learning: Training Summary Report")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")

    # Overall statistics
    lines.append("OVERALL STATISTICS")
    lines.append("-" * 40)
    lines.append(f"Total Rounds: {len(rounds)}")
    lines.append(f"Total Clients: {len(pre_fl_data)}")
    lines.append("")

    # Per-client summary
    lines.append("PER-CLIENT SUMMARY")
    lines.append("-" * 40)

    for client_id in sorted(pre_fl_data.keys(), key=lambda x: int(x)):
        client_name = client_names.get(client_id, f"Client {client_id}")
        lines.append(f"\n{client_name} (ID: {client_id})")
        lines.append("~" * 30)

        # Get valid pre-FL accuracies
        _, valid_pre = filter_none_values(rounds, pre_fl_data[client_id])
        if valid_pre:
            lines.append(f"  Pre-FL Accuracy:  Start={valid_pre[0]*100:.1f}%, End={valid_pre[-1]*100:.1f}%")
        else:
            lines.append("  Pre-FL Accuracy:  No data")

        # Get valid post-FL accuracies
        _, valid_post = filter_none_values(rounds, post_fl_data[client_id])
        if valid_post:
            lines.append(f"  Post-FL Accuracy: Start={valid_post[0]*100:.1f}%, End={valid_post[-1]*100:.1f}%")
        else:
            lines.append("  Post-FL Accuracy: No data")

        # Get valid improvements
        _, valid_imp = filter_none_values(rounds, improvement_data[client_id])
        if valid_imp:
            total_improvement = sum(valid_imp)
            avg_improvement = np.mean(valid_imp)
            lines.append(f"  Total Improvement: {total_improvement*100:+.1f}%")
            lines.append(f"  Avg per Round:     {avg_improvement*100:+.1f}%")

            # Identify diminishing returns
            if len(valid_imp) >= 2:
                if valid_imp[-1] < valid_imp[0]:
                    lines.append("  Trend: Diminishing returns observed")
                else:
                    lines.append("  Trend: Consistent or increasing improvement")

    lines.append("")

    # Per-round details
    lines.append("=" * 70)
    lines.append("PER-ROUND DETAILS")
    lines.append("=" * 70)

    for i, round_num in enumerate(rounds):
        lines.append(f"\nRound {round_num}")
        lines.append("-" * 40)

        # Table header
        header = f"{'Client':<15} {'Pre-FL':>10} {'Post-FL':>10} {'Improve':>10}"
        lines.append(header)
        lines.append("-" * 45)

        for client_id in sorted(pre_fl_data.keys(), key=lambda x: int(x)):
            client_name = client_names.get(client_id, f"Client {client_id}")[:14]

            pre_acc = pre_fl_data[client_id][i]
            post_acc = post_fl_data[client_id][i]
            imp = improvement_data[client_id][i]

            pre_str = f"{pre_acc*100:.1f}%" if pre_acc is not None else "N/A"
            post_str = f"{post_acc*100:.1f}%" if post_acc is not None else "N/A"
            imp_str = f"{imp*100:+.1f}%" if imp is not None else "N/A"

            lines.append(f"{client_name:<15} {pre_str:>10} {post_str:>10} {imp_str:>10}")

        # Round averages
        valid_pre = [pre_fl_data[cid][i] for cid in pre_fl_data if pre_fl_data[cid][i] is not None]
        valid_post = [post_fl_data[cid][i] for cid in post_fl_data if post_fl_data[cid][i] is not None]
        valid_imp = [improvement_data[cid][i] for cid in improvement_data if improvement_data[cid][i] is not None]

        if valid_pre or valid_post or valid_imp:
            lines.append("-" * 45)
            avg_pre = f"{np.mean(valid_pre)*100:.1f}%" if valid_pre else "N/A"
            avg_post = f"{np.mean(valid_post)*100:.1f}%" if valid_post else "N/A"
            avg_imp = f"{np.mean(valid_imp)*100:+.1f}%" if valid_imp else "N/A"
            lines.append(f"{'AVERAGE':<15} {avg_pre:>10} {avg_post:>10} {avg_imp:>10}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    # Write to file
    summary_text = "\n".join(lines)
    path = os.path.join(output_dir, "training_summary.txt")

    with open(path, 'w') as f:
        f.write(summary_text)

    print(f"[Graphs] Saved text summary: {path}")

    # Also print to console
    print("\n" + summary_text)

    return path


def main():
    """Main entry point for graph generation."""
    parser = argparse.ArgumentParser(
        description="Generate visualization graphs for FLEX-Med FL round metrics"
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default=ROUND_METRICS_FILE_PATH,
        help=f"Path to round_metrics.json (default: {ROUND_METRICS_FILE_PATH})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=GRAPHS_OUTPUT_DIR,
        help=f"Output directory for graphs (default: {GRAPHS_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=CLIENT_INFO_FILE_PATH,
        help=f"Path to client config data.json (default: {CLIENT_INFO_FILE_PATH})"
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Disable value annotations on data points"
    )
    parser.add_argument(
        "--annotate-combined",
        action="store_true",
        help="Enable annotations on combined view (disabled by default)"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[Graphs] Output directory: {args.output_dir}")

    # Load data
    metrics = load_round_metrics(args.metrics_path)
    client_names = load_client_names(args.config_path)

    # Extract round data
    rounds, pre_fl_data, post_fl_data, improvement_data = extract_round_data(metrics)

    if not rounds:
        print("[Graphs] Error: No round data found in metrics file")
        return

    print(f"[Graphs] Processing {len(rounds)} rounds for {len(pre_fl_data)} clients")

    # Determine annotation settings
    annotate_individual = not args.no_annotate
    annotate_combined = args.annotate_combined

    print(f"[Graphs] Annotations: Individual={annotate_individual}, Combined={annotate_combined}")

    # Generate graphs
    print("\n[Graphs] Generating individual graphs per client (with annotations)...")
    generate_individual_graphs(
        rounds, pre_fl_data, post_fl_data, improvement_data,
        client_names, args.output_dir, annotate=annotate_individual
    )

    print("\n[Graphs] Generating combined view per client...")
    generate_combined_graph_per_client(
        rounds, pre_fl_data, post_fl_data, improvement_data,
        client_names, args.output_dir, annotate=annotate_combined
    )

    print("\n[Graphs] Generating text summary...")
    generate_text_summary(
        rounds, pre_fl_data, post_fl_data, improvement_data,
        client_names, args.output_dir
    )

    print("\n" + "=" * 70)
    print("[Graphs] All visualizations generated successfully!")
    print(f"[Graphs] Output location: {args.output_dir}")
    print("=" * 70)
    print("\nGenerated structure:")
    print(f"  - {args.output_dir}/training_summary.txt")
    for client_id in sorted(pre_fl_data.keys(), key=lambda x: int(x)):
        print(f"  - {args.output_dir}/client_{client_id}/")
        print(f"      - client_{client_id}_pre_fl_accuracy.png")
        print(f"      - client_{client_id}_post_fl_accuracy.png")
        print(f"      - client_{client_id}_improvement.png")
        print(f"      - client_{client_id}_combined_metrics.png")


if __name__ == "__main__":
    main()
