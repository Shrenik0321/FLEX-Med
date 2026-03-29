#!/usr/bin/env python3
"""Generate 4 ablation study notebooks for FLEX-Med federated learning.

Reads fl_simulation_standalone.ipynb and creates:
  ablations/01_local_vs_fl_baseline.ipynb
  ablations/02_kd_ablation.ipynb
  ablations/03_wrs_ablation.ipynb
  ablations/04_alpha_sensitivity.ipynb
"""

import json
import os
import copy

OUTPUT_DIR = "ablations"
ORIGINAL_NB = "fl_simulation_standalone.ipynb"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(ORIGINAL_NB) as f:
    orig = json.load(f)

# ──────────────────────── helpers ────────────────────────

def src(idx):
    return ''.join(orig['cells'][idx]['source'])

def _lines(text):
    if not text:
        return [""]
    lines = text.split('\n')
    return [l + '\n' for l in lines[:-1]] + [lines[-1]]

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}

def code(text):
    return {"cell_type": "code", "metadata": {}, "source": _lines(text),
            "execution_count": None, "outputs": []}

def ccell(idx):
    return copy.deepcopy(orig['cells'][idx])

def save(name, cells):
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
            "colab": {"provenance": []}
        },
        "cells": cells
    }
    path = os.path.join(OUTPUT_DIR, name)
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1)
    size_kb = os.path.getsize(path) / 1024
    print(f"  {path}  ({len(cells)} cells, {size_kb:.0f} KB)")

# ──────────────────────── modified task.py ────────────────────────
# Add use_wrs, alpha, seed parameters to load_private_dataset

_t = src(10)

_t = _t.replace(
    'def load_private_dataset(partition_id: int, num_partitions: int, batch_size=32,\n'
    '                         config_path: str = CLIENT_INFO_FILE_PATH):',
    'def load_private_dataset(partition_id: int, num_partitions: int, batch_size=32,\n'
    '                         config_path: str = CLIENT_INFO_FILE_PATH, use_wrs: bool = True,\n'
    '                         alpha: float = None, seed: int = None):',
)
_t = _t.replace(
    '    if not os.path.exists(LOCAL_TRAIN_DATASET_PATH):\n'
    '        raise FileNotFoundError(f"Shared training dataset not found at {LOCAL_TRAIN_DATASET_PATH}")\n'
    '\n'
    '    cache_key = (LOCAL_TRAIN_DATASET_PATH, num_partitions, DIRICHLET_ALPHA, DIRICHLET_SEED)',
    '    if not os.path.exists(LOCAL_TRAIN_DATASET_PATH):\n'
    '        raise FileNotFoundError(f"Shared training dataset not found at {LOCAL_TRAIN_DATASET_PATH}")\n'
    '\n'
    '    alpha = alpha if alpha is not None else DIRICHLET_ALPHA\n'
    '    seed = seed if seed is not None else DIRICHLET_SEED\n'
    '\n'
    '    cache_key = (LOCAL_TRAIN_DATASET_PATH, num_partitions, alpha, seed)',
)
_t = _t.replace(
    '        partitioner, full_dataset = create_dirichlet_partitioner(\n'
    '            LOCAL_TRAIN_DATASET_PATH, num_partitions, DIRICHLET_ALPHA, DIRICHLET_SEED\n'
    '        )',
    '        partitioner, full_dataset = create_dirichlet_partitioner(\n'
    '            LOCAL_TRAIN_DATASET_PATH, num_partitions, alpha, seed\n'
    '        )',
)
_t = _t.replace(
    '    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)\n'
    '\n'
    '    trainloader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=2)',
    '    if use_wrs:\n'
    '        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)\n'
    '        trainloader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=2)\n'
    '    else:\n'
    '        trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)',
)
MOD_TASK = _t

# ──────────────────────── base cells ────────────────────────

def base_cells():
    """Return cells 1-9 from original + modified task.py cell."""
    cells = [ccell(i) for i in range(1, 10)]   # setup → helpers.py
    cells.append(code(MOD_TASK))                # modified task.py
    return cells

# ──────────────────────── health check (simplified) ────────────────────────

HEALTH_CHECK = r'''import os
from flex_med.utils.config import (
    PUBLIC_ANCHOR_DATASET_PATH, PUBLIC_TEST_DATASET_PATH,
    LOCAL_TRAIN_DATASET_PATH, DIRICHLET_ALPHA, DIRICHLET_SEED
)

print("=" * 60)
print("FLEX-Med Ablation Study — Health Check")
print("=" * 60)

for name, path in [("Public Anchor", PUBLIC_ANCHOR_DATASET_PATH),
                    ("Public Test", PUBLIC_TEST_DATASET_PATH),
                    ("Local Train", LOCAL_TRAIN_DATASET_PATH)]:
    exists = os.path.exists(path)
    status = "OK" if exists else "MISSING"
    count = ""
    if exists:
        total = sum(len(files) for _, _, files in os.walk(path))
        count = f" ({total} files)"
    print(f"  [{status}] {name}: {path}{count}")

print(f"\n  Dirichlet: alpha={DIRICHLET_ALPHA}, seed={DIRICHLET_SEED}")
print("=" * 60)
'''

# ──────────────────────── experiment runner ────────────────────────

RUNNER = r'''import torch
import numpy as np
import os
import time
import random
from flex_med.utils.config import (
    NUM_CLASSES, DIRICHLET_ALPHA, DIRICHLET_SEED
)
from flex_med.task import (
    create_dirichlet_partitioner, load_public_dataset, load_public_test_dataset,
    load_private_dataset, get_public_logits, distill_knowledge, train, test,
    compute_consensus, PARTITIONER_CACHE, LOCAL_TRAIN_DATASET_PATH
)
from flex_med.utils.helpers import (
    get_model_by_type, apply_freeze_strategy, save_model, PHASE2_START_ROUND
)

def set_seeds(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Default client configuration (matches production system)
CLIENT_CONFIGS = [
    {"id": 0, "client_name": "Asiri",  "model_type": "efficientnet_b0"},
    {"id": 1, "client_name": "Delmon", "model_type": "efficientnet_b1"},
    {"id": 2, "client_name": "Lanka",  "model_type": "efficientnet_b2"},
]
NUM_CLIENTS = len(CLIENT_CONFIGS)

def get_data_distribution(alpha=1.5, seed=42):
    """Print and return data distribution for each client partition."""
    PARTITIONER_CACHE.clear()
    dists = {}
    for i in range(NUM_CLIENTS):
        _, _, cc = load_private_dataset(i, NUM_CLIENTS, batch_size=32,
                                        alpha=alpha, seed=seed)
        total = int(sum(cc))
        dists[i] = {
            "total": total, "leukemia": int(cc[0]), "healthy": int(cc[1]),
            "leukemia_pct": round(cc[0] / total * 100, 1),
            "healthy_pct":  round(cc[1] / total * 100, 1),
        }
        print(f"  Client {i} ({CLIENT_CONFIGS[i]['client_name']}): "
              f"Total={total} | Leukemia={int(cc[0])} ({dists[i]['leukemia_pct']}%) | "
              f"Healthy={int(cc[1])} ({dists[i]['healthy_pct']}%)")
    PARTITIONER_CACHE.clear()
    return dists


def run_fl_experiment(
    num_rounds=10, local_epochs=2, batch_size=32,
    lr=0.0005, lr_decay=0.9,
    distill_lr=0.0005, distill_epochs=2, temperature=3.0,
    use_kd=True, use_wrs=True,
    alpha=1.5, seed=42,
    checkpoint_prefix="default",
):
    """
    Run a complete FL experiment with configurable components.

    Returns dict with final_results (per-client), avg_metrics, round_history,
    data_distribution, and config.
    """
    PARTITIONER_CACHE.clear()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seeds(seed)

    print(f"\n{'='*70}")
    print(f"FL EXPERIMENT: {checkpoint_prefix}")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Rounds={num_rounds}, LocalEpochs={local_epochs}, LR={lr}, "
          f"DistillLR={distill_lr}, Temp={temperature}")
    print(f"Flags: use_kd={use_kd}, use_wrs={use_wrs}, alpha={alpha}, seed={seed}")

    # ---- Data distribution ----
    print(f"\nData Distribution (alpha={alpha}):")
    data_dist = get_data_distribution(alpha, seed)

    # ---- Initialise models (pretrained) ----
    models = [get_model_by_type(cc['model_type']) for cc in CLIENT_CONFIGS]

    # ---- Public dataset & initial consensus ----
    public_loader = load_public_dataset(batch_size)
    num_public = len(public_loader.dataset)
    consensus = np.zeros((num_public, NUM_CLASSES), dtype=np.float32)

    round_history = []

    # ==================== FL ROUND LOOP ====================
    for rnd in range(1, num_rounds + 1):
        print(f"\n--- Round {rnd}/{num_rounds} ---")
        lr_factor = lr_decay ** (rnd - 1)
        cur_lr = lr * lr_factor
        cur_dlr = distill_lr * lr_factor

        round_logits, round_nexamples = [], []
        round_info = {"round": rnd, "clients": {}}

        for i, model in enumerate(models):
            cc = CLIENT_CONFIGS[i]
            model.to(device)
            model = apply_freeze_strategy(model, cc['model_type'], rnd, num_rounds)

            # Discriminative LRs
            if rnd < PHASE2_START_ROUND:
                cls_lr, bb_lr = cur_lr, 0.0
            else:
                cls_lr, bb_lr = cur_lr * 0.6, cur_lr * 0.03

            # -- Knowledge Distillation --
            d_loss = 0.0
            if use_kd and np.any(consensus != 0):
                d_cls_lr = cur_dlr
                d_bb_lr = 0.0 if rnd < PHASE2_START_ROUND else cur_dlr * 0.03
                d_loss = distill_knowledge(
                    model, public_loader, consensus, device, distill_epochs,
                    classifier_lr=d_cls_lr, backbone_lr=d_bb_lr,
                    temperature=temperature,
                    current_round=rnd, total_rounds=num_rounds,
                )

            # -- Local Training --
            trainloader, valloader, _ = load_private_dataset(
                i, NUM_CLIENTS, batch_size, use_wrs=use_wrs, alpha=alpha, seed=seed)
            t0 = time.time()
            tl, vl, ta, va = train(
                model, trainloader, local_epochs, cls_lr, bb_lr,
                device, valloader, rnd, num_rounds)
            dt = time.time() - t0

            print(f"  [{cc['client_name']}] distill={d_loss:.4f}  "
                  f"train_loss={tl:.4f} acc={ta:.2%}  "
                  f"val_loss={vl:.4f} acc={va:.2%}  ({dt:.1f}s)")

            # -- Public logits --
            logits = get_public_logits(model, public_loader, device)
            round_logits.append(logits)
            round_nexamples.append(len(trainloader.dataset))
            round_info["clients"][i] = dict(
                distill_loss=d_loss, train_loss=tl, val_loss=vl,
                train_acc=ta, val_acc=va)
            models[i] = model

        # -- Consensus --
        cm = [{"num-examples": n} for n in round_nexamples]
        consensus, _ = compute_consensus(round_logits, cm, CLIENT_CONFIGS, rnd)
        round_history.append(round_info)

    # ==================== FINAL EVALUATION ====================
    print(f"\n{'='*70}")
    print("FINAL EVALUATION (Public Test Set)")
    print(f"{'='*70}")
    test_loader = load_public_test_dataset(batch_size)
    final = {}
    for i, model in enumerate(models):
        cc = CLIENT_CONFIGS[i]
        model.to(device)
        r = test(model, test_loader, device, return_detailed=True)
        r['balanced_accuracy'] = round((r['recall'] + r['specificity']) / 2, 6)
        final[i] = r
        cm = r['confusion_matrix']
        print(f"  {cc['client_name']} ({cc['model_type']}):  "
              f"Acc={r['accuracy']:.4f}  Prec={r['precision']:.4f}  "
              f"Rec={r['recall']:.4f}  F1={r['f1_score']:.4f}  "
              f"Spec={r['specificity']:.4f}  BalAcc={r['balanced_accuracy']:.4f}")
        print(f"    CM: TP={cm['TP']} FP={cm['FP']} FN={cm['FN']} TN={cm['TN']}")

    # averages
    avg = {}
    for m in ['accuracy','precision','recall','f1_score','specificity','balanced_accuracy','loss']:
        avg[m] = round(sum(final[i][m] for i in range(NUM_CLIENTS)) / NUM_CLIENTS, 6)
    print(f"\n  AVERAGE: " + "  ".join(f"{k}={v:.4f}" for k, v in avg.items()))

    # save checkpoints
    ckdir = f"checkpoints/{checkpoint_prefix}"
    os.makedirs(ckdir, exist_ok=True)
    for i, model in enumerate(models):
        save_model(model, f"{ckdir}/{CLIENT_CONFIGS[i]['client_name']}.pt",
                   CLIENT_CONFIGS[i]['model_type'])

    return dict(final_results=final, avg_metrics=avg,
                round_history=round_history, data_distribution=data_dist,
                config=dict(num_rounds=num_rounds, local_epochs=local_epochs,
                            lr=lr, distill_lr=distill_lr, temperature=temperature,
                            use_kd=use_kd, use_wrs=use_wrs, alpha=alpha, seed=seed))


def run_local_only_experiment(
    total_epochs=20, batch_size=32,
    lr=0.0005, lr_decay=0.9,
    use_wrs=True, alpha=1.5, seed=42,
    checkpoint_prefix="local_only",
    num_rounds_equiv=10,
):
    """
    Local-only baseline: each client trains independently, no FL / distillation.
    Training budget is matched to FL: total_epochs = rounds * local_epochs.
    Freeze schedule is mapped proportionally to epoch blocks.
    """
    PARTITIONER_CACHE.clear()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seeds(seed)

    epochs_per_block = max(1, total_epochs // num_rounds_equiv)

    print(f"\n{'='*70}")
    print(f"LOCAL-ONLY EXPERIMENT: {checkpoint_prefix}")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"TotalEpochs={total_epochs} ({num_rounds_equiv} blocks x {epochs_per_block} epochs)")
    print(f"LR={lr}, use_wrs={use_wrs}, alpha={alpha}, seed={seed}")

    print(f"\nData Distribution (alpha={alpha}):")
    data_dist = get_data_distribution(alpha, seed)

    final = {}

    for ci, cc in enumerate(CLIENT_CONFIGS):
        print(f"\n{'='*60}")
        print(f"Client {ci} ({cc['client_name']}, {cc['model_type']})")
        print(f"{'='*60}")

        set_seeds(seed)  # Reset seed per client for reproducibility
        model = get_model_by_type(cc['model_type'])
        model.to(device)

        trainloader, valloader, _ = load_private_dataset(
            ci, NUM_CLIENTS, batch_size, use_wrs=use_wrs, alpha=alpha, seed=seed)

        for blk in range(num_rounds_equiv):
            pseudo_round = blk + 1
            model = apply_freeze_strategy(model, cc['model_type'],
                                          pseudo_round, num_rounds_equiv)
            lr_factor = lr_decay ** blk
            cur_lr = lr * lr_factor
            if pseudo_round < PHASE2_START_ROUND:
                cls_lr, bb_lr = cur_lr, 0.0
            else:
                cls_lr, bb_lr = cur_lr * 0.6, cur_lr * 0.03

            tl, vl, ta, va = train(
                model, trainloader, epochs_per_block, cls_lr, bb_lr,
                device, valloader, pseudo_round, num_rounds_equiv)
            print(f"  Block {pseudo_round}/{num_rounds_equiv}: "
                  f"train_loss={tl:.4f} acc={ta:.2%}  val_loss={vl:.4f} acc={va:.2%}")

        # Evaluate
        test_loader = load_public_test_dataset(batch_size)
        r = test(model, test_loader, device, return_detailed=True)
        r['balanced_accuracy'] = round((r['recall'] + r['specificity']) / 2, 6)
        final[ci] = r
        cm = r['confusion_matrix']
        print(f"\n  FINAL: Acc={r['accuracy']:.4f}  F1={r['f1_score']:.4f}  "
              f"Rec={r['recall']:.4f}  Spec={r['specificity']:.4f}")
        print(f"    CM: TP={cm['TP']} FP={cm['FP']} FN={cm['FN']} TN={cm['TN']}")

        ckdir = f"checkpoints/{checkpoint_prefix}"
        os.makedirs(ckdir, exist_ok=True)
        save_model(model, f"{ckdir}/{cc['client_name']}.pt", cc['model_type'])

    avg = {}
    for m in ['accuracy','precision','recall','f1_score','specificity','balanced_accuracy','loss']:
        avg[m] = round(sum(final[i][m] for i in range(NUM_CLIENTS)) / NUM_CLIENTS, 6)
    print(f"\n  AVERAGE: " + "  ".join(f"{k}={v:.4f}" for k, v in avg.items()))

    return dict(final_results=final, avg_metrics=avg,
                data_distribution=data_dist,
                config=dict(total_epochs=total_epochs, lr=lr,
                            use_wrs=use_wrs, alpha=alpha, seed=seed))

print("Experiment runner loaded.")
print(f"Clients: {', '.join(c['client_name']+' ('+c['model_type']+')' for c in CLIENT_CONFIGS)}")
'''

# ──────────────────────── comparison / plotting ────────────────────────

COMPARISON = r'''import pandas as pd

METRICS_LIST = ['accuracy', 'precision', 'recall', 'f1_score',
                'specificity', 'balanced_accuracy', 'loss']

def comparison_table(res_a, res_b, label_a, label_b):
    """Build a DataFrame comparing two experiment results."""
    rows = []
    for i in range(NUM_CLIENTS):
        cc = CLIENT_CONFIGS[i]
        for m in METRICS_LIST:
            va = res_a['final_results'][i].get(m, 0)
            vb = res_b['final_results'][i].get(m, 0)
            rows.append({
                'Client': f"{cc['client_name']} ({cc['model_type']})",
                'Metric': m, label_a: round(va, 4),
                label_b: round(vb, 4), 'Delta': round(vb - va, 4)})
    for m in METRICS_LIST:
        va = res_a['avg_metrics'].get(m, 0)
        vb = res_b['avg_metrics'].get(m, 0)
        rows.append({'Client': 'AVERAGE', 'Metric': m,
                     label_a: round(va, 4), label_b: round(vb, 4),
                     'Delta': round(vb - va, 4)})
    return pd.DataFrame(rows)

def print_summary(res_a, res_b, label_a, label_b):
    """Print concise average-metric comparison."""
    print(f"\n{'='*80}")
    print(f"SUMMARY: {label_a}  vs  {label_b}")
    print(f"{'='*80}")
    print(f"{'Metric':<22} {label_a:>14} {label_b:>14} {'Delta':>14}")
    print("-" * 70)
    for m in METRICS_LIST:
        va = res_a['avg_metrics'].get(m, 0)
        vb = res_b['avg_metrics'].get(m, 0)
        print(f"{m:<22} {va:>14.4f} {vb:>14.4f} {vb-va:>+14.4f}")
    print("=" * 70)

print("Comparison utilities loaded.")
'''

PLOTTING = r'''import matplotlib
matplotlib.use("Agg")          # non-interactive backend for Colab
import matplotlib.pyplot as plt
import numpy as np

def plot_comparison_bars(res_a, res_b, label_a, label_b, title="Comparison",
                         save_path=None):
    """Grouped bar charts comparing two experiments across all metrics."""
    metrics = ['accuracy','precision','recall','f1_score','specificity','balanced_accuracy']
    labels  = ['Accuracy','Precision','Recall','F1','Specificity','Bal. Acc.']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    for idx, (m, lab) in enumerate(zip(metrics, labels)):
        ax = axes[idx // 3][idx % 3]
        x = np.arange(NUM_CLIENTS + 1)
        w = 0.35
        va = [res_a['final_results'][i].get(m, 0) for i in range(NUM_CLIENTS)]
        va.append(res_a['avg_metrics'].get(m, 0))
        vb = [res_b['final_results'][i].get(m, 0) for i in range(NUM_CLIENTS)]
        vb.append(res_b['avg_metrics'].get(m, 0))
        names = [c['client_name'] for c in CLIENT_CONFIGS] + ['Average']

        ax.bar(x - w/2, va, w, label=label_a, color='#2196F3', alpha=0.8)
        ax.bar(x + w/2, vb, w, label=label_b, color='#FF9800', alpha=0.8)
        ax.set_ylabel(lab); ax.set_title(lab)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha='right')
        ax.legend(); ax.set_ylim(0, 1.05); ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_confusion_matrices(results, label, save_path=None):
    """Plot confusion matrices for all clients."""
    fig, axes = plt.subplots(1, NUM_CLIENTS, figsize=(6*NUM_CLIENTS, 5))
    if NUM_CLIENTS == 1: axes = [axes]
    for i, ax in enumerate(axes):
        cm = results['final_results'][i]['confusion_matrix']
        mat = np.array([[cm['TP'], cm['FN']], [cm['FP'], cm['TN']]])
        im = ax.imshow(mat, cmap='Blues', interpolation='nearest')
        ax.set_title(f"{CLIENT_CONFIGS[i]['client_name']}\n({CLIENT_CONFIGS[i]['model_type']})")
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(['Leukemia','Healthy'])
        ax.set_yticklabels(['Leukemia','Healthy'])
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(mat[r,c]), ha='center', va='center',
                        fontsize=14, fontweight='bold',
                        color='white' if mat[r,c] > mat.max()/2 else 'black')
        plt.colorbar(im, ax=ax)
    fig.suptitle(f"Confusion Matrices — {label}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_round_history(res, label, metric='val_acc', save_path=None):
    """Plot a training metric across rounds for each client."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(NUM_CLIENTS):
        vals = [rh['clients'][i][metric] for rh in res['round_history']]
        ax.plot(range(1, len(vals)+1), vals, marker='o',
                label=f"{CLIENT_CONFIGS[i]['client_name']}")
    ax.set_xlabel('Round'); ax.set_ylabel(metric)
    ax.set_title(f"{metric} per Round — {label}")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_alpha_comparison(all_results, save_path=None):
    """Line plot of average metrics vs alpha for sensitivity analysis."""
    alphas = sorted(all_results.keys())
    metrics = ['accuracy','precision','recall','f1_score','specificity','balanced_accuracy']
    labels  = ['Accuracy','Precision','Recall','F1','Specificity','Bal. Acc.']

    fig, ax = plt.subplots(figsize=(10, 6))
    for m, lab in zip(metrics, labels):
        vals = [all_results[a]['avg_metrics'].get(m, 0) for a in alphas]
        ax.plot(alphas, vals, marker='o', linewidth=2, label=lab)

    ax.set_xlabel('Dirichlet Alpha (lower = more heterogeneous)', fontsize=12)
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Data Heterogeneity Sensitivity Analysis', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    ax.set_xticks(alphas)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

print("Plotting utilities loaded.")
'''

# ──────────────────────── config logging cell ────────────────────────

CONFIG_LOG = r'''print("=" * 60)
print("EXPERIMENT CONFIGURATION")
print("=" * 60)
print(f"  Seed:            42")
print(f"  Dirichlet Alpha: 1.5 (default)")
print(f"  Num Clients:     {NUM_CLIENTS}")
for cc in CLIENT_CONFIGS:
    print(f"    - {cc['client_name']}: {cc['model_type']}")
print(f"  FL Rounds:       10")
print(f"  Local Epochs:    2 per round")
print(f"  Batch Size:      32")
print(f"  Learning Rate:   0.0005 (decay 0.9/round)")
print(f"  Distill LR:      0.0005")
print(f"  Distill Epochs:  2")
print(f"  Temperature:     3.0")
print("=" * 60)
'''

# ══════════════════════════════════════════════════════════════════════
# NOTEBOOK 1 — LOCAL vs FL BASELINE
# ══════════════════════════════════════════════════════════════════════

NB1_HEADER = r'''# Experiment 1: Local-Only Baseline vs Federated Learning

## Objective
Determine whether Federated Learning (FL) with FedMD-style knowledge distillation improves classification performance over independent local training, given the same total training budget and data partitions.

## Controlled Variables
- Dirichlet alpha: 1.5
- Random seed: 42
- Number of clients: 3
- Models: EfficientNet-B0 (Asiri), EfficientNet-B1 (Delmon), EfficientNet-B2 (Lanka)
- Total training epochs per client: 20 (10 rounds × 2 local epochs)
- Learning rate: 0.0005 with 0.9 decay per round
- Batch size: 32
- WeightedRandomSampler: enabled
- Progressive unfreezing: 2-phase strategy (head-only → 50% backbone)
- Data partitions: identical (same alpha & seed)

## Changed Variable
**Communication paradigm**: Local-only (no FL) vs Full FL pipeline (local training + logit distillation + consensus aggregation).

## Default Configuration
- Alpha: 1.5
- Seed: 42
- Number of Clients: 3
- Models: EfficientNet-B0, B1, B2
- Rounds: 10
- Local Epochs: 2 per round (20 total)

## Evaluation Metrics
- Accuracy, Precision (Leukemia = positive class), Recall, F1-score
- Specificity, Balanced Accuracy, Confusion Matrix, Loss

## Notes for Examiners
This notebook is part of a controlled experimental evaluation. Only one variable is changed (presence/absence of federated communication) while all other parameters are held constant. The local-only baseline uses an equivalent training budget of 20 epochs, with the same progressive unfreezing schedule mapped to proportional epoch blocks.'''

NB1_RUN_LOCAL = r'''# ========================================
# Experiment A — Local-Only Training
# ========================================
# Each client trains INDEPENDENTLY on its Dirichlet partition.
# NO communication, NO distillation, NO consensus.
# Training budget: 20 epochs (10 blocks × 2 epochs), matching FL.

print("Starting Experiment A: Local-Only Training...")
print("Each client trains independently — no FL communication.\n")

results_local = run_local_only_experiment(
    total_epochs=20,         # 10 rounds × 2 local epochs
    batch_size=32,
    lr=0.0005,
    lr_decay=0.9,
    use_wrs=True,
    alpha=1.5,
    seed=42,
    checkpoint_prefix="nb01_local_only",
    num_rounds_equiv=10,
)
print("\nExperiment A (Local-Only) complete.")
'''

NB1_RUN_FL = r'''# ========================================
# Experiment B — Full Federated Learning
# ========================================
# Standard FL pipeline: local training + consensus + distillation.
# Same data partitions, same models, same hyperparameters.

print("Starting Experiment B: Full Federated Learning...")
print("Standard FL pipeline with knowledge distillation.\n")

results_fl = run_fl_experiment(
    num_rounds=10,
    local_epochs=2,
    batch_size=32,
    lr=0.0005,
    lr_decay=0.9,
    distill_lr=0.0005,
    distill_epochs=2,
    temperature=3.0,
    use_kd=True,
    use_wrs=True,
    alpha=1.5,
    seed=42,
    checkpoint_prefix="nb01_fl_full",
)
print("\nExperiment B (FL) complete.")
'''

NB1_COMPARE = r'''# ========================================
# Results Comparison: Local vs FL
# ========================================

df = comparison_table(results_local, results_fl, "Local-Only", "FL (Full)")
print("\n--- Per-Client Detailed Comparison ---")
display(df)

print_summary(results_local, results_fl, "Local-Only", "FL (Full)")
'''

NB1_PLOT = r'''# ========================================
# Visualizations
# ========================================

plot_comparison_bars(results_local, results_fl, "Local-Only", "FL (Full)",
                     title="Local-Only vs Federated Learning",
                     save_path="nb01_local_vs_fl_bars.png")

plot_confusion_matrices(results_local, "Local-Only",
                        save_path="nb01_cm_local.png")
plot_confusion_matrices(results_fl, "FL (Full)",
                        save_path="nb01_cm_fl.png")

# Round-by-round training curves (FL only — local has no rounds)
plot_round_history(results_fl, "FL (Full)", metric='val_acc',
                   save_path="nb01_fl_round_history.png")
plot_round_history(results_fl, "FL (Full)", metric='train_loss',
                   save_path="nb01_fl_round_train_loss.png")
'''

NB1_CONCLUSION = r'''# ========================================
# Conclusion
# ========================================
#
# This experiment compares independent local training against federated
# learning with knowledge distillation (FedMD).
#
# Key questions answered:
# 1. Does FL improve over local-only training?
# 2. Which clients benefit most from FL?
# 3. Is the improvement consistent across metrics?
#
# The comparison uses identical data partitions, model architectures,
# training budget, and hyperparameters — the ONLY difference is the
# presence of federated communication (consensus + distillation).

delta_acc = results_fl['avg_metrics']['accuracy'] - results_local['avg_metrics']['accuracy']
delta_f1  = results_fl['avg_metrics']['f1_score'] - results_local['avg_metrics']['f1_score']
delta_bal = results_fl['avg_metrics']['balanced_accuracy'] - results_local['avg_metrics']['balanced_accuracy']

print(f"FL improvement over Local-Only:")
print(f"  Accuracy:          {delta_acc:+.4f}")
print(f"  F1 Score:          {delta_f1:+.4f}")
print(f"  Balanced Accuracy: {delta_bal:+.4f}")
if delta_acc > 0:
    print("\n  => Federated Learning IMPROVES over local-only training.")
else:
    print("\n  => Local-only training matches or exceeds FL in this setting.")
'''

# ══════════════════════════════════════════════════════════════════════
# NOTEBOOK 2 — KD ABLATION
# ══════════════════════════════════════════════════════════════════════

NB2_HEADER = r'''# Experiment 2: Impact of Knowledge Distillation (KD Ablation)

## Objective
Quantify the contribution of logit-based knowledge distillation to federated learning performance. By disabling KD while keeping all other FL components intact, we isolate the effect of consensus-based distillation.

## Controlled Variables
- Dirichlet alpha: 1.5
- Random seed: 42
- Number of clients: 3
- Models: EfficientNet-B0 (Asiri), EfficientNet-B1 (Delmon), EfficientNet-B2 (Lanka)
- FL rounds: 10
- Local epochs: 2 per round
- Learning rate: 0.0005 with 0.9 decay
- Batch size: 32
- WeightedRandomSampler: enabled
- Progressive unfreezing: 2-phase strategy
- Data partitions: identical (same alpha & seed)
- Consensus computation: runs in both cases (logits still collected)

## Changed Variable
**Knowledge Distillation**: Enabled (full KL-divergence distillation from consensus) vs Disabled (no distillation step, only local supervised training + logit sharing).

## Default Configuration
- Alpha: 1.5
- Seed: 42
- Number of Clients: 3
- Models: EfficientNet-B0, B1, B2
- Rounds: 10
- Local Epochs: 2 per round

## Evaluation Metrics
- Accuracy, Precision (Leukemia = positive class), Recall, F1-score
- Specificity, Balanced Accuracy, Confusion Matrix, Loss

## Notes for Examiners
This notebook is part of a controlled experimental evaluation. Only one variable is changed (presence/absence of knowledge distillation) while all other parameters are held constant. Both experiments still participate in FL rounds and share public logits; the only difference is whether each client distills from the consensus.'''

NB2_RUN_KD = r'''# ========================================
# Experiment A — FL with Knowledge Distillation (Full Pipeline)
# ========================================

print("Starting Experiment A: FL with KD (Full Pipeline)...")
print("Standard FL: local training + consensus + KL-divergence distillation.\n")

results_with_kd = run_fl_experiment(
    num_rounds=10, local_epochs=2, batch_size=32,
    lr=0.0005, lr_decay=0.9,
    distill_lr=0.0005, distill_epochs=2, temperature=3.0,
    use_kd=True,
    use_wrs=True, alpha=1.5, seed=42,
    checkpoint_prefix="nb02_with_kd",
)
print("\nExperiment A (With KD) complete.")
'''

NB2_RUN_NOKD = r'''# ========================================
# Experiment B — FL WITHOUT Knowledge Distillation
# ========================================
# The distillation step is entirely skipped.
# Clients still share logits and consensus is computed,
# but no client uses the consensus for training.

print("Starting Experiment B: FL without KD...")
print("Distillation disabled — only local supervised training.\n")

results_no_kd = run_fl_experiment(
    num_rounds=10, local_epochs=2, batch_size=32,
    lr=0.0005, lr_decay=0.9,
    distill_lr=0.0005, distill_epochs=2, temperature=3.0,
    use_kd=False,       # <-- ONLY CHANGE
    use_wrs=True, alpha=1.5, seed=42,
    checkpoint_prefix="nb02_no_kd",
)
print("\nExperiment B (No KD) complete.")
'''

NB2_COMPARE = r'''# ========================================
# Results Comparison: With KD vs Without KD
# ========================================

df = comparison_table(results_no_kd, results_with_kd, "No KD", "With KD")
print("\n--- Per-Client Detailed Comparison ---")
display(df)

print_summary(results_no_kd, results_with_kd, "No KD", "With KD")
'''

NB2_PLOT = r'''# ========================================
# Visualizations
# ========================================

plot_comparison_bars(results_no_kd, results_with_kd, "No KD", "With KD",
                     title="Impact of Knowledge Distillation",
                     save_path="nb02_kd_ablation_bars.png")

plot_confusion_matrices(results_with_kd, "With KD",
                        save_path="nb02_cm_with_kd.png")
plot_confusion_matrices(results_no_kd, "No KD",
                        save_path="nb02_cm_no_kd.png")

plot_round_history(results_with_kd, "With KD", metric='val_acc',
                   save_path="nb02_round_val_acc_kd.png")
plot_round_history(results_no_kd, "No KD", metric='val_acc',
                   save_path="nb02_round_val_acc_nokd.png")
'''

NB2_CONCLUSION = r'''# ========================================
# Conclusion
# ========================================

delta_acc = results_with_kd['avg_metrics']['accuracy'] - results_no_kd['avg_metrics']['accuracy']
delta_f1  = results_with_kd['avg_metrics']['f1_score'] - results_no_kd['avg_metrics']['f1_score']
delta_bal = results_with_kd['avg_metrics']['balanced_accuracy'] - results_no_kd['avg_metrics']['balanced_accuracy']

print(f"Impact of Knowledge Distillation:")
print(f"  Accuracy:          {delta_acc:+.4f}")
print(f"  F1 Score:          {delta_f1:+.4f}")
print(f"  Balanced Accuracy: {delta_bal:+.4f}")

# Per-client impact
print(f"\nPer-Client KD Impact (accuracy delta):")
for i in range(NUM_CLIENTS):
    cc = CLIENT_CONFIGS[i]
    d = results_with_kd['final_results'][i]['accuracy'] - results_no_kd['final_results'][i]['accuracy']
    print(f"  {cc['client_name']} ({cc['model_type']}): {d:+.4f}")

if delta_acc > 0:
    print(f"\n  => Knowledge Distillation IMPROVES FL performance by {delta_acc:.4f} accuracy.")
else:
    print(f"\n  => Knowledge Distillation does not improve FL in this setting.")
'''

# ══════════════════════════════════════════════════════════════════════
# NOTEBOOK 3 — WRS ABLATION
# ══════════════════════════════════════════════════════════════════════

NB3_HEADER = r'''# Experiment 3: Impact of Weighted Random Sampling (WRS Ablation)

## Objective
Evaluate the contribution of WeightedRandomSampler (WRS) as a class imbalance handling strategy in the federated learning pipeline. WRS oversamples the minority class during training; disabling it reveals how much performance depends on this balancing mechanism.

## Controlled Variables
- Dirichlet alpha: 1.5
- Random seed: 42
- Number of clients: 3
- Models: EfficientNet-B0 (Asiri), EfficientNet-B1 (Delmon), EfficientNet-B2 (Lanka)
- FL rounds: 10
- Local epochs: 2 per round
- Learning rate: 0.0005 with 0.9 decay
- Batch size: 32
- Knowledge Distillation: enabled
- Temperature: 3.0
- Progressive unfreezing: 2-phase strategy
- Data partitions: identical (same alpha & seed)

## Changed Variable
**Sampling strategy**: WeightedRandomSampler (minority oversampling) vs standard shuffled DataLoader (no class weighting).

## Default Configuration
- Alpha: 1.5
- Seed: 42
- Number of Clients: 3
- Models: EfficientNet-B0, B1, B2
- Rounds: 10
- Local Epochs: 2 per round

## Evaluation Metrics
- Accuracy, Precision (Leukemia = positive class), Recall, F1-score
- Specificity, Balanced Accuracy, Confusion Matrix, Loss
- **Focus**: Recall vs Specificity gap (class imbalance indicator)

## Notes for Examiners
This notebook is part of a controlled experimental evaluation. Only one variable is changed (sampling strategy) while all other parameters are held constant. Special attention is given to Recall (Leukemia detection) and Specificity (Healthy identification), as these are most affected by class imbalance handling.'''

NB3_RUN_WRS = r'''# ========================================
# Experiment A — FL with WeightedRandomSampler (WRS)
# ========================================

print("Starting Experiment A: FL with WRS...")
print("WeightedRandomSampler oversamples minority class.\n")

results_with_wrs = run_fl_experiment(
    num_rounds=10, local_epochs=2, batch_size=32,
    lr=0.0005, lr_decay=0.9,
    distill_lr=0.0005, distill_epochs=2, temperature=3.0,
    use_kd=True,
    use_wrs=True,       # <-- WRS enabled
    alpha=1.5, seed=42,
    checkpoint_prefix="nb03_with_wrs",
)
print("\nExperiment A (With WRS) complete.")
'''

NB3_RUN_NOWRS = r'''# ========================================
# Experiment B — FL WITHOUT WeightedRandomSampler
# ========================================
# Standard shuffled DataLoader replaces WRS.
# No class weighting — natural class distribution preserved.

print("Starting Experiment B: FL without WRS...")
print("Standard shuffled DataLoader, no class balancing.\n")

results_no_wrs = run_fl_experiment(
    num_rounds=10, local_epochs=2, batch_size=32,
    lr=0.0005, lr_decay=0.9,
    distill_lr=0.0005, distill_epochs=2, temperature=3.0,
    use_kd=True,
    use_wrs=False,      # <-- ONLY CHANGE
    alpha=1.5, seed=42,
    checkpoint_prefix="nb03_no_wrs",
)
print("\nExperiment B (No WRS) complete.")
'''

NB3_COMPARE = r'''# ========================================
# Results Comparison: With WRS vs Without WRS
# ========================================

df = comparison_table(results_no_wrs, results_with_wrs, "No WRS", "With WRS")
print("\n--- Per-Client Detailed Comparison ---")
display(df)

print_summary(results_no_wrs, results_with_wrs, "No WRS", "With WRS")
'''

NB3_PLOT = r'''# ========================================
# Visualizations
# ========================================

plot_comparison_bars(results_no_wrs, results_with_wrs, "No WRS", "With WRS",
                     title="Impact of Weighted Random Sampling",
                     save_path="nb03_wrs_ablation_bars.png")

plot_confusion_matrices(results_with_wrs, "With WRS",
                        save_path="nb03_cm_with_wrs.png")
plot_confusion_matrices(results_no_wrs, "No WRS",
                        save_path="nb03_cm_no_wrs.png")

# Focus: class gap analysis
print("\n" + "="*70)
print("CLASS GAP ANALYSIS (Recall vs Specificity)")
print("="*70)
print(f"{'Client':<30} {'WRS Gap':>12} {'No-WRS Gap':>12} {'Delta':>12}")
print("-" * 70)
for i in range(NUM_CLIENTS):
    cc = CLIENT_CONFIGS[i]
    gap_wrs = abs(results_with_wrs['final_results'][i]['recall'] -
                  results_with_wrs['final_results'][i]['specificity'])
    gap_nowrs = abs(results_no_wrs['final_results'][i]['recall'] -
                    results_no_wrs['final_results'][i]['specificity'])
    print(f"{cc['client_name']+' ('+cc['model_type']+')' :<30} "
          f"{gap_wrs:>12.4f} {gap_nowrs:>12.4f} {gap_nowrs-gap_wrs:>+12.4f}")
avg_gap_wrs = sum(abs(results_with_wrs['final_results'][i]['recall'] -
                      results_with_wrs['final_results'][i]['specificity'])
                  for i in range(NUM_CLIENTS)) / NUM_CLIENTS
avg_gap_nowrs = sum(abs(results_no_wrs['final_results'][i]['recall'] -
                        results_no_wrs['final_results'][i]['specificity'])
                    for i in range(NUM_CLIENTS)) / NUM_CLIENTS
print(f"{'AVERAGE':<30} {avg_gap_wrs:>12.4f} {avg_gap_nowrs:>12.4f} "
      f"{avg_gap_nowrs-avg_gap_wrs:>+12.4f}")
'''

NB3_CONCLUSION = r'''# ========================================
# Conclusion
# ========================================

delta_rec  = results_with_wrs['avg_metrics']['recall'] - results_no_wrs['avg_metrics']['recall']
delta_spec = results_with_wrs['avg_metrics']['specificity'] - results_no_wrs['avg_metrics']['specificity']
delta_bal  = results_with_wrs['avg_metrics']['balanced_accuracy'] - results_no_wrs['avg_metrics']['balanced_accuracy']
delta_f1   = results_with_wrs['avg_metrics']['f1_score'] - results_no_wrs['avg_metrics']['f1_score']

print(f"Impact of Weighted Random Sampling:")
print(f"  Recall (Leukemia):       {delta_rec:+.4f}")
print(f"  Specificity (Healthy):   {delta_spec:+.4f}")
print(f"  Balanced Accuracy:       {delta_bal:+.4f}")
print(f"  F1 Score:                {delta_f1:+.4f}")

if delta_bal > 0:
    print(f"\n  => WRS IMPROVES class balance (balanced accuracy +{delta_bal:.4f}).")
else:
    print(f"\n  => WRS does not improve class balance in this setting.")

print(f"\n  Class gap with WRS:    {avg_gap_wrs:.4f}")
print(f"  Class gap without WRS: {avg_gap_nowrs:.4f}")
if avg_gap_wrs < avg_gap_nowrs:
    print(f"  => WRS reduces the Recall-Specificity gap by {avg_gap_nowrs-avg_gap_wrs:.4f}.")
'''

# ══════════════════════════════════════════════════════════════════════
# NOTEBOOK 4 — ALPHA SENSITIVITY
# ══════════════════════════════════════════════════════════════════════

NB4_HEADER = r'''# Experiment 4: Data Heterogeneity Sensitivity (Alpha Test)

## Objective
Evaluate the robustness of the federated learning pipeline under varying levels of data heterogeneity, controlled by the Dirichlet alpha parameter. Lower alpha values produce more skewed (non-IID) data distributions across clients, simulating real-world hospital data heterogeneity.

## Controlled Variables
- Random seed: 42
- Number of clients: 3
- Models: EfficientNet-B0 (Asiri), EfficientNet-B1 (Delmon), EfficientNet-B2 (Lanka)
- FL rounds: 10
- Local epochs: 2 per round
- Learning rate: 0.0005 with 0.9 decay
- Batch size: 32
- Knowledge Distillation: enabled
- WeightedRandomSampler: enabled
- Temperature: 3.0
- Progressive unfreezing: 2-phase strategy

## Changed Variable
**Dirichlet alpha**: Controls the degree of data heterogeneity across clients.
- Alpha = 1.5 (baseline, moderate heterogeneity)
- Alpha = 1.0 (increased heterogeneity)
- Alpha = 0.5 (high heterogeneity, strongly non-IID)

## Default Configuration
- Seed: 42
- Number of Clients: 3
- Models: EfficientNet-B0, B1, B2
- Rounds: 10
- Local Epochs: 2 per round

## Evaluation Metrics
- Accuracy, Precision (Leukemia = positive class), Recall, F1-score
- Specificity, Balanced Accuracy, Confusion Matrix, Loss
- Per-client class distributions logged for each alpha

## Notes for Examiners
This notebook is part of a controlled experimental evaluation. Only the Dirichlet alpha parameter is changed between runs. NEW data partitions are generated for each alpha value (as required by different distribution parameters), but all other configuration is held constant. The seed ensures deterministic partitioning within each alpha setting.'''

NB4_RUN = r'''# ========================================
# Run FL for each alpha value
# ========================================
# NEW partitions are generated for each alpha (required by design).
# All other parameters remain constant.

ALPHAS = [1.5, 1.0, 0.5]
all_alpha_results = {}

for alpha in ALPHAS:
    print(f"\n{'#'*70}")
    print(f"# ALPHA = {alpha}")
    print(f"{'#'*70}")

    results = run_fl_experiment(
        num_rounds=10, local_epochs=2, batch_size=32,
        lr=0.0005, lr_decay=0.9,
        distill_lr=0.0005, distill_epochs=2, temperature=3.0,
        use_kd=True, use_wrs=True,
        alpha=alpha,
        seed=42,
        checkpoint_prefix=f"nb04_alpha_{str(alpha).replace('.','_')}",
    )
    all_alpha_results[alpha] = results
    print(f"\nAlpha={alpha} complete.")

print(f"\nAll {len(ALPHAS)} alpha experiments complete.")
'''

NB4_COMPARE = r'''# ========================================
# Results Comparison Across Alpha Values
# ========================================

print("\n" + "="*90)
print("ALPHA SENSITIVITY — SUMMARY TABLE")
print("="*90)

metrics = ['accuracy','precision','recall','f1_score','specificity','balanced_accuracy','loss']

# Header
header = f"{'Metric':<22}"
for a in ALPHAS:
    header += f" {'α='+str(a):>14}"
header += f" {'Δ(1.5→0.5)':>14}"
print(header)
print("-" * (22 + 15 * (len(ALPHAS) + 1)))

for m in metrics:
    row = f"{m:<22}"
    vals = []
    for a in ALPHAS:
        v = all_alpha_results[a]['avg_metrics'].get(m, 0)
        vals.append(v)
        row += f" {v:>14.4f}"
    delta = vals[-1] - vals[0]  # alpha=0.5 minus alpha=1.5
    row += f" {delta:>+14.4f}"
    print(row)

print("=" * (22 + 15 * (len(ALPHAS) + 1)))

# Per-client breakdown
for a in ALPHAS:
    print(f"\n--- Alpha = {a} ---")
    for i in range(NUM_CLIENTS):
        cc = CLIENT_CONFIGS[i]
        r = all_alpha_results[a]['final_results'][i]
        dist = all_alpha_results[a]['data_distribution'][i]
        print(f"  {cc['client_name']} ({cc['model_type']}): "
              f"Acc={r['accuracy']:.4f}  F1={r['f1_score']:.4f}  "
              f"Rec={r['recall']:.4f}  Spec={r['specificity']:.4f}  |  "
              f"Data: L={dist['leukemia_pct']}% H={dist['healthy_pct']}%")

# Data distribution comparison
print(f"\n\n{'='*90}")
print("DATA DISTRIBUTIONS ACROSS ALPHA VALUES")
print(f"{'='*90}")
for a in ALPHAS:
    print(f"\n  Alpha = {a}:")
    for i in range(NUM_CLIENTS):
        d = all_alpha_results[a]['data_distribution'][i]
        print(f"    {CLIENT_CONFIGS[i]['client_name']}: "
              f"Total={d['total']}  Leukemia={d['leukemia']} ({d['leukemia_pct']}%)  "
              f"Healthy={d['healthy']} ({d['healthy_pct']}%)")
'''

NB4_PLOT = r'''# ========================================
# Visualizations
# ========================================

# Alpha vs performance line plot
plot_alpha_comparison(all_alpha_results, save_path="nb04_alpha_sensitivity.png")

# Pairwise bar comparisons
plot_comparison_bars(
    all_alpha_results[1.5], all_alpha_results[0.5],
    "α=1.5 (Baseline)", "α=0.5 (High Het.)",
    title="Alpha 1.5 vs 0.5: High Heterogeneity Impact",
    save_path="nb04_alpha_1.5_vs_0.5.png")

# Confusion matrices for each alpha
for a in ALPHAS:
    plot_confusion_matrices(all_alpha_results[a], f"α={a}",
                            save_path=f"nb04_cm_alpha_{str(a).replace('.','_')}.png")

# Round-by-round comparison
fig, axes = plt.subplots(1, len(ALPHAS), figsize=(7*len(ALPHAS), 5))
for idx, a in enumerate(ALPHAS):
    ax = axes[idx]
    res = all_alpha_results[a]
    for i in range(NUM_CLIENTS):
        vals = [rh['clients'][i]['val_acc'] for rh in res['round_history']]
        ax.plot(range(1, len(vals)+1), vals, marker='o',
                label=CLIENT_CONFIGS[i]['client_name'])
    ax.set_xlabel('Round'); ax.set_ylabel('Val Accuracy')
    ax.set_title(f'α={a}'); ax.legend(); ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
plt.suptitle("Training Curves Across Alpha Values", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("nb04_round_curves_all_alphas.png", dpi=150, bbox_inches='tight')
plt.show()
'''

NB4_CONCLUSION = r'''# ========================================
# Conclusion
# ========================================

print("PERFORMANCE DEGRADATION ANALYSIS")
print("="*60)

baseline = all_alpha_results[1.5]['avg_metrics']
for a in ALPHAS:
    if a == 1.5:
        continue
    cur = all_alpha_results[a]['avg_metrics']
    print(f"\nAlpha {a} vs Baseline (1.5):")
    for m in ['accuracy', 'f1_score', 'balanced_accuracy']:
        delta = cur[m] - baseline[m]
        pct = (delta / baseline[m]) * 100 if baseline[m] != 0 else 0
        print(f"  {m}: {delta:+.4f} ({pct:+.1f}%)")

# Robustness score: how much does performance drop from α=1.5 to α=0.5?
drop_acc = all_alpha_results[0.5]['avg_metrics']['accuracy'] - baseline['accuracy']
drop_f1  = all_alpha_results[0.5]['avg_metrics']['f1_score'] - baseline['f1_score']

print(f"\nRobustness Summary (α=1.5 → α=0.5):")
print(f"  Accuracy drop:  {drop_acc:+.4f}")
print(f"  F1 Score drop:  {drop_f1:+.4f}")

if abs(drop_acc) < 0.05:
    print(f"\n  => Model is ROBUST to data heterogeneity (accuracy drop < 5%).")
elif abs(drop_acc) < 0.10:
    print(f"\n  => Moderate sensitivity to data heterogeneity (5-10% accuracy drop).")
else:
    print(f"\n  => HIGH sensitivity to data heterogeneity (>10% accuracy drop).")
'''

# ══════════════════════════════════════════════════════════════════════
# ASSEMBLE AND SAVE NOTEBOOKS
# ══════════════════════════════════════════════════════════════════════

print("Generating ablation study notebooks...\n")

# ---- NB 1 ----
cells_1 = [md(NB1_HEADER)]
cells_1 += base_cells()
cells_1 += [
    md("---\n## Health Check"),
    code(HEALTH_CHECK),
    md("---\n## Experiment Runner & Utilities"),
    code(RUNNER),
    code(COMPARISON),
    code(PLOTTING),
    code(CONFIG_LOG),
    md("---\n## Experiment A — Local-Only Baseline"),
    code(NB1_RUN_LOCAL),
    md("---\n## Experiment B — Federated Learning (Full Pipeline)"),
    code(NB1_RUN_FL),
    md("---\n## Results Comparison"),
    code(NB1_COMPARE),
    md("---\n## Visualizations"),
    code(NB1_PLOT),
    md("---\n## Conclusion"),
    code(NB1_CONCLUSION),
]
save("01_local_vs_fl_baseline.ipynb", cells_1)

# ---- NB 2 ----
cells_2 = [md(NB2_HEADER)]
cells_2 += base_cells()
cells_2 += [
    md("---\n## Health Check"),
    code(HEALTH_CHECK),
    md("---\n## Experiment Runner & Utilities"),
    code(RUNNER),
    code(COMPARISON),
    code(PLOTTING),
    code(CONFIG_LOG),
    md("---\n## Experiment A — FL with Knowledge Distillation"),
    code(NB2_RUN_KD),
    md("---\n## Experiment B — FL WITHOUT Knowledge Distillation"),
    code(NB2_RUN_NOKD),
    md("---\n## Results Comparison"),
    code(NB2_COMPARE),
    md("---\n## Visualizations"),
    code(NB2_PLOT),
    md("---\n## Conclusion"),
    code(NB2_CONCLUSION),
]
save("02_kd_ablation.ipynb", cells_2)

# ---- NB 3 ----
cells_3 = [md(NB3_HEADER)]
cells_3 += base_cells()
cells_3 += [
    md("---\n## Health Check"),
    code(HEALTH_CHECK),
    md("---\n## Experiment Runner & Utilities"),
    code(RUNNER),
    code(COMPARISON),
    code(PLOTTING),
    code(CONFIG_LOG),
    md("---\n## Experiment A — FL with WeightedRandomSampler"),
    code(NB3_RUN_WRS),
    md("---\n## Experiment B — FL WITHOUT WeightedRandomSampler"),
    code(NB3_RUN_NOWRS),
    md("---\n## Results Comparison"),
    code(NB3_COMPARE),
    md("---\n## Visualizations & Class Gap Analysis"),
    code(NB3_PLOT),
    md("---\n## Conclusion"),
    code(NB3_CONCLUSION),
]
save("03_wrs_ablation.ipynb", cells_3)

# ---- NB 4 ----
cells_4 = [md(NB4_HEADER)]
cells_4 += base_cells()
cells_4 += [
    md("---\n## Health Check"),
    code(HEALTH_CHECK),
    md("---\n## Experiment Runner & Utilities"),
    code(RUNNER),
    code(COMPARISON),
    code(PLOTTING),
    code(CONFIG_LOG),
    md("---\n## Run FL for Alpha = 1.5, 1.0, 0.5"),
    code(NB4_RUN),
    md("---\n## Results Comparison"),
    code(NB4_COMPARE),
    md("---\n## Visualizations"),
    code(NB4_PLOT),
    md("---\n## Conclusion"),
    code(NB4_CONCLUSION),
]
save("04_alpha_sensitivity.ipynb", cells_4)

print("\nDone! All 4 ablation notebooks generated in ablations/")
