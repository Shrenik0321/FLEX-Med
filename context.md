Context │
│ │
│ The FL system (FedMD with Dirichlet partitioning, alpha=1.0, 3 clients) fails to meet targets of 80%+ accuracy and <10% class gap. Current │
│ results: │
│ │
│ ┌────────┬──────────┬───────────┬─────────────────┐ │
│ │ Client │ Accuracy │ Class Gap │ Imbalance Ratio │ │
│ ├────────┼──────────┼───────────┼─────────────────┤ │
│ │ Asiri │ 69.6% │ 45.8% │ 6.7:1 │ │
│ ├────────┼──────────┼───────────┼─────────────────┤ │
│ │ Delmon │ 67.7% │ 58.8% │ 21:1 │ │
│ ├────────┼──────────┼───────────┼─────────────────┤ │
│ │ Lanka │ 81.1% │ 4.6% │ 1.28:1 │ │
│ └────────┴──────────┴───────────┴─────────────────┘ │
│ │
│ Lanka (nearly balanced) already hits targets. Asiri and Delmon fail specifically because of data imbalance - their models predict "leukemia" │
│ for everything. Additionally, rounds 9-10 show overfitting (val loss rising, accuracy declining). │
│ │
│ The single biggest problem: There is no WeightedRandomSampler in the training DataLoader. With 21:1 imbalance, batches are overwhelmingly │
│ leukemia samples. Focal loss alpha alone cannot compensate for this frequency gap. │
│ │
│ --- │
│ Changes (4 total, 4 files) │
│ │
│ 1. Add WeightedRandomSampler to training DataLoader [CRITICAL] │
│ │
│ File: backend/federated_learning/flex_med/task.py │
│ │
│ Line 11 - Add WeightedRandomSampler to existing import: │
│ from torch.utils.data import DataLoader, WeightedRandomSampler │
│ │
│ Lines 321-322 - Replace the trainloader creation: │
│ # BEFORE: │
│ trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2) │
│ │
│ # AFTER: │
│ class_sample_counts = torch.bincount(torch.tensor(train_targets), minlength=2).float() │
│ class_weights = 1.0 / class_sample_counts.clamp(min=1) │
│ sample_weights = class_weights[torch.tensor(train_targets)] │
│ sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True) │
│ trainloader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=2) │
│ │
│ Why: Converts effective batch distribution from 21:1 to ~1:1. This is the standard PyTorch approach for class-imbalanced datasets. Works │
│ regardless of Dirichlet alpha value. Note: sampler and shuffle=True are mutually exclusive in PyTorch. │
│ │
│ --- │
│ 2. Simplify focal alpha calculation │
│ │
│ File: backend/federated_learning/flex_med/client_app.py │
│ │
│ Lines 168-212 - Replace the entire dynamic focal alpha block (damping, boost, symmetric shift) with: │
│ n_leukemia = class_counts.get(0, 0) │
│ n_healthy = class_counts.get(1, 0) │
│ total_samples = n_leukemia + n_healthy │
│ │
│ if total_samples > 0 and n_leukemia > 0 and n_healthy > 0: │
│ # With WeightedRandomSampler handling frequency balance, │
│ # focal alpha only needs mild residual correction │
│ focal_alpha = n_healthy / total_samples │
│ focal_alpha = max(0.25, min(0.75, focal_alpha)) │
│ else: │
│ focal_alpha = 0.50 │
│ │
│ print(f"[{display_id} | {client_name}] Focal Alpha: {focal_alpha:.4f} (L:{n_leukemia}, H:{n_healthy})") │
│ │
│ Why: The sampler now handles the heavy lifting of class balance. The complex damping/boost/symmetric-shift logic was producing insufficient │
│ corrections (4.5:1 effective weight for 21:1 imbalance) and added instability. Clamping to [0.25, 0.75] prevents extreme values. │
│ │
│ --- │
│ 3. Add CosineAnnealingLR within training + steeper inter-round decay │
│ │
│ File 1: backend/federated_learning/flex_med/task.py │
│ │
│ In train() function, after optimizer creation (line 534), add: │
│ scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( │
│ optimizer, T_max=epochs, eta_min=effective_lr \* 0.1 │
│ ) │
│ │
│ After the validation block (line 571, where # REMOVED: scheduler.step(epoch_val_loss) comment is), replace comment with: │
│ scheduler.step() │
│ │
│ File 2: backend/federated_learning/pyproject.toml │
│ │
│ Line ~40 - Change: │
│ lr-decay = 0.93 # was 0.98 │
│ │
│ Why: Addresses the post-round-8 overfitting. CosineAnnealingLR decays LR smoothly within each round's local training. Steeper inter-round │
│ decay (0.93^9 = 0.52 vs 0.98^9 = 0.83) ensures LR is meaningfully reduced by late rounds. Together they prevent the "accuracy declining + │
│ loss increasing" pattern seen in rounds 9-10. │
│ │
│ --- │
│ 4. Tune consensus momentum and distillation weight │
│ │
│ File: backend/app/config.py │
│ │
│ Line 189 - Reduce consensus momentum: │
│ consensus_momentum: float = 0.25 # was 0.40 │
│ │
│ Lines 201-202 - Increase distillation strength: │
│ distill_weight_base: float = float(os.getenv("FLEX_MED_DISTILL_WEIGHT", "0.55")) # was 0.45 │
│ distill_decay_rate: float = float(os.getenv("FLEX_MED_DISTILL_DECAY", "0.12")) # was 0.2 │
│ │
│ Why: │
│ - Momentum 0.40 -> 0.25: Consensus currently drags 40% of old (poor) predictions forward. At 0.25, consensus is 75% driven by current round │
│ logits, more responsive to improving models. │
│ - Distill weight 0.45 -> 0.55, decay 0.2 -> 0.12: At round 10, distillation weight goes from 0.37 (current) to 0.49 (proposed). Stronger │
│ sustained distillation prevents imbalanced clients from drifting too far toward their skewed local distributions in late rounds. The │
│ consensus (built from balanced public data) acts as a regularizer. │
│ │
│ --- │
│ What is intentionally NOT changed │
│ │
│ - Freeze strategy (Phase 1/Phase 2 split is reasonable) │
│ - Gradient clipping, adaptive dropout, data augmentation │
│ - Quality gating in compute_consensus │
│ - Batch size, local epochs, temperature │
│ - Model architectures │
│ │
│ Verification │
│ │
│ 1. Run FL simulation with alpha=1.0 (current baseline) and compare: │
│ - All clients should achieve >80% accuracy │
│ - Class gap should be <10-15% even for the 21:1 client │
│ - Val loss should decrease monotonically (no R9-10 spike) │
│ 2. Run with alpha=0.5 (harder heterogeneity) to confirm robustness │
│ 3. Check confusion matrix: TN should be significantly higher for Asiri/Delmon
