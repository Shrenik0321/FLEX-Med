ontext │
│ │
│ Problem: Log5 (fixed focal alpha=0.5) shows oscillations in rounds 7-10 and overfitting. Log6 (data-driven focal alpha) │
│ made things SIGNIFICANTLY WORSE — final eval dropped from 79-90% (log5) to 72-85% (log6). Lanka was EXCLUDED from │
│ consensus in R1 of log6. │
│ │
│ Root cause of log6 regression: Data-driven focal alpha sets α≈0.21-0.27 based on minority ratio. Combined with │
│ WeightedRandomSampler (which already over-samples minority class in each batch), this DOUBLE-COMPENSATES — producing │
│ extremely high Healthy accuracy (88-98%) but crashing Leukemia accuracy (55-82%). │
│ │
│ Root cause of log5 oscillations: │
│ - LR decay 0.95 is too gentle — R10 LR is still 0.00063 (needs to be ~0.00035) │
│ - Consensus momentum is fixed at 0.20 — mature consensus in later rounds swings too much │
│ - Distillation targets become unstable as rounds progress │
│ │
│ 3 High-Impact Changes (Priority Order) │
│ │
│ Change 1: Revert Focal Alpha to Fixed 0.5 (CRITICAL) │
│ │
│ File: backend/federated_learning/flex_med/client_app.py (lines 180-217) │
│ │
│ Set USE_DATA_DRIVEN_FOCAL_ALPHA = False. This reverts to the log5 behavior which was clearly better. │
│ │
│ Why: WeightedRandomSampler handles class imbalance at the BATCH level (more minority samples per batch). Focal alpha │
│ handles it at the LOSS level. Doing both is redundant and harmful. With fixed α=0.5, focal loss focuses purely on hard │
│ examples via gamma=2.0, which is its actual strength. │
│ │
│ Change 2: Stronger LR Decay (0.95 → 0.88) │
│ │
│ File: backend/federated_learning/pyproject.toml (line 40) │
│ │
│ Change lr-decay = 0.95 to lr-decay = 0.88. │
│ │
│ Effect on learning rates: │
│ │
│ ┌───────┬────────────────┬────────────┐ │
│ │ Round │ Current (0.95) │ New (0.88) │ │
│ ├───────┼────────────────┼────────────┤ │
│ │ R1 │ 0.00100 │ 0.00100 │ │
│ ├───────┼────────────────┼────────────┤ │
│ │ R5 │ 0.00081 │ 0.00060 │ │
│ ├───────┼────────────────┼────────────┤ │
│ │ R8 │ 0.00070 │ 0.00040 │ │
│ ├───────┼────────────────┼────────────┤ │
│ │ R10 │ 0.00063 │ 0.00028 │ │
│ └───────┴────────────────┴────────────┘ │
│ │
│ This directly reduces oscillation amplitude in later rounds where models are already at 85-90% accuracy and need │
│ fine-tuning, not aggressive updates. The key: rounds 7-10 in log5 show val loss bouncing (0.0500→0.0514→0.0487→0.0522 for │
│ Asiri) — this is the LR being too high for the loss landscape at that stage. │
│ │
│ Change 3: Progressive Consensus Momentum (0.20 → 0.15-0.45) │
│ │
│ File: backend/federated_learning/flex_med/task.py — compute_consensus() function (around line 862) │
│ │
│ Currently: consensus_logits = momentum _ last_consensus + (1 - momentum) _ new_consensus with fixed momentum=0.20. │
│ │
│ Change to progressive momentum that increases with round number: │
│ # Early rounds: trust new data more (low momentum) │
│ # Late rounds: stabilize consensus (high momentum) │
│ progress = (server_round - 1) / max(total_rounds - 1, 1) │
│ effective_momentum = 0.15 + 0.30 \* progress # 0.15 at R1 → 0.45 at R10 │
│ │
│ Why this matters: The consensus is the DISTILLATION TARGET — every client learns from it. If it swings too much between │
│ rounds, clients chase a moving target, creating oscillation. In later rounds, the consensus should be more stable (higher │
│ momentum = more weight on previous consensus). │
│ │
│ Implementation: Pass total_rounds into compute_consensus() from FLEXMedStrategy.aggregate_train(). The strategy already │
│ knows num_rounds. │
│ │
│ Files to Modify │
│ │
│ 1. backend/federated_learning/flex_med/client_app.py — Line 180: set USE_DATA_DRIVEN_FOCAL_ALPHA = False │
│ 2. backend/federated_learning/pyproject.toml — Line 40: lr-decay = 0.88 │
│ 3. backend/federated_learning/flex_med/task.py — compute_consensus() function: add progressive momentum logic. Also │
│ update FLEXMedStrategy.aggregate_train() to pass num_rounds context. │
│ │
│ What NOT to Change │
│ │
│ - Freeze strategy: 2-phase (20% classifier only, 80% last block+classifier) is working well │
│ - Batch size, local epochs: These are fine at 32 and 3 │
│ - Distillation weight/decay: Current 0.45 base / 0.2 decay is reasonable │
│ - Minority boost (0.70): This is a sensible moderate value for WeightedRandomSampler │
│ - AdaptiveDropout: Working correctly, responds to val loss trends │
│ │
│ Verification │
│ │
│ Run a 10-round FL simulation with Dirichlet α=2.5 and compare: │
│ - Training accuracy curves should be monotonically increasing (or near-monotonic) │
│ - Val loss should decrease steadily, not oscillate in R7-10 │
│ - Public test EVAL gaps should stay below 15% in later rounds │
│ - No clients should be EXCLUDED from consensus │
│ - Final global eval should match or exceed log5 (Asiri ~80%, Lanka ~87%, Delmon ~90%) │
│ │
│ Dirichlet Sensitivity Notes │
│ │
│ For α=1.0 (more heterogeneous): consensus momentum should start higher (0.25→0.50) and distillation weight base should be │
│ increased (0.55-0.60) to provide stronger federated guidance against local drift. │
│ │
│ For α=5.0 (less heterogeneous): current settings would work well since data is near-IID. Could even reduce consensus │
│ momentum range to (0.10→0.30). │
│ │
│ The architecture is NOT only optimal for α=2.5 — the progressive momentum approach naturally adapts. The LR decay and │
│ focal loss settings are data-distribution-agnostic.
