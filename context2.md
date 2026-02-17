Context │
│ │
│ Comparing the old run (complete, 10 rounds) against the new run (in progress, 5/10 rounds complete) with the same data distribution │
│ (Dirichlet alpha=1.0, seed=42, 3 clients). The new run removed Symmetric Boost and Dynamic Focal Alpha in favor of simpler focal alpha │
│ clamped to [0.25, 0.75], added WeightedRandomSampler, cosine annealing LR, and simplified focal loss. │
│ │
│ --- │
│ 1. ROUND-BY-ROUND COMPARISON TABLES │
│ │
│ Asiri (MobileNetV2) - 2564 samples, Ratio 6.7:1 │
│ │
│ ┌───────┬───────────────┬───────────────┬──────────────┬──────────────┬──────────┬──────────┬─────────┬─────────┐ │
│ │ Round │ OLD Train Acc │ NEW Train Acc │ OLD Val Loss │ NEW Val Loss │ OLD Eval │ NEW Eval │ OLD Gap │ NEW Gap │ │
│ ├───────┼───────────────┼───────────────┼──────────────┼──────────────┼──────────┼──────────┼─────────┼─────────┤ │
│ │ 1 │ 83.97% │ 59.65% │ 0.0583 │ 0.0520 │ 69.9% │ 75.0% │ 41.8% │ 23.9% │ │
│ ├───────┼───────────────┼───────────────┼──────────────┼──────────────┼──────────┼──────────┼─────────┼─────────┤ │
│ │ 2 │ 84.55% │ 61.71% │ 0.0663 │ 0.0532 │ 79.1% │ 75.0% │ 23.2% │ 12.2% │ │
│ ├───────┼───────────────┼───────────────┼──────────────┼──────────────┼──────────┼──────────┼─────────┼─────────┤ │
│ │ 3 │ 86.75% │ 70.22% │ 0.0491 │ 0.0553 │ 85.6% │ 83.4% │ 15.0% │ 6.3% │ │
│ ├───────┼───────────────┼───────────────┼──────────────┼──────────────┼──────────┼──────────┼─────────┼─────────┤ │
│ │ 4 │ 88.79% │ 73.31% │ 0.0731 │ 0.0542 │ 86.5% │ 86.1% │ 2.7% │ 3.3% │ │
│ ├───────┼───────────────┼───────────────┼──────────────┼──────────────┼──────────┼──────────┼─────────┼─────────┤ │
│ │ 5 │ 89.17% │ 75.91% │ 0.0791 │ 0.0506 │ 87.3% │ 79.4% │ 7.0% │ 27.7% │ │
│ └───────┴───────────────┴───────────────┴──────────────┴──────────────┴──────────┴──────────┴─────────┴─────────┘ │
│ │
│ Delmon (ResNet50) - 2270 samples, Ratio 21.3:1 │
│ │
│ ┌───────┬───────────────┬───────────────┬──────────────┬──────────────┬──────────┬──────────┬─────────┬─────────┐ │
│ │ Round │ OLD Train Acc │ NEW Train Acc │ OLD Val Loss │ NEW Val Loss │ OLD Eval │ NEW Eval │ OLD Gap │ NEW Gap │ │
│ ├───────┼───────────────┼───────────────┼──────────────┼──────────────┼──────────┼──────────┼─────────┼─────────┤ │
│ │ 1 │ 94.71% │ 75.14% │ 0.0156 │ 0.0547 │ 94.4% │ 54.7% │ 0.7% │ 47.5% │ │
│ ├───────┼───────────────┼───────────────┼──────────────┼──────────────┼──────────┼──────────┼─────────┼─────────┤ │
│ │ 2 │ 94.40% │ 77.26% │ 0.0131 │ 0.0524 │ 88.6% │ 47.4% │ 12.0% │ 55.2% │ │
│ ├───────┼───────────────┼───────────────┼──────────────┼──────────────┼──────────┼──────────┼─────────┼─────────┤ │
│ │ 3 │ 94.40% │ 89.63% │ 0.0079 │ 0.0316 │ 96.2% │ 87.1% │ 2.6% │ 13.5% │ │
│ ├───────┼───────────────┼───────────────┼──────────────┼──────────────┼──────────┼──────────┼─────────┼─────────┤ │
│ │ 4 │ 95.82% │ 93.79% │ 0.0082 │ 0.0178 │ 96.8% │ 90.1% │ 3.2% │ 10.4% │ │
│ ├───────┼───────────────┼───────────────┼──────────────┼──────────────┼──────────┼──────────┼─────────┼─────────┤ │
│ │ 5 │ 96.28% │ 95.07% │ 0.0069 │ 0.0283 │ 98.5% │ 90.6% │ 5.0% │ 9.8% │ │
│ └───────┴───────────────┴───────────────┴──────────────┴──────────────┴──────────┴──────────┴─────────┴─────────┘ │
│ │
│ Lanka (DenseNet121) - 3940 samples, Ratio 1.3:1 │
│ │
│ ┌───────┬───────────────┬───────────────┬──────────────┬──────────────┬──────────┬──────────┬─────────┬─────────┐ │
│ │ Round │ OLD Train Acc │ NEW Train Acc │ OLD Val Loss │ NEW Val Loss │ OLD Eval │ NEW Eval │ OLD Gap │ NEW Gap │ │
│ ├───────┼───────────────┼───────────────┼──────────────┼──────────────┼──────────┼──────────┼─────────┼─────────┤ │
│ │ 1 │ 75.07% │ 76.27% │ 0.0679 │ 0.0623 │ 88.8% │ 67.3% │ 79.1% │ 19.2% │ │
│ ├───────┼───────────────┼───────────────┼──────────────┼──────────────┼──────────┼──────────┼─────────┼─────────┤ │
│ │ 2 │ 78.15% │ 76.60% │ 0.0610 │ 0.0620 │ 88.6% │ 63.1% │ 76.5% │ 33.2% │ │
│ ├───────┼───────────────┼───────────────┼──────────────┼──────────────┼──────────┼──────────┼─────────┼─────────┤ │
│ │ 3 │ 83.32% │ 82.58% │ 0.0681 │ 0.0622 │ 90.1% │ 63.6% │ 53.0% │ 32.6% │ │
│ ├───────┼───────────────┼───────────────┼──────────────┼──────────────┼──────────┼──────────┼─────────┼─────────┤ │
│ │ 4 │ 84.92% │ 85.00% │ 0.0464 │ 0.0438 │ 88.8% │ 72.7% │ 14.7% │ 22.1% │ │
│ ├───────┼───────────────┼───────────────┼──────────────┼──────────────┼──────────┼──────────┼─────────┼─────────┤ │
│ │ 5 │ 85.92% │ 85.67% │ 0.0433 │ 0.0592 │ 91.2% │ 74.5% │ 61.1% │ 15.5% │ │
│ └───────┴───────────────┴───────────────┴──────────────┴──────────────┴──────────┴──────────┴─────────┴─────────┘ │
│ │
│ --- │
│ 2. CRITICAL FINDINGS │
│ │
│ GOOD NEWS - What's Working │
│ │
│ A. Val Loss Stability (Asiri) - Major Improvement │
│ - OLD: Val loss climbed relentlessly: 0.0583 -> 0.0663 -> 0.0491 -> 0.0731 -> 0.0791 -> ... -> 0.0799 (round 10). Clear overfitting. │
│ - NEW: Val loss is remarkably stable: 0.0520 -> 0.0532 -> 0.0553 -> 0.0542 -> 0.0506. No overfitting signal at all. │
│ - This is the single biggest improvement. The cosine annealing + simplified focal loss is working. │
│ │
│ B. Class Gap (Lanka) - Dramatically Better │
│ - OLD Lanka was a disaster: Gap was 79.1% -> 76.5% -> 53% -> 14.7% -> 61.1% -> 13.8% -> 45.9% -> 42.7% -> 25% -> 31.5%. Wildly oscillating, │
│ never converged. │
│ - NEW Lanka: Gap is 19.2% -> 33.2% -> 32.6% -> 22.1% -> 15.5%. Clear downward trend, much more stable. │
│ - The WeightedRandomSampler is doing its job for the nearly-balanced Lanka client. │
│ │
│ C. Class Gap (Asiri) - Better Through Round 4 │
│ - OLD Asiri gap oscillated wildly: 41.8% -> 23.2% -> 15.0% -> 2.7% -> 7.0% -> 21.8% -> 0.3% -> 13.0% -> 23.1% -> 5.1% │
│ - NEW Asiri gap was steadily declining: 23.9% -> 12.2% -> 6.3% -> 3.3% (then spiked to 27.7% in R5 - see concerns below) │
│ │
│ D. No Crash Risk │
│ - The old run completed all 10 rounds successfully with the same data/infra │
│ - No OOM indicators, no Ray failures, no GPU issues │
│ - The simulation will complete barring external factors (Colab timeout, etc.) │
│ │
│ BAD NEWS - Concerns │
│ │
│ A. Asiri Round 5 Regression (HIGH CONCERN) │
│ - Eval accuracy dropped from 86.1% (R4) to 79.4% (R5) │
│ - Gap spiked from 3.3% to 27.7% (Leukemia dropped to 67.3%, Healthy jumped to 95.0%) │
│ - This coincides with dropout DECREASING from 0.60 to 0.50 (adaptive dropout detected "improvement" in val loss and reduced dropout) │
│ - Root cause: The model flipped its bias - it was learning both classes well, then suddenly biased toward Healthy. This pattern appeared in │
│ old run too (R6: gap swung from 2.7% to 21.8%). │
│ - Prediction: This should partially recover in R6-7 as distillation corrects it, but expect oscillation │
│ │
│ B. Delmon Slow Start (MEDIUM CONCERN - Recovering) │
│ - Old Delmon started at 94.71% train acc; New started at 75.14% │
│ - Old eval was 94.4% in R1; New was 54.7% (nearly random) │
│ - BUT: Delmon is catching up fast - 95.07% train acc by R5, eval 90.6% │
│ - Root cause: Without the old "Symmetric Boost 2.0x" and the old Dynamic Focal Alpha (0.1812), the 21.3:1 imbalance ratio hits much harder. │
│ The WeightedRandomSampler alone wasn't enough in early rounds with frozen backbone. │
│ - Val loss went UP in R5 (0.0178 -> 0.0283) which needs monitoring │
│ │
│ C. Lanka's Val Loss Spike in R5 (MEDIUM CONCERN) │
│ - Val loss: 0.0623 -> 0.0620 -> 0.0622 -> 0.0438 -> 0.0592 (round 5 spike) │
│ - After finally dropping in R4, it jumped back up in R5 │
│ - Train accuracy still improved (85.67%) but the val loss divergence is an early overfitting signal │
│ - In the old run, Lanka's val loss also oscillated (0.0464 -> 0.0433 -> 0.0525 -> 0.0435 -> 0.0479 -> 0.0540 -> 0.0435), so this may be │
│ characteristic of the DenseNet121 architecture with this data │
│ │
│ D. Overall Eval Accuracy is Lower Than Old (for now) │
│ - At Round 5: OLD average eval = (87.3 + 98.5 + 91.2)/3 = 92.3%. NEW average eval = (79.4 + 90.6 + 74.5)/3 = 81.5% │
│ - ~10 percentage points behind old at the same round │
│ - But the old run had serious issues with class gap that you won't see in the aggregate numbers. The old Lanka had 99.1% Leukemia / 38% │
│ Healthy at R5 - that's not a useful model despite showing 91.2% overall. │
│ │
│ OVERFITTING RISK ASSESSMENT │
│ │
│ Client: Asiri │
│ Overfitting Risk: LOW │
│ Evidence: Val loss stable at ~0.05, no climbing trend. Cosine annealing working well. │
│ ──────────────────────────────────────── │
│ Client: Delmon │
│ Overfitting Risk: LOW-MEDIUM │
│ Evidence: Val loss increased R4->R5 (0.0178->0.0283). With 8 more rounds at this pace it could climb. But train loss still decreasing. │
│ Monitor │
│ closely. │
│ ──────────────────────────────────────── │
│ Client: Lanka │
│ Overfitting Risk: MEDIUM │
│ Evidence: Val loss spike in R5 (0.0438->0.0592). The oscillation pattern matches old run. DenseNet121 may need more regularization at this │
│ data │
│ distribution. │
│ │
│ --- │
│ 3. WILL YOU HIT YOUR TARGETS? │
│ │
│ Target: 75%+ accuracy with alpha=1.0 │
│ │
│ Asiri: Currently at 75.91% train, 79.4% eval (but volatile). Likely will hit 78-82% eval by round 10. WILL MEET TARGET but with some │
│ round-to-round volatility. │
│ │
│ Delmon: Currently at 95.07% train, 90.6% eval. Strong upward trajectory. Likely 93-97% by round 10. WILL EXCEED TARGET. │
│ │
│ Lanka: Currently at 85.67% train, 74.5% eval. The gap between train and eval suggests the model doesn't generalize as well to the public │
│ test set. May reach 76-80% eval by round 10. ON THE EDGE - could go either way. │
│ │
│ Target: Declining val loss (no overfitting) │
│ │
│ Asiri: YES - Val loss has been remarkably stable/declining (0.052 -> 0.051). Best performer here. │
│ │
│ Delmon: PARTIALLY - Was declining nicely until R5 uptick. Need to watch R6-10. │
│ │
│ Lanka: AT RISK - Val loss spiked in R5. If it continues rising in R6-7, this signals overfitting. │
│ │
│ Target: Gap < 15% best case │
│ │
│ Asiri: Hit 3.3% gap in R4 but regressed to 27.7% in R5. Volatile. In best rounds, YES. Consistently? NO. │
│ │
│ Delmon: Gap has been declining (47.5% -> 9.8%). Currently 9.8%. ON TRACK. │
│ │
│ Lanka: Gap declining (19.2% -> 15.5%). Almost there. CLOSE TO TARGET. │
│ │
│ --- │
│ 4. KEY DIFFERENCES BETWEEN OLD AND NEW RUNS EXPLAINED │
│ │
│ Change: Removed Symmetric Boost │
│ Impact: Delmon's slow start (21:1 ratio no longer gets 2x boost). But also means less artificial inflation of metrics. │
│ ──────────────────────────────────────── │
│ Change: Simplified Focal Alpha (clamped [0.25, 0.75]) │
│ Impact: Old used 0.3479 with 1.17x boost for Asiri, 0.1812 with 2.0x boost for Delmon. New uses 0.25 for both (clamped minimum). This │
│ reduces │
│ Leukemia bias, improving Healthy accuracy at cost of raw numbers. │
│ ──────────────────────────────────────── │
│ Change: WeightedRandomSampler │
│ Impact: Better intra-batch balance. Explains why Lanka's gap is dramatically better (nearly balanced client benefits most). │
│ ──────────────────────────────────────── │
│ Change: Cosine Annealing │
│ Impact: Smoother LR decay within each round. Explains Asiri's val loss stability. │
│ ──────────────────────────────────────── │
│ Change: No more extreme bias │
│ Impact: Old Lanka R1: 99.1% Leukemia / 20% Healthy. New Lanka R1: 64.8% / 84.0%. The model now sees both classes. │
│ │
│ --- │
│ 5. PREDICTIONS FOR ROUNDS 6-10 │
│ │
│ Based on the old run's trajectory and current trends: │
│ │
│ 1. Asiri will oscillate between 78-86% eval accuracy. The gap will swing. Expected final: ~82-85% eval with gap around 5-15%. │
│ 2. Delmon will continue climbing and likely hit 94-97% eval. Val loss should stabilize. Expected final: ~95-97% eval with gap < 10%. │
│ 3. Lanka is the wildcard. If val loss continues climbing, expect eval to plateau at 74-78%. If val loss stabilizes, could reach 80%+. │
│ Expected final: ~76-80% eval with gap 10-20%. │
│ 4. No crashes expected - same infra completed 10 rounds before. │
│ │
│ --- │
│ 6. BOTTOM LINE VERDICT │
│ │
│ The new run is headed in the RIGHT direction, but with trade-offs: │
│ │
│ - BETTER: Class balance, val loss stability, no catastrophic gap (old Lanka had 79% gap!) │
│ - WORSE: Lower raw accuracy numbers (especially early rounds), Delmon's slow start │
│ - SAME: Both runs show round-to-round volatility characteristic of FedMD with heterogeneous data │
│ │
│ For your thesis/paper: The new run tells a much better story. The old run had a Lanka model that was 99% on Leukemia and 20% on Healthy - │
│ that's clinically useless. The new run's Lanka at 72.5% Leukemia / 88% Healthy with declining gap is actually a more balanced, deployable │
│ model. │
│ │
│ Will it crash? No. Same infrastructure completed before. │
│ Will it overfit? Lanka is at risk. Asiri looks safe. Delmon looks safe. │
│ Will it hit 75%+? Asiri and Delmon: yes. Lanka: borderline, but the trend is positive.

      Lanka's real issue isn't just val loss - look at the distillation loss trend:

- R2: 0.0060, R3: 0.0045, R4: 0.0089, R5: 0.0115  


The distillation loss is increasing - Lanka is drifting away from federated consensus toward its local data bias. In Phase 2, 2.7M DenseNet params
get unfrozen and 3 epochs of private training overwhelms the distillation signal. The model "forgets" what it learned from consensus.

Two Changes (both generalizable)

1. lr-decay: 0.93 -> 0.88 (pyproject.toml)

This is your biggest lever. Current effective LR at each round:

┌───────┬──────────────────────┬───────────────────────┐
│ Round │ 0.93 decay (current) │ 0.88 decay (proposed) │
├───────┼──────────────────────┼───────────────────────┤
│ 1 │ 0.001000 │ 0.001000 │
├───────┼──────────────────────┼───────────────────────┤
│ 3 │ 0.000865 │ 0.000774 │
├───────┼──────────────────────┼───────────────────────┤
│ 5 │ 0.000748 │ 0.000610 │
├───────┼──────────────────────┼───────────────────────┤
│ 7 │ 0.000647 │ 0.000481 │
├───────┼──────────────────────┼───────────────────────┤
│ 10 │ 0.000520 │ 0.000316 │
└───────┴──────────────────────┴───────────────────────┘

By round 5 (where Lanka spiked), the LR is 18% lower. By round 10, it's 39% lower. This directly reduces how much private training can overwrite
distillation knowledge.

For alpha 2.5/5.0: Even better. Balanced data converges faster, so a lower late-round LR prevents overshooting once the model has already learned
well. You don't need aggressive LR when the data is well-distributed.

2. weight_decay: 0.01 -> 0.02 (task.py, lines 548 and 694)

More L2 regularization penalizes large weight updates, which is exactly what causes the distillation drift. DenseNet121's dense connections
especially benefit from this - all those skip-connection weights compound small updates.

For alpha 2.5/5.0: Stronger regularization is harmless with balanced data (model already learns well) and actively prevents the same overfitting
pattern from emerging at higher accuracy levels.

Why These Are Generalizable

- They don't target Lanka specifically - they apply to ALL clients uniformly
- They don't interfere with your class-balancing mechanisms (focal loss, WeightedRandomSampler)
- Lower LR + stronger regularization is the standard approach when you want to preserve convergence while reducing overfitting
- With more balanced data (alpha 2.5/5.0), both changes become even more appropriate because the models converge faster and need less aggressive
  learning

What I'd NOT Change

- Don't reduce local-epochs from 3 - that would hurt Asiri and Delmon at alpha=1.0 where they need more training per round
- Don't touch the distillation config (DISTILL_WEIGHT_BASE=0.45, DISTILL_DECAY_RATE=0.2) - those values are reasonable
- Don't change the freeze schedule - Phase 1/Phase 2 split is working well

● User declined to answer questions

● Now I have both complete logs. The new run (log2.txt) has actually completed all 10 rounds. Let me give you a thorough critical analysis.          
   
  ---                                                                                                                                                
  Training Progression Summary

  NEW Run (log2.txt) - Per Client Across Rounds

  Asiri (MobileNetV2, 6.7:1 ratio):

  ┌───────┬────────────┬───────────┬──────────┬──────────┬──────────┐
  │ Round │ Train Loss │ Train Acc │ Val Loss │ EVAL Acc │ EVAL Gap │
  ├───────┼────────────┼───────────┼──────────┼──────────┼──────────┤
  │ 1     │ 0.0701     │ 59.65%    │ 0.0520   │ 75.0%    │ 23.9%    │
  ├───────┼────────────┼───────────┼──────────┼──────────┼──────────┤
  │ 3     │ 0.0531     │ 70.22%    │ 0.0553   │ 83.4%    │ 6.3%     │
  ├───────┼────────────┼───────────┼──────────┼──────────┼──────────┤
  │ 5     │ 0.0469     │ 75.91%    │ 0.0506   │ 79.4%    │ 27.7%    │
  ├───────┼────────────┼───────────┼──────────┼──────────┼──────────┤
  │ 7     │ 0.0447     │ 78.37%    │ 0.0595   │ 85.1%    │ 11.3%    │
  ├───────┼────────────┼───────────┼──────────┼──────────┼──────────┤
  │ 9     │ 0.0444     │ 78.26%    │ 0.0607   │ 85.3%    │ 0.7%     │
  ├───────┼────────────┼───────────┼──────────┼──────────┼──────────┤
  │ 10    │ 0.0426     │ 79.32%    │ 0.0638   │ 83.4%    │ 19.1%    │
  ├───────┼────────────┼───────────┼──────────┼──────────┼──────────┤
  │ Final │            │           │          │ 70.13%   │          │
  └───────┴────────────┴───────────┴──────────┴──────────┴──────────┘

  Delmon (ResNet50, 21.3:1 ratio):

  ┌───────┬────────────┬───────────┬──────────┬──────────┬──────────┐
  │ Round │ Train Loss │ Train Acc │ Val Loss │ EVAL Acc │ EVAL Gap │
  ├───────┼────────────┼───────────┼──────────┼──────────┼──────────┤
  │ 1     │ 0.0486     │ 75.14%    │ 0.0547   │ 54.7%    │ 47.5%    │
  ├───────┼────────────┼───────────┼──────────┼──────────┼──────────┤
  │ 3     │ 0.0244     │ 89.63%    │ 0.0316   │ 87.1%    │ 13.5%    │
  ├───────┼────────────┼───────────┼──────────┼──────────┼──────────┤
  │ 5     │ 0.0145     │ 95.07%    │ 0.0283   │ 90.6%    │ 9.8%     │
  ├───────┼────────────┼───────────┼──────────┼──────────┼──────────┤
  │ 7     │ 0.0098     │ 96.65%    │ 0.0122   │ 95.0%    │ 5.2%     │
  ├───────┼────────────┼───────────┼──────────┼──────────┼──────────┤
  │ 9     │ 0.0105     │ 96.58%    │ 0.0122   │ 93.9%    │ 6.4%     │
  ├───────┼────────────┼───────────┼──────────┼──────────┼──────────┤
  │ 10    │ 0.0082     │ 97.29%    │ 0.0082   │ 96.8%    │ 3.4%     │
  ├───────┼────────────┼───────────┼──────────┼──────────┼──────────┤
  │ Final │            │           │          │ 96.78%   │          │
  └───────┴────────────┴───────────┴──────────┴──────────┴──────────┘

  Lanka (DenseNet121, 1.3:1 ratio):

  ┌───────┬────────────┬───────────┬──────────┬──────────┬──────────┐
  │ Round │ Train Loss │ Train Acc │ Val Loss │ EVAL Acc │ EVAL Gap │
  ├───────┼────────────┼───────────┼──────────┼──────────┼──────────┤
  │ 1     │ 0.0671     │ 76.27%    │ 0.0623   │ 67.3%    │ 19.2%    │
  ├───────┼────────────┼───────────┼──────────┼──────────┼──────────┤
  │ 3     │ 0.0513     │ 82.58%    │ 0.0622   │ 63.6%    │ 32.6%    │
  ├───────┼────────────┼───────────┼──────────┼──────────┼──────────┤
  │ 5     │ 0.0430     │ 85.67%    │ 0.0592   │ 74.5%    │ 15.5%    │
  ├───────┼────────────┼───────────┼──────────┼──────────┼──────────┤
  │ 7     │ 0.0397     │ 87.04%    │ 0.0461   │ 69.6%    │ 25.7%    │
  ├───────┼────────────┼───────────┼──────────┼──────────┼──────────┤
  │ 9     │ 0.0382     │ 87.91%    │ 0.0468   │ 80.8%    │ 6.0%     │
  ├───────┼────────────┼───────────┼──────────┼──────────┼──────────┤
  │ 10    │ 0.0366     │ 88.17%    │ 0.0518   │ 70.1%    │ 22.8%    │
  ├───────┼────────────┼───────────┼──────────┼──────────┼──────────┤
  │ Final │            │           │          │ 83.45%   │          │
  └───────┴────────────┴───────────┴──────────┴──────────┴──────────┘

  ---
  Critical Feedback

  1. OVERFITTING: Asiri is overfitting from Round 5 onwards

  This is the biggest red flag. Asiri's val loss trajectory:

  R1: 0.0520 → R5: 0.0506 (good decline) → R6: 0.0628 → R8: 0.0641 → R10: 0.0638

  Train loss kept dropping (0.0701→0.0426) while val loss reversed upward from round 5. The train-val loss gap widened from ~0.003 to ~0.021. This is
   textbook overfitting. The adaptive dropout ramped to 0.60 but isn't sufficient.

  Impact: Asiri's final model eval is 70.13% - the worst of all clients and below your 75% target. Compare: its training accuracy was 79.32%, a 9.2%
  gap to final eval.

  2. Lanka's EVAL accuracy is volatile and decoupled from training

  Lanka's training metrics look solid (monotonically improving), but its public test EVAL oscillates wildly:

  R4: 72.7% → R5: 74.5% → R6: 74.0% → R7: 69.6% → R8: 76.4% → R9: 80.8% → R10: 70.1%

  The final eval dropped 10.7% from round 9 to round 10. This suggests Lanka is learning its private data distribution well but the knowledge isn't
  generalizing to the public test set consistently. The distillation may be causing interference.

  3. Overall accuracy vs. your OLD run is significantly worse

  ┌────────┬───────────┬───────────┬────────┐
  │ Client │ OLD Final │ NEW Final │ Delta  │
  ├────────┼───────────┼───────────┼────────┤
  │ Asiri  │ 91.43%    │ 70.13%    │ -21.3% │
  ├────────┼───────────┼───────────┼────────┤
  │ Delmon │ 99.42%    │ 96.78%    │ -2.6%  │
  ├────────┼───────────┼───────────┼────────┤
  │ Lanka  │ 89.36%    │ 83.45%    │ -5.9%  │
  └────────┴───────────┴───────────┴────────┘

  The simplification of focal loss and removal of the Symmetric Boost hurt Asiri the most. In the old run, Asiri had Dynamic Focal Alpha=0.3479 +
  1.17x Symmetric Boost. In the new run, it's clamped to 0.2500 with no boost. That difference matters for a 6.7:1 ratio.

  4. Consensus post-FL test results - better class balance, but gaps still fail your target

  NEW consensus tests:
  - Test 1: 72.8% overall, Gap 28.0%
  - Test 2: 72.2% overall, Gap 41.9%
  - Test 3: 78.5% overall, Gap 23.2%

  OLD consensus tests:
  - Test 1: 69.6% overall, Gap 45.8%
  - Test 2: 67.7% overall, Gap 58.8%
  - Test 3: 81.1% overall, Gap 4.6%

  The NEW run is better on Tests 1-2 for both accuracy and gap. But none meet your <15% gap target. The old run's Test 3 was much better (4.6% gap vs
   23.2%).

  5. What's going RIGHT

  - No crash risk: Completed cleanly. No NaN, no OOM, no Ray failures.
  - Delmon recovered remarkably: Despite starting at 75.14% (vs 94.71% old), it reached 97.29% training accuracy and 96.78% eval. The simplified
  focal loss with 0.25 alpha floor works for Delmon.
  - Training losses are monotonically declining for all clients - no divergence.
  - Distillation losses are low and stable (~0.005-0.012 range) - the consensus mechanism is working.
  - Consensus improved where it matters: Tests 1 and 2 accuracy went up ~3-5% while gap decreased ~17-20%. The class balance trade-off is genuinely
  better.

  6. The fundamental tension you're seeing

  The old run had Asiri excluded from consensus in rounds 1-2 (prediction gap >80%) which let the other clients dominate consensus early. The new run
   qualifies everyone from round 1, which means Asiri's poor early predictions pollute the consensus for Lanka and Delmon.

  The old Asiri had inflated train accuracy (~90%) but terrible class balance (EVAL gaps of 31-79%). The new Asiri has lower but more honest accuracy
   (~78%) with better balance (EVAL gaps mostly 3-27%). This is actually a more truthful model, but it drags down the final eval score.

  ---
  Verdict for your goals

  ┌─────────────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────┐
  │                Goal                 │                                        Status                                         │
  ├─────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ 75%+ average accuracy               │ Asiri: 70.1% FAIL, Delmon: 96.8% PASS, Lanka: 83.5% PASS, Avg: ~83.5% PASS but uneven │
  ├─────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ Declining val loss (no overfitting) │ Asiri: FAILING (val loss rising), Delmon: PASS, Lanka: BORDERLINE                     │
  ├─────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ <15% class gap                      │ FAIL on all 3 consensus tests                                                         │
  ├─────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────┤
  │ No crashes                          │ PASS                                                                                  │
  └─────────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────┘

  What to consider for alpha=2.5 and 5.0 runs

  With higher alpha (less heterogeneity), Asiri's 6.7:1 issue will reduce naturally, so you'll likely hit your 80%+ and <15% gap targets there. The
  current alpha=1.0 results are a reasonable "worst case" demonstration.

  For the paper/presentation: Frame the alpha=1.0 result as showing that FedMD does improve all clients over rounds (train accuracy improved for
  everyone), but extreme data heterogeneity (21:1) still presents challenges for consensus generalization. The consensus tests showing 72-78% with
  23-42% gaps at alpha=1.0 vs your alpha=2.5/5.0 results will make a compelling heterogeneity spectrum argument.

  If you want to improve before the alpha=2.5/5.0 runs

  The biggest win would be addressing Asiri's overfitting. Options to discuss:
  1. More aggressive LR decay (cosine annealing might need a lower initial LR for high-imbalance clients)
  2. Restore some form of boosting for extreme imbalance clients (the 0.25 floor for focal alpha is too conservative for 6.7:1)
  3. Early stopping per client based on val loss trend

  But if the goal is just to prove improvement exists across heterogeneity levels, this alpha=1.0 run already provides that evidence. The training
  curves clearly show learning. The weakness is in generalization to the public test set, which is expected at extreme heterogeneity.