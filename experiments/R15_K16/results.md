# R15 — Group size K=16 (W&B-eval only)

## Idea
Extend the equal-rollout-budget K-sweep (K=2→4→8) to K=16 to test the variance-reduction
prediction Var(Â)=(K−1)/K, Var(ĝ)~O(1/K): more samples per prompt give a lower-variance advantage
estimate and therefore more stable training, not necessarily a large accuracy gain.

## Method
- NUM_GENERATIONS=16, μ=1 (clip dormant), β=0.08, seed 42, NUM_BATCHES=467 ⇒ **MAX_STEPS=420**,
  ≈0.9·467·16 = 6720 rollouts ≈ the 6728 of K=2/4/8 (equal rollout budget).
- Equal rollout budget implies higher K trains *fewer* steps (420 at K=16 vs 841 at K=8, 1682 at K=4).
- **Mechanism.** Increasing K reduces the variance of the advantage estimate (Var(Â)=(K−1)/K, per-sample gradient variance ~O(1/K)), which should improve training stability rather than necessarily raise accuracy.

## Results
**Protocol caveat:** the VM was deleted before a greedy `eval_ckpt.py` run, so K=16 has **no greedy
accuracy** and is excluded from the greedy accuracy table. It is reported only on the W&B
in-training eval (temperature-sampled reward components), where it is comparable to the other runs.

W&B in-training eval, mean reward across the sweep:

| K | run | step | score/mean | eval_len | KL | pg_clipfrac |
|---|---|---|---|---|---|---|
| 2 (seed 1) | R5 | 3364 | 4.43 | 390 | 0.58 | 0 |
| 4 | R1 | 1682 | 6.26 | 295 | 0.12 | 0 |
| 8 | R6 | 841 | 6.73 | 287 | 0.22 | 0 |
| **16** | **R15** | **420** | **7.37** | **260** | **0.05** | **0** |

The K=16 eval-reward trajectory is a stable plateau (≈7.0→7.5 over steps 64→384), with no collapse.

## Discussion
On the W&B-eval metric the sweep is monotone increasing in K, and K=16 is the most stable run:
highest reward (7.37), lowest KL (0.05), healthy length (~260), clip dormant (pg_clipfrac=0). This
is the clearest confirmation of the variance reading (larger K ⇒ lower-variance gradient ⇒ stabler
updates). We do **not** infer a diminishing-returns curve from these numbers, because the equal
rollout budget gives each run a different step count, which confounds a final-step marginal-gain
comparison. The defensible claim is stability, not a greedy-accuracy ranking.

## Provenance & reproduction
- W&B run: `R11_K16`, state finished, 420 steps (folder renumbered R11→R15; W&B run name unchanged).
- Artifacts: `wandb_results/R11_K16_summary.json`, `wandb_results/R11_K16_eval_trajectory.csv`;
  sweep-wide `wandb_results/wandb_summary_ksweep.json`.
```bash
bash ~/tpu-2026/scripts/run_k16.sh                 # K=16, MAX_STEPS=420, equal rollout budget
# re-pull W&B evidence (read-only, no VM):
python3 -m pip install wandb && python3 -m wandb login
cd experiments_repo && python3 analysis/fetch_k16.py
```
