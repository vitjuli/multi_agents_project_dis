# R0_rerun_s42 — Baseline reproducibility (seed 42)

## Idea
Determine whether the K=2 baseline accuracy collapse (R0_baseline) is a deterministic
training-dynamics outcome or a one-off failure, by re-training the as-shipped baseline from
scratch under the identical configuration and shuffle seed.

## Method
- Identical to R0_baseline: NUM_GENERATIONS=2 (K), μ=1, β=0.08, ε=0.2, RANK=ALPHA=64,
  LR=3e-6, MAX_GRAD_NORM=0.1, NUM_BATCHES=3738 ⇒ MAX_STEPS=3364, shuffle seed 42.
- Plain `python train.py` (no env overrides); checkpoints retained at steps 2000/2500/3000/3364.
- Each retained checkpoint evaluated greedily with `eval_ckpt.py`.
- **Mechanism.** With a fixed shuffle seed the training dynamics are deterministic, so the collapse trajectory is exactly reproducible.

## Results
Held-out GSM8K, greedy, 64 prompts. Trajectory is identical to R0_baseline at every checkpoint:

| step | base | 2000 | 2500 | 3000 | 3364 (final) |
|---|---|---|---|---|---|
| accuracy | 48.44% | 28.12% | 20.31% | 6.25% | **3.12%** |
| format | 9.4% | 35.9% | 31.3% | 12.5% | 12.5% |

Final-step 95% CI: 3.12% [0.0, 7.8] (identical to R0_baseline).

## Discussion
The collapse reproduces bit-for-bit: identical accuracy at each checkpoint, not only at the
final step. The K=2 baseline collapse is therefore **deterministic for a fixed shuffle seed**,
not a transient failure. Combined with R5_baseline_s1 (seed 1 → 40.62%, no collapse), the precise
statement is that the K=2 outcome is **seed-dependent yet deterministic per seed**. This supports
the high-variance reading Var(Â)=(K−1)/K=½ (I.3(d) / I.4 Q1).

## Provenance & reproduction
- Same baseline recipe as R0_baseline; default CKPT_DIR=/tmp/content/ckpts/.
- Artifacts: `eval_results/R0_rerun_s42_step{2000,2500,3000,3364}_summary.json` (+ per-prompt CSVs);
  `_rerun_artifacts.tgz` (transfer provenance).
```bash
source ~/venvs/tunix/bin/activate
cd ~/tpu-2026/scripts && set -a; . ./.env; set +a
rm -rf data && python train.py                 # seed 42, CKPT_DIR=/tmp/content/ckpts/
for s in 2000 2500 3000 3364; do
  rm -rf data && python eval_ckpt.py --step=$s --ckpt=/tmp/content/ckpts/actor --out=$HOME/r$s
done
```
