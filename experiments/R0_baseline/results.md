# R0 — Baseline GRPO (K=2), as-shipped

**This is the reference run.** Stock recipe, no changes (the `scripts/` here is the unmodified baseline).

## Config (what defines this experiment)
- `NUM_GENERATIONS = 2` (group size K), `NUM_ITERATIONS = 1` (μ=1 ⇒ PPO clip inactive)
- `BETA = 0.08` (KL), `EPSILON = 0.2`, `RANK = ALPHA = 64`, `LR = 3e-6`, `MAX_GRAD_NORM = 0.1`
- `NUM_BATCHES = 3738` ⇒ `MAX_STEPS = 3364`; learner = `GRPOLearner` / `GRPOConfig`
- `CKPT_DIR = /tmp/content/ckpts_R0_baseline/`

## Provenance
- W&B: `R0_baseline_s0` — https://wandb.ai/chfazhvi/chfazhvi-grpo/runs/lbypybq2
- Baseline commit: `77c5a67` · Steps: 3364 (~5 h on v6e-1)

## Results (held-out GSM8K, greedy, 64 test prompts; via `scripts/eval_ckpt.py`)
| step | base | 2000 | 2500 | 3000 | 3364 (final) |
|------|------|------|------|------|------|
| accuracy | 48.44% | 28.12% | 20.31% | 6.25% | **3.12%** |
| format | 9.38% | 35.94% | 31.25% | 12.50% | 12.50% |

**Reward (final):** ~1.56 mean, **extremely noisy** (−2 … 10). `pg_clipfrac = 0` throughout (clip inactive at μ=1).
Some eval completions empty (`eval/min_length ≈ 0`). W&B held-out `rewards/eval/mean` peaks ≈step 800 (~1.38) then decays.

## 🔴 Finding
GRPO at K=2 **over-optimises** the format shaping reward and **collapses** GSM8K accuracy (48% → 3%). Root cause:
highest advantage variance `Var(Â)=(K−1)/K=½` + frequent degenerate groups → unstable updates. This motivates
both improvements (Dr.GRPO, K=4) and an **early stop** (~step 800 would have kept a far better model).

## Eval artifacts (machine-generated — not hand-typed)
`eval_results/` holds the script output that produced the numbers above:
- `R0_baseline_summary.json` — metrics + provenance (model, ckpt, `restored_step`, timestamp).
- `R0_baseline_per_prompt.csv` — all 64 test prompts: gold, extracted answer, correct/partial/format, Q+response snippet.
- `base_summary.json` / `base_per_prompt.csv` — the untrained base (the comparison point).

**Bootstrap 95% CI** (10k resamples, 64 prompts): R0 final **3.12%** [0.0, 7.8] vs base **48.44%** [35.9, 60.9]
— non-overlapping ⇒ the collapse is statistically real. Full table: `../../figs/accuracy_ci.md` (`../../bootstrap_ci.py`).

## How to reproduce
```bash
source ~/venvs/tunix/bin/activate
cd scripts && set -a; . ./.env; set +a   # WANDB/HF/PROTOCOL_BUFFERS=python
rm -rf data
python train.py                          # MAX_STEPS=3364
# eval (after training, TPU free):
python eval_ckpt.py --policy base --preset greedy
python eval_ckpt.py --policy lora --ckpt /tmp/content/ckpts_R0_baseline/actor --preset greedy
```
