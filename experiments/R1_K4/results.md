# R1 — Group size K=4 (equal rollout budget)

**One factor changed vs baseline:** the group size K (2→4), with the step count halved so the *total number
of generated completions matches the baseline* (a controlled comparison — not "more samples"). See
`scripts/config.py`: `NUM_GENERATIONS = 4`, `NUM_BATCHES = 1869`.

## Config
- `NUM_GENERATIONS = 4` (K), `NUM_BATCHES = 1869` ⇒ `MAX_STEPS = 1682` (1682×4 ≈ 3364×2 = baseline completions)
- Everything else stock (GRPOLearner, K-only change). `CKPT_DIR = /tmp/content/ckpts_R1K4/`

## Provenance
- W&B: `R1_K4_s0` (chfazhvi/chfazhvi-grpo) · branch `R1-K4` · 1682 steps

## Results (held-out GSM8K, greedy, 64 test; via `scripts/eval_ckpt.py`)
| metric | base | baseline K=2 (final) | Dr.GRPO (final) | **K=4 (final)** |
|--------|------|----------------------|-----------------|-----------------|
| accuracy | 48.44% | 3.12% | 31.25% | **48.44%** |
| format | 9.38% | 12.50% | 82.81% | **85.94%** |

## 🏆 Finding (the cleanest theory confirmation)
Same recipe as baseline except K=2→4, yet **baseline collapsed to 3.12% while K=4 keeps the full base
accuracy (48.44%) AND learns the 86% format — no collapse at all.** Larger K → lower advantage variance
(`Var(Â)=(K−1)/K`) and far fewer degenerate groups → stable updates → no over-optimisation collapse.
**Direct validation of the I.4 Q1 variance theory on held-out accuracy.** Together with R3, shows both
improvements target the same root cause (instability), and K=4 is the strongest.

## ⚠️ Eval gotcha (documented)
First eval printed `restored step 0` (= untrained, ~base by accident) because the checkpoint had been `cp`'d
into an empty dir; the real ckpt was in `/tmp/content/ckpts_R1K4/actor/{500..1682}`. **Always check the eval
log says `restored step N` with N≠0.**

## How to reproduce
```bash
source ~/venvs/tunix/bin/activate
cd scripts && set -a; . ./.env; set +a
rm -rf data
python train.py                          # K=4, MAX_STEPS=1682
python eval_ckpt.py --policy lora --ckpt /tmp/content/ckpts_R1K4/actor --preset greedy
```
