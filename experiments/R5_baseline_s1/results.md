# R5 — baseline, second seed (rigor)

**One factor changed vs baseline:** the training **data-shuffle seed** (42 → 1). Identical config otherwise
(K=2, β=0.08). Env-driven: `data.py` reads `SHUFFLE_SEED`. Launched via `run_overnight.sh` (R5 stage).

## Config
- K=2, μ=1, β=0.08, ε=0.2, **SHUFFLE_SEED=1**, NUM_BATCHES=3738 ⇒ MAX_STEPS=3364. `CKPT_DIR=ckpts_R5_baseline_s1`.
- W&B: `R5_baseline_s1`.

## Results (held-out GSM8K, greedy, 64 test; `eval_results/`)
| Model | Accuracy | Format | n/64 |
|-------|----------|--------|------|
| baseline K=2 **seed 42** (R0) | **3.12%** | 12.5% | 2 |
| baseline K=2 **seed 1** (this run) | **40.62%** | 65.6% | 26 |

## 🔴 Finding (the key rigor result)
**The K=2 baseline collapse is SEED-DEPENDENT.** Same recipe, only the data order differs: seed 42 collapses
to 3.1%, seed 1 does **not** (40.6%). So K=2 is a high-variance regime whose outcome (collapse or not) flips
with the seed — exactly why the brief asks for *seeds and a measure of uncertainty*. This second seed (a) is a
controlled rigor check, and (b) caveats our β-sweep finding: part of the β=0-vs-β=0.08 gap is this
seed-sensitivity, not β alone. The seed-robust conclusion is that a larger K (R6) removes the fragility.

## How to reproduce
```bash
cd scripts && tmux new -s r5
SHUFFLE_SEED=1 CKPT_DIR=/tmp/content/ckpts_R5_baseline_s1/ WANDB_NAME=R5_baseline_s1 python train.py
# (or: bash run_overnight.sh — runs R5 then R6)
# eval: python eval_ckpt.py --policy lora --ckpt ~/ckpts_R5_baseline_s1/actor --out eval_results/R5_baseline_s1
```
