# R4 (β=0.12) — tighter KL leash

**One factor changed vs baseline:** `BETA = 0.12` (tighter reference-KL leash, vs 0.08). Else K=2 baseline.
Env-driven; launched with `run_beta.sh 0.12`.

## Config
- K=2, μ=1, **β=0.12**, ε=0.2, NUM_BATCHES=3738 ⇒ MAX_STEPS=3364. `CKPT_DIR=ckpts_R4_beta0.12`.
- W&B: `R4_beta0.12_s0`.

## Results (held-out GSM8K, greedy, 64 test; `eval_results/`)
| step | 2000 | 2500 | 3000 | 3364 (final) |
|------|------|------|------|------|
| accuracy | 0.00% | 0.00% | 0.00% | **0.00%** |
| format | 0% | 0% | 0% | 0% |

🔴 **Total collapse, and early** — already 0%/0% format by step 2000 (the model emits nothing parseable).

## 🔴 Finding
Completes the β-sweep: **β=0 → 46.9%, β=0.08 → 3.1%, β=0.12 → 0.0%** (at seed 42). Monotone: *more* KL penalty
→ *worse* at K=2 — the opposite of the "tighter leash protects" prediction. Together with R5 (the K=2 collapse
is seed-dependent) the honest reading is: K=2 is a fragile, high-variance regime where both β and the seed
decide whether training collapses; the robust fix is a larger K (see R6).

## How to reproduce
```bash
cd scripts && tmux new -s r4b12
bash run_beta.sh 0.12
# eval: python eval_ckpt.py --policy lora --ckpt ~/ckpts_R4_beta12/actor --out eval_results/R4_beta12
```
