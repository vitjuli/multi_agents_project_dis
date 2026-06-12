# R4 (β=0) — KL penalty OFF

**One factor changed vs baseline:** `BETA = 0` (no reference-KL leash). Everything else is the K=2 baseline.
Config is env-driven now: `config.py` reads `BETA` from the environment. Launched with `run_beta.sh 0.0`.

## Config
- K=2, μ=1, **β=0** (vs baseline 0.08), ε=0.2, NUM_BATCHES=3738 ⇒ MAX_STEPS=3364. `CKPT_DIR=ckpts_R4_beta0.0`.
- W&B: `R4_beta0.0_s0` (run id 2pdmjfb6).

## Results (held-out GSM8K, greedy, 64 test; `eval_results/`)
| step | 2000 | 2500 | 3000 | 3364 (final) |
|------|------|------|------|------|
| accuracy | 53.12% | 51.56% | 42.19% | **46.88%** |
| format | 93.8% | 96.9% | 96.9% | 96.9% |

🔴 **NO collapse** — stays ~42–53% the whole run (contrast the baseline's 28→3%). Final **46.88%**, format 96.9%.

## 🔴 Finding (contradicted prediction)
The collapse coincides with KL growth, so we predicted that *removing* the leash (β=0) would be **worse**.
The opposite: β=0 is among the best (46.9%, no collapse). At K=2 the noisy KL-penalty gradient appears to be
*destabilising*, not protective. **Caveat (see R5):** the K=2 collapse is seed-dependent, so part of the
β=0-vs-β=0.08 gap is the seed-sensitivity of the K=2 regime, not β alone.

## How to reproduce
```bash
cd scripts && tmux new -s r4b0
bash run_beta.sh 0.0        # exports BETA=0, CKPT_DIR, WANDB_NAME; runs train.py
# eval: python eval_ckpt.py --policy lora --ckpt ~/ckpts_R4_beta0/actor --out eval_results/R4_beta0
```
