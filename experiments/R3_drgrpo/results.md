# R3 — Dr.GRPO (drops σ_r-norm + length-norm)

**One factor changed vs baseline:** the learner. `scripts/train.py` uses `DrGRPOConfig` / `DrGRPOLearner`
(import added at line 24). Everything else is the baseline (still K=2). See `scripts/train.py` lines 24/162/172.

## Config
- Learner = **`DrGRPOLearner` / `DrGRPOConfig`** (advantage estimator `drgrpo`: no ÷σ_r; loss agg
  `sequence-mean-token-scale`: no length normalisation). K=2, everything else stock.
- `CKPT_DIR = /tmp/content/ckpts_R3_drgrpo/`

## Provenance
- W&B: `R3_drgrpo_s0` (chfazhvi/chfazhvi-grpo) · branch `R3-drgrpo` · commit `174752c` · 3364 steps

## Results (held-out GSM8K, greedy, 64 test; via `scripts/eval_ckpt.py`)
| metric | base | baseline K=2 (final) | **Dr.GRPO (final)** |
|--------|------|----------------------|---------------------|
| accuracy | 48.44% | 3.12% | **31.25%** |
| format | 9.38% | 12.50% | **82.81%** |
| `completions/eval/mean_length` | — | ≈150 | **≈287** |
| `completions/eval/min_length` | — | ≈0 (empty!) | ≈34 (no empties) |

## 🔴 Findings (two)
1. **Contradicted prediction (length).** Dr.GRPO is pitched as a *length-bias fix* → we predicted **shorter**
   responses. Observed: **LONGER** (~287 vs ~150, train+eval agree). Likely because Dr.GRPO was validated on
   7B + long-CoT (AIME/MATH); at 1B + GSM8K the length regime inverts.
2. **Stability win.** Dr.GRPO held **31.25%** vs baseline's collapsed **3.12%** (10×) and learned the format
   (82.81%) with no empty completions. Removing the two normalisations made training far more stable. Still
   < base (48%), so on GSM8K the recovery is partial.

## Eval artifacts (machine-generated — not hand-typed)
`eval_results/` holds the script output that produced the numbers above:
- `R3_drgrpo_summary.json` — metrics + provenance (model, ckpt, `restored_step=3364`, timestamp).
- `R3_drgrpo_per_prompt.csv` — all 64 test prompts: gold, extracted answer, correct/partial/format, Q+response snippet.
- `base_*` — the untrained base (comparison point).

## How to reproduce
```bash
source ~/venvs/tunix/bin/activate
cd scripts && set -a; . ./.env; set +a
rm -rf data
python train.py                          # DrGRPOLearner, MAX_STEPS=3364
python eval_ckpt.py --policy lora --ckpt /tmp/content/ckpts_R3_drgrpo/actor --preset greedy
```
