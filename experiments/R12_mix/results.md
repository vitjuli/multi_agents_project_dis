# R12 — Mix in a second math source (ASDiv) — the capstone data run

**The final data-direction run.** Blend the GSM8K train split with a second grade-school math source —
**ASDiv** (`EleutherAI/asdiv`, 2305 problems) — at `MIX_ALPHA` (default **0.3**), then train the stock
baseline (K=2, `MAX_STEPS=3364`). Train count stays 3364 (≈2355 GSM8K + ≈1009 ASDiv) so compute is unchanged;
**eval stays GSM8K test** so the number is comparable to every other run.

> **Status:** ✅ done — **32.81%** acc / 73.44% format (21/64), 3364 steps, W&B `R12_mix` (`e7h8o1es`). Avoids
> the collapse but does not beat the no-collapse baseline on GSM8K test — details under Finding.

## Why this is clean to implement
GRPO is RL from a programmatic reward — it never sees the second source's *solution text*, only checks the
final number. So ASDiv only needs `(question, gold_number)`; we wrap each question in the same template and
store the answer as `#### <number>` so the shared `extract_hash_answer()` recovers it exactly like GSM8K.
ASDiv answers like `"9 (apples)"` → `9` via `_first_number()`.

## Theory / motivation (σ_r framing)
ASDiv is **easier** and more varied than GSM8K (mostly 1–2 step arithmetic word problems). Adding it raises the
fraction of problems the base model can already solve, so more groups have *some* correct rollouts → the
**correctness reward varies within the group (σ_r > 0)** instead of being all-zero → a real learning signal
early, when the baseline only has format-shaping signal. This is the same lever as the easy-first curriculum
(R9), reached by changing the *data distribution* rather than the *order*. Trade-off: distribution shift away
from the GSM8K test set could cap or hurt GSM8K accuracy even as training stabilises — a tension to report.

## Config
- `SECOND_SOURCE=asdiv`, `MIX_ALPHA=0.3`, `MIX_SEED=42`; `CURRICULUM=shuffle`, no difficulty filter.
- Train = 2355 GSM8K (seed-42 membership) + 1009 ASDiv, shuffled; val = GSM8K val (374, unchanged).
- Stock otherwise: `NUM_GENERATIONS=2`, `MAX_STEPS=3364`, `BETA=0.08`, `EPSILON=0.2`, `RANK=ALPHA=64`.
- Checkpoints: `~/r9_runs/ckpts_R12_mix/` (launched with `EXP_TAG=R12_mix`).

## Provenance
- W&B (planned): `R12_mix` — chfazhvi/chfazhvi-grpo. Forks baseline commit `77c5a67`.
- Second source: **ASDiv** — https://huggingface.co/datasets/EleutherAI/asdiv (Miao et al. 2020).

## Prediction (to be tested)
- More winnable problems early → **lower degenerate-group fraction**, more stable training (no seed-42 collapse),
  like R9.
- GSM8K-test accuracy: could **rise** (better-conditioned training) or **plateau/fall** (the policy spends
  capacity on easier ASDiv-style problems → distribution shift from GSM8K). Either way the σ_r diagnostic is
  the controlled signal; the accuracy direction is the honest empirical question.
- Sweep handle: `MIX_ALPHA` ∈ {0.15, 0.3, 0.5} if time allows (one extra cheap axis).

## Results (held-out GSM8K, greedy, 64 test; via `scripts/eval_ckpt.py`)
| metric | base | R0 baseline | R9 easy2hard | R10 hard2easy | **R12 mix ASDiv α=0.3** |
|--------|------|-------------|--------------|---------------|-------------------------|
| accuracy | 48.44% | 3.12% | 31.25% | 42.19% | **32.81%** |
| format   | 9.38%  | 12.50% | 75.00% | 71.88% | **73.44%** |
| 95% CI (acc) | [35.9,60.9] | [0.0,7.8] | [20.3,42.2] | [29.7,54.7] | **[21.9,43.8]** |

## Finding
**Predicted plateau/fall confirmed.** Mixing ASDiv **avoids the collapse** (32.81% vs R0's 3.12%, CIs clear of
R0) — consistent with the σ_r story: adding easier, winnable problems gives non-degenerate correctness groups
and stabilises training. But it does **not beat** the no-collapse GSM8K-only baselines (R5 40.6%, R10 42.2%) on
GSM8K *test* — R12's CI [21.9, 43.8] overlaps all of them, so it's within noise of R9/R11 and slightly below
R10. The most likely reason is the **distribution-shift trade-off we flagged**: ~30% of training is now easier
ASDiv-style problems, so policy capacity shifts away from the harder GSM8K test distribution. Net: a second
source buys *stability*, not *accuracy*, at this mix ratio — an honest result. A smaller `MIX_ALPHA` (0.15) or
curriculum-ordered mix are the natural follow-ups if chasing a GSM8K gain.

## How to reproduce
```bash
source ~/venvs/tunix/bin/activate
cd scripts && set -a; . ./.env; set +a
rm -rf data
SECOND_SOURCE=asdiv MIX_ALPHA=0.3 EXP_TAG=R12_mix WANDB_NAME=R12_mix python train.py
rm -rf data
python eval_ckpt.py --policy lora --ckpt ~/r9_runs/ckpts_R12_mix/actor --preset greedy --out eval_results/R12_mix
```
See [`../R9_curriculum/results.md`](../R9_curriculum/results.md) for the shared σ_r / degenerate-group theory.
