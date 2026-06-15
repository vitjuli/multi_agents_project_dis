# R11 — Difficulty filtering (GSM8K-only)

**Data-direction run #3** (after R9 easy→hard, R10 hard→easy). Same stock baseline (K=2, `MAX_STEPS=3364`,
GSM8K test eval), but the TRAIN pool is **filtered to a mid-difficulty band** by the `<<>>` calculator-step
proxy: keep `DIFFICULTY_MIN ≤ steps ≤ DIFFICULTY_MAX`, default **[2, 6]**. Train count stays 3364 (drawn from
the filtered pool) so compute is unchanged.

> **Status:** ✅ done — **21.88%** acc / 59.38% format (14/64), 3364 steps, W&B `R11_filter`. The filter
> **hurt** (below R9/R10 and base) — a second honest-negative result; details under Finding.

## Theory / motivation
A group is *degenerate* (`σ_r = 0`, no advantage signal) when all K rollouts get the same reward — which
happens at the two extremes: **trivial** problems the model always solves (all-K-correct) and **near-impossible**
ones it never solves (all-K-fail). Filtering to the mid band keeps the prompts where rollouts are most likely
to *disagree* → **maximises within-group σ_r** → most informative gradient per step. This is the "filter"
complement to R9/R10's "reorder," using the same difficulty proxy.

## Config
- `DIFFICULTY_MIN=2`, `DIFFICULTY_MAX=6` (env-overridable). Band [2,6] keeps **6894 / 7473 (92%)** of GSM8K
  train — drops trivial 1-step (404) and hard 7–9-step (175). First 3738 of the filtered, seed-42 order →
  3364 train / 374 val. `CURRICULUM=shuffle`, no mix.
- Everything else stock: `NUM_GENERATIONS=2`, `MAX_STEPS=3364`, `BETA=0.08`, `EPSILON=0.2`, `RANK=ALPHA=64`.
- Checkpoints: `~/r9_runs/ckpts_R11_filter/` (launched with `EXP_TAG=R11_filter`).

## Provenance
- W&B (planned): `R11_filter` — chfazhvi/chfazhvi-grpo. Forks baseline commit `77c5a67`.

## Prediction (to be tested)
- **Lower degenerate-group fraction than R0** (we removed the degenerate-prone extremes) and possibly higher /
  steadier reward per step.
- **Effect size is likely small**: step-count is a coarse proxy and the extremes are only ~8% of data, so the
  filtered set is close to the full one. If R11 ≈ R0/R9, that is an honest negative result — the informative
  band is better captured by the *model's* pass-rate than by gold step-count (worth saying explicitly).
- Tighter bands (e.g. `[3,6]` → 63% kept) are the more aggressive alternative if the effect is too weak.

## Results (held-out GSM8K, greedy, 64 test; via `scripts/eval_ckpt.py`)
| metric | base | R0 baseline | R9 easy2hard | R10 hard2easy | **R11 filter [2,6]** |
|--------|------|-------------|--------------|---------------|----------------------|
| accuracy | 48.44% | 3.12% | 31.25% | 42.19% | **21.88%** |
| format   | 9.38%  | 12.50% | 75.00% | 71.88% | **59.38%** |
| 95% CI (acc) | [35.9,60.9] | [0.0,7.8] | [20.3,42.2] | [29.7,54.7] | **[12.5,32.8]** |

## Finding
The filter **hurt**: R11 (21.88%) is below R9 (31%), R10 (42%), and base (48%) — the predicted "small effect"
turned out to be a small *negative* one (still above R0's collapse, so it didn't destabilise). Diagnosis: the
[2,6] band drops the **trivial 1-step problems the base model can actually solve**, which were a useful source
of *correct* rollouts (non-degenerate, σ_r>0) early on; removing them — plus the loss of data diversity — left
a harder, narrower training set without the compensating signal we hoped for. Lesson for the write-up: gold
**step-count is a poor proxy for what makes a group degenerate for *this* model** (the model's own pass-rate
would be the right axis), so filtering on it is the weakest of the three data levers. Honest negative — in scope.

## How to reproduce
```bash
source ~/venvs/tunix/bin/activate
cd scripts && set -a; . ./.env; set +a
rm -rf data
DIFFICULTY_MIN=2 DIFFICULTY_MAX=6 EXP_TAG=R11_filter WANDB_NAME=R11_filter python train.py
rm -rf data
python eval_ckpt.py --policy lora --ckpt ~/r9_runs/ckpts_R11_filter/actor --preset greedy --out eval_results/R11_filter
```
See [`../R9_curriculum/results.md`](../R9_curriculum/results.md) for the shared σ_r / degenerate-group theory.
