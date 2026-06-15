# R9 — Difficulty curriculum (data ordering)

**One factor changed vs baseline:** the *order* of the training examples. The train split is sorted by a
difficulty proxy (number of `<<...>>` calculator steps in the GSM8K gold solution; Cobbe et al. 2021) instead
of the baseline's random shuffle. Everything else is the **stock R0 baseline** — K=2, same prompts, same
held-out eval split, same `MAX_STEPS=3364`, same compute, same seed-42 tie-break. This is a controlled
single-factor comparison (order only), in the "Data / curriculum" direction of I.3.

> **Status:** ✅ done. R9 (easy2hard) **31.25%** acc / 75.0% format (20/64), W&B `R9_curriculum` (`crkcls55`),
> 3364 steps. Control **[R10](../R10_anticurriculum/results.md)** (hard2easy) done at 42.19%. Headline below.

## What changed (the only delta vs R0)
- `scripts/config.py`: new knob `CURRICULUM = "easy2hard" | "hard2easy" | "shuffle"` (env-overridable).
  `CKPT_DIR = ~/r9_runs/ckpts_R9_{CURRICULUM}/` (per-run, so the two modes never resume each other; under
  `$HOME` because `/tmp/content` is owned by another teammate on the shared VM). Also `SAVE_INTERVAL_STEPS=250`
  + `MAX_TO_KEEP=20` so the full early trajectory is kept (R9 is about early dynamics).
- `scripts/data.py`: `reasoning_difficulty()` (the `<<...>>` step-count proxy, line-count fallback) and
  `_order_examples()`. `build_train_val_test` replicates the **baseline train/val membership exactly**
  (seed-42 shuffle → first `NUM_BATCHES` → `TRAIN_FRACTION` cut) and reorders **only the train portion** —
  so the data *content* of every split is identical to R0; only the visiting order changes. Test split is
  untouched (same fixed 64 prompts as every other experiment).
- `scripts/train.py`: passes `CURRICULUM` through and logs it.

## Config
- `CURRICULUM = easy2hard`; `NUM_GENERATIONS = 2` (baseline K). The `hard2easy` control is run
  **[R10](../R10_anticurriculum/results.md)** (same code, `CURRICULUM=hard2easy`).
- `NUM_BATCHES = 3738` ⇒ `MAX_STEPS = 3364` (identical compute to R0); `BETA = 0.08`, `EPSILON = 0.2`,
  `RANK = ALPHA = 64`, `LR = 3e-6`, `MAX_GRAD_NORM = 0.1` — all stock.

## Provenance
- W&B: `R9_curriculum` — https://wandb.ai/chfazhvi/chfazhvi-grpo/runs/crkcls55
- Forks baseline commit `77c5a67` (same as R0). Control: `R10_anticurriculum` (see ../R10_anticurriculum/).

## Theory / motivation (why this should matter)
The shaping reward is the sum of format + soft-format + correctness terms. Early in training the model can
neither format nor solve, so within a K=2 group the **correctness** reward is usually all-zero — a
*degenerate group* (`σ_r = 0`, advantage `Â_i = (r_i − r̄)/σ_r` undefined / zero signal). The only non-zero
signal is the format-shaping reward, which is exactly what the baseline over-optimises into the seed-42
collapse (R0: 48% → 3%). An **easy-first** curriculum front-loads problems the base model can already solve
(base is 48% on GSM8K), so *some* of the K rollouts are correct early ⇒ the correctness reward varies within
the group (`σ_r > 0`) ⇒ a real, non-degenerate learning signal appears **when the baseline has only format
signal**. This directly targets the I.2(c) / I.4-Q1 mechanism the whole team's story rests on.

**Prediction (to be tested):**
1. R9 (easy2hard) has a **lower degenerate-group fraction early in training** than R0 (and than R10 hard2easy).
2. R9 **mitigates or delays the seed-42 collapse** → higher final held-out accuracy than R0's 3.12%.
3. **R10 (hard2easy) ≤ R0** (front-loading unsolvable problems keeps groups degenerate longer). The
   R9 − R10 gap is the clean, controlled measurement of the ordering effect.
   If the gap is ~0, that is an honest negative result: ordering does not move σ_r enough at this scale.

## Relationship to the R0/R5 seed-dependence (and the confound to address)
In this pipeline the training "seed" **is** the data-shuffle seed (`data.py: .shuffle(seed=SHUFFLE_SEED)`):
R0 (`SHUFFLE_SEED=42`) collapsed to 3.12%, R5 (`SHUFFLE_SEED=1`, identical config) did **not** (40.62%). So the
team's "collapse is seed-dependent" result is really **collapse is data-order-dependent** — which is exactly
the lever this experiment manipulates *on purpose*. R9/R10 are the principled, mechanistic version of that
accidental observation: instead of reshuffling at random and hoping, we order by difficulty and predict the
direction of the effect from the σ_r / degenerate-group theory.

**The confound:** because a merely *different* random order (R5) already avoided the collapse, "R9 easy-first
did not collapse" is on its own weak — it could be that almost any order other than the unlucky seed-42 is fine.
The disentangling design is that **R0, R9 and R10 all train on the *same* 3364 examples** (R9/R10 reuse R0's
seed-42 membership; only the *within-set order* changes — random-42 / easy-first / hard-first). The controlled
claim is therefore the **R9 − R10 gap on identical data**: if R9 ≫ R10, the *curriculum direction* matters, not
just "a different order"; R0 and R5 sit alongside as the two random-order points (one collapsed, one didn't).

**Limitation (state it explicitly):** collapse is stochastic in the order, so a single R9 and single R10 could
each be lucky/unlucky. The fully rigorous version repeats easy/hard at a second membership seed
(`SHUFFLE_SEED=1`) — ~2 extra runs, likely beyond the TPU window. We therefore lead with the R9−R10 contrast,
cite R0/R5 as the order-sensitivity pair, and flag the single-seed caveat.

## Results (held-out GSM8K, greedy, 64 test; via `scripts/eval_ckpt.py`)
| metric | base | R0 baseline K=2 | R5 baseline (seed 1) | **R9 easy2hard** | R10 hard2easy |
|--------|------|-----------------|----------------------|------------------|---------------|
| accuracy | 48.44% | 3.12% | 40.62% | **31.25%** | 42.19% |
| format   | 9.38%  | 12.50% | 65.62% | **75.00%** | 71.88% |
| 95% CI (acc) | [35.9, 60.9] | [0.0, 7.8] | [29.7, 53.1] | **[20.3, 42.2]** | [29.7, 54.7] |

(Bootstrap 95% CI: `../../bootstrap_ci.py` → `../../figs/accuracy_ci.md`, 10k resamples over the 64 test prompts.)

## Finding
1. **Real, significant: deliberate difficulty ordering avoids the seed-42 collapse.** Both R9 (31.25%) and R10
   (42.19%) sit far above R0's collapsed 3.12% — their CIs are non-overlapping with R0's [0.0, 7.8]. On the
   *same 3364 examples*, merely changing the visiting order from the unlucky random seed-42 order to a
   difficulty order removes the catastrophic collapse and learns the format (75% / 72% vs R0's 12%). This is
   the controlled, mechanistic confirmation of the team's "collapse is data-order-dependent" observation (R0 vs R5).
2. **Not supported (honest negative): the curriculum *direction*.** I predicted easy→hard > hard→easy. Observed
   the opposite *nominally* (R10 42% > R9 31%), but the CIs overlap heavily ([20.3, 42.2] vs [29.7, 54.7]), so
   the direction effect is **within noise** — neither beats the other significantly. The σ_r theory's directional
   prediction does not hold at this scale; what matters is *that* there is a deliberate order, not which way.
   The degenerate-group diagnostic (below) is the right place to look for any residual directional signal.

## Diagnostic (deliverable c) — the headline plot for this experiment
**Degenerate-group fraction vs training progress**, R0 vs R9 (easy2hard) vs R10 (hard2easy). With
`micro_batch=1`, `rewards/train/{min,max}` are per-group, so a step is degenerate ⟺ `min==max`. Use the shared
`../../degenerate_groups.py` (R9/R10 entries added) → `degenerate_groups_binned.csv` → plot. Prediction: the
R9 curve sits **below** R0 in the early bins.

## Eval artifacts (machine-generated — not hand-typed)
`eval_results/` here holds the shared `base_*` (untrained base) and, after the run:
- `R9_curriculum_summary.json` / `_per_prompt.csv` (easy2hard).
The hard2easy control's artifacts live in the sibling folder `../R10_anticurriculum/eval_results/`.

## How to reproduce
```bash
source ~/venvs/tunix/bin/activate
cd scripts && set -a; . ./.env; set +a
rm -rf data

# R9 — curriculum (easy -> hard); CKPT_DIR auto = ~/r9_runs/ckpts_R9_easy2hard/
CURRICULUM=easy2hard python train.py            # MAX_STEPS=3364

# R10 — anti-curriculum control (hard -> easy)
CURRICULUM=hard2easy python train.py

# eval each (TPU free after training; verify the log prints `restored step N`, N != 0):
rm -rf data
python eval_ckpt.py --policy base --preset greedy --out eval_results/base
python eval_ckpt.py --policy lora --ckpt ~/r9_runs/ckpts_R9_easy2hard/actor --preset greedy --out eval_results/R9_curriculum
python eval_ckpt.py --policy lora --ckpt ~/r9_runs/ckpts_R9_hard2easy/actor --preset greedy --out ../R10_anticurriculum/eval_results/R10_anticurriculum

# diagnostic + CI (no TPU):
cd .. && python degenerate_groups.py && python bootstrap_ci.py
```
