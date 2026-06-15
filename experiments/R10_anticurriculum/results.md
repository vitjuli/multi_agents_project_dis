# R10 — Anti-curriculum (hard→easy data ordering) — control for R9

**The control run for the R9 curriculum experiment.** Identical code and config to
[`R9_curriculum`](../R9_curriculum/results.md); the only difference is `CURRICULUM=hard2easy` — the GSM8K
train split is visited **hardest-first** instead of easiest-first. Same examples, same held-out eval split,
same `MAX_STEPS=3364`, same K=2, same seed-42 tie-break. The R9 (easy→hard) vs R10 (hard→easy) gap is the
clean, controlled measurement of the *ordering* effect; R0 is the shuffled baseline.

> **Status:** ✅ done — **42.19%** acc / 71.88% format (27/64), 3364 steps. W&B `R10_anticurriculum`
> (re-uploaded as `r10anticurric` after an accidental delete; original id tombstoned by W&B). Survived a
> mid-run TPU host reboot — resumed from step 2500 (checkpoints in `$HOME`).

## What this run is
- Same modification as R9 (difficulty ordering by the `<<...>>` calculator-step proxy), **reversed**.
- Scripts here are byte-identical to R9's; this run is produced purely by the env var `CURRICULUM=hard2easy`
  (which `config.py` reads). Kept as a self-contained folder so it is independently reproducible.

## Config
- `CURRICULUM = hard2easy`; `NUM_GENERATIONS = 2`, `NUM_BATCHES = 3738` ⇒ `MAX_STEPS = 3364`;
  `BETA = 0.08`, `EPSILON = 0.2`, `RANK = ALPHA = 64`, `LR = 3e-6`, `MAX_GRAD_NORM = 0.1` — all stock.
- Checkpoints: `~/r9_runs/ckpts_R9_hard2easy/actor/` (the `R9_` path literal is just the shared run root).

## Provenance
- W&B (planned): `R10_anticurriculum` — chfazhvi/chfazhvi-grpo
- Forks baseline commit `77c5a67` (same as R0/R9).

## Prediction (to be tested)
Front-loading the *hardest* problems keeps the early groups degenerate for longer (the base model fails all K
rollouts ⇒ correctness `σ_r = 0`), so R10 should be **≤ R0 baseline** and clearly **worse than R9**. If instead
R10 ≈ R9, the ordering does not move σ_r enough at this scale — an honest negative result that weakens the
curriculum story (report it either way).

## Results (held-out GSM8K, greedy, 64 test; via `scripts/eval_ckpt.py`)
| metric | base | R0 baseline (final) | R9 easy2hard (final) | **R10 hard2easy (final)** |
|--------|------|---------------------|----------------------|---------------------------|
| accuracy | 48.44% | 3.12% | 31.25% | **42.19%** |
| format   | 9.38%  | 12.50% | 75.00% | **71.88%** |
| 95% CI (acc) | [35.9, 60.9] | [0.0, 7.8] | [20.3, 42.2] | **[29.7, 54.7]** |

**Finding:** R10 (hard→easy) does **not** collapse — 42.19% vs R0's 3.12% (non-overlapping CIs), so any
deliberate difficulty ordering avoids the seed-42 collapse. The prediction that hard-first ≤ R0 / ≤ R9 is
**not supported**: R10 nominally *beats* easy-first R9 (42% vs 31%), but the CIs overlap, so the direction
effect is within noise. Joint discussion in [`../R9_curriculum/results.md`](../R9_curriculum/results.md#finding).

## How to reproduce
```bash
source ~/venvs/tunix/bin/activate
cd scripts && set -a; . ./.env; set +a
rm -rf data
CURRICULUM=hard2easy python train.py            # MAX_STEPS=3364
rm -rf data
python eval_ckpt.py --policy lora --ckpt ~/r9_runs/ckpts_R9_hard2easy/actor --preset greedy --out eval_results/R10_anticurriculum
```

See [`R9_curriculum/results.md`](../R9_curriculum/results.md) for the full motivation, the σ_r / degenerate-group
theory, and the joint diagnostic (degenerate-group fraction vs training progress: R0 vs R9 vs R10).
