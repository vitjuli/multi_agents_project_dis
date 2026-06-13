# R7 — K=8 with higher learning rate (`R7_K8_lr5e6`)

## Purpose

This experiment tests whether the stable K=8 GRPO regime can support a larger optimiser step and make better use of the KL budget.

## Change from R6_K8

Starting point: R6_K8.

One factor changed:

- `LEARNING_RATE`: `3e-6` -> `5e-6`

Held fixed:

- `NUM_GENERATIONS=8`
- `NUM_BATCHES=935`
- `BETA=0.08`
- `EPSILON=0.2`
- `RANK=64`
- `ALPHA=64.0`
- `TOTAL_GENERATION_STEPS=768`
- reward functions unchanged
- evaluation setup unchanged

## Theory motivation

R6 showed that K=8 gave a more stable GRPO run than the K=2 baseline. This suggests that the larger group size improved the effective sample size and reduced the variance of the group-relative policy-gradient estimate.

This run tests whether the stable K=8 regime can support a larger optimiser step. Increasing the learning rate should move the LoRA policy further per update and spend the KL budget faster. If reward and accuracy improve while KL remains controlled, R6 may have under-used its KL budget. If KL grows without accuracy improvement, the higher learning rate is too aggressive and R6 was already close to the safe step size.

## Run provenance

- Branch:
- Pre-run commit:
- Post-run commit:
- Command:
- Start time:
- End time:
- Wall-clock time:
- W&B run link:
- Checkpoint path:
- Final restored step:
- Notes:

## Config changes

- `LEARNING_RATE=5e-6`
- `NUM_GENERATIONS=8`
- `NUM_BATCHES=935`
- `BETA=0.08`
- `EPSILON=0.2`
- `RANK=64`
- `ALPHA=64.0`
- `TOTAL_GENERATION_STEPS=768`

## Training behaviour

Fill after run:

- Mean reward curve:
- KL curve:
- Any instability/collapse:
- Response length behaviour if available:
- Checkpoint saving:

## Evaluation results

Held-out GSM8K, greedy, 64 prompts.

| Model | Accuracy | Partial accuracy | Format accuracy | n/64 |
|-------|----------|------------------|-----------------|------|
| R6_K8 | 56.25% | 59.38% | 87.50% | 36 |
| R7_K8_lr5e6 | | | | |

## Diagnostics

Fill after evaluation:

- Bootstrap CI:
- KL vs R6:
- Reward vs R6:
- Response length vs R6 if available:
- Any intermediate checkpoint evaluation:

## Interpretation

Fill after results:

- Did higher LR improve accuracy?
- Did KL increase in a controlled way?
- Did reward improve without over-optimising format?
- Was the KL budget under-used in R6 or already near the safe step size?

## Limitations

- 64-prompt evaluation is small, so report bootstrap CI.
- One seed/run only, unless time permits repeat.
- W&B curves are needed to interpret reward/KL dynamics.

## Report takeaway

Fill after results.
