# R7_K8_lr5e6

## Purpose

R7 tests whether the stable K=8 GRPO setting from R6 can support a larger optimiser step size. The only intended algorithmic change from R6_K8 is increasing the learning rate from 3e-6 to 5e-6.

This is an optimisation experiment: the aim is to see whether a larger learning rate makes better use of the KL budget, or whether it destabilises the policy update despite the improved effective sample size from K=8 generations.

## Controlled change from R6_K8

Changed:

- LEARNING_RATE: 3e-6 -> 5e-6

Held fixed:

- NUM_GENERATIONS = 8
- NUM_BATCHES = 935
- BETA = 0.08
- EPSILON = 0.2
- LoRA rank = 64
- LoRA alpha = 64.0
- TOTAL_GENERATION_STEPS = 768
- Reward functions unchanged
- Evaluation setup unchanged

## Run provenance

- Experiment folder: experiments/R7_K8_lr5e6
- Branch: run/R7_K8_lr5e6
- Commit used: 2edcd9a8d58a0236b2ebcc43b91adeb1c37d17e5
- Model: google/gemma-3-1b-it
- Policy: LoRA
- Checkpoint path: /home/ext_nabeelch_1265_gmail_com/ckpts_R7_K8_lr5e6/actor
- Restored checkpoint step for evaluation: 841
- W&B run: https://wandb.ai/chfazhvi/chfazhvi-grpo/runs/iv88s7va
- Evaluation timestamp UTC: 2026-06-14T14:46:56Z

## Training notes

The successful run completed to step 841.

Final W&B training summary:

- actor/train/kl = 0.27155
- actor/eval/kl = 0.15575
- actor/train/loss = 0.02219
- actor/eval/loss = 0.01495
- actor/train/pg_clipfrac = 0
- actor/eval/pg_clipfrac = 0
- completions/eval/max_length = 768

There were two failed launch attempts before the successful run:

- W&B run yqye359y: failed due to TensorBoard path permission issue under /tmp/content.
- W&B run tffqsrrw: failed due to local TFDS/protobuf cache issue.

Both failed attempts are ignored as training results. The final successful run is iv88s7va.

## Evaluation result

Evaluation was run using eval_ckpt.py, restoring the trained LoRA checkpoint from step 841.

Final held-out GSM8K evaluation on 64 prompts:

| Run | Accuracy | Partial | Format | Correct / Total |
|---|---:|---:|---:|---:|
| R7_K8_lr5e6 | 53.12% | 53.12% | 81.25% | 34 / 64 |

Summary JSON:

```json
{
  "model": "google/gemma-3-1b-it",
  "policy": "lora",
  "ckpt": "/home/ext_nabeelch_1265_gmail_com/ckpts_R7_K8_lr5e6/actor",
  "restored_step": 841,
  "preset": "greedy",
  "dataset": "gsm8k:test",
  "n_prompts_requested": 64,
  "n_correct": 34,
  "n_total": 64,
  "accuracy_pct": 53.12,
  "partial_pct": 53.12,
  "format_pct": 81.25,
  "timestamp_utc": "2026-06-14T14:46:56Z"
}
```

## Interpretation

R7 did not improve over R6_K8. R6_K8 achieved 56.25% accuracy, while R7_K8_lr5e6 achieved 53.12%.

This suggests that increasing the learning rate from 3e-6 to 5e-6 did not make better use of the K=8 effective sample size. Instead, the larger optimiser step appears to slightly reduce final task accuracy and format compliance. This is consistent with the idea that, even when the gradient estimate is stabilised by a larger group size, increasing the step size can move the policy too aggressively within the KL-constrained update.

The useful conclusion for the report is therefore negative but principled: K=8 improves stability relative to smaller group settings, but the R6 learning rate of 3e-6 appears closer to the useful regime than 5e-6 for this setup.

## Files produced

- Training log: experiments/R7_K8_lr5e6/train_R7_K8_lr5e6.log
- Evaluation log: experiments/R7_K8_lr5e6/eval_R7_K8_lr5e6.log
- Summary JSON: experiments/R7_K8_lr5e6/eval_results/R7_K8_lr5e6_summary.json
- Per-prompt CSV: experiments/R7_K8_lr5e6/eval_results/R7_K8_lr5e6_per_prompt.csv

## Bootstrap confidence interval

Bootstrap uncertainty was computed using 10,000 resamples over the 64 held-out GSM8K prompts.

| Run | Accuracy | 95% CI | Correct / Total |
|---|---:|---:|---:|
| R6 K=8 | 56.25% | [43.75%, 68.75%] | 36 / 64 |
| R7 K=8 lr5e-6 | 53.12% | [40.62%, 65.62%] | 34 / 64 |

The confidence intervals overlap strongly. Therefore, R7 should not be presented as a clear degradation, but rather as a learning-rate increase that did not produce evidence of improvement over R6. The main conclusion is that the K=8 setting appears to benefit from the original 3e-6 learning rate more than the larger 5e-6 step size.

