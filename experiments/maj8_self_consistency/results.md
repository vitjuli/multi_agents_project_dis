# maj@8 — Self-consistency (test-time scaling)

## Idea
Quantify how much test-time compute lifts accuracy, and whether the GRPO improvement survives under
self-consistency. No training: instead of greedy single-sample decoding, draw k=8 completions per
prompt and majority-vote the final numeric answer.

## Method
- Eval-only, held-out 64 prompts, via `eval_majk.py`. k=8, temperature 0.7 (`standard` preset).
- Compared models: untrained base, and the best GRPO checkpoint (K=8, R6, restored_step 841).
- Greedy single-sample accuracy is the reference point for each model.
- **Mechanism.** Majority voting over k=8 samples reduces single-sample variance and approximates the model's marginal answer distribution, so it measures the reasoning capability already latent in the policy.

## Results
Held-out GSM8K, 64 prompts:

| model | greedy (1 sample) | maj@8 | 95% CI (maj@8) | Δ |
|---|---|---|---|---|
| base (untrained) | 48.44% | **59.38%** (38/64) | [46.9, 71.9] | +10.9 |
| K=8 GRPO (R6) | 56.25% | **59.38%** (38/64) | [46.9, 71.9] | +3.1 |

## Discussion
Self-consistency lifts the base by +10.9 points but the GRPO K=8 model by only +3.1, and both
converge to the same 59.38%. At this scale GRPO and test-time sampling behave as **substitutes, not
complements**: GRPO mainly improves the single-sample (greedy) output (48→56), but adds little
reasoning capability beyond what majority voting already extracts from the base. 59.38% is the
highest single-run accuracy figure in the project. Both measurements are at n=64, so the tie should be read as
≈ rather than exact.

## Provenance & reproduction
- No W&B run (eval-only). K=8 checkpoint from R6 (restored_step 841).
- Artifacts: `eval_results/{base_maj8,R6_K8_maj8}_summary.json` (+ per-prompt CSVs); each carries
  `method: "self-consistency maj@8"`, k, accuracy, timestamp.
```bash
A=~/ckpts_R6_K8/actor
python eval_majk.py --ckpt=$A     --out=$HOME/R6_K8_maj8 --k=8 --preset=standard
python eval_majk.py --policy=base --out=$HOME/base_maj8  --k=8 --preset=standard
```
