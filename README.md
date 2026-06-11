# MAS Practical Exam — GRPO experiments (team `chfazhvi`)

GRPO finetuning of `gemma-3-1b-it` on GSM8K (Tunix/JAX + LoRA). Part I.1–I.3.
**W&B:** https://wandb.ai/chfazhvi/chfazhvi-grpo

## Why folders, not branches
Each experiment lives in its **own self-contained folder** under `experiments/` with a **full copy of the
code it ran** (`scripts/`) + its `results.md`. Nothing is shared, so there is **no risk of an accidental
merge/overwrite** clobbering another experiment. Each folder is independently reproducible.

## 🏆 Headline results (held-out GSM8K, greedy, 64 test prompts)
| Model | Accuracy | Format | Verdict |
|-------|----------|--------|---------|
| base `gemma-3-1b-it` | 48.44% | 9.38% | strong base, ignores template |
| **R0** baseline GRPO K=2 (final) | **3.12%** | 12.50% | 🔴 collapsed |
| **R3** Dr.GRPO (final) | **31.25%** | 82.81% | 🟢 10× more stable + format |
| **R1** K=4 (final) | **48.44%** | 85.94% | 🏆 no collapse: base acc + format |

> 📊 **Figures are individual.** This shared repo holds the *data* (numbers, `eval_results/`, raw CSVs) but
> **not** rendered plots — each teammate builds their own figures for their own report from this data
> (see `bootstrap_ci.py` and the W&B project). Don't reuse someone else's plots.

**Story:** vanilla GRPO at K=2 has the highest advantage variance (`Var(Â)=(K−1)/K`) + most degenerate groups
→ it over-optimises the format shaping reward and **collapses** math accuracy (48%→3% over training). **Both**
improvements fix the instability: **Dr.GRPO** (drop σ_r-norm + length-norm) recovers to 31%; **K=4** (lower
variance) prevents the collapse entirely. → directly validates the I.4 Q1 variance theory on accuracy.

## Layout
```
README.md                     # this file
figs/accuracy_ci.md           # bootstrap 95% CI table (numbers; rendered plots are individual, not shared)
experiments/
  R0_baseline/  scripts/ (stock) + results.md + eval_results/   # reference run, K=2
  R3_drgrpo/    scripts/ (DrGRPO) + results.md + eval_results/   # improvement #1
  R1_K4/        scripts/ (K=4)   + results.md + eval_results/    # improvement #2 (best)
```
Each `results.md` has: exact config, W&B/commit provenance, accuracy numbers, the finding, a reproduce block.
Each `eval_results/` holds the **machine-generated** proof the numbers are real (not hand-typed):
`<exp>_summary.json` (metrics + model/ckpt/step/timestamp) and `<exp>_per_prompt.csv` (all 64 test prompts:
gold, extracted answer, correct/partial/format, question + response snippet) — produced by `eval_ckpt.py --out`.

## Eval tooling (in every `scripts/`)
- `eval_ckpt.py --policy {base,lora} --ckpt <…/actor> [--step N]` — restores the trained LoRA adapters and
  scores GSM8K. 🔑 The Orbax checkpoint stores **only the ~312 LoRA-adapter leaves** (not the frozen base),
  so restore uses `CheckpointManager.maybe_restore(lora, restore_only_lora_params=True)`. The shipped
  `evaluate.py` never restores (measures base) — `eval_ckpt.py` is our fix.
- `eval_debug.py` — prints the model vs checkpoint trees (how we diagnosed the restore).
- Before each eval: `rm -rf data` (tfds/protobuf reload crash). Verify the log prints `restored step N`, N≠0.

## Planned (not yet run)
R6 clip-higher (`epsilon_high=0.28`) · R4 β sweep (0 / 0.12) · R5 baseline seed-2 (CI) · R1b K=8 ·
reward-rebalance (now high-priority — the failure is format reward-hacking). See `../project_docs/` for the
full plan + theory.
