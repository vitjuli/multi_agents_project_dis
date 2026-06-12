# MAS Practical Exam — GRPO experiments (team `chfazhvi`)

GRPO finetuning of `gemma-3-1b-it` on GSM8K (Tunix/JAX + LoRA). Part I.1–I.3.
**W&B:** https://wandb.ai/chfazhvi/chfazhvi-grpo

## Why folders, not branches
Each experiment lives in its **own self-contained folder** under `experiments/` with a **full copy of the
code it ran** (`scripts/`) + its `results.md`. Nothing is shared, so there is **no risk of an accidental
merge/overwrite** clobbering another experiment. Each folder is independently reproducible.

## 🏆 Headline results (held-out GSM8K, greedy, 64 test prompts) — 8 runs
| Model | Accuracy | Format | Verdict |
|-------|----------|--------|---------|
| base `gemma-3-1b-it` | 48.44% | 9.38% | strong base, ignores template |
| **R0** baseline K=2 (seed 42) | **3.12%** | 12.50% | 🔴 collapsed |
| **R5** baseline K=2 (seed 1) | **40.62%** | 65.62% | ⚠️ NOT collapsed → **collapse is seed-dependent** |
| **R3** Dr.GRPO K=2 | **31.25%** | 82.81% | 🟢 recovers |
| **R1** K=4 | **48.44%** | 85.94% | 🟢 stable |
| **R6** K=8 | **56.25%** | 87.50% | 🏆 **best, > base** |
| **R4** β=0 / β=0.12 | 46.88% / 0.00% | 96.9% / 0% | β-sweep: more KL penalty → worse (at seed 42) |

> 📊 **Figures are individual.** This shared repo holds the *data* (numbers, `eval_results/`, raw CSVs) but
> **not** rendered plots — each teammate builds their own figures for their own report from this data
> (see `bootstrap_ci.py` and the W&B project). Don't reuse someone else's plots.

**Story:** at K=2 the advantage variance `Var(Â)=(K−1)/K` is highest, so training is fragile — it can
over-optimise the format shaping reward and **collapse** (48→3%), but whether it does is **seed-dependent**
(R5: seed 1 stays at 40.6%) and **β-dependent** (R4: β=0→47%, β=0.08→3%, β=0.12→0%). The clean, **seed-robust
fix is a larger group**: K=4 is stable at base accuracy and **K=8 exceeds it (56%)** — the cleanest validation
of the variance theory on held-out accuracy. Dr.GRPO (drop σ_r/length norms) also recovers (31%) and, against
our prediction, makes responses *longer* not shorter.

## Layout
```
README.md                     # this file
figs/accuracy_ci.md           # bootstrap 95% CI table (numbers; rendered plots are individual, not shared)
experiments/
  R0_baseline/      K=2 reference (seed 42, collapses)
  R5_baseline_s1/   K=2, 2nd seed (seed 1, does NOT collapse — rigor)
  R3_drgrpo/        Dr.GRPO (drop σ_r + length norms)
  R1_K4/            K=4 (stable)
  R6_K8/            K=8 (best, > base)
  R4_beta0/  R4_beta12/   β-sweep (KL penalty off / tighter)
  (each = scripts/ + results.md + eval_results/)
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
