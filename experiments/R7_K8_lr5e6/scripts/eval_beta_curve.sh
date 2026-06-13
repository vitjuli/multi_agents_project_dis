#!/usr/bin/env bash
# Eval the β-sweep collapse curves: intermediate steps for β=0 and β=0.12,
# each WITH --out so we get summary.json + per_prompt.csv artifacts.
# Finals (step 3364) already done: eval_results/R4_beta0, eval_results/R4_beta12.
#
# Run on the VM:  bash eval_beta_curve.sh
source ~/venvs/tunix/bin/activate
cd ~/tpu-2026/scripts

for s in 2000 2500 3000; do
  echo "=== β=0  step $s ==="
  rm -rf data
  python eval_ckpt.py --ckpt ~/ckpts_R4_beta0/actor --step "$s" \
      --out "eval_results/R4_beta0_s${s}" --preset greedy
done

for s in 2000 2500 3000; do
  echo "=== β=0.12  step $s ==="
  rm -rf data
  python eval_ckpt.py --ckpt ~/ckpts_R4_beta12/actor --step "$s" \
      --out "eval_results/R4_beta12_s${s}" --preset greedy
done

echo "DONE — 6 step evals written under eval_results/ (R4_beta0_s* and R4_beta12_s*)."
