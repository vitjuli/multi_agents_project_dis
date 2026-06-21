#!/usr/bin/env bash
# R10 = mu=2 (PPO inner iterations) on the stable K=4 — the FIRST run that activates the clip.
# The whole project so far ran mu=1 (=> rho==1 => clip dormant, pg_clipfrac=0). mu=2 reuses each
# rollout batch twice, so on the 2nd inner pass rho != 1 and the PPO clip finally engages.
#
# Equal rollout budget vs R1_K4 (same K=4, same rollouts, only mu differs):
#   NB=1869, K=4 -> 0.9*1869=1682 rollout-batches x4 = 6728 rollouts (== baseline budget)
#   mu=2 doubles gradient steps -> MAX_STEPS = 1869 * 2 * 0.9 = 3364.
# Compare the result against R1_K4 (48%, mu=1).  Run INSIDE tmux:
#   tmux new -s mu2 ;  bash ~/tpu-2026/scripts/run_mu2.sh ;  detach: Ctrl-b d
set -euo pipefail
source ~/venvs/tunix/bin/activate
cd ~/tpu-2026/scripts
set -a; . ./.env; set +a
export NUM_ITERATIONS=2 NUM_GENERATIONS=4 NUM_BATCHES=1869 BETA=0.08 \
       SHUFFLE_SEED=42 CKPT_DIR=/tmp/content/ckpts_R10_mu2_K4/ WANDB_NAME=R10_mu2_K4
unset WANDB_RUN_ID 2>/dev/null || true

echo "================ R10_mu2_K4  PRE-FLIGHT  $(date) ================"
echo "  mu(NUM_ITERATIONS)=$NUM_ITERATIONS   K(NUM_GENERATIONS)=$NUM_GENERATIONS   NUM_BATCHES=$NUM_BATCHES"
echo "  expect: MAX_STEPS=3364, and pg_clipfrac>0 (clip now ACTIVE) — this is the whole point."
echo "  CKPT_DIR=$CKPT_DIR   WANDB_NAME=$WANDB_NAME"
# fail loudly if the override did not take (the R7 lesson):
[ "$NUM_ITERATIONS" = "2" ] || { echo "ABORT: NUM_ITERATIONS!=2"; exit 1; }
grep -q "NUM_ITERATIONS = int(os.environ" config.py \
  || { echo "ABORT: config.py has no NUM_ITERATIONS env-override -> scp the updated config.py first!"; exit 1; }
echo "================================================================"
sleep 3
rm -rf data
python train.py 2>&1 | tee ~/run_R10_mu2_K4.log
echo "[R10] done $(date) — eval:  python eval_ckpt.py --policy=lora --ckpt=${CKPT_DIR%/}/actor --step=3364 --out=\$HOME/R10_final"
