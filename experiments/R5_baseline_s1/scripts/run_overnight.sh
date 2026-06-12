#!/usr/bin/env bash
# Two sequential GRPO runs, unattended (overnight). Run INSIDE tmux:
#   tmux new -s overnight
#   bash run_overnight.sh
#   # then detach: Ctrl-b  d   (DO NOT Ctrl-C)
#
#   R5 = 2nd seed of the baseline (rigor: is the K=2 collapse reproducible across data order?)
#   R6 = K=8 (extends the K-sweep K=2->4->8; equal rollout budget). ⚠ may OOM on v6e-1 —
#        if it crashes that is itself a documented result; R5 is already safe by then.
source ~/venvs/tunix/bin/activate
cd ~/tpu-2026/scripts
set -a; . ./.env; set +a        # WANDB / HF / PROTOCOL_BUFFERS=python

run_one () {                    # $1 = tag; env (CKPT_DIR etc.) set by caller
  echo "=====================  START $1  $(date)  ====================="
  echo "[overnight] K=$NUM_GENERATIONS NB=$NUM_BATCHES BETA=$BETA SEED=$SHUFFLE_SEED CKPT=$CKPT_DIR NAME=$WANDB_NAME"
  rm -rf data
  python train.py 2>&1 | tee ~/run_${1}.log
  cp -r "${CKPT_DIR%/}" ~/ckpts_${1} 2>/dev/null && echo "[overnight] saved ckpt -> ~/ckpts_${1}"
  echo "=====================  END   $1  $(date)  ====================="
}

# --- R5: baseline, second seed (shuffle seed 1 instead of 42) ---
export SHUFFLE_SEED=1 NUM_GENERATIONS=2 NUM_BATCHES=3738 BETA=0.08 \
       CKPT_DIR=/tmp/content/ckpts_R5_baseline_s1/ WANDB_NAME=R5_baseline_s1
run_one R5_baseline_s1          # MAX_STEPS=3364

# --- R6: K=8 (equal rollout budget -> NUM_BATCHES quartered) ---
export SHUFFLE_SEED=42 NUM_GENERATIONS=8 NUM_BATCHES=935 BETA=0.08 \
       CKPT_DIR=/tmp/content/ckpts_R6_K8/ WANDB_NAME=R6_K8
run_one R6_K8                   # MAX_STEPS=841 ; 841*8 = 6728 = baseline 3364*2

echo "=====================  OVERNIGHT DONE  $(date)  ====================="
