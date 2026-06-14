#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

source ~/venvs/tunix/bin/activate

# New run, not resume
unset WANDB_RUN_ID

# W&B team project. Change only if your team uses different names.
export WANDB_ENTITY="${WANDB_ENTITY:-chfazhvi}"
export WANDB_PROJECT="${WANDB_PROJECT:-chfazhvi-grpo}"
export WANDB_NAME="${WANDB_NAME:-R7_K8_lr5e6}"

# Match R6_K8 equal-rollout setup
export NUM_GENERATIONS=8
export NUM_BATCHES=935

# Unique R7 checkpoint path
export CKPT_DIR="$HOME/ckpts_R7_K8_lr5e6/"

echo "Starting R7_K8_lr5e6"
echo "Time: $(date)"
echo "WANDB_ENTITY=$WANDB_ENTITY"
echo "WANDB_PROJECT=$WANDB_PROJECT"
echo "WANDB_NAME=$WANDB_NAME"
echo "NUM_GENERATIONS=$NUM_GENERATIONS"
echo "NUM_BATCHES=$NUM_BATCHES"
echo "CKPT_DIR=$CKPT_DIR"
echo "Working directory: $(pwd)"

python -u train.py 2>&1 | tee ../train_R7_K8_lr5e6.log

echo "Finished R7_K8_lr5e6"
echo "Time: $(date)"
