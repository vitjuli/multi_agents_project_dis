#!/usr/bin/env bash
# R11 = K=16, extending the equal-rollout-budget K-sweep (K=2/4/8 -> 16).
# Tests the lecture / I.4-Q1 prediction of DIMINISHING RETURNS ("8->16 helps less"):
# Var(g) ~ O(1/K), so K=16 should NOT beat K=8 by much.
#
# Equal rollout budget (total rollouts ~= 6728, like the rest of the sweep):
#   K=16 -> NUM_BATCHES=467 -> 0.9*467=420 rollout-batches x16 = 6720 rollouts ~= 6728.
#   mu=1 (clip dormant) -> MAX_STEPS = 467 * 1 * 0.9 = 420.
# Caveats baked into the writeup: (a) only ~420 steps at this budget (fewer than K=8's 841),
# so a flat/low result also CONFIRMS diminishing returns; (b) K=16 may OOM on v6e-1 — an OOM
# crash is itself a documented result. Run INSIDE tmux:
#   tmux new -s k16 ;  bash ~/tpu-2026/scripts/run_k16.sh ;  detach: Ctrl-b d
set -euo pipefail
source ~/venvs/tunix/bin/activate
cd ~/tpu-2026/scripts
set -a; . ./.env; set +a
export NUM_ITERATIONS=1 NUM_GENERATIONS=16 NUM_BATCHES=467 BETA=0.08 \
       SHUFFLE_SEED=42 CKPT_DIR=/tmp/content/ckpts_R11_K16/ WANDB_NAME=R11_K16
unset WANDB_RUN_ID 2>/dev/null || true

echo "================ R11_K16  PRE-FLIGHT  $(date) ================"
echo "  K(NUM_GENERATIONS)=$NUM_GENERATIONS   mu=$NUM_ITERATIONS   NUM_BATCHES=$NUM_BATCHES"
echo "  expect: MAX_STEPS=420, ~6720 rollouts (equal budget). Watch for OOM on v6e-1."
echo "  CKPT_DIR=$CKPT_DIR   WANDB_NAME=$WANDB_NAME"
[ "$NUM_GENERATIONS" = "16" ] || { echo "ABORT: K!=16"; exit 1; }
echo "============================================================="
sleep 3
rm -rf data
python train.py 2>&1 | tee ~/run_R11_K16.log
echo "[R11] done $(date) — eval:  python eval_ckpt.py --ckpt=${CKPT_DIR%/}/actor --out=\$HOME/R11_final --policy=lora"
