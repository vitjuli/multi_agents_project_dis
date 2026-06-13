#!/usr/bin/env bash
# β-sweep launcher (R4) — tests whether the KL leash controls the over-optimisation collapse.
#
# Run INSIDE a fresh tmux session (so it survives disconnect):
#   tmux new -s r4b0
#   bash run_beta.sh 0.0       # KL OFF  -> predict: collapses earlier/harder than baseline
#   # (later, in a new tmux)
#   bash run_beta.sh 0.12      # tighter leash -> predict: collapse delayed / milder
#
# Baseline β=0.08 is R0 (already done) — the middle point of the sweep.
# Everything else is stock baseline (K=2, NUM_BATCHES=3738 -> MAX_STEPS=3364, ~5 h each).
B="${1:?usage: bash run_beta.sh <BETA>   e.g. 0.0  or  0.12}"
TAG="beta${B}"

source ~/venvs/tunix/bin/activate
cd ~/tpu-2026/scripts
set -a; . ./.env; set +a                       # WANDB / HF / PROTOCOL_BUFFERS=python
rm -rf data                                    # avoid the tfds reload crash

export BETA="$B"                               # config.py reads this
export CKPT_DIR="/tmp/content/ckpts_R4_${TAG}/"  # per-run dir (else Tunix resumes -> instant finish)
export WANDB_NAME="R4_${TAG}_s0"

echo "[run_beta] BETA=$BETA  CKPT_DIR=$CKPT_DIR  WANDB_NAME=$WANDB_NAME"
echo "[run_beta] expect: MAX_STEPS=3364 (~5h). Detach tmux with Ctrl-b d; DO NOT Ctrl-C."
python train.py 2>&1 | tee ~/run_R4_${TAG}.log
