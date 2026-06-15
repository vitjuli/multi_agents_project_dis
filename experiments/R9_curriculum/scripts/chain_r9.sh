#!/usr/bin/env bash
# Autonomous R9 chainer: waits for the easy2hard run to finish CLEANLY, then
# launches the hard2easy control. Runs in its own tmux session (r9_chain) so it
# survives SSH/laptop disconnects. Does nothing destructive; only launches.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
C="$HERE/chain.log"
log(){ echo "[chain $(date '+%F %T')] $*" | tee -a "$C"; }

log "watching easy2hard; will launch hard2easy when it finishes cleanly."
# Wait until the training process is gone (poll every 60s).
while pgrep -f "python -u train.py" >/dev/null 2>&1; do sleep 60; done
log "train.py process ended."

if grep -q "Training finished." "$HERE/train_easy2hard.log"; then
  log "easy2hard finished cleanly -> launching hard2easy (run R10)."
  "$HERE/run_r9.sh" hard2easy R10_anticurriculum 2>&1 | tee -a "$C"
  log "hard2easy launched (tmux r9_hard2easy, W&B name R10_anticurriculum)."
else
  log "WARNING: easy2hard did NOT finish cleanly (no 'Training finished.' in log). NOT launching hard2easy — inspect train_easy2hard.log."
fi
