#!/usr/bin/env bash
# Launch one R9 training run in a detached tmux session (survives SSH logout).
#   ./run_r9.sh easy2hard R9_curriculum
#   ./run_r9.sh hard2easy R9_anticurriculum
# Reattach:  tmux attach -t r9_<curriculum>
set -euo pipefail
CUR="${1:?usage: run_r9.sh <easy2hard|hard2easy> <wandb_name>}"
NAME="${2:?usage: run_r9.sh <curriculum> <wandb_name>}"
SESSION="r9_${CUR}"
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="$HOME/venvs/tunix"
LOG="$HERE/train_${CUR}.log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session $SESSION already running — attach with: tmux attach -t $SESSION"
  exit 0
fi

INNER="cd $HERE && source $VENV/bin/activate && set -a && source ./.env && set +a && rm -rf data && CURRICULUM=$CUR WANDB_NAME=$NAME python -u train.py 2>&1 | tee '$LOG'; echo; echo '--- process exited (\$?) ---'; exec bash"
tmux new-session -d -s "$SESSION" "bash -lc \"$INNER\""
echo "Launched tmux '$SESSION' (CURRICULUM=$CUR, WANDB_NAME=$NAME)."
echo "Log: $LOG"
echo "Watch: tmux attach -t $SESSION   |   tail -f $LOG"
