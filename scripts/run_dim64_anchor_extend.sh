#!/usr/bin/env bash
# Extend dim64+anchor training: wait for the current 5000-step run to finish,
# then resume from last.pt with max_steps=8000 + reset_scheduler (LR re-anneals).
set -u
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
cd "$ROOT_DIR"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
LOG="$ROOT_DIR/logs/train_dim64_anchor_extend.log"
mkdir -p "$(dirname "$LOG")"

# Wait for the original training process to exit (it owns GPU0).
while pgrep -f "train.py --config configs/baselines_500/ours_dim64_anchor" >/dev/null 2>&1; do
  sleep 30
done
echo "[$(date --iso-8601=seconds)] original run finished, resuming with 8000-step config" | tee -a "$LOG"

"$PYTHON" train.py \
  --config configs/baselines_500/ours_dim64_anchor/seed13.json \
  --device cuda --num-workers 8 \
  --resume runs/baselines_500/ours_dim64_anchor/seed13/last.pt \
  >>"$LOG" 2>&1
echo "[$(date --iso-8601=seconds)] extend run exit code: $?" | tee -a "$LOG"
