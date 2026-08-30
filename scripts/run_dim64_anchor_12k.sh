#!/usr/bin/env bash
# Wait for dim64 8000-step run to finish, reset LR for warm restart, resume to 12000.
set -u
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
cd "$ROOT_DIR"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
LOG="$ROOT_DIR/logs/train_dim64_anchor_12k.log"
mkdir -p "$(dirname "$LOG")"

while pgrep -f "train.py --config configs/baselines_500/ours_dim64_anchor" >/dev/null 2>&1; do
  sleep 30
done
echo "[$(date --iso-8601=seconds)] dim64 8000-step run finished; resetting LR" | tee -a "$LOG"

"$PYTHON" - <<'PY'
import torch
p = "runs/baselines_500/ours_dim64_anchor/seed13/last.pt"
ck = torch.load(p, map_location="cpu", weights_only=False)
ck["optimizer"]["param_groups"][0]["lr"] = 2e-4
ck["optimizer"]["param_groups"][0]["initial_lr"] = 2e-4
torch.save(ck, p)
print("dim64 last.pt lr -> 2e-4 (step %s)" % ck.get("step"))
PY

"$PYTHON" train.py \
  --config configs/baselines_500/ours_dim64_anchor/seed13.json \
  --device cuda --num-workers 8 \
  --resume runs/baselines_500/ours_dim64_anchor/seed13/last.pt \
  >>"$LOG" 2>&1
echo "[$(date --iso-8601=seconds)] dim64 12k run exit code: $?" | tee -a "$LOG"
