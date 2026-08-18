#!/usr/bin/env bash
# Launch all v0.4 identity experiments: seeds 13/37/73 in parallel on
# gpu0/gpu1/gpu3, then run the v0.4 analysis once everything completes.
# Usage: bash scripts/run_v04_all.sh [gpu13] [gpu37] [gpu73]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
GPU13="${1:-0}"
GPU37="${2:-1}"
GPU73="${3:-3}"

cd "$ROOT_DIR"
mkdir -p logs/v04_identity
echo "[$(date --iso-8601=seconds)] launching v04 seed13->GPU$GPU13, seed37->GPU$GPU37, seed73->GPU$GPU73"

for entry in "13 $GPU13" "37 $GPU37" "73 $GPU73"; do
  set -- $entry
  seed="$1"; gpu="$2"
  if pgrep -f "run_v04_identity_multiseed.sh $gpu $seed" >/dev/null 2>&1; then
    echo "seed$seed on GPU$gpu already running; skipping launch"
  else
    nohup bash scripts/run_v04_identity_multiseed.sh "$gpu" "$seed" \
      >>"logs/v04_identity/seed$seed.orchestrator.log" 2>&1 &
    echo "launched seed$seed (PID $!) on GPU$gpu"
  fi
done

wait
echo "[$(date --iso-8601=seconds)] all v04 training queues finished"

"$PYTHON" scripts/analyze_v04_identity.py \
  --root "$ROOT_DIR/runs/v04_identity" \
  --incumbent-root "$ROOT_DIR/runs/v03_degradation" \
  --control-root "$ROOT_DIR/runs/ablation_v02_multiseed" \
  --seeds 13 37 73
