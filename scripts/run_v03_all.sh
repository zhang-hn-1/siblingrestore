#!/usr/bin/env bash
# Launch all v0.3 degradation experiments: seed 13/37/73 in parallel on
# gpu0/gpu1/gpu3, then run the v0.3 analysis once everything completes.
# Usage: bash scripts/run_v03_all.sh [gpu_for_seed13] [gpu_for_seed37] [gpu_for_seed73]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
GPU13="${1:-0}"
GPU37="${2:-1}"
GPU73="${3:-3}"

cd "$ROOT_DIR"
mkdir -p logs/v03_degradation
echo "[$(date --iso-8601=seconds)] launching seed13->GPU$GPU13, seed37->GPU$GPU37, seed73->GPU$GPU73"

for entry in "13 $GPU13" "37 $GPU37" "73 $GPU73"; do
  set -- $entry
  seed="$1"; gpu="$2"
  if pgrep -f "run_v03_degradation_multiseed.sh $gpu $seed" >/dev/null 2>&1; then
    echo "seed$seed on GPU$gpu already running; skipping launch"
  else
    nohup bash scripts/run_v03_degradation_multiseed.sh "$gpu" "$seed" \
      >>"logs/v03_degradation/seed$seed.orchestrator.log" 2>&1 &
    echo "launched seed$seed (PID $!) on GPU$gpu"
  fi
done

wait
echo "[$(date --iso-8601=seconds)] all v03 training queues finished"

"$PYTHON" scripts/analyze_v03_degradation.py \
  --root "$ROOT_DIR/runs/v03_degradation" \
  --control-root "$ROOT_DIR/runs/ablation_v02_multiseed" \
  --seeds 13 37 73
