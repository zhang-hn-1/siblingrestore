#!/usr/bin/env bash
# v0.5 orchestrator: enforce at most two physical GPUs and named screen
# sessions; run verifier setup first, audit it, evaluate controls with the
# independent evaluator, then launch the two restoration queues (seeds 13/37
# first, seed 73 on whichever queue frees up), then run the analyzer.
# Usage: bash scripts/run_v05_all.sh [gpu0] [gpu1]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
RUN_ROOT="$ROOT_DIR/runs/v05_frozen_verifier"
LOG_ROOT="$ROOT_DIR/logs/v05_frozen_verifier"
VERIFIER_STATUS="$RUN_ROOT/verifier_status.json"
VERIFIER_SESSION="siblingrestore-v05-verifier"
CONTROLS="independent group degradation_001 degradation_001_output_001"
SEEDS="13 37 73"

if ! command -v screen >/dev/null 2>&1; then
  echo "screen is required for v0.5 formal runs" >&2
  exit 2
fi

GPU_A="${1:-0}"
GPU_B="${2:-1}"
for gpu in "$GPU_A" "$GPU_B"; do
  if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
    echo "invalid GPU id: $gpu" >&2
    exit 2
  fi
done
if [ "$GPU_A" = "$GPU_B" ]; then
  echo "duplicate GPU id rejected: $GPU_A" >&2
  exit 2
fi

cd "$ROOT_DIR"
mkdir -p "$RUN_ROOT/queues" "$LOG_ROOT"
echo "[$(date --iso-8601=seconds)] v0.5 orchestrator start: queues on physical GPU$GPU_A and GPU$GPU_B"

screen_exists() {
  screen -ls 2>/dev/null | grep -q "$1\." || return 1
}

wait_status() {
  local status_file="$1"; local name="$2"
  for _ in $(seq 1 720); do
    if [ -f "$status_file" ]; then
      if grep -q '"status": "completed"' "$status_file"; then
        return 0
      fi
      if grep -q '"status": "failed"\|"status": "stopped"' "$status_file"; then
        echo "$name failed or stopped: $(cat "$status_file")" >&2
        return 1
      fi
    fi
    sleep 30
  done
  echo "$name did not complete in time" >&2
  return 1
}

if screen_exists "$VERIFIER_SESSION"; then
  echo "verifier screen $VERIFIER_SESSION already exists; not relaunching" >&2
else
  screen -dmS "$VERIFIER_SESSION" bash -lc \
    "cd '$ROOT_DIR' && bash scripts/run_v05_verifiers.sh $GPU_A $VERIFIER_SESSION"
  echo "launched verifier screen $VERIFIER_SESSION on physical GPU$GPU_A"
fi

echo "[$(date --iso-8601=seconds)] waiting for verifier audit..."
if ! wait_status "$VERIFIER_STATUS" "verifier setup"; then
  exit 4
fi
echo "[$(date --iso-8601=seconds)] verifier setup audited OK"

TEACHER_SHA="$("$PYTHON" - <<'PY'
import json
from pathlib import Path
fp = json.loads(Path("runs/v05_frozen_verifier/teacher/checkpoint_fingerprint.json").read_text())
print(fp["sha256"])
PY
)"
"$PYTHON" scripts/make_v05_configs.py --teacher-sha "$TEACHER_SHA"
echo "[$(date --iso-8601=seconds)] restoration configs regenerated with teacher sha $TEACHER_SHA"

echo "[$(date --iso-8601=seconds)] evaluating controls with independent evaluator on GPU$GPU_A"
for method in $CONTROLS; do
  for seed in $SEEDS; do
    output_dir="$RUN_ROOT/evaluations/$method/seed$seed"
    out_json="$output_dir/frozen_verifier_val.json"
    if [ -f "$out_json" ]; then
      echo "skip existing control evaluation $method seed$seed"
      continue
    fi
    case "$method" in
      independent) ckpt="runs/v02_backbone_decision/independent/seed$seed/best.pt" ;;
      degradation_001_output_001) ckpt="runs/v03_degradation/$method/seed$seed/best.pt" ;;
      *) ckpt="runs/ablation_v02_multiseed/seed$seed/$method/best.pt" ;;
    esac
    mkdir -p "$output_dir"
    echo "[$(date --iso-8601=seconds)] control eval $method seed$seed"
    CUDA_VISIBLE_DEVICES="$GPU_A" "$PYTHON" scripts/evaluate_frozen_verifier.py \
      --checkpoint "$ckpt" --verifier "$RUN_ROOT/evaluator/best.pt" \
      --data-root data/plamd_vari_grip_pilot --output "$out_json" \
      --split val --device cuda --method "$method" --seed "$seed"
  done
done
echo "[$(date --iso-8601=seconds)] control evaluations complete"

launch_queue() {
  local seed="$1"; local gpu="$2"; local session="$3"
  if screen_exists "$session"; then
    echo "session $session already exists; not relaunching" >&2
    return 0
  fi
  screen -dmS "$session" bash -lc \
    "cd '$ROOT_DIR' && bash scripts/run_v05_frozen_anchor_multiseed.sh $gpu $seed $session"
  echo "launched seed$seed on physical GPU$gpu (session $session)"
}

SESSION_A="siblingrestore-v05-gpu$GPU_A"
SESSION_B="siblingrestore-v05-gpu$GPU_B"

launch_queue 13 "$GPU_A" "$SESSION_A"
launch_queue 37 "$GPU_B" "$SESSION_B"
wait_status "$RUN_ROOT/queues/seed13.gpu$GPU_A.status.json" "seed13 gpu$GPU_A" || exit 5
wait_status "$RUN_ROOT/queues/seed37.gpu$GPU_B.status.json" "seed37 gpu$GPU_B" || exit 5
echo "[$(date --iso-8601=seconds)] round 1 complete; scheduling seed73 on GPU$GPU_A"
launch_queue 73 "$GPU_A" "$SESSION_A"
wait_status "$RUN_ROOT/queues/seed73.gpu$GPU_A.status.json" "seed73 gpu$GPU_A" || exit 5
echo "[$(date --iso-8601=seconds)] all restoration queues completed"

"$PYTHON" scripts/analyze_v05_frozen_anchor.py \
  --root "$RUN_ROOT" --seeds 13 37 73
echo "[$(date --iso-8601=seconds)] analysis complete; see $RUN_ROOT/v05_report.md"
echo "attach: screen -ls ; screen -r $VERIFIER_SESSION ; screen -r $SESSION_A ; screen -r $SESSION_B"
