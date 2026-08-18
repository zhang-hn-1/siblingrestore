#!/usr/bin/env bash
# Train and independently evaluate one seed for both v0.5 frozen-anchor methods
# on one physical GPU. Designed to run inside a named screen session.
# Usage: scripts/run_v05_frozen_anchor_multiseed.sh <GPU_ID> <SEED> <SESSION_NAME>
set -euo pipefail

GPU_ID="$1"
SEED="$2"
SESSION_NAME="$3"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
CONFIG_ROOT="$ROOT_DIR/configs/v05_frozen_verifier"
RUN_ROOT="$ROOT_DIR/runs/v05_frozen_verifier"
LOG_ROOT="$ROOT_DIR/logs/v05_frozen_verifier"
EVALUATOR_CHECKPOINT="$RUN_ROOT/evaluator/best.pt"
STATUS_FILE="$RUN_ROOT/queues/seed$SEED.gpu$GPU_ID.status.json"
METHODS="frozen_anchor_001 frozen_anchor_pcgrad_001"

if [ ! -x "$PYTHON" ]; then
  echo "python executable not found: $PYTHON" >&2
  exit 2
fi
if [ ! -f "$EVALUATOR_CHECKPOINT" ]; then
  echo "evaluator checkpoint missing: $EVALUATOR_CHECKPOINT" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
mkdir -p "$RUN_ROOT/queues" "$LOG_ROOT"
cd "$ROOT_DIR"
exec 9>"$LOG_ROOT/seed$SEED.gpu$GPU_ID.lock"
if ! flock -n 9; then
  echo "queue seed$SEED gpu$GPU_ID is already running" >&2
  exit 3
fi

STOPPED=0
write_status() {
  local status="$1"; local code="$2"
  STATUS="$status" CODE="$code" GPU_ID="$GPU_ID" SEED="$SEED" SESSION="$SESSION_NAME" CURRENT="$CURRENT_EXPERIMENT" STATUS_FILE="$STATUS_FILE" "$PYTHON" - <<'PY'
import json, os
from datetime import datetime, timezone
path = os.environ["STATUS_FILE"]
payload = {
    "status": os.environ["STATUS"],
    "exit_code": int(os.environ["CODE"]),
    "physical_gpu": os.environ["GPU_ID"],
    "seed": int(os.environ["SEED"]),
    "session_name": os.environ["SESSION"],
    "current_experiment": os.environ.get("CURRENT", ""),
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
}
open(path, "w", encoding="utf-8").write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
PY
}

write_running() {
  STATUS="running" CODE=0 GPU_ID="$GPU_ID" SEED="$SEED" SESSION="$SESSION_NAME" CURRENT="$CURRENT_EXPERIMENT" STATUS_FILE="$STATUS_FILE" "$PYTHON" - <<'PY'
import json, os
from datetime import datetime, timezone
path = os.environ["STATUS_FILE"]
payload = {
    "status": "running",
    "exit_code": 0,
    "physical_gpu": os.environ["GPU_ID"],
    "seed": int(os.environ["SEED"]),
    "session_name": os.environ["SESSION"],
    "current_experiment": os.environ.get("CURRENT", ""),
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
}
open(path, "w", encoding="utf-8").write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
PY
}

on_exit() {
  local code=$?
  if [ "$STOPPED" -eq 1 ]; then
    write_status stopped 143
  elif [ "$code" -eq 0 ]; then
    write_status completed 0
  else
    write_status failed "$code"
  fi
  exit "$code"
}
trap 'STOPPED=1; exit 143' TERM INT
trap on_exit EXIT

is_complete() {
  "$PYTHON" - "$RUN_ROOT/$1/seed$SEED" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
required = ("best.pt", "last.pt", "best_validation.json", "last_validation.json", "validation_history.jsonl", "train_log.jsonl", "resolved_config.json", "summary.json", "frozen_verifier_val.json", "frozen_verifier_val.csv", "frozen_verifier_val_pairs.csv")
if not all((p / name).exists() for name in required): raise SystemExit(1)
if int(json.loads((p / "summary.json").read_text())["steps"]) != 5000: raise SystemExit(1)
f = json.loads((p / "frozen_verifier_val.json").read_text())
if f.get("split") != "val" or f.get("sources") != 16 or f.get("restored_views") != 96: raise SystemExit(1)
PY
}

CURRENT_EXPERIMENT=""
write_running
for method in $METHODS; do
  CURRENT_EXPERIMENT="${method}_seed$SEED"
  write_running
  config_path="$CONFIG_ROOT/$method/seed$SEED.json"
  output_dir="$RUN_ROOT/$method/seed$SEED"
  train_log="$LOG_ROOT/$method.seed$SEED.train.log"
  eval_log="$LOG_ROOT/$method.seed$SEED.verifier_eval.log"
  mkdir -p "$output_dir"
  echo "[$(date --iso-8601=seconds)] START $method seed$SEED GPU$GPU_ID session=$SESSION_NAME" | tee -a "$train_log"
  if is_complete "$method"; then
    echo "[$(date --iso-8601=seconds)] SKIP completed $method seed$SEED" | tee -a "$train_log"
  elif [ -f "$output_dir/last.pt" ]; then
    echo "[$(date --iso-8601=seconds)] RESUME $output_dir/last.pt" | tee -a "$train_log"
    "$PYTHON" train.py --config "$config_path" --device cuda --num-workers 8 --resume "$output_dir/last.pt" >>"$train_log" 2>&1
  else
    echo "[$(date --iso-8601=seconds)] FRESH $output_dir" | tee -a "$train_log"
    "$PYTHON" train.py --config "$config_path" --device cuda --num-workers 8 >>"$train_log" 2>&1
  fi
  if [ ! -f "$output_dir/frozen_verifier_val.json" ] || ! is_complete "$method"; then
    "$PYTHON" scripts/evaluate_frozen_verifier.py \
      --checkpoint "$output_dir/best.pt" \
      --verifier "$EVALUATOR_CHECKPOINT" \
      --data-root data/plamd_vari_grip_pilot \
      --output "$output_dir/frozen_verifier_val.json" \
      --split val --device cuda --method "$method" --seed "$SEED" >>"$eval_log" 2>&1
  fi
  if ! is_complete "$method"; then
    echo "[$(date --iso-8601=seconds)] VALIDATION FAILED $method seed$SEED" >&2
    exit 4
  fi
  echo "[$(date --iso-8601=seconds)] END $method seed$SEED" | tee -a "$train_log"
done
CURRENT_EXPERIMENT=""
write_running
echo "[$(date --iso-8601=seconds)] DONE all methods seed$SEED gpu$GPU_ID" | tee -a "$LOG_ROOT/seed$SEED.gpu$GPU_ID.overall.log"
