#!/usr/bin/env bash
# Train one baseline model family for all seeds on one physical GPU, then run
# the independent frozen-verifier evaluation. Designed for screen sessions.
# Usage: scripts/run_baseline_multiseed.sh <GPU_ID> <FAMILY> <SESSION_NAME>
set -euo pipefail

GPU_ID="$1"
FAMILY="$2"
SESSION_NAME="$3"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
CONFIG_ROOT="$ROOT_DIR/configs/baselines/$FAMILY"
RUN_ROOT="$ROOT_DIR/runs/baselines/$FAMILY"
LOG_ROOT="$ROOT_DIR/logs/baselines/$FAMILY"
STATUS_FILE="$ROOT_DIR/runs/baselines/$FAMILY.queue.status.json"
SEEDS="13"

if [ ! -x "$PYTHON" ]; then
  echo "python executable not found: $PYTHON" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT_DIR"
exec 9>"$LOG_ROOT/$FAMILY.queue.lock"
if ! flock -n 9; then
  echo "queue $FAMILY already running" >&2
  exit 3
fi

STOPPED=0
write_status() {
  local status="$1"; local code="$2"
  STATUS="$status" CODE="$code" GPU_ID="$GPU_ID" FAMILY="$FAMILY" SESSION="$SESSION_NAME" STATUS_FILE="$STATUS_FILE" CURRENT="$CURRENT_EXPERIMENT" "$PYTHON" - <<'PY'
import json, os
from datetime import datetime, timezone
payload = {"status": os.environ["STATUS"], "exit_code": int(os.environ["CODE"]), "physical_gpu": os.environ["GPU_ID"], "family": os.environ["FAMILY"], "session_name": os.environ["SESSION"], "current_experiment": os.environ.get("CURRENT", ""), "timestamp_utc": datetime.now(timezone.utc).isoformat()}
open(os.environ["STATUS_FILE"], "w", encoding="utf-8").write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
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
  "$PYTHON" - "$RUN_ROOT/seed$1" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
required = ("best.pt", "last.pt", "best_validation.json", "last_validation.json", "validation_history.jsonl", "train_log.jsonl", "resolved_config.json", "summary.json", "frozen_verifier_val.json", "frozen_verifier_val.csv", "frozen_verifier_val_pairs.csv")
if not all((p / name).exists() for name in required): raise SystemExit(1)
if int(json.loads((p / "summary.json").read_text())["steps"]) != 5000: raise SystemExit(1)
PY
}

CURRENT_EXPERIMENT=""
write_status running 0
for seed in $SEEDS; do
  CURRENT_EXPERIMENT="${FAMILY}_seed$seed"
  write_status running 0
  config_path="$CONFIG_ROOT/seed$seed.json"
  output_dir="$RUN_ROOT/seed$seed"
  train_log="$LOG_ROOT/seed$seed.train.log"
  eval_log="$LOG_ROOT/seed$seed.verifier_eval.log"
  mkdir -p "$output_dir"
  echo "[$(date --iso-8601=seconds)] START $FAMILY seed$seed GPU$GPU_ID session=$SESSION_NAME" | tee -a "$train_log"
  if is_complete "$seed"; then
    echo "[$(date --iso-8601=seconds)] SKIP completed $FAMILY seed$seed" | tee -a "$train_log"
  elif [ -f "$output_dir/last.pt" ]; then
    echo "[$(date --iso-8601=seconds)] RESUME $output_dir/last.pt" | tee -a "$train_log"
    "$PYTHON" train.py --config "$config_path" --device cuda --num-workers 8 --resume "$output_dir/last.pt" >>"$train_log" 2>&1
  else
    echo "[$(date --iso-8601=seconds)] FRESH $output_dir" | tee -a "$train_log"
    "$PYTHON" train.py --config "$config_path" --device cuda --num-workers 8 >>"$train_log" 2>&1
  fi
  if [ ! -f "$output_dir/frozen_verifier_val.json" ] || ! is_complete "$seed"; then
    "$PYTHON" scripts/evaluate_frozen_verifier.py \
      --checkpoint "$output_dir/best.pt" \
      --verifier "$ROOT_DIR/../v05_frozen_verifier/evaluator/best.pt" \
      --data-root data/plamd_vari_grip_pilot \
      --output "$output_dir/frozen_verifier_val.json" \
      --split val --device cuda --method "$FAMILY" --seed "$seed" >>"$eval_log" 2>&1
  fi
  if ! is_complete "$seed"; then
    echo "[$(date --iso-8601=seconds)] VALIDATION FAILED $FAMILY seed$seed" >&2
    exit 4
  fi
  echo "[$(date --iso-8601=seconds)] END $FAMILY seed$seed" | tee -a "$train_log"
done
CURRENT_EXPERIMENT=""
write_status running 0
echo "[$(date --iso-8601=seconds)] DONE $FAMILY" | tee -a "$LOG_ROOT/$FAMILY.overall.log"
