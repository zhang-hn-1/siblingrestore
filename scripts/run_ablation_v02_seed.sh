#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 GPU_ID SEED" >&2
  exit 2
fi

GPU_ID="$1"
SEED="$2"
case "$SEED" in
  37|73) ;;
  *) echo "only seed 37 or 73 is allowed; seed13 is reused, not retrained" >&2; exit 2 ;;
esac

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
CONFIG_ROOT="$ROOT_DIR/configs/ablation_v02_multiseed/seed$SEED"
RUN_ROOT="$ROOT_DIR/runs/ablation_v02_multiseed/seed$SEED"
LOG_ROOT="$ROOT_DIR/logs/ablation_v02_multiseed/seed$SEED"

if [ ! -x "$PYTHON" ]; then
  echo "python executable not found: $PYTHON" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT_DIR"

exec 9>"$LOG_ROOT/queue.lock"
if ! flock -n 9; then
  echo "seed$SEED queue is already running" >&2
  exit 3
fi

write_status() {
  local status="$1"
  local code="$2"
  if [ -z "$code" ]; then code=0; fi
  STATUS="$status" CODE="$code" GPU_ID="$GPU_ID" SEED="$SEED" CURRENT="$CURRENT_EXPERIMENT" "$PYTHON" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

path = Path("runs/ablation_v02_multiseed") / f"seed{os.environ['SEED']}" / "queue_status.json"
payload = {
    "status": os.environ["STATUS"],
    "exit_code": int(os.environ["CODE"]),
    "seed": int(os.environ["SEED"]),
    "physical_gpu_requested": os.environ["GPU_ID"],
    "current_experiment": os.environ.get("CURRENT", ""),
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
}
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

on_exit() {
  local code=$?
  if [ "$code" -eq 0 ]; then
    write_status completed 0
  else
    write_status failed "$code"
  fi
  exit "$code"
}
trap on_exit EXIT

is_complete() {
  "$PYTHON" - "$RUN_ROOT/$1" <<'PY'
import json
import sys
from pathlib import Path

directory = Path(sys.argv[1])
required = ("best.pt", "last.pt", "best_validation.json", "last_validation.json", "validation_history.jsonl", "source_fidelity_val.json", "source_fidelity_val.csv", "summary.json")
if not all((directory / name).exists() for name in required):
    raise SystemExit(1)
summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
if int(summary.get("steps", 0)) < 5000:
    raise SystemExit(1)
fidelity = json.loads((directory / "source_fidelity_val.json").read_text(encoding="utf-8"))
if fidelity.get("sources") != 16 or fidelity.get("restored_views") != 96:
    raise SystemExit(1)
PY
}

validate_experiment() {
  "$PYTHON" - "$RUN_ROOT/$1" "$SEED" "$GPU_ID" <<'PY'
import json
import math
import sys
from pathlib import Path

directory = Path(sys.argv[1])
seed = int(sys.argv[2])
gpu = sys.argv[3]
summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
assert int(summary["steps"]) == 5000, summary
assert summary["device"] == "cuda", summary
config = json.loads((directory / "resolved_config.json").read_text(encoding="utf-8"))
assert int(config["seed"]) == seed, config
assert int(config["num_workers"]) == 8, config
assert int(config["crop_size"]) == 256, config
assert int(config["max_steps"]) == 5000, config
assert int(config["validation_interval_steps"]) == 500, config
logs = [json.loads(line) for line in (directory / "train_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
assert logs and int(logs[-1]["step"]) == 5000, len(logs)
for row in logs:
    for value in row.values():
        if isinstance(value, (int, float)):
            assert math.isfinite(float(value)), (directory, row)
fidelity = json.loads((directory / "source_fidelity_val.json").read_text(encoding="utf-8"))
assert fidelity["sources"] == 16 and fidelity["restored_views"] == 96, fidelity
print(json.dumps({"seed": seed, "gpu": gpu, "experiment": directory.name, "steps": 5000, "validation_points": len((directory / "validation_history.jsonl").read_text(encoding="utf-8").splitlines()), "fidelity_views": 96}, ensure_ascii=False))
PY
}

all_complete() {
  for name in group source_0003 degradation_001; do
    if ! is_complete "$name"; then return 1; fi
  done
}

CURRENT_EXPERIMENT=""
write_status running 0
for name in group source_0003 degradation_001; do
  CURRENT_EXPERIMENT="$name"
  output_dir="$RUN_ROOT/$name"
  train_log="$LOG_ROOT/$name.train.log"
  eval_log="$LOG_ROOT/$name.source_fidelity.log"
  was_complete=0
  echo "[$(date --iso-8601=seconds)] START seed$SEED/$name GPU$GPU_ID" | tee -a "$train_log"
  if is_complete "$name"; then
    was_complete=1
    echo "[$(date --iso-8601=seconds)] SKIP completed seed$SEED/$name" | tee -a "$train_log"
  else
    if [ -f "$output_dir/last.pt" ]; then
      echo "[$(date --iso-8601=seconds)] RESUME $output_dir/last.pt" | tee -a "$train_log"
      "$PYTHON" train.py --config "$CONFIG_ROOT/$name.json" --device cuda --num-workers 8 --resume "$output_dir/last.pt" >>"$train_log" 2>&1
    else
      echo "[$(date --iso-8601=seconds)] FRESH $output_dir" | tee -a "$train_log"
      "$PYTHON" train.py --config "$CONFIG_ROOT/$name.json" --device cuda --num-workers 8 >>"$train_log" 2>&1
    fi
  fi
  if [ "$was_complete" -eq 0 ] || [ ! -f "$output_dir/source_fidelity_val.json" ]; then
    "$PYTHON" evaluate_source_fidelity.py --checkpoint "$output_dir/best.pt" --data-root data/plamd_vari_grip_pilot --split val --output "$output_dir/source_fidelity_val.json" --device cuda >>"$eval_log" 2>&1
  else
    echo "[$(date --iso-8601=seconds)] SKIP existing source fidelity" | tee -a "$eval_log"
  fi
  validate_experiment "$name" | tee -a "$train_log"
  echo "[$(date --iso-8601=seconds)] END seed$SEED/$name" | tee -a "$train_log"
done

CURRENT_EXPERIMENT=""
if all_complete && [ -d "$ROOT_DIR/runs/ablation_v02_multiseed/seed13/group" ] && [ -d "$ROOT_DIR/runs/ablation_v02_multiseed/seed37/group" ] && [ -d "$ROOT_DIR/runs/ablation_v02_multiseed/seed73/group" ]; then
  "$PYTHON" analyze_loss_ablation.py --multiseed-root "$ROOT_DIR/runs/ablation_v02_multiseed" --seeds 13 37 73 --bootstrap-rounds 10000
fi
