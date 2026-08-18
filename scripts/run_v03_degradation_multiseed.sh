#!/usr/bin/env bash
# Train one seed across all four v0.3 degradation methods on one GPU.
# Usage: scripts/run_v03_degradation_multiseed.sh <GPU_ID> <SEED>
set -euo pipefail

GPU_ID="$1"
SEED="$2"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
CONFIG_ROOT="$ROOT_DIR/configs/v03_degradation"
RUN_ROOT="$ROOT_DIR/runs/v03_degradation"
LOG_ROOT="$ROOT_DIR/logs/v03_degradation"
METHODS="degradation_003 degradation_005 degradation_001_output_001 degradation_cond_001"

if [ ! -x "$PYTHON" ]; then
  echo "python executable not found: $PYTHON" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT_DIR"
exec 9>"$LOG_ROOT/seed$SEED.queue.lock"
if ! flock -n 9; then
  echo "v03 seed$SEED queue is already running" >&2
  exit 3
fi

write_status() {
  local status="$1"
  local code="$2"
  STATUS="$status" CODE="$code" GPU_ID="$GPU_ID" SEED="$SEED" CURRENT="$CURRENT_EXPERIMENT" "$PYTHON" - <<'PY'
import json, os
from datetime import datetime, timezone
from pathlib import Path
path = Path(f"runs/v03_degradation/seed{os.environ['SEED']}.queue_status.json")
path.write_text(json.dumps({
    "status": os.environ["STATUS"],
    "exit_code": int(os.environ["CODE"]),
    "physical_gpu_requested": os.environ["GPU_ID"],
    "seed": int(os.environ["SEED"]),
    "current_experiment": os.environ.get("CURRENT", ""),
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

on_exit() {
  local code=$?
  if [ "$code" -eq 0 ]; then write_status completed 0; else write_status failed "$code"; fi
  exit "$code"
}
trap on_exit EXIT

is_complete() {
  "$PYTHON" - "$RUN_ROOT/$1/seed$SEED" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
required = ("best.pt", "last.pt", "best_validation.json", "last_validation.json", "validation_history.jsonl", "train_log.jsonl", "resolved_config.json", "summary.json", "source_fidelity_val.json", "source_fidelity_val.csv")
if not all((p / name).exists() for name in required): raise SystemExit(1)
if int(json.loads((p / "summary.json").read_text())["steps"]) != 5000: raise SystemExit(1)
f = json.loads((p / "source_fidelity_val.json").read_text())
if f.get("split") != "val" or f.get("sources") != 16 or f.get("restored_views") != 96: raise SystemExit(1)
PY
}

is_trained() {
  "$PYTHON" - "$RUN_ROOT/$1/seed$SEED" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
required = ("best.pt", "last.pt", "best_validation.json", "last_validation.json", "validation_history.jsonl", "train_log.jsonl", "resolved_config.json", "summary.json")
if not all((p / name).exists() for name in required): raise SystemExit(1)
if int(json.loads((p / "summary.json").read_text())["steps"]) != 5000: raise SystemExit(1)
PY
}

validate_seed() {
  "$PYTHON" - "$RUN_ROOT/$1/seed$SEED" "$1" "$SEED" "$GPU_ID" <<'PY'
import json, math, sys
from pathlib import Path
p = Path(sys.argv[1])
method = sys.argv[2]
seed = int(sys.argv[3])
gpu = sys.argv[4]
config = json.loads((p / "resolved_config.json").read_text())
summary = json.loads((p / "summary.json").read_text())
assert config["mode"] == "sibling" and int(config["seed"]) == seed
assert config["device"] == "cuda" and int(config["num_workers"]) == 8
assert int(config["crop_size"]) == 256 and int(config["max_steps"]) == 5000
assert int(config["validation_interval_steps"]) == 500
assert int(config["batch_size"]) == 4 and int(config["sibling_count"]) == 2
assert int(summary["steps"]) == 5000
expected_params = 1222689 if bool(config["model"].get("degradation_conditioned", False)) else 1193121
assert int(summary["parameters"]) == expected_params, (summary["parameters"], expected_params)
logs = [json.loads(x) for x in (p / "train_log.jsonl").read_text().splitlines() if x.strip()]
assert logs and int(logs[-1]["step"]) == 5000
for row in logs:
    for value in row.values():
        if isinstance(value, (int, float)): assert math.isfinite(float(value))
f = json.loads((p / "source_fidelity_val.json").read_text())
assert f["split"] == "val" and f["sources"] == 16 and f["restored_views"] == 96
print(json.dumps({"method": method, "seed": seed, "gpu": gpu, "steps": 5000, "parameters": summary["parameters"], "validation_points": len((p / "validation_history.jsonl").read_text().splitlines()), "fidelity_views": 96}, ensure_ascii=False))
PY
}

CURRENT_EXPERIMENT=""
write_status running 0
for method in $METHODS; do
  CURRENT_EXPERIMENT="${method}_seed$SEED"
  config_path="$CONFIG_ROOT/$method/seed$SEED.json"
  output_dir="$RUN_ROOT/$method/seed$SEED"
  train_log="$LOG_ROOT/$method.seed$SEED.train.log"
  eval_log="$LOG_ROOT/$method.seed$SEED.source_fidelity.log"
  mkdir -p "$output_dir"
  echo "[$(date --iso-8601=seconds)] START $method seed$SEED GPU$GPU_ID" | tee -a "$train_log"
  if is_complete "$method"; then
    echo "[$(date --iso-8601=seconds)] SKIP completed $method seed$SEED" | tee -a "$train_log"
  elif is_trained "$method"; then
    echo "[$(date --iso-8601=seconds)] TRAINED; running fidelity eval for $method seed$SEED" | tee -a "$train_log"
  elif [ -f "$output_dir/last.pt" ]; then
    echo "[$(date --iso-8601=seconds)] RESUME $output_dir/last.pt" | tee -a "$train_log"
    "$PYTHON" train.py --config "$config_path" --device cuda --num-workers 8 --resume "$output_dir/last.pt" >>"$train_log" 2>&1
  else
    echo "[$(date --iso-8601=seconds)] FRESH $output_dir" | tee -a "$train_log"
    "$PYTHON" train.py --config "$config_path" --device cuda --num-workers 8 >>"$train_log" 2>&1
  fi
  if [ ! -f "$output_dir/source_fidelity_val.json" ] || ! is_complete "$method"; then
    "$PYTHON" evaluate_source_fidelity.py --checkpoint "$output_dir/best.pt" --data-root data/plamd_vari_grip_pilot --split val --output "$output_dir/source_fidelity_val.json" --device cuda >>"$eval_log" 2>&1
  fi
  validate_seed "$method" | tee -a "$train_log"
  echo "[$(date --iso-8601=seconds)] END $method seed$SEED" | tee -a "$train_log"
done
CURRENT_EXPERIMENT=""
echo "[$(date --iso-8601=seconds)] DONE all methods seed$SEED" | tee -a "$LOG_ROOT/seed$SEED.overall.log"
