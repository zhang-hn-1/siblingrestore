#!/usr/bin/env bash
set -euo pipefail

GPU_ID="$1"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
CONFIG_ROOT="$ROOT_DIR/configs/v02_backbone_decision/independent"
RUN_ROOT="$ROOT_DIR/runs/v02_backbone_decision/independent"
LOG_ROOT="$ROOT_DIR/logs/v02_backbone_decision/independent"

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
  echo "independent queue is already running" >&2
  exit 3
fi

write_status() {
  local status="$1"
  local code="$2"
  STATUS="$status" CODE="$code" GPU_ID="$GPU_ID" CURRENT="$CURRENT_EXPERIMENT" "$PYTHON" - <<'PY'
import json, os
from datetime import datetime, timezone
from pathlib import Path
path = Path("runs/v02_backbone_decision/independent/queue_status.json")
path.write_text(json.dumps({
    "status": os.environ["STATUS"],
    "exit_code": int(os.environ["CODE"]),
    "physical_gpu_requested": os.environ["GPU_ID"],
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
  "$PYTHON" - "$RUN_ROOT/seed$1" <<'PY'
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

validate_seed() {
  "$PYTHON" - "$RUN_ROOT/seed$1" "$1" "$GPU_ID" <<'PY'
import json, math, sys
from pathlib import Path
p = Path(sys.argv[1])
seed = int(sys.argv[2])
gpu = sys.argv[3]
config = json.loads((p / "resolved_config.json").read_text())
summary = json.loads((p / "summary.json").read_text())
assert config["mode"] == "independent" and int(config["seed"]) == seed
assert config["device"] == "cuda" and int(config["num_workers"]) == 8
assert int(config["crop_size"]) == 256 and int(config["max_steps"]) == 5000
assert int(config["validation_interval_steps"]) == 500
assert int(summary["steps"]) == 5000 and int(summary["parameters"]) == 1193121
logs = [json.loads(x) for x in (p / "train_log.jsonl").read_text().splitlines() if x.strip()]
assert logs and int(logs[-1]["step"]) == 5000
for row in logs:
    for value in row.values():
        if isinstance(value, (int, float)): assert math.isfinite(float(value))
f = json.loads((p / "source_fidelity_val.json").read_text())
assert f["split"] == "val" and f["sources"] == 16 and f["restored_views"] == 96
print(json.dumps({"seed": seed, "gpu": gpu, "steps": 5000, "validation_points": len((p / "validation_history.jsonl").read_text().splitlines()), "fidelity_views": 96}, ensure_ascii=False))
PY
}

CURRENT_EXPERIMENT=""
write_status running 0
for seed in 13 37 73; do
  CURRENT_EXPERIMENT="independent_seed$seed"
  output_dir="$RUN_ROOT/seed$seed"
  train_log="$LOG_ROOT/seed$seed.train.log"
  eval_log="$LOG_ROOT/seed$seed.source_fidelity.log"
  mkdir -p "$output_dir"
  echo "[$(date --iso-8601=seconds)] START independent seed$seed GPU$GPU_ID" | tee -a "$train_log"
  if is_complete "$seed"; then
    echo "[$(date --iso-8601=seconds)] SKIP completed seed$seed" | tee -a "$train_log"
  elif [ -f "$output_dir/last.pt" ]; then
    echo "[$(date --iso-8601=seconds)] RESUME $output_dir/last.pt" | tee -a "$train_log"
    "$PYTHON" train.py --config "$CONFIG_ROOT/seed$seed.json" --device cuda --num-workers 8 --resume "$output_dir/last.pt" >>"$train_log" 2>&1
  else
    echo "[$(date --iso-8601=seconds)] FRESH $output_dir" | tee -a "$train_log"
    "$PYTHON" train.py --config "$CONFIG_ROOT/seed$seed.json" --device cuda --num-workers 8 >>"$train_log" 2>&1
  fi
  if [ ! -f "$output_dir/source_fidelity_val.json" ] || ! is_complete "$seed"; then
    "$PYTHON" evaluate_source_fidelity.py --checkpoint "$output_dir/best.pt" --data-root data/plamd_vari_grip_pilot --split val --output "$output_dir/source_fidelity_val.json" --device cuda >>"$eval_log" 2>&1
  fi
  validate_seed "$seed" | tee -a "$train_log"
  echo "[$(date --iso-8601=seconds)] END independent seed$seed" | tee -a "$train_log"
done

CURRENT_EXPERIMENT=""
"$PYTHON" scripts/analyze_v02_backbone_decision.py --root "$ROOT_DIR/runs/v02_backbone_decision" --baseline-root "$ROOT_DIR/runs/ablation_v02_multiseed" --seeds 13 37 73

