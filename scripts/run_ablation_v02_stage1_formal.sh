#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
FORMAL_ROOT="$ROOT_DIR/runs/ablation_v02_formal_seed13"
LOG_ROOT="$ROOT_DIR/logs/ablation_v02_formal_seed13"
CONFIG_ROOT="$ROOT_DIR/configs/ablation_v02_formal_seed13"
EXPERIMENTS=(group source_0001 source_0003 degradation_001)
CURRENT_EXPERIMENT=""

mkdir -p "$FORMAL_ROOT" "$LOG_ROOT"
cd "$ROOT_DIR"

write_completion_status() {
  local status="$1"
  local exit_code="${2:-0}"
  STATUS="$status" EXIT_CODE="$exit_code" CURRENT="$CURRENT_EXPERIMENT" "$PYTHON" - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path
path = Path("runs/ablation_v02_formal_seed13/completion_status.json")
payload = {"status": os.environ["STATUS"], "exit_code": int(os.environ["EXIT_CODE"]), "current_experiment": os.environ.get("CURRENT", ""), "timestamp_utc": datetime.now(timezone.utc).isoformat()}
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

on_exit() {
  local code=$?
  if [ "$code" -ne 0 ]; then write_completion_status failed "$code"; fi
  exit "$code"
}
trap on_exit EXIT

is_complete() {
  "$PYTHON" - "$FORMAL_ROOT/$1" <<'PY'
import json
import sys
from pathlib import Path
directory = Path(sys.argv[1])
required = ("best.pt", "last.pt", "best_validation.json", "last_validation.json", "validation_history.jsonl")
if not all((directory / name).exists() for name in required): raise SystemExit(1)
if not (directory / "summary.json").exists(): raise SystemExit(1)
if json.loads((directory / "summary.json").read_text(encoding="utf-8"))["steps"] != 5000: raise SystemExit(1)
PY
}

validate_experiment() {
  "$PYTHON" - "$FORMAL_ROOT/$1" <<'PY'
import json
import math
import sys
from pathlib import Path
directory = Path(sys.argv[1])
summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
assert summary["steps"] == 5000, summary
for filename in ("best.pt", "last.pt", "best_validation.json", "last_validation.json", "validation_history.jsonl", "source_fidelity_val.json", "source_fidelity_val.csv"):
    assert (directory / filename).exists(), filename
history = [json.loads(line) for line in (directory / "validation_history.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
assert len(history) >= 10, len(history)
config = json.loads((directory / "resolved_config.json").read_text(encoding="utf-8"))
logs = [json.loads(line) for line in (directory / "train_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
assert all(math.isfinite(float(value)) for row in logs for value in row.values() if isinstance(value, (int, float)))
enabled = [name for name in ("source", "output", "degradation") if float(config["loss_weights"][name]) > 0]
for name in enabled: assert sum(row.get(f"grad_cos_rec_{name}") is not None for row in logs) >= 10, name
fidelity = json.loads((directory / "source_fidelity_val.json").read_text(encoding="utf-8"))
assert fidelity["sources"] == 16, fidelity
assert fidelity["restored_views"] == 96, fidelity
print(json.dumps({"experiment": directory.name, "steps": summary["steps"], "validation_points": len(history), "diagnostic_losses": enabled, "fidelity_views": fidelity["restored_views"]}, ensure_ascii=False))
PY
}

for name in "${EXPERIMENTS[@]}"; do
  CURRENT_EXPERIMENT="$name"
  output_dir="$FORMAL_ROOT/$name"
  train_log="$LOG_ROOT/${name}.train.log"
  eval_log="$LOG_ROOT/${name}.source_fidelity.log"
  echo "[$(date --iso-8601=seconds)] START $name" | tee -a "$train_log"
  if ! is_complete "$name"; then
    resume_args=()
    if [ -f "$output_dir/last.pt" ]; then resume_args=(--resume "$output_dir/last.pt"); fi
    "$PYTHON" train.py --config "$CONFIG_ROOT/$name.json" --device cuda --num-workers 8 "${resume_args[@]}" >>"$train_log" 2>&1
  else
    echo "[$(date --iso-8601=seconds)] SKIP completed $name" | tee -a "$train_log"
  fi
  if [ ! -f "$output_dir/source_fidelity_val.json" ]; then
    "$PYTHON" evaluate_source_fidelity.py --checkpoint "$output_dir/best.pt" --data-root data/plamd_vari_grip_pilot --split val --output "$output_dir/source_fidelity_val.json" --device cuda >>"$eval_log" 2>&1
  else
    echo "[$(date --iso-8601=seconds)] SKIP existing source fidelity $name" | tee -a "$eval_log"
  fi
  validate_experiment "$name" | tee -a "$train_log"
  echo "[$(date --iso-8601=seconds)] END $name" | tee -a "$train_log"
done

"$PYTHON" analyze_loss_ablation.py --root "$FORMAL_ROOT" --experiments "${EXPERIMENTS[@]}" --json-output "$FORMAL_ROOT/ablation_report_stage1.json" --markdown-output "$FORMAL_ROOT/ablation_report_stage1.md" --fairness-output "$FORMAL_ROOT/fairness_check.json"
CURRENT_EXPERIMENT=""
write_completion_status completed 0
