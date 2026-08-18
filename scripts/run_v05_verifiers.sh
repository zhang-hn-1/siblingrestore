#!/usr/bin/env bash
# Train and audit the v0.5 teacher and evaluator verifiers sequentially on one
# physical GPU. Designed to run inside a named screen session. The pipeline
# refuses to continue until verifier_status.json reports audited success.
# Usage: scripts/run_v05_verifiers.sh <GPU_ID> <SESSION_NAME>
set -euo pipefail

GPU_ID="$1"
SESSION_NAME="$2"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
CONFIG_ROOT="$ROOT_DIR/configs/v05_frozen_verifier"
RUN_ROOT="$ROOT_DIR/runs/v05_frozen_verifier"
LOG_ROOT="$ROOT_DIR/logs/v05_frozen_verifier"
STATUS_FILE="$RUN_ROOT/verifier_status.json"
ROLES="teacher evaluator"
MIN_ANCHOR_MARGIN="0.05"

if [ ! -x "$PYTHON" ]; then
  echo "python executable not found: $PYTHON" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT_DIR"

write_status() {
  local status="$1"; local code="$2"
  STATUS="$status" CODE="$code" GPU_ID="$GPU_ID" SESSION="$SESSION_NAME" STATUS_FILE="$STATUS_FILE" "$PYTHON" - <<'PY'
import json, os
from datetime import datetime, timezone
path = os.environ["STATUS_FILE"]
payload = {"status": os.environ["STATUS"], "exit_code": int(os.environ["CODE"]), "physical_gpu": os.environ["GPU_ID"], "session_name": os.environ["SESSION"], "current_role": os.environ.get("CURRENT_ROLE", ""), "timestamp_utc": datetime.now(timezone.utc).isoformat()}
open(path, "w", encoding="utf-8").write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
PY
}

on_exit() {
  local code=$?
  if [ "$code" -eq 0 ]; then write_status completed 0; else write_status failed "$code"; fi
  exit "$code"
}
trap on_exit EXIT

audit_verifier() {
  "$PYTHON" - "$RUN_ROOT/$1" "$MIN_ANCHOR_MARGIN" <<'PY'
import json, sys
from pathlib import Path
directory = Path(sys.argv[1])
minimum = float(sys.argv[2])
required = ("best.pt", "last.pt", "resolved_config.json", "summary.json", "selection_split.json", "verifier_diagnostics.json", "checkpoint_fingerprint.json")
missing = [name for name in required if not (directory / name).exists()]
if missing:
    print(json.dumps({"audit": "failed", "reason": "missing artifacts", "missing": missing}))
    raise SystemExit(1)
diagnostics = json.loads((directory / "verifier_diagnostics.json").read_text())
selection = diagnostics.get("selection", {})
summary = json.loads((directory / "summary.json").read_text())
fingerprint = json.loads((directory / "checkpoint_fingerprint.json").read_text())
margin = float(selection.get("anchor_margin", 0.0) or 0.0)
retrieval = float(selection.get("retrieval_top1", 0.0) or 0.0)
pairwise = float(selection.get("clean_anchor_pairwise_cos", 1.0) or 1.0)
print(json.dumps({"audit": "ok", "anchor_margin": margin, "retrieval_top1": retrieval, "clean_anchor_pairwise_cos": pairwise, "summary": summary, "fingerprint": fingerprint}, ensure_ascii=False))
if margin < minimum or retrieval <= 0.2 or pairwise > 0.95:
    print(json.dumps({"audit": "failed", "reason": "insufficient separation", "anchor_margin": margin, "retrieval_top1": retrieval, "clean_anchor_pairwise_cos": pairwise}))
    raise SystemExit(1)
PY
}

CURRENT_ROLE=""
write_status running 0
for role in $ROLES; do
  CURRENT_ROLE="$role"
  write_status running 0
  config_path="$CONFIG_ROOT/verifier_$role.json"
  output_dir="$RUN_ROOT/$role"
  log_path="$LOG_ROOT/$role.verifier.log"
  mkdir -p "$output_dir"
  echo "[$(date --iso-8601=seconds)] START verifier role=$role GPU$GPU_ID session=$SESSION_NAME" | tee -a "$log_path"
  "$PYTHON" train_verifier.py --config "$config_path" >>"$log_path" 2>&1
  audit_verifier "$role" | tee -a "$log_path"
  echo "[$(date --iso-8601=seconds)] END verifier role=$role (audited)" | tee -a "$log_path"
done
CURRENT_ROLE=""
write_status running 0
echo "[$(date --iso-8601=seconds)] DONE verifier setup (teacher+evaluator audited)" | tee -a "$LOG_ROOT/verifiers.overall.log"
