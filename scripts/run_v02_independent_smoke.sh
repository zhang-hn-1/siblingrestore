#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-3}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
CONFIG="$ROOT_DIR/configs/v02_backbone_decision/independent/seed13.json"
OUTPUT_DIR="$ROOT_DIR/runs/v02_backbone_decision_smoke/independent_seed13"

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
cd "$ROOT_DIR"

rm -f "$OUTPUT_DIR"/best.pt "$OUTPUT_DIR"/last.pt \
  "$OUTPUT_DIR"/best_validation.json "$OUTPUT_DIR"/last_validation.json \
  "$OUTPUT_DIR"/validation_history.jsonl "$OUTPUT_DIR"/train_log.jsonl \
  "$OUTPUT_DIR"/summary.json "$OUTPUT_DIR"/resolved_config.json

"$PYTHON" train.py \
  --config "$CONFIG" \
  --device cuda \
  --output-dir "$OUTPUT_DIR" \
  --max-steps 1 \
  --crop-size 64 \
  --num-workers 8 \
  --gradient-diagnostic-interval 1 \
  --eval-items 2

"$PYTHON" - "$OUTPUT_DIR/last.pt" <<'PY'
import json
import sys
from pathlib import Path

checkpoint = __import__("torch").load(sys.argv[1], map_location="cpu", weights_only=False)
required = {"model", "optimizer", "scheduler", "scaler", "epoch", "step", "best_psnr", "next_validation", "rng_state"}
missing = required.difference(checkpoint)
if missing:
    raise SystemExit(f"missing checkpoint fields: {sorted(missing)}")
if int(checkpoint["step"]) != 1:
    raise SystemExit(f"expected smoke checkpoint at step 1, got {checkpoint['step']}")
PY

"$PYTHON" train.py \
  --config "$CONFIG" \
  --device cuda \
  --output-dir "$OUTPUT_DIR" \
  --max-steps 2 \
  --crop-size 64 \
  --num-workers 8 \
  --gradient-diagnostic-interval 1 \
  --eval-items 2 \
  --resume "$OUTPUT_DIR/last.pt"

"$PYTHON" - "$OUTPUT_DIR" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
if int(summary["steps"]) != 2:
    raise SystemExit(f"expected resumed smoke at step 2, got {summary['steps']}")
rows = [json.loads(line) for line in (root / "train_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
if [int(row["step"]) for row in rows] != [1, 2]:
    raise SystemExit("resume log does not contain exactly steps 1 and 2")
for row in rows:
    for value in row.values():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            raise SystemExit("NaN/Inf in smoke log")
print(json.dumps({"status": "smoke_resume_passed", "steps": 2, "gpu": "cuda:" + str(__import__('os').environ.get('CUDA_VISIBLE_DEVICES'))}, ensure_ascii=False))
PY
