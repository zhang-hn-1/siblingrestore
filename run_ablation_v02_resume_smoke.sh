#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/home/zhanghangning/.venv/bin/python}"
OUTPUT_DIR="runs/ablation_v02_resume_test"
cd "$ROOT_DIR"
if [ -e "$OUTPUT_DIR" ]; then
  mv "$OUTPUT_DIR" "${OUTPUT_DIR}_previous_$(date +%Y%m%d_%H%M%S)"
fi

"$PYTHON" train.py \
  --config configs/ablation_v02/source_001.json \
  --output-dir "$OUTPUT_DIR" \
  --device cuda \
  --max-steps 1 \
  --crop-size 64 \
  --num-workers 8 \
  --eval-items 2 \
  --gradient-diagnostic-interval 1

"$PYTHON" train.py \
  --config configs/ablation_v02/source_001.json \
  --output-dir "$OUTPUT_DIR" \
  --device cuda \
  --max-steps 2 \
  --crop-size 64 \
  --num-workers 8 \
  --eval-items 2 \
  --gradient-diagnostic-interval 1 \
  --resume "$OUTPUT_DIR/last.pt"

"$PYTHON" - <<'PY'
import json
from pathlib import Path
steps = [json.loads(line)["step"] for line in Path("runs/ablation_v02_resume_test/train_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
assert steps == [1, 2], steps
print(json.dumps({"resume_steps": steps}, ensure_ascii=False))
PY
