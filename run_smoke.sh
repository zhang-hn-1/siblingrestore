#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
cd "$ROOT_DIR"

"$PYTHON" check_bundle.py
"$PYTHON" train.py --config configs/smoke.json --mode independent --output-dir runs/smoke_independent
"$PYTHON" train.py --config configs/smoke.json --mode group --output-dir runs/smoke_group
"$PYTHON" train.py --config configs/smoke.json --mode sibling --output-dir runs/smoke_sibling
"$PYTHON" evaluate_source_fidelity.py \
  --checkpoint runs/smoke_sibling/best.pt \
  --data-root data/plamd_vari_grip_pilot \
  --split val \
  --output runs/smoke_sibling/source_fidelity_smoke.json \
  --max-sources 2 \
  --tile-size 256
