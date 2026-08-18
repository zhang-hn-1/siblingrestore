#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
cd "$ROOT_DIR"

"$PYTHON" train.py --config configs/pilot_independent.json
"$PYTHON" train.py --config configs/pilot_group.json
"$PYTHON" train.py --config configs/pilot_sibling.json

for mode in independent group sibling; do
  "$PYTHON" evaluate_source_fidelity.py \
    --checkpoint "runs/pilot_${mode}/best.pt" \
    --data-root data/plamd_vari_grip_pilot \
    --split val \
    --output "runs/pilot_${mode}/source_fidelity_val.json"
done
