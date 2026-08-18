#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/home/zhanghangning/.venv/bin/python}"
cd "$ROOT_DIR"

for name in group source_0001 source_0003 source_001 output_001 degradation_001; do
  "$PYTHON" train.py --config "configs/ablation_v02/${name}.json"
done

for name in group source_0001 source_0003 source_001 output_001 degradation_001; do
  "$PYTHON" evaluate_source_fidelity.py \
    --checkpoint "runs/ablation_v02/${name}/best.pt" \
    --data-root data/plamd_vari_grip_pilot \
    --split val \
    --output "runs/ablation_v02/${name}/source_fidelity_val.json"
done

"$PYTHON" analyze_loss_ablation.py --root runs/ablation_v02
