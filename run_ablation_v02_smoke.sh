#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/home/zhanghangning/.venv/bin/python}"
cd "$ROOT_DIR"

for name in group source_0001 source_0003 source_001 output_001 degradation_001; do
  "$PYTHON" train.py \
    --config "configs/ablation_v02/${name}.json" \
    --device cuda \
    --max-steps 2 \
    --crop-size 64 \
    --num-workers 8 \
    --eval-items 2 \
    --gradient-diagnostic-interval 1
done

for name in group source_0001 source_0003 source_001 output_001 degradation_001; do
  "$PYTHON" evaluate_source_fidelity.py \
    --checkpoint "runs/ablation_v02/${name}/best.pt" \
    --data-root data/plamd_vari_grip_pilot \
    --split val \
    --max-sources 2 \
    --tile-size 64 \
    --output "runs/ablation_v02/${name}/source_fidelity_val.json" \
    --device cuda
done

"$PYTHON" analyze_loss_ablation.py --root runs/ablation_v02
