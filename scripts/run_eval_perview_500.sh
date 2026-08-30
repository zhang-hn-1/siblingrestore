#!/usr/bin/env bash
# Per-view (yesterday's) frozen-verifier evaluation on the 500-source comparison.
# Screen: eval_perview_500 | GPU3
# 1. PromptIR-500 (user requested)
# 2. GRL-500 (reference only)
set -u
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
cd "$ROOT_DIR"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=3
LOG="$ROOT_DIR/logs/eval_perview_500.log"
mkdir -p "$(dirname "$LOG")"

echo "[$(date --iso-8601=seconds)] === eval PromptIR-500 (per-view) ===" | tee -a "$LOG"
"$PYTHON" scripts/evaluate_frozen_verifier.py \
  --checkpoint runs/baselines_500/promptir/seed13/best.pt \
  --verifier runs/baselines_500/verifier_evaluator/best.pt \
  --data-root data/plamd_vari_grip_500 \
  --output runs/baselines_500/promptir/seed13/frozen_verifier_val.json \
  --split val --device cuda --method promptir --seed 13 --tile-size 512 \
  >>"$LOG" 2>&1
if [ $? -ne 0 ]; then echo "[$(date --iso-8601=seconds)] PROMPTIR EVAL FAILED" | tee -a "$LOG"; exit 1; fi
echo "[$(date --iso-8601=seconds)] PROMPTIR EVAL DONE" | tee -a "$LOG"

echo "[$(date --iso-8601=seconds)] === eval GRL-500 (per-view) ===" | tee -a "$LOG"
"$PYTHON" scripts/evaluate_frozen_verifier.py \
  --checkpoint runs/baselines_500/grl/seed13/best.pt \
  --verifier runs/baselines_500/verifier_evaluator/best.pt \
  --data-root data/plamd_vari_grip_500 \
  --output runs/baselines_500/grl/seed13/frozen_verifier_val.json \
  --split val --device cuda --method grl --seed 13 --tile-size 512 \
  >>"$LOG" 2>&1
if [ $? -ne 0 ]; then echo "[$(date --iso-8601=seconds)] GRL EVAL FAILED" | tee -a "$LOG"; exit 1; fi
echo "[$(date --iso-8601=seconds)] GRL EVAL DONE" | tee -a "$LOG"

echo "[$(date --iso-8601=seconds)] EVAL_QUEUE_DONE" | tee -a "$LOG"
