#!/usr/bin/env bash
# Batch evaluation queue for the 500-source comparison (GPU3, screen: batch_eval_500).
# 1. Verify batch pipeline vs per-view pipeline on dim48+anchor (reference: PSNR 25.386 | top1 0.8896 | margin 0.1222 | AUC 0.9871 | EER 0.0456)
# 2. Batch-evaluate PromptIR-500
# 3. Batch-evaluate GRL-500 (reference only)
set -u
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
cd "$ROOT_DIR"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=3
LOG="$ROOT_DIR/logs/batch_eval_500.log"
mkdir -p "$(dirname "$LOG")"

echo "[$(date --iso-8601=seconds)] === step 1: verify batch vs per-view (dim48+anchor) ===" | tee -a "$LOG"
"$PYTHON" scripts/evaluate_batch.py \
  --checkpoint runs/baselines_500/ours_dim48_anchor/seed13/best.pt \
  --verifier runs/baselines_500/verifier_evaluator/best.pt \
  --data-root data/plamd_vari_grip_500 \
  --output runs/baselines_500/ours_dim48_anchor/seed13/frozen_verifier_val_batch.json \
  --split val --device cuda --method ours_dim48_anchor --seed 13 \
  >>"$LOG" 2>&1
if [ $? -ne 0 ]; then echo "[$(date --iso-8601=seconds)] STEP1 FAILED" | tee -a "$LOG"; exit 1; fi
echo "[$(date --iso-8601=seconds)] === step 1 done ===" | tee -a "$LOG"

echo "[$(date --iso-8601=seconds)] === step 2: batch eval PromptIR-500 ===" | tee -a "$LOG"
"$PYTHON" scripts/evaluate_batch.py \
  --checkpoint runs/baselines_500/promptir/seed13/best.pt \
  --verifier runs/baselines_500/verifier_evaluator/best.pt \
  --data-root data/plamd_vari_grip_500 \
  --output runs/baselines_500/promptir/seed13/frozen_verifier_val.json \
  --split val --device cuda --method promptir --seed 13 \
  >>"$LOG" 2>&1
if [ $? -ne 0 ]; then echo "[$(date --iso-8601=seconds)] STEP2 FAILED" | tee -a "$LOG"; exit 1; fi
echo "[$(date --iso-8601=seconds)] === step 2 done ===" | tee -a "$LOG"

echo "[$(date --iso-8601=seconds)] === step 3: batch eval GRL-500 (reference) ===" | tee -a "$LOG"
"$PYTHON" scripts/evaluate_batch.py \
  --checkpoint runs/baselines_500/grl/seed13/best.pt \
  --verifier runs/baselines_500/verifier_evaluator/best.pt \
  --data-root data/plamd_vari_grip_500 \
  --output runs/baselines_500/grl/seed13/frozen_verifier_val.json \
  --split val --device cuda --method grl --seed 13 \
  >>"$LOG" 2>&1
if [ $? -ne 0 ]; then echo "[$(date --iso-8601=seconds)] STEP3 FAILED" | tee -a "$LOG"; exit 1; fi
echo "[$(date --iso-8601=seconds)] === step 3 done ===" | tee -a "$LOG"

echo "[$(date --iso-8601=seconds)] BATCH_EVAL_QUEUE_DONE" | tee -a "$LOG"
