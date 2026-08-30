#!/usr/bin/env bash
# 串行评估 transweather、uformer（等 GPU1 上 dfpir 评估完成后依次跑）
set -u
cd "$(dirname "$0")/.."
PYTHON="$PWD/.venv/bin/python"
VERIFIER="runs/campaigns/c001_sfr_v1/verifier_evaluator/best.pt"

wait_gpu_eval_free() {
  while true; do
    free=1
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
      if ps -o args= -p $p 2>/dev/null | grep -q "evaluate_frozen_verifier_v2"; then
        free=0; break
      fi
    done
    [ $free -eq 1 ] && break
    sleep 30
  done
  echo "[$(date +%H:%M:%S)] GPU1 eval free"
}

run_eval() {
  local model="$1"
  wait_gpu_eval_free
  echo "[$(date +%H:%M:%S)] GPU1: eval $model"
  CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 "$PYTHON" \
    scripts/evaluate_frozen_verifier_v2.py \
    --checkpoint "runs/campaigns/c005_official_group1/$model/13/best.pt" \
    --verifier "$VERIFIER" --data-root data/plamd_sfr_v1 \
    --output "results/campaigns/c005_official_group1/$model.13.test.json" \
    --split test --device cuda --method "$model" --seed 13 --lpips \
    > "results/campaigns/c005_official_group1/logs/eval:$model:13:test.log" 2>&1
  echo "[$(date +%H:%M:%S)] GPU1: done $model (exit $?)"
}

run_eval transweather
run_eval uformer
