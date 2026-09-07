#!/usr/bin/env bash
set -u
ROOT="/home/zhanghangning/siblingrestore-pilot-server"
LOG="$ROOT/artifacts/identity_evaluation/cross_verifier/evaluate_watchdog.log"
mkdir -p "$(dirname "$LOG")"
cd "$ROOT"
printf '[%s] launching cross-verifier evaluation\n' "$(date --iso-8601=seconds)" >> "$LOG"
CUDA_VISIBLE_DEVICES=1 "$ROOT/.venv/bin/python" evaluation/evaluate_cross_verifier.py --device cuda >> "$LOG" 2>&1
rc=$?
printf '[%s] cross-verifier evaluation exited rc=%s\n' "$(date --iso-8601=seconds)" "$rc" >> "$LOG"
exit $rc
