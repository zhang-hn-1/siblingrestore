#!/usr/bin/env bash
set -u
ROOT="/home/zhanghangning/siblingrestore-pilot-server"
LOG="$ROOT/artifacts/identity_evaluation/cross_verifier/train_watchdog.log"
mkdir -p "$(dirname "$LOG")"
cd "$ROOT"
printf '[%s] launching cross-verifier training\n' "$(date --iso-8601=seconds)" >> "$LOG"
"$ROOT/.venv/bin/python" evaluation/train_cross_verifier.py >> "$LOG" 2>&1
rc=$?
printf '[%s] cross-verifier training exited rc=%s\n' "$(date --iso-8601=seconds)" "$rc" >> "$LOG"
exit $rc
