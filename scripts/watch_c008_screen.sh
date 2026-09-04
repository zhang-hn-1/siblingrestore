#!/usr/bin/env bash
set -u

ROOT="/home/zhanghangning/siblingrestore-pilot-server"
SPEC="$ROOT/scripts/campaigns/c008_alcrb_hfrb_12k.json"
MANIFEST="$ROOT/results/psnr_modules_12k/c008_jobs.jsonl"
CAMPAIGN_MANIFEST="$ROOT/results/psnr_modules_12k/jobs.jsonl"
WATCH_LOG="$ROOT/results/psnr_modules_12k/c008_screen_watchdog.log"
PYTHON="$ROOT/.venv/bin/python"

cd "$ROOT"
mkdir -p "$(dirname "$WATCH_LOG")"

while true; do
  printf '[%s] launching c008 campaign runner\n' "$(date --iso-8601=seconds)" >> "$WATCH_LOG"
  "$PYTHON" scripts/run_campaign.py run "$SPEC" >> "$WATCH_LOG" 2>&1
  rc=$?
  printf '[%s] runner exited rc=%s\n' "$(date --iso-8601=seconds)" "$rc" >> "$WATCH_LOG"

  state=$($PYTHON - "$CAMPAIGN_MANIFEST" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    print("missing")
else:
    jobs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    statuses = {job["status"] for job in jobs}
    if "failed" in statuses:
        print("failed")
    elif jobs and all(job["status"] == "done" for job in jobs):
        print("done")
    else:
        print("incomplete")
PY
)
  case "$state" in
    done) exit 0 ;;
    failed|missing) exit 1 ;;
    incomplete) sleep 5 ;;
  esac
done
