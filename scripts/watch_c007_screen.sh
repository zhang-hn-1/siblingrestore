#!/usr/bin/env bash
set -u

ROOT="/home/zhanghangning/siblingrestore-pilot-server"
SPEC="$ROOT/scripts/campaigns/c007_psnr_modules_12k.json"
MANIFEST="$ROOT/results/psnr_modules_12k/jobs.jsonl"
WATCH_LOG="$ROOT/results/psnr_modules_12k/screen_watchdog.log"
PYTHON="$ROOT/.venv/bin/python"

mkdir -p "$(dirname "$WATCH_LOG")"
cd "$ROOT"

while true; do
  printf '[%s] launching campaign runner\n' "$(date --iso-8601=seconds)" >> "$WATCH_LOG"
  "$PYTHON" scripts/run_campaign.py run "$SPEC" >> "$WATCH_LOG" 2>&1
  rc=$?
  printf '[%s] campaign runner exited rc=%s\n' "$(date --iso-8601=seconds)" "$rc" >> "$WATCH_LOG"

  state=$($PYTHON - "$MANIFEST" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    print("missing")
    raise SystemExit
jobs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
statuses = {job["status"] for job in jobs}
if "failed" in statuses:
    print("failed")
elif jobs and all(job["status"] == "done" for job in jobs):
    print("done")
elif "pending" in statuses or "running" in statuses:
    print("incomplete")
else:
    print("stopped")
PY
)

  case "$state" in
    done)
      printf '[%s] campaign complete; watchdog stopping\n' "$(date --iso-8601=seconds)" >> "$WATCH_LOG"
      exit 0
      ;;
    failed|missing|stopped)
      printf '[%s] campaign state=%s; watchdog stopping for inspection\n' "$(date --iso-8601=seconds)" "$state" >> "$WATCH_LOG"
      exit 1
      ;;
    incomplete)
      printf '[%s] incomplete state detected; restarting runner after 5s\n' "$(date --iso-8601=seconds)" >> "$WATCH_LOG"
      sleep 5
      ;;
  esac
done
