#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SPEC="${1:?usage: run_controlled_group.sh scripts/campaigns/<group>.json}"
SPEC_NAME="$(basename "$SPEC" .json)"
LOCK_DIR="$ROOT_DIR/runs/.controlled-${SPEC_NAME}.lock"

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "another controlled campaign is already running: $LOCK_DIR" >&2
  exit 2
fi
cleanup() { rmdir "$LOCK_DIR" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

cd "$ROOT_DIR"
PYTHON="$ROOT_DIR/.venv/bin/python"

# Prepare is idempotent.  Exactly one runner owns the group lock, and the
# campaign runner assigns at most one job to each listed GPU.
"$PYTHON" scripts/run_campaign.py prepare "$SPEC"
set +e
"$PYTHON" scripts/run_campaign.py run "$SPEC"
status=$?
set -e
exit "$status"
