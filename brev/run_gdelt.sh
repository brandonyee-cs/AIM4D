#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -d .venv ]]; then
  source .venv/bin/activate
fi

export GDELT_WORKERS="${GDELT_WORKERS:-16}"
LOG="brev/gdelt.log"

echo "Starting GDELT download with $GDELT_WORKERS workers. Log: $LOG"
nohup python3 -u data/download_gdelt.py > "$LOG" 2>&1 &
PID=$!
echo "PID=$PID"
echo "Tail with:  tail -f $LOG"
