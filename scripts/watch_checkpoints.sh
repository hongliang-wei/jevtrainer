#!/usr/bin/env bash
# Evaluate every new step-* checkpoint of a run while it trains.
#   scripts/watch_checkpoints.sh runs/repro/intern_0.8b_v1 intern-accuracy-v1
set -uo pipefail
RUN=$1
SUITE=${2:-intern-accuracy-v1}
cd "$(dirname "$0")/.."
while true; do
  for ck in $(ls -d "$RUN"/step-* 2>/dev/null | sort -V); do
    if [ -f "$ck/readout.json" ] && [ ! -f "$ck/eval/results.json" ]; then
      echo "$(date) evaluating $ck"
      jt eval --set checkpoint="$ck" --set benchmarks="$SUITE" --set output_dir="$ck/eval" > "$ck/eval.log" 2>&1
      python scripts/intern_table.py "$RUN"
      [ -n "${WANDB_API_KEY:-}" ] && python scripts/wandb_upload.py "$RUN" > "$RUN/wandb_upload.log" 2>&1
    fi
  done
  if [ -f "$RUN/metrics.json" ]; then
    python scripts/intern_table.py "$RUN"
    [ -n "${WANDB_API_KEY:-}" ] && python scripts/wandb_upload.py "$RUN" > "$RUN/wandb_upload.log" 2>&1
    exit 0
  fi
  sleep 120
done
