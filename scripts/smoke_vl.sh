#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
jt train configs/train/marker_vl.yaml --set output_dir=runs/smoke/marker_vl
python scripts/summarize_runs.py runs/smoke
