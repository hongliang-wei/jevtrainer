#!/usr/bin/env bash
# Three readouts with LoRA plus pointer full fine-tuning on Qwen3.5-0.8B; ~200 steps each.
set -euo pipefail
cd "$(dirname "$0")/.."
for cfg in marker_lora slot_lora pointer_lora; do
  jt train configs/train/$cfg.yaml --set max_steps=200 --set output_dir=runs/smoke/$cfg
done
jt train configs/train/pointer_full.yaml --set output_dir=runs/smoke/pointer_full
python scripts/summarize_runs.py runs/smoke
