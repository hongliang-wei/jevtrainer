# Measured results

All runs on one RTX 3090 (48 GB), Qwen/Qwen3.5-0.8B, bf16 autocast, flash-linear-attention installed
(causal-conv1d not installed; the short convolution falls back to PyTorch).

## Smoke runs (`scripts/smoke_text.sh`, `scripts/smoke_vl.sh`)

Training data: banking77 + boolq + agnews, 3,000 records each (text) or ScienceQA image subset (vision).
Evaluation on 500 held-out benchmark records (300 for ScienceQA). These are 100-200 step runs meant to
show that every path trains, saves, calibrates and evaluates, not to compare readouts.

| run | finetune | steps | wall time | holdout acc | fitted T | banking77 macro-F1 | boolq acc | scienceqa_img acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| marker | LoRA r16 | 200 | 286 s | 0.908 | 1.14 | n/a (77 > 62 options) | 77.6 | |
| slot | LoRA r16 | 200 | 529 s | 0.671 | 1.14 | 40.1 | 68.0 | |
| pointer | LoRA r16 | 200 | 140 s | 0.865 | 0.96 | 68.9 | 72.4 | |
| pointer | full | 100 | 90 s | 0.724 | 1.22 | 52.8 | 63.0 | |
| marker, images | LoRA r16 | 200 | 199 s | 0.867 | 5.42 | | | 69.3 |

The marker readout, like Intern-Decision, has 62 option symbols (A-Z, a-z, 0-9); records with more
options are subsampled during training (gold kept) and skipped at evaluation, where they count as wrong.

`jt serve runs/smoke/pointer_lora` answered a choice / noul / score request made through the official
`typesafe-sdk` client with only `base_url` changed (`scripts/sdk_check.py`).

## Zero-shot Qwen3.5-0.8B on Intern-Decision accuracy-v1

Marker readout, no training (the `<decision>` embedding is the readout's untrained initialisation).

| suite | zero-shot | Intern-Decision-0.8B |
|---|---:|---:|
| JevBench Easy | 97.92 | 97.92 |
| JevBench Original | 62.50 | 80.56 |
| JevBench Hard | 39.64 | 52.25 |
| Typed Decision | 41.40 | 77.35 |
| ToolACE | 77.10 | 94.52 |
| AG News | 80.67 | 88.61 |
| WildJailBreak | 62.04 | 64.48 |
| average | 65.90 | 79.38 |

See [repro_intern_0.8b.md](repro_intern_0.8b.md) for the training runs.
