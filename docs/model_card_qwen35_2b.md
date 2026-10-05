---
license: apache-2.0
base_model: Qwen/Qwen3.5-2B
tags:
- jev
- marker
- qwen3.5
- lora
---

# jevtrainer-qwen35-2b-2026-10-05

A Jev-style decision model. One forward pass returns a probability for every option. It does not generate text. The readout is `marker`. The base `Qwen/Qwen3.5-2B` is LoRA fine-tuned (r=64, alpha=128), with the vision tower frozen. Training finished on **2026-10-05**.

The training config is [`configs/repro/qwen35_2b_lora_v4.yaml`](https://github.com/hongliang-wei/jevtrainer/blob/main/configs/repro/qwen35_2b_lora_v4.yaml) in [jevtrainer](https://github.com/hongliang-wei/jevtrainer): the same public v4 mixture as the 0.8B run, about 540k training records, one epoch. Holdout is 11,706 records, accuracy 88.19%.

This repo is the LoRA adapter plus the readout. `adapter/` holds the PEFT weights. The same directory has `readout.safetensors`, `readout.json`, `calibration.json`, and `tokenizer/`. `load()` reads those together and downloads `Qwen/Qwen3.5-2B` for the frozen base. Pass a local directory; a Hub repo id is not accepted directly.

## Evaluation

Accuracy in percent. Average is the unweighted mean of Easy, Original, Hard, Typed Decision, ToolACE, AG News, and WildJailBreak. longdoc-dev and gui-v1 are separate.

| Model | Easy | Original | Hard | Typed Decision | ToolACE | AG News | WildJailBreak | Average | longdoc-dev | gui-v1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [jevtrainer-0.8b](https://huggingface.co/weihongliang/jevtrainer-qwen35-0.8b-2026-10-02) | 100.00 | 84.72 | 50.45 | 71.05 | 95.81 | 92.17 | 83.44 | 82.52 | 80.90 | 66.71 |
| [jevtrainer-2b](https://huggingface.co/weihongliang/jevtrainer-qwen35-2b-2026-10-05) | 100.00 | 90.28 | 52.25 | 72.50 | 96.13 | 92.64 | 92.67 | **85.21** | 82.93 | 69.08 |

```bash
huggingface-cli download weihongliang/jevtrainer-qwen35-2b-2026-10-05 --local-dir runs/jev-qwen35-2b
jt eval configs/eval/intern_accuracy_v1.yaml --set checkpoint=runs/jev-qwen35-2b
```
