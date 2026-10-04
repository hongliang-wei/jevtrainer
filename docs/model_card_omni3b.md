---
license: apache-2.0
base_model: Qwen/Qwen2.5-Omni-3B
tags:
- jev
- marker
- qwen2.5-omni
---

# jev-marker-omni3b-2026-10-04

A Jev-style decision model that reads text, images, video, and audio. The readout is `marker`. LoRA (r=32, alpha=64) is trained on the thinker of `Qwen/Qwen2.5-Omni-3B`. Only the thinker is loaded; the speech output is not. Training finished on **2026-10-04**.

The training config is `configs/av/omni3b_v4.yaml` in [jevtrainer](https://github.com/hongliang-wei/jevtrainer): 79 text, image, audio, and video datasets, one epoch. The holdout has 1,610 examples, accuracy **85.03%**, skill 79.54%, ECE 0.0096.

```bash
pip install -e .   # github.com/hongliang-wei/jevtrainer
huggingface-cli download weihongliang/jev-marker-omni3b-2026-10-04 --local-dir runs/jev-omni3b
jt eval configs/eval/av_omni.yaml --set checkpoint=runs/jev-omni3b --set benchmarks=worldsense,omnibench
```

This repo is the LoRA adapter and the readout. Base weights still come from `Qwen/Qwen2.5-Omni-3B`. `readout.json` records the base model id.
