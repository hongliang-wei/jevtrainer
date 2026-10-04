---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
tags:
- jev
- marker
- qwen3.5
---

# jev-marker-qwen35-0.8b-2026-10-02

A Jev-style decision model. One forward pass returns a probability for every option. It does not generate text. The readout is `marker`. The base `Qwen/Qwen3.5-0.8B` is fully fine-tuned, with the vision tower frozen. Training finished on **2026-10-02**.

The training config is `configs/repro/intern_0.8b_v4.yaml` in [jevtrainer](https://github.com/hongliang-wei/jevtrainer): public text, Jev-format data, and images, about 540k training records, one epoch.

Intern seven-suite accuracy averages **82.52** (Intern-Decision-0.8B is 79.38). longdoc-dev averages 80.90, gui-v1 averages 66.71.

```bash
pip install -e .   # github.com/hongliang-wei/jevtrainer
huggingface-cli download weihongliang/jev-marker-qwen35-0.8b-2026-10-02 --local-dir runs/jev-qwen35-0.8b
jt eval configs/eval/intern_accuracy_v1.yaml --set checkpoint=runs/jev-qwen35-0.8b
```

Weights are in `model/`. The same directory has `readout.safetensors`, `readout.json`, and `calibration.json`, which `jt eval` reads together.
