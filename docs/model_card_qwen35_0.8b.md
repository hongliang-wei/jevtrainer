---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
tags:
- jev
- marker
- qwen3.5
---

# jev-marker-qwen35-0.8b-2026-10-02

Jev 式决策模型：一次前向给出每个选项的概率，不生成文字。读出是 `marker`，底座 `Qwen/Qwen3.5-0.8B` 全参数微调，视觉塔冻结。训练完成于 **2026-10-02**。

训练配置是 [jevtrainer](https://github.com/hongliang-wei/jevtrainer) 的 `configs/repro/intern_0.8b_v4.yaml`：公开文本、Jev 格式数据和图像，约 54 万条训练记录，1 个 epoch。

Intern 七套件准确率平均 **82.52**（Intern-Decision-0.8B 为 79.38）。longdoc-dev 平均 80.90，gui-v1 平均 66.71。

```bash
pip install -e .   # github.com/hongliang-wei/jevtrainer
huggingface-cli download weihongliang/jev-marker-qwen35-0.8b-2026-10-02 --local-dir runs/jev-qwen35-0.8b
jt eval configs/eval/intern_accuracy_v1.yaml --set checkpoint=runs/jev-qwen35-0.8b
```

权重在 `model/`。同目录还有 `readout.safetensors`、`readout.json` 和 `calibration.json`，`jt eval` 会一起读。
