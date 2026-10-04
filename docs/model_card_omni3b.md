---
license: apache-2.0
base_model: Qwen/Qwen2.5-Omni-3B
tags:
- jev
- marker
- qwen2.5-omni
---

# jev-marker-omni3b-2026-10-04

Jev 式决策模型，读文本、图像、视频和音频。读出是 `marker`，底座 `Qwen/Qwen2.5-Omni-3B` 的 thinker 上做 LoRA（r=32, alpha=64）。只加载 thinker，不加载语音输出。训练完成于 **2026-10-04**。

训练配置是 [jevtrainer](https://github.com/hongliang-wei/jevtrainer) 的 `configs/av/omni3b_v4.yaml`：79 个文本、图像、音频和视频数据集，1 个 epoch。保留集 1610 条，准确率 **85.03%**，skill 79.54%，ECE 0.0096。

```bash
pip install -e .   # github.com/hongliang-wei/jevtrainer
huggingface-cli download weihongliang/jev-marker-omni3b-2026-10-04 --local-dir runs/jev-omni3b
jt eval configs/eval/av_omni.yaml --set checkpoint=runs/jev-omni3b --set benchmarks=worldsense,omnibench
```

这里是 LoRA adapter 和读出，底座权重仍从 `Qwen/Qwen2.5-Omni-3B` 下载。`readout.json` 里记着底座的名字。
