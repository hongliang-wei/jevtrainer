<p align="center">
  <img src="docs/icon.jpg" width="160" alt="jevtrainer">
</p>

<p align="center">
  <a href="https://github.com/hongliang-wei/jevtrainer/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/hongliang-wei/jevtrainer?style=social"></a>
  <a href="https://github.com/hongliang-wei/jevtrainer/commits/main"><img alt="Last commit" src="https://img.shields.io/github/last-commit/hongliang-wei/jevtrainer"></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-blue"></a>
  <a href="https://huggingface.co/weihongliang/jev-marker-qwen35-0.8b-2026-10-02"><img alt="Hugging Face 0.8B" src="https://img.shields.io/badge/%F0%9F%A4%97-Qwen3.5%200.8B-yellow"></a>
  <a href="https://huggingface.co/weihongliang/jev-marker-omni3b-2026-10-04"><img alt="Hugging Face Omni" src="https://img.shields.io/badge/%F0%9F%A4%97-Omni%203B-yellow"></a>
</p>

<div align="center">

### 一个 YAML 文件，训练用概率作答、不生成文字的决策模型

</div>

\[ [English](README.md) | 中文 \]

Jev 模型读入一段状态（文本、JSON，以及可选的图像、视频或音频）和一组答案已经写好的问题。一次前向返回每个选项的概率。

| 题型 | 调用方给出 | 模型返回 |
| --- | --- | --- |
| `choice` | 2–255 个命名选项 | 选中的键、全部概率、置信度 |
| `score` | 2–10 个有序档位 | 期望档位、分布、置信度 |
| `noul` | 一个是否命题 | P(是) |

概率从语言模型里读出来。LoRA 和全参数微调都支持。新数据集、新读出或新底座是一个小模块，一个 YAML 文件就能开跑。

## 目录

- [能力](#能力)
- [训好的模型](#训好的模型)
- [支持的模型](#支持的模型)
- [训练方式](#训练方式)
- [数据集](#数据集)
- [环境](#环境)
- [开始使用](#开始使用)
  - [安装](#安装)
  - [最快的一条训练](#最快的一条训练)
  - [评测](#评测)
  - [服务](#服务)
- [项目结构](#项目结构)
- [许可证](#许可证)

## 能力

- **一个文件开训**：`jt init` 写出 YAML，`jt train` 跑起来。未知字段会报错，并给出相近的名字。
- **三种题型**：`choice`、`score`、`noul`，一次前向同时打分。
- **三种读出**：`marker`（Intern-Decision）、`slot`（Bosun v3.1）、`pointer`（Kev）。每种都能做 LoRA 或全参数微调。
- **模态**：文本、图像、GUI 截图、音频、视频、音视频，用能编码这些模态的底座。
- **开放底座**：[支持的模型](#支持的模型)里的每一系。其他因果语言模型不用写新代码，会自动推断。
- **333 个数据集**已经注册。`jt data list` 和 `jt bench list` 可以查看。
- **服务**：`jt serve` 提供 `POST /v1/systemone`。

## 训好的模型

下载后把目录当作 `checkpoint`。LoRA 模型仍会去拉底座权重。

| 模型 | 日期 | 底座 | 结果 |
| --- | --- | --- | --- |
| [weihongliang/jev-marker-qwen35-0.8b-2026-10-02](https://huggingface.co/weihongliang/jev-marker-qwen35-0.8b-2026-10-02) | 2026-10-02 | Qwen3.5-0.8B，全参数，`marker` | Intern 七套件平均 **82.52**（Intern-Decision-0.8B 为 79.38） |
| [weihongliang/jev-marker-omni3b-2026-10-04](https://huggingface.co/weihongliang/jev-marker-omni3b-2026-10-04) | 2026-10-04 | Qwen2.5-Omni-3B thinker，LoRA r=32，`marker` | 保留集准确率 **85.03%**（1610 条） |

```bash
huggingface-cli download weihongliang/jev-marker-qwen35-0.8b-2026-10-02 --local-dir runs/jev-qwen35-0.8b
jt eval configs/eval/intern_accuracy_v1.yaml --set checkpoint=runs/jev-qwen35-0.8b

huggingface-cli download weihongliang/jev-marker-omni3b-2026-10-04 --local-dir runs/jev-omni3b
jt eval configs/eval/av_omni.yaml --set checkpoint=runs/jev-omni3b --set benchmarks=worldsense,omnibench
```

## 支持的模型

`family: auto` 按 `config.model_type` 选下面的一行。`jt model list` 打出同一份清单。表里没有的类型仍走 `GenericFamily`。

| 模型 | `model_type` | 输入 | 说明 |
| --- | --- | --- | --- |
| Qwen2、Qwen2.5 | `qwen2` | 文本 | |
| Qwen2-MoE | `qwen2_moe` | 文本 | |
| Qwen2-VL | `qwen2_vl` | 文本、图像 | |
| Qwen2.5-VL | `qwen2_5_vl` | 文本、图像 | |
| Qwen3 | `qwen3` | 文本 | |
| Qwen3-MoE | `qwen3_moe` | 文本 | |
| Qwen3-Next | `qwen3_next` | 文本 | |
| Qwen3-VL | `qwen3_vl` | 文本、图像 | |
| Qwen3-VL-MoE | `qwen3_vl_moe` | 文本、图像 | |
| Qwen3.5 | `qwen3_5` | 文本、图像 | |
| Qwen3.5-MoE | `qwen3_5_moe` | 文本、图像 | |
| Qwen2.5-Omni | `qwen2_5_omni` | 文本、图像、视频、音频 | 只加载 thinker，不加载语音解码器 |
| Qwen3-Omni | `qwen3_omni_moe` | 文本、图像、视频、音频 | 只加载 thinker |
| Gemma、Gemma 2 | `gemma`、`gemma2` | 文本 | |
| Gemma 3 | `gemma3`、`gemma3_text` | 文本 | 文本路径测过 |
| Gemma 3n | `gemma3n`、`gemma3n_text` | 文本 | 不支持图像 |
| Gemma 4 | `gemma4`、`gemma4_text`、`gemma4_unified` | 文本、图像、视频、音频 | |
| Llama、Llama 3 | `llama` | 文本 | |
| Llama 4 | `llama4`、`llama4_text` | 文本、图像 | `llama4_text` 只有文本 |
| Llama 3.2 Vision | `mllama` | 文本、图像 | |
| Mistral、Mistral 3 | `mistral`、`mistral3` | 文本 | |
| Ministral | `ministral` | 文本 | |
| Mixtral | `mixtral` | 文本 | |
| Pixtral | `pixtral` | 文本、图像 | |
| InternVL | `internvl`、`internvl_chat` | 文本、图像 | |
| Intern-S1 | `interns1` | 文本、图像 | |
| ModernBERT | `modernbert` | 文本 | 仅 `slot` 和 `pointer` |
| BERT | `bert` | 文本 | 仅 `slot` 和 `pointer` |
| RoBERTa | `roberta` | 文本 | 仅 `slot` 和 `pointer` |
| XLM-RoBERTa | `xlm-roberta` | 文本 | 仅 `slot` 和 `pointer` |
| DeBERTa、DeBERTa-v2 | `deberta`、`deberta-v2` | 文本 | 仅 `slot` 和 `pointer` |
| 其他因果语言模型 | — | 看配置 | `GenericFamily` 推断模型类、主干和 LoRA 目标 |

编码器没有 LM head，所以不能用 `marker`。更大的模型和 4bit 加载写在 [`jevtrainer/config.py`](jevtrainer/config.py)。加一个家族：[`docs/adding_model.md`](docs/adding_model.md)。

## 训练方式

| 读出 | 怎么读 | LoRA | 全参数 |
| --- | --- | --- | --- |
| `marker` | 每个问题一个 `<decision>` 占位，读它前面选项符号的 logits | ✅ | ✅ |
| `slot` | 选项打乱到预留的 `<\|decision_000\|>` 等 token 上，用隐藏状态和 slot 向量的点积 | ✅ | ✅ |
| `pointer` | 每个选项包在 `<opt>…</opt>` 里，用 `<decide>` 和每个 `</opt>` 的双线性分数 | ✅ | ✅ |

默认值：`finetune: lora`，`lora.r: 16`，LoRA 学习率 1e-4，全参数 1e-5，以及 `epochs`、`batch_size`、`grad_accum` 和留给温度校准的 `holdout`。覆盖某一项：`jt train my.yaml --set lr=5e-5 --set lora.r=32`。

## 数据集

`jt data list` 列出全部 **333** 个数据集。`eval_only` 的数据集不进训练混合。训练样本的状态若和某条评测记录相同，会被丢掉，`--dry-run` 会报告丢掉多少。

- **文本**：意图、主题、情感、安全、NLI、阅读、知识、工具、偏好，以及一套中文数据集
- **图像和 GUI**：截图上的元素选择、动作类型、轨迹的下一步
- **音频**：情感、事件、语音指令、发音打分、会议
- **视频**：动作、问答、时间区间、步骤、字幕、屏幕录像
- **音视频**：问答、事件、情感、唇读

评测套件包括 `intern-accuracy-v1`、`core`、`gui-v1`、`zh-bench`、`av-omni`（WorldSense、Daily-Omni、OmniBench、AV-Odyssey、AV-SpeakerBench）、`audio-bench`、`video-bench`。其余用 `jt bench list` 查看。

`jt data prepare NAME --cap N` 写入 `$JEVTRAINER_CACHE`。`load` 找的是 `split-cap<cap>-v<version>.jsonl`。这个文件不在时，会用同一次划分里最新的旧缓存。训练默认 cap 是 100000。开跑前把要用的那一份准备好。

加一个数据集：[`docs/adding_dataset.md`](docs/adding_dataset.md)。

## 环境

| | 最低 |
| --- | --- |
| Python | 3.10 |
| torch | 按训练用的 CUDA 安装 |
| transformers | 5.5 |
| ffmpeg | 在 `PATH` 上，或安装 `imageio-ffmpeg`，音视频需要 |

## 开始使用

### 安装

```bash
pip install -e .
```

### 最快的一条训练

```bash
jt init --readout marker > my.yaml
jt train my.yaml --dry-run   # 条数、泄漏检查、可训练参数、一条渲染好的样本
jt train my.yaml
```

最小配置：

```yaml
model: Qwen/Qwen3.5-0.8B
readout: marker            # marker | slot | pointer
dataset: banking77,boolq   # 注册名、name:split，或 ./my.jsonl
```

音视频用能编码这些模态的底座：

```yaml
model: Qwen/Qwen2.5-Omni-3B
readout: marker
readout_options:
  video_fps: 1.0
  video_max_frames: 32
  use_audio_in_video: true
dataset: music_avqa,ravdess
```

现成的 Omni-3B 混合是 [`configs/av/omni3b_v4.yaml`](configs/av/omni3b_v4.yaml)。`JT_VIDEO_FPS=1` 大约每秒抽一帧，缓存文件名带 `-fps1`。

一次运行的目录里有 `config.yaml`、权重或 adapter、`readout.safetensors`、`readout.json`、`calibration.json`、`train_log.jsonl`、`metrics.json`。`save_steps: N` 每 N 步写一个 `step-N/`。中断后用同一条命令加上 `--set resume=true` 接着训。

### 评测

```bash
jt eval configs/eval/intern_accuracy_v1.yaml --set checkpoint=runs/my-run
```

### 服务

```bash
jt serve runs/my-run    # POST /v1/systemone
```

## 项目结构

```text
jevtrainer/schema.py        记录格式
jevtrainer/config.py        YAML
jevtrainer/media.py         图像、视频帧、音频
jevtrainer/data/            数据集注册、混合、converters/
jevtrainer/readouts/        marker.py、slot.py、pointer.py
jevtrainer/model/           加载、LoRA 或全参数、families/
jevtrainer/train/           训练、损失、温度校准
jevtrainer/eval/            指标、评测套件
jevtrainer/serve/           /v1/systemone
configs/                    训练和评测的 YAML
```

加一个读出：[`docs/adding_readout.md`](docs/adding_readout.md)。

```bash
pip install -e .[dev,serve]
pytest
```

## 许可证

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)。
