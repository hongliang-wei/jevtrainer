<p align="center">
  <img src="docs/icon.jpg" width="160" alt="jevtrainer">
</p>

# jevtrainer

用一个 YAML 文件训练 **Jev 式决策模型**。模型读入一段状态（文本、JSON，以及可选的图像、视频或音频）和一组答案已经给定的问题，一次前向返回每个选项的概率，不生成文字。

| 题型 | 调用方给出 | 模型返回 |
|---|---|---|
| `choice` | 2–255 个命名选项 | 选中的键、全部概率、置信度 |
| `score` | 2–10 个有序档位 | 期望档位、分布、置信度 |
| `noul` | 一个是否命题 | P(是) |

概率从语言模型里读出来，有三种读出，都可以做 LoRA 或全参数微调：

| 读出 | 来源 | 怎么读 |
|---|---|---|
| `marker` | Intern-Decision | 每个问题一个 `<decision>` 占位，读它前面选项符号 A、B、… 的 logits |
| `slot` | Bosun v3.1 | 选项打乱到预留的 `<\|decision_000\|>` 等 token 上，用最后一层隐藏状态和 slot 向量的点积 |
| `pointer` | Kev | 每个选项包在 `<opt>…</opt>` 里，用 `<decide>` 和每个 `</opt>` 的双线性分数 |

## 训好的模型

| 模型 | 日期 | 底座 | 说明 |
|---|---|---|---|
| [weihongliang/jev-marker-qwen35-0.8b-2026-10-02](https://huggingface.co/weihongliang/jev-marker-qwen35-0.8b-2026-10-02) | 2026-10-02 | Qwen3.5-0.8B，全参数，`marker` | 公开文本、Jev 格式数据和图像。Intern 七套件平均 **82.52**（Intern-Decision-0.8B 为 79.38） |
| [weihongliang/jev-marker-omni3b-2026-10-04](https://huggingface.co/weihongliang/jev-marker-omni3b-2026-10-04) | 2026-10-04 | Qwen2.5-Omni-3B thinker，LoRA r=32，`marker` | 文本、图像、音频、视频。保留集准确率 **85.03%**（1610 条） |

下载后当作本地 checkpoint。LoRA 模型仍会去拉底座权重。

```bash
huggingface-cli download weihongliang/jev-marker-qwen35-0.8b-2026-10-02 --local-dir runs/jev-qwen35-0.8b
jt eval configs/eval/intern_accuracy_v1.yaml --set checkpoint=runs/jev-qwen35-0.8b

huggingface-cli download weihongliang/jev-marker-omni3b-2026-10-04 --local-dir runs/jev-omni3b
jt eval configs/eval/av_omni.yaml --set checkpoint=runs/jev-omni3b --set benchmarks=worldsense,omnibench
```

## 支持的模型和数据集

`jt model list` 列出模型家族。`GenericFamily` 能推断大多数因果语言模型，新底座通常不用写代码。已经接好的包括 Qwen3.5（文本和图像）、Qwen3、Llama 结构、ModernBERT（仅 `slot` 和 `pointer`）。Qwen2.5-Omni 和 Qwen3-Omni 用家族 `omni`，只加载 thinker。更大的模型和 4bit 量化见 [`jevtrainer/config.py`](jevtrainer/config.py)。

`jt data list` 列出全部 **333** 个数据集。`eval_only` 的数据集不能进训练混合。训练时若状态和某条评测记录相同，会自动丢掉，`--dry-run` 会报告丢掉多少。

* 文本：意图、主题、情感、安全、NLI、阅读、知识、工具、偏好，以及一套中文数据集
* 图像和 GUI：截图上的元素选择、动作类型、轨迹的下一步
* 音频：情感、事件、语音指令、发音打分、会议
* 视频：动作、问答、时间区间、步骤、字幕、屏幕录像
* 音视频：问答、事件、情感、唇读

音视频评测套件：`av-omni`（WorldSense、Daily-Omni、OmniBench、AV-Odyssey、AV-SpeakerBench）、`audio-bench`、`video-bench`。文本和 GUI 还有 `intern-accuracy-v1`、`core`、`gui-v1`、`zh-bench` 等，`jt bench list` 看全部。

## 训练和评测

```bash
pip install -e .            # Python >= 3.10，torch，transformers >= 5.5
jt init --readout marker > my.yaml
jt train my.yaml --dry-run  # 条数、泄漏检查、可训练参数、一条渲染好的样本
jt train my.yaml
jt eval configs/eval/intern_accuracy_v1.yaml --set checkpoint=runs/my-run
jt serve runs/my-run        # POST /v1/systemone
```

最小配置：

```yaml
model: Qwen/Qwen3.5-0.8B
readout: marker           # marker | slot | pointer
dataset: banking77,boolq  # 注册名、name:split，或 ./my.jsonl
```

其余都有默认值：`finetune: lora`（或 `full`）、`lora.r: 16`、学习率（LoRA 1e-4，全参数 1e-5）、`epochs`、`batch_size`、`grad_accum`、`holdout`（留作温度校准）。未知字段会报错并给出相近的名字。覆盖某一项：`jt train my.yaml --set lr=5e-5 --set lora.r=32`。

音视频用能编码这些模态的底座，并准备好 ffmpeg（系统里有，或装 `imageio-ffmpeg`）：

```yaml
model: Qwen/Qwen2.5-Omni-3B
readout: marker
readout_options:
  video_fps: 1.0
  video_max_frames: 32
  use_audio_in_video: true
dataset: music_avqa,ravdess
```

现成的 Omni-3B 混合是 [`configs/av/omni3b_v4.yaml`](configs/av/omni3b_v4.yaml)。`JT_VIDEO_FPS=1` 会按大约每秒一帧抽帧，缓存文件名带 `-fps1`。

`jt data prepare NAME --cap N` 把数据写进 `$JEVTRAINER_CACHE`。`load` 找的是 `split-cap<cap>-v<version>.jsonl`。这个文件不在时，会用同一次划分里最新的旧缓存，不再重新转换。训练默认 cap 是 100000，开跑前把要用的那一份准备好。

一次运行的目录里有 `config.yaml`、权重或 adapter、`readout.safetensors`、`readout.json`、`calibration.json`、`train_log.jsonl`、`metrics.json`。`save_steps: N` 每 N 步写一个 `step-N/`。中断后用同一条命令加上 `--set resume=true` 从最近的 `step-N/state/` 接着训。

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

加数据集、读出或模型家族：[`docs/adding_dataset.md`](docs/adding_dataset.md)、[`docs/adding_readout.md`](docs/adding_readout.md)、[`docs/adding_model.md`](docs/adding_model.md)。

```bash
pip install -e .[dev,serve]
pytest
```
