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

### One YAML file to train a model that answers with probabilities, not generated text

</div>

\[ English | [中文](README_zh.md) \]

A Jev model reads a state (text, JSON, and optionally an image, a video, or audio) and questions whose options are already written. One forward pass returns a probability for every option.

| Question | The caller supplies | The model returns |
| --- | --- | --- |
| `choice` | 2–255 named options | the chosen key, the full distribution, a confidence |
| `score` | 2–10 ordered levels | the expected level, the distribution, a confidence |
| `noul` | one yes/no proposition | P(yes) |

The probabilities are read out of a language model. LoRA and full fine-tuning are both supported. A new dataset, readout, or base model is a small module, and one YAML file starts a run.

## Table of Contents

- [Features](#features)
- [Released models](#released-models)
- [Supported models](#supported-models)
- [Training](#training)
- [Datasets](#datasets)
- [Requirement](#requirement)
- [Getting started](#getting-started)
  - [Installation](#installation)
  - [Quickstart](#quickstart)
  - [Evaluate](#evaluate)
  - [Serve](#serve)
- [Layout](#layout)
- [License](#license)

## Features

- **One file to train**: `jt init` writes a YAML, `jt train` runs it. Unknown fields are rejected with a close name.
- **Three question types**: `choice`, `score`, and `noul`, scored in one forward pass.
- **Three readouts**: `marker` (Intern-Decision), `slot` (Bosun v3.1), `pointer` (Kev). Each works with LoRA or full fine-tuning.
- **Modalities**: text, images, GUI screenshots, audio, video, and audio-video, on bases that can encode them.
- **Tested checkpoints**: the models in [Supported models](#supported-models). Each one has completed a real forward and backward, or a training run.
- **333 datasets** already registered. `jt data list` and `jt bench list` show them.
- **Serve**: `jt serve` exposes `POST /v1/systemone`.

## Released models

Download a repo and pass the directory as `checkpoint`. A LoRA model still downloads its base weights.

| Model | Date | Base | Result |
| --- | --- | --- | --- |
| [weihongliang/jev-marker-qwen35-0.8b-2026-10-02](https://huggingface.co/weihongliang/jev-marker-qwen35-0.8b-2026-10-02) | 2026-10-02 | Qwen3.5-0.8B, full, `marker` | Intern seven-suite average **82.52** (Intern-Decision-0.8B is 79.38) |
| [weihongliang/jev-marker-omni3b-2026-10-04](https://huggingface.co/weihongliang/jev-marker-omni3b-2026-10-04) | 2026-10-04 | Qwen2.5-Omni-3B thinker, LoRA r=32, `marker` | Holdout accuracy **85.03%** (1,610 examples) |

```bash
huggingface-cli download weihongliang/jev-marker-qwen35-0.8b-2026-10-02 --local-dir runs/jev-qwen35-0.8b
jt eval configs/eval/intern_accuracy_v1.yaml --set checkpoint=runs/jev-qwen35-0.8b

huggingface-cli download weihongliang/jev-marker-omni3b-2026-10-04 --local-dir runs/jev-omni3b
jt eval configs/eval/av_omni.yaml --set checkpoint=runs/jev-omni3b --set benchmarks=worldsense,omnibench
```

## Supported models

These checkpoints were run on a real GPU. The smoke test is `scripts/smoke_models.py`: bf16, LoRA r=8, `marker`, three text records, and one image record when noted. A training run is listed when one has actually started.

| Checkpoint | Input | What was run |
| --- | --- | --- |
| [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | text, image | full `marker` training; LoRA smoke for `marker`, `slot`, and `pointer`, including images |
| [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | text, image | LoRA `marker` training |
| [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | text, image | smoke, including an image |
| [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | text, image | smoke, including an image |
| [Qwen/Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) | text, image, video, audio | LoRA `marker` training of the thinker |
| [google/gemma-3-270m](https://huggingface.co/google/gemma-3-270m) | text | smoke |
| [google/gemma-4-E2B](https://huggingface.co/google/gemma-4-E2B) | text | smoke; the audio and vision towers stay frozen |
| [google/gemma-4-E4B](https://huggingface.co/google/gemma-4-E4B) | text | smoke |
| [google/gemma-4-12B](https://huggingface.co/google/gemma-4-12B) | text | smoke |
| [openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) | text | smoke |
| [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) | text | smoke |
| [LiquidAI/LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M) | text | smoke |
| [LiquidAI/LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) | text | smoke |
| [Rta-AILabs/Nandi-Mini-150M](https://huggingface.co/Rta-AILabs/Nandi-Mini-150M) | text | smoke |
| [FrontiersMind/Lumma-0.6B-Base](https://huggingface.co/FrontiersMind/Lumma-0.6B-Base) | text | smoke |

`Qwen/Qwen3.8-27B` and `google/gemma-4-26B-A4B-it` were not run: each checkpoint is about 50 GB. Other sizes of these series have not been tested. `jt model list` shows which `model_type` values the code recognizes. Adding a family: [`docs/adding_model.md`](docs/adding_model.md).

## Training

| Readout | How it reads | LoRA | Full |
| --- | --- | --- | --- |
| `marker` | one `<decision>` placeholder per question; read the logits of the option letters in front of it | ✅ | ✅ |
| `slot` | options are shuffled onto reserved `<\|decision_000\|>` tokens; score is a dot product with a slot vector | ✅ | ✅ |
| `pointer` | each option is wrapped in `<opt>…</opt>`; a bilinear score between `<decide>` and each `</opt>` | ✅ | ✅ |

Defaults: `finetune: lora`, `lora.r: 16`, learning rate 1e-4 for LoRA and 1e-5 for full fine-tuning, plus `epochs`, `batch_size`, `grad_accum`, and a `holdout` kept for temperature calibration. Override a field with `jt train my.yaml --set lr=5e-5 --set lora.r=32`.

## Datasets

`jt data list` lists all **333** datasets. An `eval_only` set stays out of a training mix. A training example whose state matches an evaluation record is dropped, and `--dry-run` reports how many.

- **Text**: intent, topic, sentiment, safety, NLI, reading, knowledge, tools, preference, and a Chinese collection
- **Images and GUI**: element choice on a screenshot, action type, the next step of a trajectory
- **Audio**: emotion, events, spoken commands, pronunciation scores, meetings
- **Video**: actions, question answering, time spans, steps, subtitles, screen recordings
- **Audio-video**: question answering, events, emotion, lip reading

Benchmark suites include `intern-accuracy-v1`, `core`, `gui-v1`, `zh-bench`, `av-omni` (WorldSense, Daily-Omni, OmniBench, AV-Odyssey, AV-SpeakerBench), `audio-bench`, and `video-bench`. `jt bench list` shows the rest.

`jt data prepare NAME --cap N` writes `$JEVTRAINER_CACHE`. `load` looks for `split-cap<cap>-v<version>.jsonl`. When that file is missing, it reuses the newest older cache of the same split. The training default cap is 100000. Prepare the file a run will actually use.

Adding a dataset: [`docs/adding_dataset.md`](docs/adding_dataset.md).

## Requirement

| | Minimum |
| --- | --- |
| Python | 3.10 |
| torch | installed with the CUDA build you train on |
| transformers | 5.5 |
| ffmpeg | on `PATH`, or the `imageio-ffmpeg` package, for audio and video |

## Getting started

### Installation

```bash
pip install -e .
```

### Quickstart

```bash
jt init --readout marker > my.yaml
jt train my.yaml --dry-run   # counts, leak check, trainable parameters, one rendered example
jt train my.yaml
```

Smallest config:

```yaml
model: Qwen/Qwen3.5-0.8B
readout: marker            # marker | slot | pointer
dataset: banking77,boolq   # a registered name, name:split, or ./my.jsonl
```

Audio and video need a base that encodes them:

```yaml
model: Qwen/Qwen2.5-Omni-3B
readout: marker
readout_options:
  video_fps: 1.0
  video_max_frames: 32
  use_audio_in_video: true
dataset: music_avqa,ravdess
```

A ready-made Omni-3B mix is [`configs/av/omni3b_v4.yaml`](configs/av/omni3b_v4.yaml). `JT_VIDEO_FPS=1` extracts about one frame per second, and the cache name carries `-fps1`.

A run directory holds `config.yaml`, the weights or an adapter, `readout.safetensors`, `readout.json`, `calibration.json`, `train_log.jsonl`, and `metrics.json`. `save_steps: N` writes `step-N/` every N steps. After an interrupt, rerun the same command with `--set resume=true`.

### Evaluate

```bash
jt eval configs/eval/intern_accuracy_v1.yaml --set checkpoint=runs/my-run
```

### Serve

```bash
jt serve runs/my-run    # POST /v1/systemone
```

## Layout

```text
jevtrainer/schema.py        record format
jevtrainer/config.py        YAML
jevtrainer/media.py         images, video frames, audio
jevtrainer/data/            dataset registry, mixes, converters/
jevtrainer/readouts/        marker.py, slot.py, pointer.py
jevtrainer/model/           loading, LoRA or full fine-tune, families/
jevtrainer/train/           training, loss, temperature calibration
jevtrainer/eval/            metrics, benchmark suites
jevtrainer/serve/           /v1/systemone
configs/                    training and evaluation YAML
```

Adding a readout: [`docs/adding_readout.md`](docs/adding_readout.md).

```bash
pip install -e .[dev,serve]
pytest
```

## License

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
