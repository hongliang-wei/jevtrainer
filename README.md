<p align="center">
  <img src="docs/icon.jpg" width="160" alt="jevtrainer">
</p>

<p align="center"><a href="README_zh.md">中文</a></p>

# jevtrainer

Train a **Jev-style decision model** from one YAML file. The model reads a state (text, JSON, and optionally images, video, or audio) and a set of questions whose answers are already given. One forward pass returns a probability for every option. It does not generate text.

| Question | The caller supplies | The model returns |
|---|---|---|
| `choice` | 2–255 named options | the chosen key, the full distribution, a confidence |
| `score` | 2–10 ordered levels | the expected level, the distribution, a confidence |
| `noul` | one yes/no proposition | P(yes) |

Probabilities are read out of a language model. Three readouts are included. Each can be trained with LoRA or full fine-tuning.

| Readout | From | How it reads |
|---|---|---|
| `marker` | Intern-Decision | one `<decision>` placeholder per question; read the logits of the option letters A, B, … in front of it |
| `slot` | Bosun v3.1 | options are shuffled onto reserved `<\|decision_000\|>` tokens; score is the dot product of the last hidden state and a slot vector |
| `pointer` | Kev | each option is wrapped in `<opt>…</opt>`; a bilinear score between `<decide>` and each `</opt>` |

## Trained models

| Model | Date | Base | Notes |
|---|---|---|---|
| [weihongliang/jev-marker-qwen35-0.8b-2026-10-02](https://huggingface.co/weihongliang/jev-marker-qwen35-0.8b-2026-10-02) | 2026-10-02 | Qwen3.5-0.8B, full fine-tune, `marker` | Public text, Jev-format data, and images. Intern seven-suite average **82.52** (Intern-Decision-0.8B is 79.38) |
| [weihongliang/jev-marker-omni3b-2026-10-04](https://huggingface.co/weihongliang/jev-marker-omni3b-2026-10-04) | 2026-10-04 | Qwen2.5-Omni-3B thinker, LoRA r=32, `marker` | Text, images, audio, and video. Holdout accuracy **85.03%** (1,610 examples) |

Download a repo and pass it as a local checkpoint. A LoRA model still downloads its base weights.

```bash
huggingface-cli download weihongliang/jev-marker-qwen35-0.8b-2026-10-02 --local-dir runs/jev-qwen35-0.8b
jt eval configs/eval/intern_accuracy_v1.yaml --set checkpoint=runs/jev-qwen35-0.8b

huggingface-cli download weihongliang/jev-marker-omni3b-2026-10-04 --local-dir runs/jev-omni3b
jt eval configs/eval/av_omni.yaml --set checkpoint=runs/jev-omni3b --set benchmarks=worldsense,omnibench
```

## Models and datasets

`jt model list` lists model families. `GenericFamily` infers most causal language models, so a new base usually needs no new code. Wired up already: Qwen3.5 (text and images), Qwen3, Llama-style models, and ModernBERT (`slot` and `pointer` only). Qwen2.5-Omni and Qwen3-Omni use the `omni` family and load the thinker only. Larger models and 4-bit quantization are in [`jevtrainer/config.py`](jevtrainer/config.py).

`jt data list` lists all **333** datasets. An `eval_only` dataset cannot enter a training mix. A training example whose state matches an evaluation record is dropped. `--dry-run` reports how many were dropped.

* Text: intent, topic, sentiment, safety, NLI, reading, knowledge, tools, preference, and a Chinese collection
* Images and GUI: element choice on a screenshot, action type, the next step of a trajectory
* Audio: emotion, events, spoken commands, pronunciation scores, meetings
* Video: actions, question answering, time spans, steps, subtitles, screen recordings
* Audio-video: question answering, events, emotion, lip reading

Audio-video suites: `av-omni` (WorldSense, Daily-Omni, OmniBench, AV-Odyssey, AV-SpeakerBench), `audio-bench`, `video-bench`. Text and GUI also have `intern-accuracy-v1`, `core`, `gui-v1`, `zh-bench`, and others. `jt bench list` shows them all.

## Train and evaluate

```bash
pip install -e .            # Python >= 3.10, torch, transformers >= 5.5
jt init --readout marker > my.yaml
jt train my.yaml --dry-run  # counts, leak check, trainable parameters, one rendered example
jt train my.yaml
jt eval configs/eval/intern_accuracy_v1.yaml --set checkpoint=runs/my-run
jt serve runs/my-run        # POST /v1/systemone
```

Smallest config:

```yaml
model: Qwen/Qwen3.5-0.8B
readout: marker           # marker | slot | pointer
dataset: banking77,boolq  # a registered name, name:split, or ./my.jsonl
```

Everything else has a default: `finetune: lora` (or `full`), `lora.r: 16`, learning rate (1e-4 for LoRA, 1e-5 for full), `epochs`, `batch_size`, `grad_accum`, and `holdout` (kept for temperature calibration). An unknown field is an error, and the message suggests a close name. Override one field with `jt train my.yaml --set lr=5e-5 --set lora.r=32`.

Audio and video need a base that encodes those modalities, and ffmpeg (on the system, or via `imageio-ffmpeg`):

```yaml
model: Qwen/Qwen2.5-Omni-3B
readout: marker
readout_options:
  video_fps: 1.0
  video_max_frames: 32
  use_audio_in_video: true
dataset: music_avqa,ravdess
```

The ready-made Omni-3B mix is [`configs/av/omni3b_v4.yaml`](configs/av/omni3b_v4.yaml). `JT_VIDEO_FPS=1` extracts about one frame per second, and the cache filename carries `-fps1`.

`jt data prepare NAME --cap N` writes records into `$JEVTRAINER_CACHE`. `load` looks for `split-cap<cap>-v<version>.jsonl`. When that file is missing, it reuses the newest older cache of the same split and does not convert again. The training default cap is 100000. Prepare the file you intend to train on before the run starts.

A run directory holds `config.yaml`, the weights or an adapter, `readout.safetensors`, `readout.json`, `calibration.json`, `train_log.jsonl`, and `metrics.json`. `save_steps: N` writes `step-N/` every N steps. After an interrupt, rerun the same command with `--set resume=true` to continue from the latest `step-N/state/`.

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

Adding a dataset, a readout, or a model family: [`docs/adding_dataset.md`](docs/adding_dataset.md), [`docs/adding_readout.md`](docs/adding_readout.md), [`docs/adding_model.md`](docs/adding_model.md).

```bash
pip install -e .[dev,serve]
pytest
```
