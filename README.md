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

`jt data list` lists all **333** datasets. A name marked `(eval)` is `eval_only` and stays out of a training mix. A training example whose state matches an evaluation record is dropped, and `--dry-run` reports how many.

<details><summary>Intent</summary>

* atis
* banking77
* bitext_support
* clinc150
* massive_intent
* massive_intent_zh
* massive_scenario
* massive_scenario_zh
* trec

</details>

<details><summary>Topic</summary>

* agnews
* dbpedia14
* newsgroups20
* yahoo_topics

</details>

<details><summary>Sentiment</summary>

* amazon_polarity
* amazon_stars
* amazon_zh
* chnsenticorp
* emotion
* go_emotions
* imdb
* rotten_tomatoes
* sst2
* sst5
* tweet_emotion
* tweet_irony
* tweet_sentiment
* weibo_emotion
* weibo_senti
* yelp
* zh_jdreview
* zh_sentiment3
* zh_shopping
* zh_waimai

</details>

<details><summary>Natural language inference</summary>

* anli
* cb
* cmnli
* mnli
* ocnli
* qnli
* rte
* scitail
* snli

</details>

<details><summary>Reading</summary>

* boolq
* c3
* multirc
* quality
* race

</details>

<details><summary>Knowledge</summary>

* agieval_zh (eval)
* arc
* ceval_val (eval)
* cmmlu (eval)
* gpqa_diamond (eval)
* gpqa_main (eval)
* hle (eval)
* mmlu (eval)
* mmlu_aux
* mmlu_pro (eval)
* openbookqa
* qasc
* sciq
* truthfulqa (eval)

</details>

<details><summary>Commonsense</summary>

* copa
* csqa
* hellaswag
* piqa
* social_iqa
* winogrande

</details>

<details><summary>Reasoning</summary>

* bbh (eval)
* cladder (eval)
* logiqa
* musr (eval)
* reclor
* strategyqa

</details>

<details><summary>Language</summary>

* chid
* cluewsc
* cola
* csc
* subjectivity
* text_correction_zh
* wic

</details>

<details><summary>Dialogue</summary>

* cdconv
* esconv

</details>

<details><summary>Preference</summary>

* cvalues_rlhf
* dpo_pairs_zh
* dpo_zh
* helpsteer2
* helpsteer3
* hh_rlhf
* shp
* skywork_pref
* ultrafeedback
* ultrafeedback_zh
* zhihu_rlhf

</details>

<details><summary>Judgement</summary>

* mt_bench_human
* ppe_ifeval
* reward_bench
* reward_bench2

</details>

<details><summary>Safety</summary>

* aegis2
* beavertails
* chinese_safetyqa (eval)
* civil_comments
* cold
* hate_offensive
* jailbreak_classification
* pku_saferlhf
* safety_prompts_zh
* salad
* tc260
* toxic_chat
* toxicn
* tweet_hate
* tweet_offensive
* wildjailbreak
* xd_violence
* xd_violence_type

</details>

<details><summary>Security</summary>

* fake_jobs
* phishing
* prompt_injection
* safeguard_injection

</details>

<details><summary>Classification</summary>

* bias_in_bios
* hyperpartisan
* iflytek
* patents
* thucnews
* tnews

</details>

<details><summary>Similarity</summary>

* mrpc
* paws
* qqp
* stsb

</details>

<details><summary>Sentence pairs</summary>

* afqmc
* atec
* bq_corpus
* lcqmc
* pawsx_zh
* zh_stsb

</details>

<details><summary>Retrieval</summary>

* esci
* mmarco_rerank_zh
* qbqtc
* t2_rerank
* wiki_qa

</details>

<details><summary>Retrieval-augmented generation</summary>

* ragtruth

</details>

<details><summary>Multi-hop</summary>

* hotpotqa
* musique
* wiki2mh
* wikihop

</details>

<details><summary>Fact checking</summary>

* hover
* liar2

</details>

<details><summary>Verification</summary>

* docnli
* halueval_summ
* wice

</details>

<details><summary>Numeric</summary>

* drop
* tabfact

</details>

<details><summary>Math</summary>

* gsm8k_mc

</details>

<details><summary>Rules</summary>

* sharc

</details>

<details><summary>Tools</summary>

* bfcl (eval)
* glaive_toolcall_zh
* glaive_tools
* hermes_tools
* toolace
* when2call (eval)

</details>

<details><summary>Jev-format decisions</summary>

* intern/agnews_test (eval)
* intern/jevbench_easy (eval)
* intern/jevbench_hard (eval)
* intern/jevbench_original (eval)
* intern/toolace_test (eval)
* intern/typed_decisions_test (eval)
* intern/wildjailbreak_test (eval)
* jebadiah_synth
* kev_suites
* mojev_mix
* onejev
* pngwn_system_one
* pngwn_typed_v2
* this_that_complex (eval)
* typed_decisions
* typed_decisions_hf_test (eval)

</details>

<details><summary>Games</summary>

* nanojev

</details>

<details><summary>Calibration</summary>

* intern/known_distribution_pilot (eval)

</details>

<details><summary>Code</summary>

* cruxeval (eval)

</details>

<details><summary>Legal</summary>

* cail2018
* case_hold
* contract_nli
* ecthr
* jecqa (eval)
* ledgar
* legal_case_zh
* legalbench_consumer_contracts
* legalbench_cuad
* legalbench_rules
* maud
* unfair_tos

</details>

<details><summary>Finance</summary>

* fin_news_topic
* fin_phrasebank
* fin_tweets
* financeiq (eval)
* fincuge_news
* fincuge_sentiment
* finqa
* tatqa

</details>

<details><summary>Medical</summary>

* chip_sts
* cmb
* cmedqa1_rerank
* cmedqa_rerank
* cmexam
* kuake_qic
* kuake_qqr
* kuake_qtr
* medmcqa
* medqa
* pubmedqa (eval)

</details>

<details><summary>Science</summary>

* csl
* csl_discipline
* qasper

</details>

<details><summary>Spam</summary>

* enron_spam
* sms_spam

</details>

<details><summary>Detection</summary>

* hc3_zh

</details>

<details><summary>Stance</summary>

* c_stance

</details>

<details><summary>Support</summary>

* support_tickets

</details>

<details><summary>Temporal</summary>

* timeqa

</details>

<details><summary>Spatial</summary>

* this_that_spatial (eval)

</details>

<details><summary>Images</summary>

* cauldron_ai2d
* cauldron_aokvqa
* cauldron_iconqa
* cauldron_scienceqa
* cauldron_tqa
* cmmmu (eval)
* mmbench_cn (eval)
* scienceqa_img

</details>

<details><summary>GUI and agents</summary>

* aguvis_aitw
* aguvis_amex
* aguvis_android_control
* aguvis_coat
* aguvis_gui_odyssey
* aguvis_guide
* aguvis_miniwob
* cagui (eval)
* guiact_smartphone
* guiact_web_multi
* guiact_web_single
* mind2web
* mm_mind2web
* omniact
* s1_mini
* swe_agent
* weblinx

</details>

<details><summary>Audio</summary>

* aishell1_gender
* ami_gender
* ami_same_speaker
* audioset
* crema_d
* emodb
* esc50
* esd
* fleurs_langid
* fsd50k
* minds14
* mmar (eval)
* mmau_mini (eval)
* mmsu (eval)
* ravdess
* savee
* slurp
* speech_commands
* speechocean762_accuracy
* speechocean762_fluency
* speechocean762_prosodic
* speechocean762_total
* tess
* urbansound8k
* voicebench_mmsu (eval)
* voicebench_openbookqa (eval)
* voxconverse_overlap
* voxconverse_speakers

</details>

<details><summary>Video</summary>

* activitynet_captions
* activitynet_qa
* charades
* charades_sta
* clevrer_mc
* coin
* egoschema (eval)
* epic_kitchens
* genvidbench
* genvideo
* gui_world
* gui_world_env
* gui_world_goal
* hmdb51
* kinetics400
* longvideobench (eval)
* lsvq
* m4_vitevqa
* msrvtt
* msrvtt_qa
* msvd_qa
* mvbench (eval)
* nexar
* nextqa
* perception_test (eval)
* perceptiontest_val (eval)
* qvhighlights
* ssv2
* star
* tempcompass (eval)
* tgif_qa
* ucf101
* vatex
* vatex_zh
* video_mme (eval)
* video_mme_sub (eval)
* videogui_goal
* videogui_plan
* youcook2

</details>

<details><summary>Audio-video</summary>

* av_odyssey (eval)
* av_speakerbench (eval)
* ave
* ave_match
* avqa
* avsbench
* chsims
* chsims2
* chsims_nonverbal
* cmu_mosei
* crema_d_video
* daily_omni (eval)
* lrs3_transcript
* meld
* mintrec
* music_avqa
* mustard
* omnibench (eval)
* urfunny
* vggsound
* worldsense (eval)

</details>

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
