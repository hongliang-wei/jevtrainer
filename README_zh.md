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
- **测过的检查点**：[支持的模型](#支持的模型)里的每一个都在真实 GPU 上完成过一次前向加反向，或一次训练。
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

下面这些检查点在真实 GPU 上跑过。冒烟测试是 `scripts/smoke_models.py`：bf16、LoRA r=8、`marker`、三条文本，注明的再加一条带图记录。有训练记录的，是已经真正开跑过的。

| 检查点 | 输入 | 跑过什么 |
| --- | --- | --- |
| [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | 文本、图像 | 全参数 `marker` 训练；`marker`、`slot`、`pointer` 的 LoRA 冒烟，含图像 |
| [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | 文本、图像 | LoRA `marker` 训练 |
| [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | 文本、图像 | 冒烟，含一条图像 |
| [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 文本、图像 | 冒烟，含一条图像 |
| [Qwen/Qwen2.5-Omni-3B](https://huggingface.co/Qwen/Qwen2.5-Omni-3B) | 文本、图像、视频、音频 | thinker 上的 LoRA `marker` 训练 |
| [google/gemma-3-270m](https://huggingface.co/google/gemma-3-270m) | 文本 | 冒烟 |
| [google/gemma-4-E2B](https://huggingface.co/google/gemma-4-E2B) | 文本 | 冒烟；音频塔和视觉塔保持冻结 |
| [google/gemma-4-E4B](https://huggingface.co/google/gemma-4-E4B) | 文本 | 冒烟 |
| [google/gemma-4-12B](https://huggingface.co/google/gemma-4-12B) | 文本 | 冒烟 |
| [openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B) | 文本 | 冒烟 |
| [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) | 文本 | 冒烟 |
| [LiquidAI/LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M) | 文本 | 冒烟 |
| [LiquidAI/LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) | 文本 | 冒烟 |
| [Rta-AILabs/Nandi-Mini-150M](https://huggingface.co/Rta-AILabs/Nandi-Mini-150M) | 文本 | 冒烟 |
| [FrontiersMind/Lumma-0.6B-Base](https://huggingface.co/FrontiersMind/Lumma-0.6B-Base) | 文本 | 冒烟 |

`Qwen/Qwen3.8-27B` 和 `google/gemma-4-26B-A4B-it` 没有跑：权重大约 50 GB。同一系列的其他尺寸没有测过。`jt model list` 列出代码能识别的 `model_type`。加一个家族：[`docs/adding_model.md`](docs/adding_model.md)。

## 训练方式

| 读出 | 怎么读 | LoRA | 全参数 |
| --- | --- | --- | --- |
| `marker` | 每个问题一个 `<decision>` 占位，读它前面选项符号的 logits | ✅ | ✅ |
| `slot` | 选项打乱到预留的 `<\|decision_000\|>` 等 token 上，用隐藏状态和 slot 向量的点积 | ✅ | ✅ |
| `pointer` | 每个选项包在 `<opt>…</opt>` 里，用 `<decide>` 和每个 `</opt>` 的双线性分数 | ✅ | ✅ |

默认值：`finetune: lora`，`lora.r: 16`，LoRA 学习率 1e-4，全参数 1e-5，以及 `epochs`、`batch_size`、`grad_accum` 和留给温度校准的 `holdout`。覆盖某一项：`jt train my.yaml --set lr=5e-5 --set lora.r=32`。

## 数据集

`jt data list` 列出全部 **333** 个数据集。标了 `(eval)` 的是 `eval_only`，不能进训练混合。训练样本的状态若和某条评测记录相同，会被丢掉，`--dry-run` 会报告丢掉多少。

<details><summary>意图</summary>

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

<details><summary>主题</summary>

* agnews
* dbpedia14
* newsgroups20
* yahoo_topics

</details>

<details><summary>情感</summary>

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

<details><summary>自然语言推理</summary>

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

<details><summary>阅读</summary>

* boolq
* c3
* multirc
* quality
* race

</details>

<details><summary>知识</summary>

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

<details><summary>常识</summary>

* copa
* csqa
* hellaswag
* piqa
* social_iqa
* winogrande

</details>

<details><summary>推理</summary>

* bbh (eval)
* cladder (eval)
* logiqa
* musr (eval)
* reclor
* strategyqa

</details>

<details><summary>语言</summary>

* chid
* cluewsc
* cola
* csc
* subjectivity
* text_correction_zh
* wic

</details>

<details><summary>对话</summary>

* cdconv
* esconv

</details>

<details><summary>偏好</summary>

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

<details><summary>评判</summary>

* mt_bench_human
* ppe_ifeval
* reward_bench
* reward_bench2

</details>

<details><summary>安全</summary>

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

<details><summary>安全检测</summary>

* fake_jobs
* phishing
* prompt_injection
* safeguard_injection

</details>

<details><summary>分类</summary>

* bias_in_bios
* hyperpartisan
* iflytek
* patents
* thucnews
* tnews

</details>

<details><summary>相似度</summary>

* mrpc
* paws
* qqp
* stsb

</details>

<details><summary>句对</summary>

* afqmc
* atec
* bq_corpus
* lcqmc
* pawsx_zh
* zh_stsb

</details>

<details><summary>检索</summary>

* esci
* mmarco_rerank_zh
* qbqtc
* t2_rerank
* wiki_qa

</details>

<details><summary>检索增强</summary>

* ragtruth

</details>

<details><summary>多跳</summary>

* hotpotqa
* musique
* wiki2mh
* wikihop

</details>

<details><summary>事实核查</summary>

* hover
* liar2

</details>

<details><summary>核验</summary>

* docnli
* halueval_summ
* wice

</details>

<details><summary>数值</summary>

* drop
* tabfact

</details>

<details><summary>数学</summary>

* gsm8k_mc

</details>

<details><summary>规则</summary>

* sharc

</details>

<details><summary>工具</summary>

* bfcl (eval)
* glaive_toolcall_zh
* glaive_tools
* hermes_tools
* toolace
* when2call (eval)

</details>

<details><summary>Jev 格式决策</summary>

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

<details><summary>游戏</summary>

* nanojev

</details>

<details><summary>校准</summary>

* intern/known_distribution_pilot (eval)

</details>

<details><summary>代码</summary>

* cruxeval (eval)

</details>

<details><summary>法律</summary>

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

<details><summary>金融</summary>

* fin_news_topic
* fin_phrasebank
* fin_tweets
* financeiq (eval)
* fincuge_news
* fincuge_sentiment
* finqa
* tatqa

</details>

<details><summary>医学</summary>

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

<details><summary>科学</summary>

* csl
* csl_discipline
* qasper

</details>

<details><summary>垃圾信息</summary>

* enron_spam
* sms_spam

</details>

<details><summary>检测</summary>

* hc3_zh

</details>

<details><summary>立场</summary>

* c_stance

</details>

<details><summary>客服</summary>

* support_tickets

</details>

<details><summary>时间</summary>

* timeqa

</details>

<details><summary>空间</summary>

* this_that_spatial (eval)

</details>

<details><summary>图像</summary>

* cauldron_ai2d
* cauldron_aokvqa
* cauldron_iconqa
* cauldron_scienceqa
* cauldron_tqa
* cmmmu (eval)
* mmbench_cn (eval)
* scienceqa_img

</details>

<details><summary>GUI 与智能体</summary>

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

<details><summary>音频</summary>

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

<details><summary>视频</summary>

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

<details><summary>音视频</summary>

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
