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

| Model | Model size |
| --- | --- |
| [Qwen3.5](https://huggingface.co/Qwen) | 0.8B/2B/4B/9B |
| [Qwen2.5-Omni](https://huggingface.co/Qwen) | 3B |
| [Gemma 3](https://huggingface.co/google) | 270M |
| [Gemma 4](https://huggingface.co/google) | E2B/E4B/12B |
| [MiniCPM5](https://huggingface.co/openbmb) | 1B/2B |
| [LFM2.5](https://huggingface.co/LiquidAI) | 350M/2.6B |
| [Nandi-Mini](https://huggingface.co/Rta-AILabs) | 150M |
| [Lumma](https://huggingface.co/FrontiersMind) | 0.6B |

Larger models and 4-bit loading are set in [`jevtrainer/config.py`](jevtrainer/config.py). Adding a family: [`docs/adding_model.md`](docs/adding_model.md).


## Training

| Readout | How it reads | LoRA | Full |
| --- | --- | --- | --- |
| `marker` | one `<decision>` placeholder per question; read the logits of the option letters in front of it | ✅ | ✅ |
| `slot` | options are shuffled onto reserved `<\|decision_000\|>` tokens; score is a dot product with a slot vector | ✅ | ✅ |
| `pointer` | each option is wrapped in `<opt>…</opt>`; a bilinear score between `<decide>` and each `</opt>` | ✅ | ✅ |

Defaults: `finetune: lora`, `lora.r: 16`, learning rate 1e-4 for LoRA and 1e-5 for full fine-tuning, plus `epochs`, `batch_size`, `grad_accum`, and a `holdout` kept for temperature calibration. Override a field with `jt train my.yaml --set lr=5e-5 --set lora.r=32`.

## Datasets

`jt data list` lists all **333** datasets. Each name is a registered converter: put it in `dataset:` and the library builds the records. A name marked `(eval)` is `eval_only` and stays out of a training mix. A training example whose state matches an evaluation record is dropped, and `--dry-run` reports how many.

<details><summary>Intent</summary>

- [atis](https://huggingface.co/datasets/tuetschek/atis)
- [banking77](https://huggingface.co/datasets/mteb/banking77)
- [bitext_support](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- [clinc150](https://huggingface.co/datasets/clinc/clinc_oos)
- [massive_intent](https://huggingface.co/datasets/mteb/amazon_massive_intent)
- [massive_intent_zh](https://huggingface.co/datasets/mteb/amazon_massive_intent)
- [massive_scenario](https://huggingface.co/datasets/mteb/amazon_massive_scenario)
- [massive_scenario_zh](https://huggingface.co/datasets/mteb/amazon_massive_scenario)
- [trec](https://huggingface.co/datasets/CogComp/trec)

</details>

<details><summary>Topic</summary>

- [agnews](https://huggingface.co/datasets/fancyzhx/ag_news)
- [dbpedia14](https://huggingface.co/datasets/fancyzhx/dbpedia_14)
- [newsgroups20](https://huggingface.co/datasets/SetFit/20_newsgroups)
- [yahoo_topics](https://huggingface.co/datasets/community-datasets/yahoo_answers_topics)

</details>

<details><summary>Sentiment</summary>

- [amazon_polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity)
- [amazon_stars](https://huggingface.co/datasets/SetFit/amazon_reviews_multi_en)
- [amazon_zh](https://huggingface.co/datasets/SetFit/amazon_reviews_multi_zh)
- [chnsenticorp](https://huggingface.co/datasets/lansinuote/ChnSentiCorp)
- [emotion](https://huggingface.co/datasets/dair-ai/emotion)
- [go_emotions](https://huggingface.co/datasets/google-research-datasets/go_emotions)
- [imdb](https://huggingface.co/datasets/stanfordnlp/imdb)
- [rotten_tomatoes](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes)
- [sst2](https://huggingface.co/datasets/stanfordnlp/sst2)
- [sst5](https://huggingface.co/datasets/SetFit/sst5)
- [tweet_emotion](https://huggingface.co/datasets/cardiffnlp/tweet_eval)
- [tweet_irony](https://huggingface.co/datasets/cardiffnlp/tweet_eval)
- [tweet_sentiment](https://huggingface.co/datasets/cardiffnlp/tweet_eval)
- [weibo_emotion](https://huggingface.co/datasets/souljoy/COVID-19_weibo_emotion)
- [weibo_senti](https://huggingface.co/datasets/dirtycomputer/weibo_senti_100k)
- [yelp](https://huggingface.co/datasets/Yelp/yelp_review_full)
- [zh_jdreview](https://huggingface.co/datasets/C-MTEB/JDReview-classification)
- [zh_sentiment3](https://huggingface.co/datasets/C-MTEB/MultilingualSentiment-classification)
- [zh_shopping](https://huggingface.co/datasets/C-MTEB/OnlineShopping-classification)
- [zh_waimai](https://huggingface.co/datasets/C-MTEB/Waimai-classification)

</details>

<details><summary>Natural language inference</summary>

- [anli](https://huggingface.co/datasets/facebook/anli)
- [cb](https://huggingface.co/datasets/aps/super_glue)
- [cmnli](https://huggingface.co/datasets/clue/clue)
- [mnli](https://huggingface.co/datasets/nyu-mll/multi_nli)
- [ocnli](https://huggingface.co/datasets/clue/clue)
- [qnli](https://huggingface.co/datasets/nyu-mll/glue)
- [rte](https://huggingface.co/datasets/nyu-mll/glue)
- [scitail](https://huggingface.co/datasets/allenai/scitail)
- [snli](https://huggingface.co/datasets/stanfordnlp/snli)

</details>

<details><summary>Reading</summary>

- [boolq](https://huggingface.co/datasets/google/boolq)
- [c3](https://huggingface.co/datasets/clue/clue)
- [multirc](https://huggingface.co/datasets/aps/super_glue)
- [quality](https://huggingface.co/datasets/emozilla/quality)
- [race](https://huggingface.co/datasets/ehovy/race)

</details>

<details><summary>Knowledge</summary>

- [agieval_zh](https://huggingface.co/datasets/hails/agieval) (eval)
- [arc](https://huggingface.co/datasets/allenai/ai2_arc)
- [ceval_val](https://huggingface.co/datasets/ceval/ceval-exam) (eval)
- [cmmlu](https://huggingface.co/datasets/haonan-li/cmmlu) (eval)
- [gpqa_diamond](https://huggingface.co/datasets/Idavidrein/gpqa) (eval)
- [gpqa_main](https://huggingface.co/datasets/Idavidrein/gpqa) (eval)
- [hle](https://huggingface.co/datasets/cais/hle) (eval)
- [mmlu](https://huggingface.co/datasets/cais/mmlu) (eval)
- [mmlu_aux](https://huggingface.co/datasets/cais/mmlu)
- [mmlu_pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) (eval)
- [openbookqa](https://huggingface.co/datasets/allenai/openbookqa)
- [qasc](https://huggingface.co/datasets/allenai/qasc)
- [sciq](https://huggingface.co/datasets/allenai/sciq)
- [truthfulqa](https://huggingface.co/datasets/truthfulqa/truthful_qa) (eval)

</details>

<details><summary>Commonsense</summary>

- [copa](https://huggingface.co/datasets/aps/super_glue)
- [csqa](https://huggingface.co/datasets/tau/commonsense_qa)
- [hellaswag](https://huggingface.co/datasets/Rowan/hellaswag)
- [piqa](https://huggingface.co/datasets/baber/piqa)
- [social_iqa](https://huggingface.co/datasets/allenai/social_i_qa)
- [winogrande](https://huggingface.co/datasets/allenai/winogrande)

</details>

<details><summary>Reasoning</summary>

- [bbh](https://huggingface.co/datasets/lukaemon/bbh) (eval)
- [cladder](https://huggingface.co/datasets/causalnlp/CLadder) (eval)
- [logiqa](https://huggingface.co/datasets/lucasmccabe/logiqa)
- [musr](https://huggingface.co/datasets/TAUR-Lab/MuSR) (eval)
- [reclor](https://huggingface.co/datasets/tasksource/reclor)
- [strategyqa](https://huggingface.co/datasets/ChilleD/StrategyQA)

</details>

<details><summary>Language</summary>

- [chid](https://huggingface.co/datasets/clue/clue)
- [cluewsc](https://huggingface.co/datasets/clue/clue)
- [cola](https://huggingface.co/datasets/nyu-mll/glue)
- [csc](https://huggingface.co/datasets/shibing624/CSC)
- [subjectivity](https://huggingface.co/datasets/SetFit/subj)
- [text_correction_zh](https://huggingface.co/datasets/shibing624/chinese_text_correction)
- [wic](https://huggingface.co/datasets/aps/super_glue)

</details>

<details><summary>Dialogue</summary>

- [cdconv](https://huggingface.co/datasets/thu-coai/cdconv)
- [esconv](https://huggingface.co/datasets/thu-coai/esconv)

</details>

<details><summary>Preference</summary>

- [cvalues_rlhf](https://huggingface.co/datasets/Skepsun/cvalues_rlhf)
- [dpo_pairs_zh](https://huggingface.co/datasets/wenbopan/Chinese-dpo-pairs)
- [dpo_zh](https://huggingface.co/datasets/shibing624/DPO-En-Zh-20k-Preference)
- [helpsteer2](https://huggingface.co/datasets/nvidia/HelpSteer2)
- [helpsteer3](https://huggingface.co/datasets/nvidia/HelpSteer3)
- [hh_rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [shp](https://huggingface.co/datasets/stanfordnlp/SHP)
- [skywork_pref](https://huggingface.co/datasets/Skywork/Skywork-Reward-Preference-80K-v0.2)
- [ultrafeedback](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)
- [ultrafeedback_zh](https://huggingface.co/datasets/opencsg/UltraFeedback-chinese)
- [zhihu_rlhf](https://huggingface.co/datasets/liyucheng/zhihu_rlhf_3k)

</details>

<details><summary>Judgement</summary>

- [mt_bench_human](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments)
- [ppe_ifeval](https://huggingface.co/datasets/lmarena-ai/PPE-IFEval-Best-of-K)
- [reward_bench](https://huggingface.co/datasets/allenai/reward-bench)
- [reward_bench2](https://huggingface.co/datasets/allenai/reward-bench-2)

</details>

<details><summary>Safety</summary>

- [aegis2](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0)
- [beavertails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails)
- [chinese_safetyqa](https://huggingface.co/datasets/OpenStellarTeam/Chinese-SafetyQA) (eval)
- [civil_comments](https://huggingface.co/datasets/google/civil_comments)
- [cold](https://huggingface.co/datasets/thu-coai/cold)
- [hate_offensive](https://huggingface.co/datasets/tdavidson/hate_speech_offensive)
- [jailbreak_classification](https://huggingface.co/datasets/jackhhao/jailbreak-classification)
- [pku_saferlhf](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF)
- [safety_prompts_zh](https://huggingface.co/datasets/thu-coai/Safety-Prompts)
- [salad](https://huggingface.co/datasets/OpenSafetyLab/Salad-Data)
- [tc260](https://huggingface.co/datasets/BBBBBBBBBBBQ/TC260-Chinese-Safety-Prompts)
- [toxic_chat](https://huggingface.co/datasets/lmsys/toxic-chat)
- [toxicn](https://huggingface.co/datasets/JunyuLu/ToxiCN)
- [tweet_hate](https://huggingface.co/datasets/cardiffnlp/tweet_eval)
- [tweet_offensive](https://huggingface.co/datasets/cardiffnlp/tweet_eval)
- [wildjailbreak](https://huggingface.co/datasets/allenai/wildjailbreak)
- [xd_violence](https://huggingface.co/datasets/jherng/xd-violence)
- [xd_violence_type](https://huggingface.co/datasets/jherng/xd-violence)

</details>

<details><summary>Security</summary>

- [fake_jobs](https://huggingface.co/datasets/victor/real-or-fake-fake-jobposting-prediction)
- [phishing](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset)
- [prompt_injection](https://huggingface.co/datasets/deepset/prompt-injections)
- [safeguard_injection](https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection)

</details>

<details><summary>Classification</summary>

- [bias_in_bios](https://huggingface.co/datasets/LabHC/bias_in_bios)
- [hyperpartisan](https://huggingface.co/datasets/SemEvalWorkshop/hyperpartisan_news_detection)
- [iflytek](https://huggingface.co/datasets/clue/clue)
- [patents](https://huggingface.co/datasets/ccdv/patent-classification)
- [thucnews](https://huggingface.co/datasets/Tongjilibo/THUCNews)
- [tnews](https://huggingface.co/datasets/clue/clue)

</details>

<details><summary>Similarity</summary>

- [mrpc](https://huggingface.co/datasets/nyu-mll/glue)
- [paws](https://huggingface.co/datasets/google-research-datasets/paws)
- [qqp](https://huggingface.co/datasets/nyu-mll/glue)
- [stsb](https://huggingface.co/datasets/sentence-transformers/stsb)

</details>

<details><summary>Sentence pairs</summary>

- [afqmc](https://huggingface.co/datasets/clue/clue)
- [atec](https://huggingface.co/datasets/C-MTEB/ATEC)
- [bq_corpus](https://huggingface.co/datasets/C-MTEB/BQ)
- [lcqmc](https://huggingface.co/datasets/C-MTEB/LCQMC)
- [pawsx_zh](https://huggingface.co/datasets/C-MTEB/PAWSX)
- [zh_stsb](https://huggingface.co/datasets/C-MTEB/STSB)

</details>

<details><summary>Retrieval</summary>

- [esci](https://huggingface.co/datasets/tasksource/esci)
- [mmarco_rerank_zh](https://huggingface.co/datasets/C-MTEB/Mmarco-reranking)
- [qbqtc](https://huggingface.co/datasets/C-MTEB/QBQTC)
- [t2_rerank](https://huggingface.co/datasets/C-MTEB/T2Reranking)
- [wiki_qa](https://huggingface.co/datasets/microsoft/wiki_qa)

</details>

<details><summary>Retrieval-augmented generation</summary>

- [ragtruth](https://huggingface.co/datasets/wandb/RAGTruth-processed)

</details>

<details><summary>Multi-hop</summary>

- [hotpotqa](https://huggingface.co/datasets/hotpotqa/hotpot_qa)
- [musique](https://huggingface.co/datasets/bdsaglam/musique)
- [wiki2mh](https://huggingface.co/datasets/framolfese/2WikiMultihopQA)
- [wikihop](https://huggingface.co/datasets/QAngaroo/wiki_hop)

</details>

<details><summary>Fact checking</summary>

- [hover](https://huggingface.co/datasets/Dzeniks/hover)
- [liar2](https://huggingface.co/datasets/chengxuphd/liar2)

</details>

<details><summary>Verification</summary>

- [docnli](https://huggingface.co/datasets/tasksource/doc-nli)
- [halueval_summ](https://huggingface.co/datasets/pminervini/HaluEval)
- [wice](https://huggingface.co/datasets/tasksource/wice)

</details>

<details><summary>Numeric</summary>

- [drop](https://huggingface.co/datasets/ucinlp/drop)
- [tabfact](https://huggingface.co/datasets/table-benchmark/tabfact)

</details>

<details><summary>Math</summary>

- [gsm8k_mc](https://huggingface.co/datasets/openai/gsm8k)

</details>

<details><summary>Rules</summary>

- [sharc](https://huggingface.co/datasets/tasksource/sharc)

</details>

<details><summary>Tools</summary>

- [bfcl](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) (eval)
- [glaive_toolcall_zh](https://huggingface.co/datasets/llamafactory/glaive_toolcall_zh)
- [glaive_tools](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)
- [hermes_tools](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)
- [toolace](https://huggingface.co/datasets/Team-ACE/ToolACE)
- [when2call](https://huggingface.co/datasets/nvidia/When2Call) (eval)

</details>

<details><summary>Jev-format decisions</summary>

- [intern/agnews_test](https://github.com/internlm/Intern-Decision) (eval)
- [intern/jevbench_easy](https://github.com/internlm/Intern-Decision) (eval)
- [intern/jevbench_hard](https://github.com/internlm/Intern-Decision) (eval)
- [intern/jevbench_original](https://github.com/internlm/Intern-Decision) (eval)
- [intern/toolace_test](https://github.com/internlm/Intern-Decision) (eval)
- [intern/typed_decisions_test](https://github.com/internlm/Intern-Decision) (eval)
- [intern/wildjailbreak_test](https://github.com/internlm/Intern-Decision) (eval)
- [jebadiah_synth](https://huggingface.co/datasets/frontier-infra/jebadiah-synth-v2)
- [kev_suites](https://huggingface.co/datasets/jaredpalmer/kev-suites)
- [mojev_mix](https://huggingface.co/datasets/MoLeMo-Lab/mojev-mix)
- [onejev](https://huggingface.co/datasets/OmniJev/OneJev-Data)
- [pngwn_system_one](https://huggingface.co/datasets/pngwn/system-one-decisions)
- [pngwn_typed_v2](https://huggingface.co/datasets/pngwn/typed-decisions-v2-system-one)
- [this_that_complex](https://huggingface.co/datasets/limberc/this-that-complex-decisions) (eval)
- [typed_decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
- [typed_decisions_hf_test](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) (eval)

</details>

<details><summary>Games</summary>

- [nanojev](https://huggingface.co/datasets/C-Tianyu/NanoJev-Data)

</details>

<details><summary>Calibration</summary>

- [intern/known_distribution_pilot](https://github.com/internlm/Intern-Decision) (eval)

</details>

<details><summary>Code</summary>

- [cruxeval](https://huggingface.co/datasets/cruxeval-org/cruxeval) (eval)

</details>

<details><summary>Legal</summary>

- [cail2018](https://huggingface.co/datasets/china-ai-law-challenge/cail2018)
- [case_hold](https://huggingface.co/datasets/coastalcph/lex_glue)
- [contract_nli](https://huggingface.co/datasets/kiddothe2b/contract-nli)
- [ecthr](https://huggingface.co/datasets/coastalcph/lex_glue)
- [jecqa](https://huggingface.co/datasets/hails/agieval-jec-qa-kd) (eval)
- [ledgar](https://huggingface.co/datasets/coastalcph/lex_glue)
- [legal_case_zh](https://huggingface.co/datasets/gehits/Chinese-Legal-Case-Classification-Dataset)
- [legalbench_consumer_contracts](https://huggingface.co/datasets/nguha/legalbench)
- [legalbench_cuad](https://huggingface.co/datasets/nguha/legalbench)
- [legalbench_rules](https://huggingface.co/datasets/nguha/legalbench)
- [maud](https://huggingface.co/datasets/theatticusproject/maud)
- [unfair_tos](https://huggingface.co/datasets/coastalcph/lex_glue)

</details>

<details><summary>Finance</summary>

- [fin_news_topic](https://huggingface.co/datasets/zeroshot/twitter-financial-news-topic)
- [fin_phrasebank](https://huggingface.co/datasets/atrost/financial_phrasebank)
- [fin_tweets](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)
- [financeiq](https://huggingface.co/datasets/Duxiaoman-DI/FinanceIQ) (eval)
- [fincuge_news](https://huggingface.co/datasets/Maciel/FinCUGE-Instruction)
- [fincuge_sentiment](https://huggingface.co/datasets/Maciel/FinCUGE-Instruction)
- [finqa](https://huggingface.co/datasets/dreamerdeo/finqa)
- [tatqa](https://huggingface.co/datasets/next-tat/TAT-QA)

</details>

<details><summary>Medical</summary>

- [chip_sts](https://huggingface.co/datasets/dirtycomputer/CBLUE-CHIP-STS)
- [cmb](https://huggingface.co/datasets/FreedomIntelligence/CMB)
- [cmedqa1_rerank](https://huggingface.co/datasets/C-MTEB/CMedQAv1-reranking)
- [cmedqa_rerank](https://huggingface.co/datasets/C-MTEB/CMedQAv2-reranking)
- [cmexam](https://huggingface.co/datasets/fzkuji/CMExam)
- [kuake_qic](https://huggingface.co/datasets/wyp/CBlue-KUAKE-QIC)
- [kuake_qqr](https://huggingface.co/datasets/dirtycomputer/CBLUE-KUAKE-QQR)
- [kuake_qtr](https://huggingface.co/datasets/dirtycomputer/CBLUE-KUAKE-QTR)
- [medmcqa](https://huggingface.co/datasets/openlifescienceai/medmcqa)
- [medqa](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options)
- [pubmedqa](https://huggingface.co/datasets/qiaojin/PubMedQA) (eval)

</details>

<details><summary>Science</summary>

- [csl](https://huggingface.co/datasets/clue/clue)
- [csl_discipline](https://huggingface.co/datasets/C-MTEB/CLSClusteringP2P)
- [qasper](https://huggingface.co/datasets/allenai/qasper-yesno)

</details>

<details><summary>Spam</summary>

- [enron_spam](https://huggingface.co/datasets/SetFit/enron_spam)
- [sms_spam](https://huggingface.co/datasets/ucirvine/sms_spam)

</details>

<details><summary>Detection</summary>

- [hc3_zh](https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese)

</details>

<details><summary>Stance</summary>

- [c_stance](https://huggingface.co/datasets/yfhe/C-STANCE-A)

</details>

<details><summary>Support</summary>

- [support_tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)

</details>

<details><summary>Temporal</summary>

- [timeqa](https://huggingface.co/datasets/hugosousa/TimeQA)

</details>

<details><summary>Spatial</summary>

- [this_that_spatial](https://huggingface.co/datasets/limberc/this-that-spatial-bench) (eval)

</details>

<details><summary>Images</summary>

- [cauldron_ai2d](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron)
- [cauldron_aokvqa](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron)
- [cauldron_iconqa](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron)
- [cauldron_scienceqa](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron)
- [cauldron_tqa](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron)
- [cmmmu](https://huggingface.co/datasets/lmms-lab/CMMMU) (eval)
- [mmbench_cn](https://huggingface.co/datasets/lmms-lab/MMBench_CN) (eval)
- [scienceqa_img](https://huggingface.co/datasets/derek-thomas/ScienceQA)

</details>

<details><summary>GUI and agents</summary>

- [aguvis_aitw](https://huggingface.co/datasets/xlangai/aguvis-stage2)
- [aguvis_amex](https://huggingface.co/datasets/xlangai/aguvis-stage2)
- [aguvis_android_control](https://huggingface.co/datasets/xlangai/aguvis-stage2)
- [aguvis_coat](https://huggingface.co/datasets/xlangai/aguvis-stage2)
- [aguvis_gui_odyssey](https://huggingface.co/datasets/xlangai/aguvis-stage2)
- [aguvis_guide](https://huggingface.co/datasets/xlangai/aguvis-stage2)
- [aguvis_miniwob](https://huggingface.co/datasets/xlangai/aguvis-stage2)
- [cagui](https://huggingface.co/datasets/openbmb/CAGUI) (eval)
- [guiact_smartphone](https://huggingface.co/datasets/yiye2023/GUIAct)
- [guiact_web_multi](https://huggingface.co/datasets/yiye2023/GUIAct)
- [guiact_web_single](https://huggingface.co/datasets/yiye2023/GUIAct)
- [mind2web](https://huggingface.co/datasets/osunlp/Mind2Web)
- [mm_mind2web](https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web)
- [omniact](https://huggingface.co/datasets/Writer/omniact)
- [s1_mini](https://huggingface.co/datasets/DavidHatley/system-one-mini-data)
- [swe_agent](https://huggingface.co/datasets/nebius/SWE-agent-trajectories)
- [weblinx](https://huggingface.co/datasets/McGill-NLP/WebLINX)

</details>

<details><summary>Audio</summary>

- [aishell1_gender](https://huggingface.co/datasets/AISHELL/AISHELL-1)
- [ami_gender](https://huggingface.co/datasets/edinburghcstr/ami)
- [ami_same_speaker](https://huggingface.co/datasets/edinburghcstr/ami)
- [audioset](https://huggingface.co/datasets/agkphysics/AudioSet)
- [crema_d](https://huggingface.co/datasets/confit/cremad-parquet)
- [emodb](https://huggingface.co/datasets/confit/emodb-parquet)
- [esc50](https://huggingface.co/datasets/ashraq/esc50)
- [esd](https://huggingface.co/datasets/AbstractTTS/ESD_english)
- [fleurs_langid](https://huggingface.co/datasets/mteb/fleurs)
- [fsd50k](https://huggingface.co/datasets/philgzl/fsd50k)
- [minds14](https://huggingface.co/datasets/PolyAI/minds14)
- [mmar](https://huggingface.co/datasets/BoJack/MMAR) (eval)
- [mmau_mini](https://huggingface.co/datasets/gamma-lab-umd/MMAU-test-mini) (eval)
- [mmsu](https://huggingface.co/datasets/ddwang2000/MMSU) (eval)
- [ravdess](https://huggingface.co/datasets/confit/ravdess-parquet)
- [savee](https://huggingface.co/datasets/AbstractTTS/SAVEE)
- [slurp](https://huggingface.co/datasets/marcel-gohsen/slurp)
- [speech_commands](https://huggingface.co/datasets/pollen-robotics/speech-commands-v0.02)
- [speechocean762_accuracy](https://huggingface.co/datasets/mispeech/speechocean762)
- [speechocean762_fluency](https://huggingface.co/datasets/mispeech/speechocean762)
- [speechocean762_prosodic](https://huggingface.co/datasets/mispeech/speechocean762)
- [speechocean762_total](https://huggingface.co/datasets/mispeech/speechocean762)
- [tess](https://huggingface.co/datasets/AbstractTTS/TESS)
- [urbansound8k](https://huggingface.co/datasets/danavery/urbansound8K)
- [voicebench_mmsu](https://huggingface.co/datasets/hlt-lab/voicebench) (eval)
- [voicebench_openbookqa](https://huggingface.co/datasets/hlt-lab/voicebench) (eval)
- [voxconverse_overlap](https://huggingface.co/datasets/diarizers-community/voxconverse)
- [voxconverse_speakers](https://huggingface.co/datasets/diarizers-community/voxconverse)

</details>

<details><summary>Video</summary>

- [activitynet_captions](https://huggingface.co/datasets/friedrichor/ActivityNet_Captions)
- [activitynet_qa](https://huggingface.co/datasets/lmms-eval/ActivityNetQA)
- [charades](https://prior.allenai.org/projects/charades)
- [charades_sta](https://huggingface.co/datasets/VideoSearchR1/charades-stage1_data)
- [clevrer_mc](https://huggingface.co/datasets/ngqtrung/video-r1-clevrer-mc-v1)
- [coin](https://coin-dataset.github.io)
- [egoschema](https://huggingface.co/datasets/lmms-eval/egoschema) (eval)
- [epic_kitchens](https://huggingface.co/datasets/lightly-ai/epic-kitchens-100-clips)
- [genvidbench](https://huggingface.co/datasets/jian-0/GenVidBench)
- [genvideo](https://huggingface.co/datasets/sshaiton/GenVideo-50k-shard)
- [gui_world](https://huggingface.co/datasets/ONE-Lab/GUI-World)
- [gui_world_env](https://huggingface.co/datasets/ONE-Lab/GUI-World)
- [gui_world_goal](https://huggingface.co/datasets/ONE-Lab/GUI-World)
- [hmdb51](https://huggingface.co/datasets/divm/hmdb51)
- [kinetics400](https://huggingface.co/datasets/kiyoonkim/kinetics-400-targz)
- [longvideobench](https://huggingface.co/datasets/Jialuo21/LongVideoBench) (eval)
- [lsvq](https://huggingface.co/datasets/teowu/LSVQ-videos)
- [m4_vitevqa](https://huggingface.co/datasets/yankie123/tea-m4vitevqa)
- [msrvtt](https://huggingface.co/datasets/VLM2Vec/MSR-VTT)
- [msrvtt_qa](https://huggingface.co/datasets/Rohollah903/msrvtt_qa)
- [msvd_qa](https://huggingface.co/datasets/Rohollah903/msvd_qa)
- [mvbench](https://huggingface.co/datasets/OpenGVLab/MVBench) (eval)
- [nexar](https://huggingface.co/datasets/akhil0790/nexar_collision_prediction)
- [nextqa](https://huggingface.co/datasets/lmms-eval/NExTQA)
- [perception_test](https://huggingface.co/datasets/advaitgupta/perception_test_mcq) (eval)
- [perceptiontest_val](https://huggingface.co/datasets/lmms-eval/PerceptionTest_Val) (eval)
- [qvhighlights](https://huggingface.co/datasets/ayushsdev/qvhighlights-videos)
- [ssv2](https://huggingface.co/datasets/morpheushoc/something-something-v2)
- [star](https://huggingface.co/datasets/tdat1465/star-charades)
- [tempcompass](https://huggingface.co/datasets/lmms-eval/TempCompass) (eval)
- [tgif_qa](https://huggingface.co/datasets/Xiaodong/TGIF_Zero_Shot_QA_)
- [ucf101](https://huggingface.co/datasets/quchenyuan/UCF101-ZIP)
- [vatex](https://huggingface.co/datasets/Blazewild/Vatex_reencoded)
- [vatex_zh](https://huggingface.co/datasets/lmms-eval/VATEX_ZH)
- [video_mme](https://huggingface.co/datasets/lmms-lab/Video-MME) (eval)
- [video_mme_sub](https://huggingface.co/datasets/lmms-lab/Video-MME) (eval)
- [videogui_goal](https://huggingface.co/datasets/VideoGUI/VideoGUI-High-Plan)
- [videogui_plan](https://huggingface.co/datasets/VideoGUI/VideoGUI-High-Plan)
- [youcook2](https://huggingface.co/datasets/VLM2Vec/YouCook2)

</details>

<details><summary>Audio-video</summary>

- [av_odyssey](https://huggingface.co/datasets/AV-Odyssey/AV_Odyssey_Bench) (eval)
- [av_speakerbench](https://huggingface.co/datasets/plnguyen2908/AV-SpeakerBench) (eval)
- [ave](https://huggingface.co/datasets/UnFaZeD07/AVE-Dataset)
- [ave_match](https://huggingface.co/datasets/UnFaZeD07/AVE-Dataset)
- [avqa](https://huggingface.co/datasets/juyil/AVQA-videos)
- [avsbench](https://huggingface.co/datasets/UnFaZeD07/AVSBench)
- [chsims](https://huggingface.co/datasets/tamb2203579/CH-SIMS)
- [chsims2](https://huggingface.co/datasets/tamb2203579/CH-SIMSv2)
- [chsims_nonverbal](https://huggingface.co/datasets/tamb2203579/CH-SIMS)
- [cmu_mosei](https://huggingface.co/datasets/tamb2203579/CMU-MOSEI)
- [crema_d_video](https://huggingface.co/datasets/NoahMartinezXiang/2014_CREMA-D)
- [daily_omni](https://huggingface.co/datasets/liarliar/Daily-Omni) (eval)
- [lrs3_transcript](https://huggingface.co/datasets/mattymchen/lrs3-test)
- [meld](https://huggingface.co/datasets/THUIAR/MMLA-Datasets)
- [mintrec](https://huggingface.co/datasets/THUIAR/MMLA-Datasets)
- [music_avqa](https://huggingface.co/datasets/UnFaZeD07/Music-AVQA)
- [mustard](https://huggingface.co/datasets/THUIAR/MMLA-Datasets)
- [omnibench](https://huggingface.co/datasets/m-a-p/OmniBench) (eval)
- [urfunny](https://huggingface.co/datasets/THUIAR/MMLA-Datasets)
- [vggsound](https://huggingface.co/datasets/11hu83/vggsound)
- [worldsense](https://huggingface.co/datasets/lmms-lab/WorldSense) (eval)

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
