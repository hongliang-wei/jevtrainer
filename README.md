<p align="center">
  <img src="docs/icon.jpg" width="160" alt="jevtrainer">
</p>

<p align="center">
  <a href="https://github.com/hongliang-wei/jevtrainer/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/hongliang-wei/jevtrainer?style=social"></a>
  <a href="https://github.com/hongliang-wei/jevtrainer/commits/main"><img alt="Last commit" src="https://img.shields.io/github/last-commit/hongliang-wei/jevtrainer"></a>
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-blue"></a>
  <a href="https://huggingface.co/weihongliang/jevtrainer-qwen35-0.8b-2026-10-02"><img alt="Hugging Face 0.8B" src="https://img.shields.io/badge/%F0%9F%A4%97-Qwen3.5%200.8B-yellow"></a>
  <a href="https://huggingface.co/weihongliang/jevtrainer-omni3b-2026-10-04"><img alt="Hugging Face Omni" src="https://img.shields.io/badge/%F0%9F%A4%97-Omni%203B-yellow"></a>
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

| Model | Date | Base | Training config |
| --- | --- | --- | --- |
| [weihongliang/jevtrainer-qwen35-0.8b-2026-10-02](https://huggingface.co/weihongliang/jevtrainer-qwen35-0.8b-2026-10-02) | 2026-10-02 | Qwen3.5-0.8B, full, `marker` | [`configs/repro/intern_0.8b_v4.yaml`](configs/repro/intern_0.8b_v4.yaml) |
| [weihongliang/jevtrainer-omni3b-2026-10-04](https://huggingface.co/weihongliang/jevtrainer-omni3b-2026-10-04) | 2026-10-04 | Qwen2.5-Omni-3B thinker, LoRA r=32, `marker` | [`configs/av/omni3b_v4.yaml`](configs/av/omni3b_v4.yaml) |

**Evaluation of [weihongliang/jevtrainer-qwen35-0.8b-2026-10-02](https://huggingface.co/weihongliang/jevtrainer-qwen35-0.8b-2026-10-02).** Accuracy in percent.

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="8">Seven suites. Average is the unweighted mean of these seven.</th>
<th rowspan="2">longdoc-dev</th>
<th rowspan="2">gui-v1</th>
</tr>
<tr>
<th>Easy</th>
<th>Original</th>
<th>Hard</th>
<th>Typed Decision</th>
<th>ToolACE</th>
<th>AG News</th>
<th>WildJailBreak</th>
<th>Average</th>
</tr>
</thead>
<tbody>
<tr>
<td>Intern-Decision-0.8B</td>
<td align="right">97.92</td>
<td align="right">80.56</td>
<td align="right">52.25</td>
<td align="right">77.35</td>
<td align="right">94.52</td>
<td align="right">88.61</td>
<td align="right">64.48</td>
<td align="right">79.38</td>
<td></td>
<td></td>
</tr>
<tr>
<td>Qwen/Qwen3.5-0.8B</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td align="right">40.37</td>
<td align="right">48.29</td>
</tr>
<tr>
<td>weihongliang/jevtrainer-qwen35-0.8b-2026-10-02</td>
<td align="right">100.00</td>
<td align="right">84.72</td>
<td align="right">50.45</td>
<td align="right">71.05</td>
<td align="right">95.81</td>
<td align="right">92.17</td>
<td align="right">83.44</td>
<td align="right"><b>82.52</b></td>
<td align="right">80.90</td>
<td align="right">66.71</td>
</tr>
</tbody>
</table>

Each file's `dataset:` block is the training mix (names, caps, and grouping). Hyperparameters are in the same file.

```bash
huggingface-cli download weihongliang/jevtrainer-qwen35-0.8b-2026-10-02 --local-dir runs/jev-qwen35-0.8b
jt eval configs/eval/intern_accuracy_v1.yaml --set checkpoint=runs/jev-qwen35-0.8b

huggingface-cli download weihongliang/jevtrainer-omni3b-2026-10-04 --local-dir runs/jev-omni3b
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

`jt data list` lists all **333** datasets in three groups. Each name is a registered converter: put it in `dataset:` and the library builds the records. A name marked `(eval)` is `eval_only` and stays out of a training mix. A training example whose state matches an evaluation record is dropped, and `--dry-run` reports how many.

Image questions (one picture, no audio and no video) are listed under Text, with the text sets. GUI actions, tool choice, and agent trajectories are under Agent. Audio, video, and audio-video are the third group.

<details><summary>Text (222)</summary>

- [aegis2](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0)
- [afqmc](https://huggingface.co/datasets/clue/clue)
- [agieval_zh](https://huggingface.co/datasets/hails/agieval) (eval)
- [agnews](https://huggingface.co/datasets/fancyzhx/ag_news)
- [amazon_polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity)
- [amazon_stars](https://huggingface.co/datasets/SetFit/amazon_reviews_multi_en)
- [amazon_zh](https://huggingface.co/datasets/SetFit/amazon_reviews_multi_zh)
- [anli](https://huggingface.co/datasets/facebook/anli)
- [arc](https://huggingface.co/datasets/allenai/ai2_arc)
- [atec](https://huggingface.co/datasets/C-MTEB/ATEC)
- [atis](https://huggingface.co/datasets/tuetschek/atis)
- [banking77](https://huggingface.co/datasets/mteb/banking77)
- [bbh](https://huggingface.co/datasets/lukaemon/bbh) (eval)
- [beavertails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails)
- [bias_in_bios](https://huggingface.co/datasets/LabHC/bias_in_bios)
- [bitext_support](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- [boolq](https://huggingface.co/datasets/google/boolq)
- [bq_corpus](https://huggingface.co/datasets/C-MTEB/BQ)
- [c3](https://huggingface.co/datasets/clue/clue)
- [c_stance](https://huggingface.co/datasets/yfhe/C-STANCE-A)
- [cail2018](https://huggingface.co/datasets/china-ai-law-challenge/cail2018)
- [case_hold](https://huggingface.co/datasets/coastalcph/lex_glue)
- [cauldron_ai2d](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron)
- [cauldron_aokvqa](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron)
- [cauldron_iconqa](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron)
- [cauldron_scienceqa](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron)
- [cauldron_tqa](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron)
- [cb](https://huggingface.co/datasets/aps/super_glue)
- [cdconv](https://huggingface.co/datasets/thu-coai/cdconv)
- [ceval_val](https://huggingface.co/datasets/ceval/ceval-exam) (eval)
- [chid](https://huggingface.co/datasets/clue/clue)
- [chinese_safetyqa](https://huggingface.co/datasets/OpenStellarTeam/Chinese-SafetyQA) (eval)
- [chip_sts](https://huggingface.co/datasets/dirtycomputer/CBLUE-CHIP-STS)
- [chnsenticorp](https://huggingface.co/datasets/lansinuote/ChnSentiCorp)
- [civil_comments](https://huggingface.co/datasets/google/civil_comments)
- [cladder](https://huggingface.co/datasets/causalnlp/CLadder) (eval)
- [clinc150](https://huggingface.co/datasets/clinc/clinc_oos)
- [cluewsc](https://huggingface.co/datasets/clue/clue)
- [cmb](https://huggingface.co/datasets/FreedomIntelligence/CMB)
- [cmedqa1_rerank](https://huggingface.co/datasets/C-MTEB/CMedQAv1-reranking)
- [cmedqa_rerank](https://huggingface.co/datasets/C-MTEB/CMedQAv2-reranking)
- [cmexam](https://huggingface.co/datasets/fzkuji/CMExam)
- [cmmlu](https://huggingface.co/datasets/haonan-li/cmmlu) (eval)
- [cmmmu](https://huggingface.co/datasets/lmms-lab/CMMMU) (eval)
- [cmnli](https://huggingface.co/datasets/clue/clue)
- [cola](https://huggingface.co/datasets/nyu-mll/glue)
- [cold](https://huggingface.co/datasets/thu-coai/cold)
- [contract_nli](https://huggingface.co/datasets/kiddothe2b/contract-nli)
- [copa](https://huggingface.co/datasets/aps/super_glue)
- [cruxeval](https://huggingface.co/datasets/cruxeval-org/cruxeval) (eval)
- [csc](https://huggingface.co/datasets/shibing624/CSC)
- [csl](https://huggingface.co/datasets/clue/clue)
- [csl_discipline](https://huggingface.co/datasets/C-MTEB/CLSClusteringP2P)
- [csqa](https://huggingface.co/datasets/tau/commonsense_qa)
- [cvalues_rlhf](https://huggingface.co/datasets/Skepsun/cvalues_rlhf)
- [dbpedia14](https://huggingface.co/datasets/fancyzhx/dbpedia_14)
- [docnli](https://huggingface.co/datasets/tasksource/doc-nli)
- [dpo_pairs_zh](https://huggingface.co/datasets/wenbopan/Chinese-dpo-pairs)
- [dpo_zh](https://huggingface.co/datasets/shibing624/DPO-En-Zh-20k-Preference)
- [drop](https://huggingface.co/datasets/ucinlp/drop)
- [ecthr](https://huggingface.co/datasets/coastalcph/lex_glue)
- [emotion](https://huggingface.co/datasets/dair-ai/emotion)
- [enron_spam](https://huggingface.co/datasets/SetFit/enron_spam)
- [esci](https://huggingface.co/datasets/tasksource/esci)
- [esconv](https://huggingface.co/datasets/thu-coai/esconv)
- [fake_jobs](https://huggingface.co/datasets/victor/real-or-fake-fake-jobposting-prediction)
- [fin_news_topic](https://huggingface.co/datasets/zeroshot/twitter-financial-news-topic)
- [fin_phrasebank](https://huggingface.co/datasets/atrost/financial_phrasebank)
- [fin_tweets](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)
- [financeiq](https://huggingface.co/datasets/Duxiaoman-DI/FinanceIQ) (eval)
- [fincuge_news](https://huggingface.co/datasets/Maciel/FinCUGE-Instruction)
- [fincuge_sentiment](https://huggingface.co/datasets/Maciel/FinCUGE-Instruction)
- [finqa](https://huggingface.co/datasets/dreamerdeo/finqa)
- [go_emotions](https://huggingface.co/datasets/google-research-datasets/go_emotions)
- [gpqa_diamond](https://huggingface.co/datasets/Idavidrein/gpqa) (eval)
- [gpqa_main](https://huggingface.co/datasets/Idavidrein/gpqa) (eval)
- [gsm8k_mc](https://huggingface.co/datasets/openai/gsm8k)
- [halueval_summ](https://huggingface.co/datasets/pminervini/HaluEval)
- [hate_offensive](https://huggingface.co/datasets/tdavidson/hate_speech_offensive)
- [hc3_zh](https://huggingface.co/datasets/Hello-SimpleAI/HC3-Chinese)
- [hellaswag](https://huggingface.co/datasets/Rowan/hellaswag)
- [helpsteer2](https://huggingface.co/datasets/nvidia/HelpSteer2)
- [helpsteer3](https://huggingface.co/datasets/nvidia/HelpSteer3)
- [hh_rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [hle](https://huggingface.co/datasets/cais/hle) (eval)
- [hotpotqa](https://huggingface.co/datasets/hotpotqa/hotpot_qa)
- [hover](https://huggingface.co/datasets/Dzeniks/hover)
- [hyperpartisan](https://huggingface.co/datasets/SemEvalWorkshop/hyperpartisan_news_detection)
- [iflytek](https://huggingface.co/datasets/clue/clue)
- [imdb](https://huggingface.co/datasets/stanfordnlp/imdb)
- [intern/agnews_test](https://github.com/internlm/Intern-Decision) (eval)
- [intern/jevbench_easy](https://github.com/internlm/Intern-Decision) (eval)
- [intern/jevbench_hard](https://github.com/internlm/Intern-Decision) (eval)
- [intern/jevbench_original](https://github.com/internlm/Intern-Decision) (eval)
- [intern/known_distribution_pilot](https://github.com/internlm/Intern-Decision) (eval)
- [intern/toolace_test](https://github.com/internlm/Intern-Decision) (eval)
- [intern/typed_decisions_test](https://github.com/internlm/Intern-Decision) (eval)
- [intern/wildjailbreak_test](https://github.com/internlm/Intern-Decision) (eval)
- [jailbreak_classification](https://huggingface.co/datasets/jackhhao/jailbreak-classification)
- [jebadiah_synth](https://huggingface.co/datasets/frontier-infra/jebadiah-synth-v2)
- [jecqa](https://huggingface.co/datasets/hails/agieval-jec-qa-kd) (eval)
- [kev_suites](https://huggingface.co/datasets/jaredpalmer/kev-suites)
- [kuake_qic](https://huggingface.co/datasets/wyp/CBlue-KUAKE-QIC)
- [kuake_qqr](https://huggingface.co/datasets/dirtycomputer/CBLUE-KUAKE-QQR)
- [kuake_qtr](https://huggingface.co/datasets/dirtycomputer/CBLUE-KUAKE-QTR)
- [lcqmc](https://huggingface.co/datasets/C-MTEB/LCQMC)
- [ledgar](https://huggingface.co/datasets/coastalcph/lex_glue)
- [legal_case_zh](https://huggingface.co/datasets/gehits/Chinese-Legal-Case-Classification-Dataset)
- [legalbench_consumer_contracts](https://huggingface.co/datasets/nguha/legalbench)
- [legalbench_cuad](https://huggingface.co/datasets/nguha/legalbench)
- [legalbench_rules](https://huggingface.co/datasets/nguha/legalbench)
- [liar2](https://huggingface.co/datasets/chengxuphd/liar2)
- [logiqa](https://huggingface.co/datasets/lucasmccabe/logiqa)
- [massive_intent](https://huggingface.co/datasets/mteb/amazon_massive_intent)
- [massive_intent_zh](https://huggingface.co/datasets/mteb/amazon_massive_intent)
- [massive_scenario](https://huggingface.co/datasets/mteb/amazon_massive_scenario)
- [massive_scenario_zh](https://huggingface.co/datasets/mteb/amazon_massive_scenario)
- [maud](https://huggingface.co/datasets/theatticusproject/maud)
- [medmcqa](https://huggingface.co/datasets/openlifescienceai/medmcqa)
- [medqa](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options)
- [mmarco_rerank_zh](https://huggingface.co/datasets/C-MTEB/Mmarco-reranking)
- [mmbench_cn](https://huggingface.co/datasets/lmms-lab/MMBench_CN) (eval)
- [mmlu](https://huggingface.co/datasets/cais/mmlu) (eval)
- [mmlu_aux](https://huggingface.co/datasets/cais/mmlu)
- [mmlu_pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) (eval)
- [mnli](https://huggingface.co/datasets/nyu-mll/multi_nli)
- [mojev_mix](https://huggingface.co/datasets/MoLeMo-Lab/mojev-mix)
- [mrpc](https://huggingface.co/datasets/nyu-mll/glue)
- [mt_bench_human](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments)
- [multirc](https://huggingface.co/datasets/aps/super_glue)
- [musique](https://huggingface.co/datasets/bdsaglam/musique)
- [musr](https://huggingface.co/datasets/TAUR-Lab/MuSR) (eval)
- [nanojev](https://huggingface.co/datasets/C-Tianyu/NanoJev-Data)
- [newsgroups20](https://huggingface.co/datasets/SetFit/20_newsgroups)
- [ocnli](https://huggingface.co/datasets/clue/clue)
- [onejev](https://huggingface.co/datasets/OmniJev/OneJev-Data)
- [openbookqa](https://huggingface.co/datasets/allenai/openbookqa)
- [patents](https://huggingface.co/datasets/ccdv/patent-classification)
- [paws](https://huggingface.co/datasets/google-research-datasets/paws)
- [pawsx_zh](https://huggingface.co/datasets/C-MTEB/PAWSX)
- [phishing](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset)
- [piqa](https://huggingface.co/datasets/baber/piqa)
- [pku_saferlhf](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF)
- [pngwn_system_one](https://huggingface.co/datasets/pngwn/system-one-decisions)
- [pngwn_typed_v2](https://huggingface.co/datasets/pngwn/typed-decisions-v2-system-one)
- [ppe_ifeval](https://huggingface.co/datasets/lmarena-ai/PPE-IFEval-Best-of-K)
- [prompt_injection](https://huggingface.co/datasets/deepset/prompt-injections)
- [pubmedqa](https://huggingface.co/datasets/qiaojin/PubMedQA) (eval)
- [qasc](https://huggingface.co/datasets/allenai/qasc)
- [qasper](https://huggingface.co/datasets/allenai/qasper-yesno)
- [qbqtc](https://huggingface.co/datasets/C-MTEB/QBQTC)
- [qnli](https://huggingface.co/datasets/nyu-mll/glue)
- [qqp](https://huggingface.co/datasets/nyu-mll/glue)
- [quality](https://huggingface.co/datasets/emozilla/quality)
- [race](https://huggingface.co/datasets/ehovy/race)
- [ragtruth](https://huggingface.co/datasets/wandb/RAGTruth-processed)
- [reclor](https://huggingface.co/datasets/tasksource/reclor)
- [reward_bench](https://huggingface.co/datasets/allenai/reward-bench)
- [reward_bench2](https://huggingface.co/datasets/allenai/reward-bench-2)
- [rotten_tomatoes](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes)
- [rte](https://huggingface.co/datasets/nyu-mll/glue)
- [safeguard_injection](https://huggingface.co/datasets/xTRam1/safe-guard-prompt-injection)
- [safety_prompts_zh](https://huggingface.co/datasets/thu-coai/Safety-Prompts)
- [salad](https://huggingface.co/datasets/OpenSafetyLab/Salad-Data)
- [scienceqa_img](https://huggingface.co/datasets/derek-thomas/ScienceQA)
- [sciq](https://huggingface.co/datasets/allenai/sciq)
- [scitail](https://huggingface.co/datasets/allenai/scitail)
- [sharc](https://huggingface.co/datasets/tasksource/sharc)
- [shp](https://huggingface.co/datasets/stanfordnlp/SHP)
- [skywork_pref](https://huggingface.co/datasets/Skywork/Skywork-Reward-Preference-80K-v0.2)
- [sms_spam](https://huggingface.co/datasets/ucirvine/sms_spam)
- [snli](https://huggingface.co/datasets/stanfordnlp/snli)
- [social_iqa](https://huggingface.co/datasets/allenai/social_i_qa)
- [sst2](https://huggingface.co/datasets/stanfordnlp/sst2)
- [sst5](https://huggingface.co/datasets/SetFit/sst5)
- [strategyqa](https://huggingface.co/datasets/ChilleD/StrategyQA)
- [stsb](https://huggingface.co/datasets/sentence-transformers/stsb)
- [subjectivity](https://huggingface.co/datasets/SetFit/subj)
- [support_tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)
- [t2_rerank](https://huggingface.co/datasets/C-MTEB/T2Reranking)
- [tabfact](https://huggingface.co/datasets/table-benchmark/tabfact)
- [tatqa](https://huggingface.co/datasets/next-tat/TAT-QA)
- [tc260](https://huggingface.co/datasets/BBBBBBBBBBBQ/TC260-Chinese-Safety-Prompts)
- [text_correction_zh](https://huggingface.co/datasets/shibing624/chinese_text_correction)
- [this_that_complex](https://huggingface.co/datasets/limberc/this-that-complex-decisions) (eval)
- [this_that_spatial](https://huggingface.co/datasets/limberc/this-that-spatial-bench) (eval)
- [thucnews](https://huggingface.co/datasets/Tongjilibo/THUCNews)
- [timeqa](https://huggingface.co/datasets/hugosousa/TimeQA)
- [tnews](https://huggingface.co/datasets/clue/clue)
- [toxic_chat](https://huggingface.co/datasets/lmsys/toxic-chat)
- [toxicn](https://huggingface.co/datasets/JunyuLu/ToxiCN)
- [trec](https://huggingface.co/datasets/CogComp/trec)
- [truthfulqa](https://huggingface.co/datasets/truthfulqa/truthful_qa) (eval)
- [tweet_emotion](https://huggingface.co/datasets/cardiffnlp/tweet_eval)
- [tweet_hate](https://huggingface.co/datasets/cardiffnlp/tweet_eval)
- [tweet_irony](https://huggingface.co/datasets/cardiffnlp/tweet_eval)
- [tweet_offensive](https://huggingface.co/datasets/cardiffnlp/tweet_eval)
- [tweet_sentiment](https://huggingface.co/datasets/cardiffnlp/tweet_eval)
- [typed_decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
- [typed_decisions_hf_test](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) (eval)
- [ultrafeedback](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)
- [ultrafeedback_zh](https://huggingface.co/datasets/opencsg/UltraFeedback-chinese)
- [unfair_tos](https://huggingface.co/datasets/coastalcph/lex_glue)
- [weibo_emotion](https://huggingface.co/datasets/souljoy/COVID-19_weibo_emotion)
- [weibo_senti](https://huggingface.co/datasets/dirtycomputer/weibo_senti_100k)
- [wic](https://huggingface.co/datasets/aps/super_glue)
- [wice](https://huggingface.co/datasets/tasksource/wice)
- [wiki2mh](https://huggingface.co/datasets/framolfese/2WikiMultihopQA)
- [wiki_qa](https://huggingface.co/datasets/microsoft/wiki_qa)
- [wikihop](https://huggingface.co/datasets/QAngaroo/wiki_hop)
- [wildjailbreak](https://huggingface.co/datasets/allenai/wildjailbreak)
- [winogrande](https://huggingface.co/datasets/allenai/winogrande)
- [xd_violence](https://huggingface.co/datasets/jherng/xd-violence)
- [xd_violence_type](https://huggingface.co/datasets/jherng/xd-violence)
- [yahoo_topics](https://huggingface.co/datasets/community-datasets/yahoo_answers_topics)
- [yelp](https://huggingface.co/datasets/Yelp/yelp_review_full)
- [zh_jdreview](https://huggingface.co/datasets/C-MTEB/JDReview-classification)
- [zh_sentiment3](https://huggingface.co/datasets/C-MTEB/MultilingualSentiment-classification)
- [zh_shopping](https://huggingface.co/datasets/C-MTEB/OnlineShopping-classification)
- [zh_stsb](https://huggingface.co/datasets/C-MTEB/STSB)
- [zh_waimai](https://huggingface.co/datasets/C-MTEB/Waimai-classification)
- [zhihu_rlhf](https://huggingface.co/datasets/liyucheng/zhihu_rlhf_3k)

</details>

<details><summary>Agent (23)</summary>

- [aguvis_aitw](https://huggingface.co/datasets/xlangai/aguvis-stage2)
- [aguvis_amex](https://huggingface.co/datasets/xlangai/aguvis-stage2)
- [aguvis_android_control](https://huggingface.co/datasets/xlangai/aguvis-stage2)
- [aguvis_coat](https://huggingface.co/datasets/xlangai/aguvis-stage2)
- [aguvis_gui_odyssey](https://huggingface.co/datasets/xlangai/aguvis-stage2)
- [aguvis_guide](https://huggingface.co/datasets/xlangai/aguvis-stage2)
- [aguvis_miniwob](https://huggingface.co/datasets/xlangai/aguvis-stage2)
- [bfcl](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) (eval)
- [cagui](https://huggingface.co/datasets/openbmb/CAGUI) (eval)
- [glaive_toolcall_zh](https://huggingface.co/datasets/llamafactory/glaive_toolcall_zh)
- [glaive_tools](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)
- [guiact_smartphone](https://huggingface.co/datasets/yiye2023/GUIAct)
- [guiact_web_multi](https://huggingface.co/datasets/yiye2023/GUIAct)
- [guiact_web_single](https://huggingface.co/datasets/yiye2023/GUIAct)
- [hermes_tools](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1)
- [mind2web](https://huggingface.co/datasets/osunlp/Mind2Web)
- [mm_mind2web](https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web)
- [omniact](https://huggingface.co/datasets/Writer/omniact)
- [s1_mini](https://huggingface.co/datasets/DavidHatley/system-one-mini-data)
- [swe_agent](https://huggingface.co/datasets/nebius/SWE-agent-trajectories)
- [toolace](https://huggingface.co/datasets/Team-ACE/ToolACE)
- [weblinx](https://huggingface.co/datasets/McGill-NLP/WebLINX)
- [when2call](https://huggingface.co/datasets/nvidia/When2Call) (eval)

</details>

<details><summary>Audio and video (88)</summary>

- [activitynet_captions](https://huggingface.co/datasets/friedrichor/ActivityNet_Captions)
- [activitynet_qa](https://huggingface.co/datasets/lmms-eval/ActivityNetQA)
- [aishell1_gender](https://huggingface.co/datasets/AISHELL/AISHELL-1)
- [ami_gender](https://huggingface.co/datasets/edinburghcstr/ami)
- [ami_same_speaker](https://huggingface.co/datasets/edinburghcstr/ami)
- [audioset](https://huggingface.co/datasets/agkphysics/AudioSet)
- [av_odyssey](https://huggingface.co/datasets/AV-Odyssey/AV_Odyssey_Bench) (eval)
- [av_speakerbench](https://huggingface.co/datasets/plnguyen2908/AV-SpeakerBench) (eval)
- [ave](https://huggingface.co/datasets/UnFaZeD07/AVE-Dataset)
- [ave_match](https://huggingface.co/datasets/UnFaZeD07/AVE-Dataset)
- [avqa](https://huggingface.co/datasets/juyil/AVQA-videos)
- [avsbench](https://huggingface.co/datasets/UnFaZeD07/AVSBench)
- [charades](https://prior.allenai.org/projects/charades)
- [charades_sta](https://huggingface.co/datasets/VideoSearchR1/charades-stage1_data)
- [chsims](https://huggingface.co/datasets/tamb2203579/CH-SIMS)
- [chsims2](https://huggingface.co/datasets/tamb2203579/CH-SIMSv2)
- [chsims_nonverbal](https://huggingface.co/datasets/tamb2203579/CH-SIMS)
- [clevrer_mc](https://huggingface.co/datasets/ngqtrung/video-r1-clevrer-mc-v1)
- [cmu_mosei](https://huggingface.co/datasets/tamb2203579/CMU-MOSEI)
- [coin](https://coin-dataset.github.io)
- [crema_d](https://huggingface.co/datasets/confit/cremad-parquet)
- [crema_d_video](https://huggingface.co/datasets/NoahMartinezXiang/2014_CREMA-D)
- [daily_omni](https://huggingface.co/datasets/liarliar/Daily-Omni) (eval)
- [egoschema](https://huggingface.co/datasets/lmms-eval/egoschema) (eval)
- [emodb](https://huggingface.co/datasets/confit/emodb-parquet)
- [epic_kitchens](https://huggingface.co/datasets/lightly-ai/epic-kitchens-100-clips)
- [esc50](https://huggingface.co/datasets/ashraq/esc50)
- [esd](https://huggingface.co/datasets/AbstractTTS/ESD_english)
- [fleurs_langid](https://huggingface.co/datasets/mteb/fleurs)
- [fsd50k](https://huggingface.co/datasets/philgzl/fsd50k)
- [genvidbench](https://huggingface.co/datasets/jian-0/GenVidBench)
- [genvideo](https://huggingface.co/datasets/sshaiton/GenVideo-50k-shard)
- [gui_world](https://huggingface.co/datasets/ONE-Lab/GUI-World)
- [gui_world_env](https://huggingface.co/datasets/ONE-Lab/GUI-World)
- [gui_world_goal](https://huggingface.co/datasets/ONE-Lab/GUI-World)
- [hmdb51](https://huggingface.co/datasets/divm/hmdb51)
- [kinetics400](https://huggingface.co/datasets/kiyoonkim/kinetics-400-targz)
- [longvideobench](https://huggingface.co/datasets/Jialuo21/LongVideoBench) (eval)
- [lrs3_transcript](https://huggingface.co/datasets/mattymchen/lrs3-test)
- [lsvq](https://huggingface.co/datasets/teowu/LSVQ-videos)
- [m4_vitevqa](https://huggingface.co/datasets/yankie123/tea-m4vitevqa)
- [meld](https://huggingface.co/datasets/THUIAR/MMLA-Datasets)
- [minds14](https://huggingface.co/datasets/PolyAI/minds14)
- [mintrec](https://huggingface.co/datasets/THUIAR/MMLA-Datasets)
- [mmar](https://huggingface.co/datasets/BoJack/MMAR) (eval)
- [mmau_mini](https://huggingface.co/datasets/gamma-lab-umd/MMAU-test-mini) (eval)
- [mmsu](https://huggingface.co/datasets/ddwang2000/MMSU) (eval)
- [msrvtt](https://huggingface.co/datasets/VLM2Vec/MSR-VTT)
- [msrvtt_qa](https://huggingface.co/datasets/Rohollah903/msrvtt_qa)
- [msvd_qa](https://huggingface.co/datasets/Rohollah903/msvd_qa)
- [music_avqa](https://huggingface.co/datasets/UnFaZeD07/Music-AVQA)
- [mustard](https://huggingface.co/datasets/THUIAR/MMLA-Datasets)
- [mvbench](https://huggingface.co/datasets/OpenGVLab/MVBench) (eval)
- [nexar](https://huggingface.co/datasets/akhil0790/nexar_collision_prediction)
- [nextqa](https://huggingface.co/datasets/lmms-eval/NExTQA)
- [omnibench](https://huggingface.co/datasets/m-a-p/OmniBench) (eval)
- [perception_test](https://huggingface.co/datasets/advaitgupta/perception_test_mcq) (eval)
- [perceptiontest_val](https://huggingface.co/datasets/lmms-eval/PerceptionTest_Val) (eval)
- [qvhighlights](https://huggingface.co/datasets/ayushsdev/qvhighlights-videos)
- [ravdess](https://huggingface.co/datasets/confit/ravdess-parquet)
- [savee](https://huggingface.co/datasets/AbstractTTS/SAVEE)
- [slurp](https://huggingface.co/datasets/marcel-gohsen/slurp)
- [speech_commands](https://huggingface.co/datasets/pollen-robotics/speech-commands-v0.02)
- [speechocean762_accuracy](https://huggingface.co/datasets/mispeech/speechocean762)
- [speechocean762_fluency](https://huggingface.co/datasets/mispeech/speechocean762)
- [speechocean762_prosodic](https://huggingface.co/datasets/mispeech/speechocean762)
- [speechocean762_total](https://huggingface.co/datasets/mispeech/speechocean762)
- [ssv2](https://huggingface.co/datasets/morpheushoc/something-something-v2)
- [star](https://huggingface.co/datasets/tdat1465/star-charades)
- [tempcompass](https://huggingface.co/datasets/lmms-eval/TempCompass) (eval)
- [tess](https://huggingface.co/datasets/AbstractTTS/TESS)
- [tgif_qa](https://huggingface.co/datasets/Xiaodong/TGIF_Zero_Shot_QA_)
- [ucf101](https://huggingface.co/datasets/quchenyuan/UCF101-ZIP)
- [urbansound8k](https://huggingface.co/datasets/danavery/urbansound8K)
- [urfunny](https://huggingface.co/datasets/THUIAR/MMLA-Datasets)
- [vatex](https://huggingface.co/datasets/Blazewild/Vatex_reencoded)
- [vatex_zh](https://huggingface.co/datasets/lmms-eval/VATEX_ZH)
- [vggsound](https://huggingface.co/datasets/11hu83/vggsound)
- [video_mme](https://huggingface.co/datasets/lmms-lab/Video-MME) (eval)
- [video_mme_sub](https://huggingface.co/datasets/lmms-lab/Video-MME) (eval)
- [videogui_goal](https://huggingface.co/datasets/VideoGUI/VideoGUI-High-Plan)
- [videogui_plan](https://huggingface.co/datasets/VideoGUI/VideoGUI-High-Plan)
- [voicebench_mmsu](https://huggingface.co/datasets/hlt-lab/voicebench) (eval)
- [voicebench_openbookqa](https://huggingface.co/datasets/hlt-lab/voicebench) (eval)
- [voxconverse_overlap](https://huggingface.co/datasets/diarizers-community/voxconverse)
- [voxconverse_speakers](https://huggingface.co/datasets/diarizers-community/voxconverse)
- [worldsense](https://huggingface.co/datasets/lmms-lab/WorldSense) (eval)
- [youcook2](https://huggingface.co/datasets/VLM2Vec/YouCook2)

</details>

Benchmark suites include `intern-accuracy-v1`, `core`, `gui-v1`, `zh-bench`, `av-omni` (WorldSense, Daily-Omni, OmniBench, AV-Odyssey, AV-SpeakerBench), `audio-bench`, and `video-bench`. `jt bench list` shows the rest.

`jt data prepare NAME --cap N` writes `$JEVTRAINER_CACHE`. `load` looks for `split-cap<cap>-v<version>.jsonl`. When that file is missing, it reuses the newest older cache of the same split. The training default cap is 100000. Prepare the file a run will actually use.

Adding a dataset: [`docs/adding_dataset.md`](docs/adding_dataset.md).

## Requirement

Install a CUDA build of PyTorch that matches the GPU, then this library. `pip install -e .` installs every row below except PyTorch itself. Versions are the minimum in [`pyproject.toml`](pyproject.toml).

| Package | Minimum | What it is for |
| --- | --- | --- |
| Python | 3.10 | |
| torch | 2.4 | the training step. Take the CUDA wheel from [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) |
| torchvision | installed with that torch | image tensors |
| transformers | 5.5 | loading the base model |
| peft | 0.18 | LoRA adapters |
| accelerate | 1.10 | the training loop, mixed precision, gradient accumulation |
| datasets | 4.0 | downloading an upstream set when a converter runs |
| safetensors | 0.4 | saving the readout and a LoRA adapter |
| pillow | 10 | reading images |
| numpy | 1.26 | audio samples |
| pydantic | 2.7 | checking the YAML |
| PyYAML | 6 | reading the YAML |
| typer | 0.12 | the `jt` command |

```bash
pip install torch torchvision
pip install -e .
```

Audio and video need three more packages. Install them when a dataset or the base model uses sound or frames.

| Package | What it is for |
| --- | --- |
| ffmpeg on `PATH`, or the `imageio-ffmpeg` package | a converter cuts frames and audio while it builds the cache |
| PyAV (`av`) | training and eval decode a video file into frames |
| `soundfile` | training and eval read the cached `.flac` |

```bash
pip install soundfile av imageio-ffmpeg
```

Resampling uses `librosa` when it is installed, and torch otherwise.

These packages are only needed when the YAML turns them on.

| Package | Install | Turned on by |
| --- | --- | --- |
| bitsandbytes | `pip install bitsandbytes` | `quantize: 4bit` or `8bit`, or `optim: adamw_8bit` |
| flash-linear-attention | `pip install -e ".[fast]"` | models whose attention can use it |
| fastapi, uvicorn | `pip install -e ".[serve]"` | `jt serve` |
| tensorboard or wandb | `pip install tensorboard` or `pip install wandb` | `report_to: tensorboard` or `report_to: wandb` |
| pytest | `pip install -e ".[dev]"` | `pytest` |

### GPU

One GPU, bf16. Memory from runs in this repo:

| Run | Machine |
| --- | --- |
| Qwen3.5-0.8B, full fine-tune | one RTX 3090 (48 GB) |
| Qwen3.5-2B, LoRA | one GPU, about 24 GB in use |
| Qwen2.5-Omni-3B, LoRA | one RTX 3090 (48 GB) |

`quantize: 4bit` is how a model that does not fit is loaded. Qwen3.5-9B and Gemma 4 12B completed a short forward and backward, not a full training run, so this table has no memory number for them.

## Getting started

### Installation

```bash
pip install torch torchvision   # CUDA wheel: https://pytorch.org/get-started/locally/
pip install -e .
```

Audio and video:

```bash
pip install soundfile av imageio-ffmpeg
```

`jt serve` needs `pip install -e ".[serve]"`.

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

A readout turns the model's hidden states into a probability for each option. The built-ins are `marker`, `slot`, and `pointer`. A new one is a small class: [`docs/adding_readout.md`](docs/adding_readout.md).

People changing the library install the test runner and the HTTP server, then run the tests. Training only needs `pip install -e .`.

```bash
pip install -e ".[dev,serve]"
pytest
```

## License

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
