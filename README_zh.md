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

| 模型 | 日期 | 底座 | 训练配置 | 结果 |
| --- | --- | --- | --- | --- |
| [weihongliang/jevtrainer-qwen35-0.8b-2026-10-02](https://huggingface.co/weihongliang/jevtrainer-qwen35-0.8b-2026-10-02) | 2026-10-02 | Qwen3.5-0.8B，全参数，`marker` | [`configs/repro/intern_0.8b_v4.yaml`](configs/repro/intern_0.8b_v4.yaml) | Intern 平均 **82.52**（官方 79.38）。longdoc-dev 80.90。gui-v1 66.71 |
| [weihongliang/jevtrainer-omni3b-2026-10-04](https://huggingface.co/weihongliang/jevtrainer-omni3b-2026-10-04) | 2026-10-04 | Qwen2.5-Omni-3B thinker，LoRA r=32，`marker` | [`configs/av/omni3b_v4.yaml`](configs/av/omni3b_v4.yaml) | 保留集 **85.03%**（1610 条） |

每个文件里的 `dataset:` 就是训练用的数据集（名字、采样上限、分组方式），超参也在同一份文件里。

```bash
huggingface-cli download weihongliang/jevtrainer-qwen35-0.8b-2026-10-02 --local-dir runs/jev-qwen35-0.8b
jt eval configs/eval/intern_accuracy_v1.yaml --set checkpoint=runs/jev-qwen35-0.8b

huggingface-cli download weihongliang/jevtrainer-omni3b-2026-10-04 --local-dir runs/jev-omni3b
jt eval configs/eval/av_omni.yaml --set checkpoint=runs/jev-omni3b --set benchmarks=worldsense,omnibench
```

## 支持的模型

| 模型 | 尺寸 |
| --- | --- |
| [Qwen3.5](https://huggingface.co/Qwen) | 0.8B/2B/4B/9B |
| [Qwen2.5-Omni](https://huggingface.co/Qwen) | 3B |
| [Gemma 3](https://huggingface.co/google) | 270M |
| [Gemma 4](https://huggingface.co/google) | E2B/E4B/12B |
| [MiniCPM5](https://huggingface.co/openbmb) | 1B/2B |
| [LFM2.5](https://huggingface.co/LiquidAI) | 350M/2.6B |
| [Nandi-Mini](https://huggingface.co/Rta-AILabs) | 150M |
| [Lumma](https://huggingface.co/FrontiersMind) | 0.6B |

更大的模型和 4bit 加载写在 [`jevtrainer/config.py`](jevtrainer/config.py)。加一个家族：[`docs/adding_model.md`](docs/adding_model.md)。


## 训练方式

| 读出 | 怎么读 | LoRA | 全参数 |
| --- | --- | --- | --- |
| `marker` | 每个问题一个 `<decision>` 占位，读它前面选项符号的 logits | ✅ | ✅ |
| `slot` | 选项打乱到预留的 `<\|decision_000\|>` 等 token 上，用隐藏状态和 slot 向量的点积 | ✅ | ✅ |
| `pointer` | 每个选项包在 `<opt>…</opt>` 里，用 `<decide>` 和每个 `</opt>` 的双线性分数 | ✅ | ✅ |

默认值：`finetune: lora`，`lora.r: 16`，LoRA 学习率 1e-4，全参数 1e-5，以及 `epochs`、`batch_size`、`grad_accum` 和留给温度校准的 `holdout`。覆盖某一项：`jt train my.yaml --set lr=5e-5 --set lora.r=32`。

## 数据集

`jt data list` 列出全部 **333** 个数据集，分成三组。每个名字都有转换器，写进 `dataset:` 就会生成记录。标了 `(eval)` 的是 `eval_only`，不能进训练混合。训练样本的状态若和某条评测记录相同，会被丢掉，`--dry-run` 会报告丢掉多少。

图像题（一张图，没有音频也没有视频）和文字数据集放在「文字」里。GUI 动作、工具选择、智能体轨迹放在 Agent。音频、视频、音视频是第三组。

<details><summary>文字 (222)</summary>

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

<details><summary>音视频 (88)</summary>

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

评测套件包括 `intern-accuracy-v1`、`core`、`gui-v1`、`zh-bench`、`av-omni`（WorldSense、Daily-Omni、OmniBench、AV-Odyssey、AV-SpeakerBench）、`audio-bench`、`video-bench`。其余用 `jt bench list` 查看。

`jt data prepare NAME --cap N` 写入 `$JEVTRAINER_CACHE`。`load` 找的是 `split-cap<cap>-v<version>.jsonl`。这个文件不在时，会用同一次划分里最新的旧缓存。训练默认 cap 是 100000。开跑前把要用的那一份准备好。

加一个数据集：[`docs/adding_dataset.md`](docs/adding_dataset.md)。

## 环境

先装和 GPU 匹配的 CUDA 版 PyTorch，再装本库。`pip install -e .` 会装下表里除 PyTorch 以外的每一行。版本是 [`pyproject.toml`](pyproject.toml) 里的最低要求。

| 包 | 最低 | 用途 |
| --- | --- | --- |
| Python | 3.10 | |
| torch | 2.4 | 训练。到 [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) 选 CUDA 轮子 |
| torchvision | 和这份 torch 一起装 | 图像张量 |
| transformers | 5.5 | 加载基座模型 |
| peft | 0.18 | LoRA |
| accelerate | 1.10 | 训练循环、混合精度、梯度累积 |
| datasets | 4.0 | 转换器运行时下载上游数据 |
| safetensors | 0.4 | 保存 readout 和 LoRA |
| pillow | 10 | 读图像 |
| numpy | 1.26 | 音频采样 |
| pydantic | 2.7 | 检查 YAML |
| PyYAML | 6 | 读 YAML |
| typer | 0.12 | `jt` 命令 |

```bash
pip install torch torchvision
pip install -e .
```

音视频再装三个包。数据集或基座要用声音或帧时再装。

| 包 | 用途 |
| --- | --- |
| `PATH` 上的 ffmpeg，或 `imageio-ffmpeg` | 转换器建缓存时切帧、切音频 |
| PyAV（`av`） | 训练和评测把视频解成帧 |
| `soundfile` | 训练和评测读缓存里的 `.flac` |

```bash
pip install soundfile av imageio-ffmpeg
```

重采样在装了 `librosa` 时用它，否则用 torch。

下面这些只在 YAML 打开对应功能时装。

| 包 | 安装 | 何时需要 |
| --- | --- | --- |
| bitsandbytes | `pip install bitsandbytes` | `quantize: 4bit` 或 `8bit`，或 `optim: adamw_8bit` |
| flash-linear-attention | `pip install -e ".[fast]"` | 注意力实现能用到它的模型 |
| fastapi、uvicorn | `pip install -e ".[serve]"` | `jt serve` |
| tensorboard 或 wandb | `pip install tensorboard` 或 `pip install wandb` | `report_to: tensorboard` 或 `report_to: wandb` |
| pytest | `pip install -e ".[dev]"` | 跑测试 |

### GPU

一块 GPU，bf16。下面是这个仓库里实际跑过的占用：

| 训练 | 机器 |
| --- | --- |
| Qwen3.5-0.8B，全量微调 | 一块 RTX 3090（48 GB） |
| Qwen3.5-2B，LoRA | 一块 GPU，大约用掉 24 GB |
| Qwen2.5-Omni-3B，LoRA | 一块 RTX 3090（48 GB） |

装不下时用 `quantize: 4bit`。Qwen3.5-9B 和 Gemma 4 12B 只做过一次前向和反向，没有跑完整训练，所以这里不写它们的显存。

## 开始使用

### 安装

```bash
pip install torch torchvision   # CUDA 轮子：https://pytorch.org/get-started/locally/
pip install -e .
```

音视频：

```bash
pip install soundfile av imageio-ffmpeg
```

`jt serve` 需要 `pip install -e ".[serve]"`。

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

读出（readout）把模型的隐藏状态变成每个选项的概率。库里自带 `marker`、`slot`、`pointer`。新写一个是一个小类：[`docs/adding_readout.md`](docs/adding_readout.md)。

改这个库的人装测试和 HTTP 服务，再跑测试。只训练的话，`pip install -e .` 就够了。

```bash
pip install -e ".[dev,serve]"
pytest
```

## 许可证

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)。
