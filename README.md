# jevtrainer

Train Jev-like **typed decision models** from one YAML file. A typed decision model reads a
state (text, JSON, and optional images, video or audio) and a set of questions whose answers are fixed in advance,
and returns a probability for every option in one forward pass, without generating text:

| type | caller gives | model returns |
|---|---|---|
| `choice` | 2–255 named options | chosen key, all probabilities, confidence |
| `score` | 2–10 ordered levels | expected level, distribution, confidence |
| `noul` | a yes/no proposition | P(yes) |

Three readouts (how the probabilities are read from a language model) are built in, and each
works with LoRA or full fine-tuning and any Hugging Face causal LM. Text, images, video and audio
share these readouts; video and audio need a family that can encode them (Qwen2.5-Omni and
Qwen3-Omni, family `omni`):

| readout | from | prompt | read |
|---|---|---|---|
| `marker` | Intern-Decision | assistant JSON skeleton with one `<decision>` per question; all questions in one row | LM-head logits of the option symbols A, B, … just before each placeholder |
| `slot` | Bosun v3.1 | one row per question; options shuffled onto reserved `<\|decision_000\|>`… tokens | last hidden state · slot embeddings |
| `pointer` | Kev | one row per question: `<state>… <q>… (<opt>…</opt>)* <decide>` | bilinear score of `<decide>` against each `</opt>` |

## Quick start

```bash
pip install -e .            # Python >= 3.10, torch, transformers >= 5.5
jt init --readout pointer > my.yaml
jt train my.yaml --dry-run  # data counts, leakage check, trainable params, a rendered sample
jt train my.yaml
jt eval configs/eval/intern_accuracy_v1.yaml --set checkpoint=runs/pointer-lora
jt serve runs/pointer-lora  # POST /v1/systemone, compatible with the TypeSafe SDK base URL
```

The smallest config:

```yaml
model: Qwen/Qwen3.5-0.8B
readout: pointer          # marker | slot | pointer
dataset: banking77,boolq  # registry names, name:split, or ./my.jsonl
```

Everything else has a default: `finetune: lora` (or `full`), `lora.r: 16`, `lr` (1e-4 LoRA, 1e-5
full), `epochs`, `batch_size`, `grad_accum`, `augment`, `loss: ce | ce_smooth | brier | ce_brier`,
`holdout` (kept for temperature calibration), `eval_dataset`, … See
[`jevtrainer/config.py`](jevtrainer/config.py) for the full list; unknown fields are rejected with
a suggestion. Any field can be overridden: `jt train my.yaml --set lr=5e-5 --set lora.r=32`.

A run directory holds `config.yaml`, the adapter or full weights, `readout.safetensors`,
`readout.json`, `calibration.json`, `train_log.jsonl`, `metrics.json` and `eval/results.md`.

With `save_steps: N` every N steps writes `step-N/` (weights, usable by `jt eval`). The newest one
also keeps `step-N/state/` (optimizer, scheduler, RNG and data position; older copies are deleted,
turn off with `save_state: false`). After a crash, rerun the same command with `--set resume=true`:
training continues from that step with the same batches it would have seen.

## Data

Every source becomes the same record:

```json
{"id": "…", "state": {"ticket": "…"}, "images": ["path/or/url.png"],
 "questions": {"team":  {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "charges", "tech": "bugs"}},
               "urgent": {"type": "noul",   "instructions": "Is this urgent?"},
               "anger":  {"type": "score",  "instructions": "How angry?", "criteria": ["calm", "annoyed", "furious"]}},
 "targets": {"team": {"label": "billing"}, "urgent": {"label": "yes"}, "anger": {"label": "1", "probs": {"0": 0.2, "1": 0.7, "2": 0.1}}}}
```

`jt data list` shows the registered datasets, `jt data show NAME`, `jt data prepare all` converts
everything into `$JEVTRAINER_CACHE`. `load` reads
`$JEVTRAINER_CACHE/records/<name>/<split>-cap<cap>-v<version>.jsonl`. If that exact file is missing
it reuses the newest older cache of the same split and does not convert again, for text and GUI sets
as well as audio and video. Prepare the cap a run will ask for (`jt data prepare NAME --cap N`,
training's default cap is 100000) before starting it. Datasets marked `eval_only` can never enter a
training mixture, and training records whose state (or any long text field of it) matches an
evaluation record are dropped automatically; `--dry-run` reports how many.

Built in (333 datasets, `jt data list`):

* typed-decision sets: typed_decisions, kev_suites, jebadiah_synth, mojev_mix, onejev (multimodal),
  pngwn_typed_v2 / pngwn_system_one (non-commercial); eval-only this_that_complex / this_that_spatial
* intent: banking77, clinc150, massive_intent, massive_scenario, bitext_support, trec
* topic: agnews, dbpedia14, yahoo_topics, newsgroups20
* sentiment / emotion: sst2, sst5, imdb, amazon_polarity, amazon_stars, rotten_tomatoes, yelp, emotion,
  go_emotions, tweet_sentiment, tweet_emotion, tweet_irony, fin_tweets, fin_phrasebank
* safety: wildjailbreak (gated, needs HF_TOKEN), jailbreak_classification, toxic_chat, civil_comments,
  tweet_offensive, tweet_hate, hate_offensive; spam: sms_spam, enron_spam
* NLI / similarity: mnli, anli, snli, qnli, rte, cb, scitail, contract_nli, mrpc, qqp, paws, wic, stsb
* reading / QA: boolq, multirc, race, wiki_qa, strategyqa, social_iqa, piqa, copa, logiqa
* knowledge: arc, openbookqa, csqa, qasc, sciq, mmlu_aux, medmcqa, medqa; eval-only mmlu, mmlu_pro,
  truthfulqa, pubmedqa, bbh, musr, cladder, cruxeval
* commonsense / math: hellaswag, winogrande, gsm8k_mc
* tools / agents: toolace, glaive_tools, hermes_tools, mind2web; eval-only when2call
* GUI / browser / computer use (screenshots, `data/converters/gui.py`): element choice on numbered
  candidate boxes drawn on the screenshot for mm_mind2web (splits test_task / test_website /
  test_domain), guiact_web_single / guiact_web_multi / guiact_smartphone (also action type), omniact,
  weblinx (non-commercial); next-step choice and action type on AGUVIS trajectories for aguvis_aitw,
  aguvis_android_control, aguvis_amex, aguvis_guide, aguvis_coat, aguvis_miniwob, aguvis_gui_odyssey.
  Screenshots are cached as JPG with the long side at most 1280 px
* legal / RAG / retrieval: unfair_tos, case_hold, ragtruth, esci
* preference: helpsteer2 (five rubric scores), ultrafeedback, shp, hh_rlhf
* support / misc: support_tickets, liar2, bias_in_bios, cola, subjectivity
* vision: scienceqa_img, cauldron_ai2d / aokvqa / scienceqa / iconqa / tqa, onejev
* more safety / preference: aegis2, beavertails, pku_saferlhf (safer / better), skywork_pref, salad (harm
  domain + category), helpsteer3 (multi-turn, includes code and Chinese)
* practical (`data/converters/practical.py`): prompt_injection, safeguard_injection, fake_jobs, atis,
  fin_news_topic, esconv (support strategy, non-commercial)
* Chinese (tag `zh`, Chinese instructions and option texts; `data/converters/zh*.py`, `gui_zh.py`):
  * news / apps / topics: tnews, thucnews (full articles), iflytek (119 app categories), csl_discipline,
    fincuge_news
  * intent / sentiment / stance: massive_intent_zh, massive_scenario_zh, kuake_qic, zh_shopping, zh_waimai,
    zh_jdreview, zh_sentiment3, amazon_zh (stars), weibo_senti, chnsenticorp, weibo_emotion, fincuge_sentiment,
    c_stance
  * safety / detection: cold, toxicn, tc260 and safety_prompts_zh (both synthetic), hc3_zh, cvalues_rlhf
    (harmless half)
  * matching / NLI / similarity: afqmc, lcqmc, bq_corpus, atec, pawsx_zh, chip_sts, kuake_qqr, kuake_qtr,
    zh_stsb, qbqtc, cmnli, ocnli (non-commercial), csl, cluewsc
  * retrieval: t2_rerank, mmarco_rerank_zh (non-commercial), cmedqa_rerank, cmedqa1_rerank
  * reading / exams / law / medicine: c3, chid, cail2018 (charge + sentence band), legal_case_zh,
    cmexam, cmb; tools: glaive_toolcall_zh
  * proofreading / dialogue: csc (spelling errors), text_correction_zh (law, medical, official documents),
    cdconv (chatbot contradiction type, non-commercial)
  * preference: cvalues_rlhf, zhihu_rlhf, dpo_zh, ultrafeedback_zh, dpo_pairs_zh (translated)
  * eval-only: ceval_val, cmmlu, agieval_zh, jecqa, financeiq, chinese_safetyqa, mmbench_cn, cmmmu,
    cagui (Chinese Android GUI steps: action type + numbered element)
* audio (`data/converters/audio_*.py`, `av_meeting.py`): emotion ravdess, savee, crema_d, tess, esd,
  emodb; events esc50, urbansound8k, fsd50k, audioset; speech slurp, minds14, speech_commands,
  fleurs_langid, aishell1_gender; pronunciation scores speechocean762_accuracy / fluency / prosodic /
  total; meetings ami_same_speaker, ami_gender, voxconverse_speakers, voxconverse_overlap.
  Eval-only: mmau_mini, mmar, mmsu, voicebench_mmsu, voicebench_openbookqa
* video (`data/converters/video_*.py`): action ssv2, hmdb51, kinetics400, ucf101, charades; QA nextqa,
  activitynet_qa, msvd_qa, msrvtt_qa, star, clevrer_mc, tgif_qa, m4_vitevqa; time ranges charades_sta,
  qvhighlights, activitynet_captions; procedure coin, epic_kitchens, youcook2; captions msrvtt, vatex,
  vatex_zh; quality and driving lsvq, nexar; screen video gui_world / gui_world_goal / gui_world_env,
  videogui_goal, videogui_plan; synthetic genvideo, genvidbench. Eval-only: mvbench, tempcompass,
  egoschema, longvideobench, video_mme, video_mme_sub, perception_test, perceptiontest_val
* audio-video (`data/converters/av_*.py`): QA avqa, music_avqa; events vggsound, ave, ave_match,
  avsbench; affect meld, mustard, urfunny, mintrec, chsims, chsims2, chsims_nonverbal, cmu_mosei,
  crema_d_video; lip reading lrs3_transcript; violence in the clip xd_violence, xd_violence_type.
  Eval-only: worldsense, daily_omni, omnibench, av_odyssey, av_speakerbench

Clips are a `media` list on the record, referenced from the state as `<image:N>`, `<video:N>` and
`<audio:N>`. A video item is pre-extracted frames plus 16 kHz flac, or a path decoded when the record
is loaded. Items the state does not tag are placed in front of it. See [`jevtrainer/media.py`](jevtrainer/media.py).

```yaml
model: Qwen/Qwen2.5-Omni-3B
readout: marker
readout_options:
  video_fps: 1.0          # frames kept per second of the clip; default without this is video_frames: 8
  video_max_frames: 32
  audio_max_s: 30
  use_audio_in_video: true
dataset: music_avqa,ravdess
```

`readout_options` also takes `frame_max_side`, `video_audio_max_s` and `image_pixel_budget`. The
Omni-3B mix is [`configs/av/omni3b_v4.yaml`](configs/av/omni3b_v4.yaml). Conversion needs ffmpeg on
`PATH` or the `imageio-ffmpeg` wheel. `JT_VIDEO_FPS=1` writes about one frame per second (capped by
`JT_VIDEO_MAX_FRAMES`, default 32) under `media_fps1/` and a cache file ending in `-fps1`; video and
audio-video sets then load that file. `JT_VIDEO_FPS_STRICT=1` does not fall back to the 8-frame cache.

## Benchmarks

`jt bench list`. Suites: `intern-accuracy-v1` (JevBench easy/original/hard, typed decision,
ToolACE, AG News, WildJailbreak; fetched by `jt data fetch intern`), `intern-calibration`
(known-distribution pilot, scored by TVD), `core` (16 public benchmarks), `extended` (31 more:
BBH, MuSR, CLadder, CRUXEval, TruthfulQA, ContractNLI, ESCI, When2Call, RAGTruth, ...), `vision`,
`gui-v1` (held-out GUI splits: Multimodal-Mind2Web test task / website / domain, GUIAct web-single /
web-multi / smartphone test, OmniACT test, WebLINX valid; 1,000 each), `zh-dev` (validation splits
of 26 Chinese training sources), `zh-bench` (C-Eval val, CMMLU, AGIEval Chinese, JEC-QA, FinanceIQ,
Chinese-SafetyQA), `vision-zh` (MMBench-CN, CMMMU), `gui-zh` (CAGUI), `av-omni` (WorldSense,
Daily-Omni, OmniBench, AV-Odyssey, AV-SpeakerBench), `audio-bench` (MMAU-mini, MMAR, MMSU, VoiceBench),
`video-bench` (MVBench, TempCompass, EgoSchema, LongVideoBench, Video-MME, PerceptionTest).

Sources that need extra handling live in `data/converters/recovered.py`: GPQA and HLE (gated: accept
the terms on the Hub and set `HF_TOKEN`; HLE keeps its text-only multiple-choice items), BFCL (JSON
Lines under a `.json` name with per-category columns, joined with `possible_answer/`; one noul per
function), system-one-mini (question texts taken from its generator's `SCHEMA`), NanoJev
(`unified/hard/`; about 10% of its labels come from the hosted Jev and are flagged `jev_distilled`),
HoVer (evidence text from `Dzeniks/hover`, labels from the original release). Every result
reports accuracy, chance-corrected skill, ECE (10 bins, max-probability confidence), multiclass
Brier and NLL, plus macro-F1 or case-exact where the benchmark defines it.

## Models

`jt model list`, `jt model inspect MODEL`. A `ModelFamily` tells readouts how to load a model, where
its backbone and LM head are, which linear layers get LoRA, and which modules are the vision tower.
`GenericFamily` infers all of this, so most new checkpoints need no code. Tested: Qwen3.5 (text and
VL, hybrid linear attention), Qwen3, Llama-architecture, ModernBERT (slot and pointer only).
Qwen2.5-Omni and Qwen3-Omni use family `omni`: only the thinker is loaded, and a video's own
soundtrack is interleaved with its frames when `use_audio_in_video` is set.

Also wired up, with GPU runs still pending:

| base | family | notes |
|---|---|---|
| MiniCPM5-1B / 2B | llama | |
| Qwen3.5-4B / 9B, Qwen3.8-27B | qwen | 27B: `quantize: 4bit` on a 48 GB card |
| gemma-3-270m | gemma | |
| gemma-4-E4B | gemma | per-layer embeddings; text only (audio / vision towers frozen, no LoRA) |
| gemma-4-12B | gemma | `gemma4_unified` needs transformers >= 5.17 |
| gemma-4-26B-A4B-it | gemma | MoE: router excluded from LoRA; 52 GB in bf16, so `device_map: auto` + `max_memory` |
| LFM2.5-350M / 2.6B | generic | |
| Nandi-Mini-150M, Lumma-0.6B | generic | `trust_remote_code`; factorized embedding / `lm_head_proj` handled |

Memory on one 48 GB card: full fine-tuning keeps fp32 weights + AdamW states (16 bytes per parameter),
so it stops at about 2.6B; `optim: adamw_8bit` cuts that to about 10 bytes. For larger models use
LoRA, and `quantize: 4bit` (QLoRA; LoRA only) when bf16 weights do not fit. Quantization and
`optim: adamw_8bit` need `bitsandbytes`. Checkpoints trained quantized reload quantized with the
adapter unmerged.

## Extending

* a dataset: [docs/adding_dataset.md](docs/adding_dataset.md)
* a readout: [docs/adding_readout.md](docs/adding_readout.md)
* a model family: [docs/adding_model.md](docs/adding_model.md)
* a training method (e.g. RL): subclass `jevtrainer.train.trainer.Trainer`, override `loss(logits, batch)`,
  and register it in `TRAINERS`.

## Layout

```text
jevtrainer/schema.py        record format
jevtrainer/config.py        YAML configs
jevtrainer/media.py         images, video frames, audio
jevtrainer/data/            DatasetSpec, augmentation, mixture, converters/
jevtrainer/readouts/        base.py + marker.py, slot.py, pointer.py
jevtrainer/model/           load.py (LoRA/full, save/load) + families/ (omni.py for Qwen-Omni)
jevtrainer/train/           trainer, losses, calibration, batching
jevtrainer/eval/            metrics, runner, benchmarks/
jevtrainer/serve/           /v1/systemone
```

## Development

```bash
pip install -e .[dev,serve]
pytest            # tiny random models with real tokenizers; CPU is enough
scripts/smoke_text.sh && scripts/smoke_vl.sh   # Qwen3.5-0.8B on a GPU
```
