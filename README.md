# jevtrainer

Train Jev-like **typed decision models** from one YAML file. A typed decision model reads a
state (text, JSON, optional images) and a set of questions whose answers are fixed in advance,
and returns a probability for every option in one forward pass, without generating text:

| type | caller gives | model returns |
|---|---|---|
| `choice` | 2–255 named options | chosen key, all probabilities, confidence |
| `score` | 2–10 ordered levels | expected level, distribution, confidence |
| `noul` | a yes/no proposition | P(yes) |

Three readouts (how the probabilities are read from a language model) are built in, and each
works with LoRA or full fine-tuning, text or images, and any Hugging Face causal LM:

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
everything into `$JEVTRAINER_CACHE`. Datasets marked `eval_only` can never enter a training mixture,
and training records whose state (or any long text field of it) matches an evaluation record are
dropped automatically; `--dry-run` reports how many.

Built in: intent (banking77, clinc150), topic (agnews), sentiment (sst5, yelp), spam, NLI (mnli,
anli, snli), reading (boolq), knowledge (arc, openbookqa, csqa, mmlu_aux; mmlu, mmlu_pro eval-only),
commonsense (hellaswag, winogrande), math (gsm8k_mc), typed workflows (typed_decisions), tools
(toolace, glaive_tools), safety (jailbreak_classification, toxic_chat, wildjailbreak if you have
access), agents (mind2web), vision (scienceqa_img, cauldron_ai2d/aokvqa/scienceqa/iconqa/tqa).

## Benchmarks

`jt bench list`. Suites: `intern-accuracy-v1` (JevBench easy/original/hard, typed decision,
ToolACE, AG News, WildJailbreak; fetched by `jt data fetch intern`), `intern-calibration`
(known-distribution pilot, scored by TVD), `core` (16 public benchmarks), `vision`. Every result
reports accuracy, chance-corrected skill, ECE (10 bins, max-probability confidence), multiclass
Brier and NLL, plus macro-F1 or case-exact where the benchmark defines it.

## Models

`jt model list`, `jt model inspect MODEL`. A `ModelFamily` tells readouts how to load a model, where
its backbone and LM head are, which linear layers get LoRA, and which modules are the vision tower.
`GenericFamily` infers all of this, so most new checkpoints need no code. Tested: Qwen3.5 (text and
VL, hybrid linear attention), Qwen3, Llama-architecture, ModernBERT (slot and pointer only).

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
jevtrainer/data/            DatasetSpec, augmentation, mixture, converters/
jevtrainer/readouts/        base.py + marker.py, slot.py, pointer.py
jevtrainer/model/           load.py (LoRA/full, save/load) + families/
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
