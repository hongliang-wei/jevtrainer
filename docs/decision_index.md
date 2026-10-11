# Decision Index

[Decision Index](https://github.com/apolinario/decision-index) is the public benchmark for typed decision engines: 37 benchmarks
in five areas, chance-corrected, unanswered counted as wrong. The kit rebuilds the suite, runs an engine and scores it.
This repo does not reimplement it: `jt serve` already speaks the `POST /v1/systemone` wire format, and the kit's `http` engine
drives any such server.

Only the public part (20% of the board's Full score) is reproducible. The number you get is the **public index**, not the board's Full score.
It is not the same as `configs/eval/intern_accuracy_v1.yaml` (Intern-Decision's own seven suites).

## Setup (once)

```bash
git clone https://github.com/apolinario/decision-index && cd decision-index
pip install -e ".[transformers,rebuild]"
export HF_HUB_DISABLE_XET=1            # accept the HLE terms on the Hub and log in first
python -m decision_index suite rebuild --work work    # about 7 GB download, 17 GB working space
python -m decision_index suite import \
    --rows work/artifacts/benchmark-suite/release-v2-rebuilt/selected-rows.jsonl.gz \
    --added-rows work/artifacts/benchmark-suite/release-v2-rebuilt/added-rows.jsonl.gz \
    --gsm8k-rows work/artifacts/benchmark-suite/release-v3-rebuilt/gsm8k-rows.jsonl.gz
```

The suite is hash-checked and not redistributable: build it once and copy the staged `suite-0.3/` between machines.

## Run

```bash
pip install -e ".[serve]"
bash scripts/decision_index.sh runs/my-run my-run 8009 --limit 100    # smoke test
bash scripts/decision_index.sh runs/my-run my-run 8009                # full suite, resumes after a crash
```

`jt serve` is started, waited for, and stopped by the script. Read `runs/decision_index/my-run/scores.json`
(`index`, `raw_index`, five areas, coverage, unsupported counts, latency).

## Server rules the kit relies on

- A request that does not fit (too long, too many options) answers HTTP 413 with the words "maximum context length ... options per choice".
  The kit records it as `unsupported` (counts as wrong, never retried). Anything else is an `error` and is retried on resume.
- `instructions` may be JSON in the suite; the server turns it into text. States may be strings or JSON.
- One forward at a time (a lock), so run one server per GPU.
