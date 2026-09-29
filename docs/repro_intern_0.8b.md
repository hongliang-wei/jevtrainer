# Reproducing Intern-Decision-0.8B with open data

Goal: the same model shape as Intern-Decision-0.8B (marker readout, full fine-tuning of
Qwen/Qwen3.5-0.8B, vision tower frozen, full-vocabulary cross-entropy at each placeholder) and the
same evaluation (Intern's `benchmarks/accuracy-v1`, 7 suites), trained only on public data.
Intern-Decision did not release its training data, so this is "same architecture, same test, own
data"; numbers are comparable on the test side only.

## Target

Intern-Decision README, reported probabilities (T = 2.7478, no refit):

| Easy | Original | Hard | Typed Decision | ToolACE | AG News | WildJailBreak | Average | Hard Brier | Hard ECE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 97.92 | 80.56 | 52.25 | 77.35 | 94.52 | 88.61 | 64.48 | 79.38 | 0.5295 | 0.0657 |

Stop when the 7-suite average reaches 95% of that (75.4) with Hard ECE <= 0.08, or after three
recipe rounds without improvement.

## Evaluation protocol

* `jt eval configs/eval/intern_accuracy_v1.yaml --set checkpoint=RUN` (suite `intern-accuracy-v1`).
* Decision-level accuracy; Typed Decision counts its 2,000 decisions, not 400 rows.
* ECE: 10 equal-width bins over max(P); Brier: multiclass sum of squares; both at the fitted temperature.
* Decisions that cannot be read (too long, too many options) count as wrong (`coverage`).
* Temperature is fitted on the training mixture's 2% holdout, never on a test suite.

## Leakage control

Training records are dropped when their normalised state, or any long text field inside it, equals
one in the evaluated suites (`exclude_eval_overlap: true`, reported in `data_report.json`). All seven
suites and `typed_decisions_hf_test` are `eval_only`. JevBench has no training split and is never trained on.

Differences from Intern's inputs that cannot be closed with open data:

* WildJailbreak is gated; public jailbreak and toxicity sets stand in for it.
* Intern's own mixture, teacher labels and calibration set are private.

## Runs

| run | recipe | Easy | Original | Hard | Typed | ToolACE | AG News | WJB | Avg | Hard ECE | T |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zero-shot | none | 97.92 | 62.50 | 39.64 | 41.40 | 77.10 | 80.67 | 62.04 | 65.90 | 0.126 | 1.0 |
