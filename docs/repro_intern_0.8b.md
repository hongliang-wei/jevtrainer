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
v4 adds the GUI datasets' train splits and also evaluates `gui-v1`, built from their test / valid splits
(Mind2Web's are unseen tasks, websites and domains); OneJev's Mind2Web rows come from the train split.

Differences from Intern's inputs that cannot be closed with open data:

* WildJailbreak is gated; public jailbreak and toxicity sets stand in for it.
* Intern's own mixture, teacher labels and calibration set are private.

## Runs

| run | recipe | Easy | Original | Hard | Typed | ToolACE | AG News | WJB | Avg | Hard ECE | T |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zero-shot | none | 97.92 | 62.50 | 39.64 | 41.40 | 77.10 | 80.67 | 62.04 | 65.90 | 0.126 | 1.0 |
| v1 | 21 text sources, 161k records, 2 epochs | 100.00 | 81.94 | 36.04 | 72.15 | 96.77 | 93.07 | 57.74 | 76.82 | 0.226 | 1.477 |

### v3: Jev-format and multimodal data

v2 (v1 + WildJailBreak) was stopped after one minute; v3 follows v1 directly and contains v2's
sources. `configs/repro/intern_0.8b_v3.yaml` adds:

| group | sources | records |
|---|---|---:|
| v2 text | v1's 21 sources + WildJailBreak (20k) | 184,597 |
| Jev-format text | kev_suites 37,288, jebadiah_synth 3,253, mojev_mix 100,000 (train, conversion cap), nanojev 10,898 | 151,439 |
| OneJev | all 94,707 rows (65,643 with images: screenshots, 3-4 image trajectories, 16/32-frame videos) | 94,707 |
| image MCQ | 10,000 sampled in proportion to size: scienceqa_img 1,562, cauldron iconqa 4,125, aokvqa 2,081, scienceqa 1,250, ai2d 609, tqa 373 | 10,000 |

440,743 records after overlap removal (615 removed, all from typed_decisions / toolace / agnews /
jailbreak_classification; none from the new sources); 431,929 train + 8,814 holdout, 13,498 steps at batch 16 x 4.

Images: Qwen3.5's processor allows up to 16M pixels per image; one vision token covers 32x32 pixels.
Cached images give <= 2,209 tokens each, but OneJev's 4-screenshot trajectories reach 3,520 vision
tokens, and 40% of them exceeded `max_length` 6144 and were silently skipped. The readout option
`image_pixel_budget: 2621440` shrinks a record's images together to at most 2,560 tokens in total;
single images and 16-frame clips are unchanged, 32-frame clips go from 112 to 80 tokens per frame.
Length-grouped batching now counts about 1,000 characters per image (at most 8 images) so image records are
batched together. Worst-case batches (16 rows of 5.9k tokens, 165k vision patches) peak at 30 GiB,
so batch 16 is kept.
