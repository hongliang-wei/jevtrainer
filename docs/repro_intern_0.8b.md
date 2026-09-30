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
v4's long-document sources train on their train splits only; their validation splits form `longdoc-dev`
(checkpoint selection) and their test splits are never used. The `gui-v1` suite (GUI datasets' test / valid
splits; Mind2Web's are unseen tasks, websites and domains) is evaluated after training; OneJev's Mind2Web
rows come from the train split.

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
jailbreak_classification; none from the new sources); 431,929 train + 8,814 holdout, 6,749 steps at batch 16 x 4.

v3 trains for one epoch instead of two: v1's Hard peaked around 0.8 epoch and degraded after, and two
epochs of v3 would take 75-90 h.

Images: Qwen3.5's processor allows up to 16M pixels per image; one vision token covers 32x32 pixels.
Cached images give <= 2,209 tokens each, but OneJev's 4-screenshot trajectories reach 3,520 vision
tokens, and 40% of them exceeded `max_length` 6144 and were silently skipped. The readout option
`image_pixel_budget: 2621440` shrinks a record's images together to at most 2,560 tokens in total;
single images and 16-frame clips are unchanged, 32-frame clips go from 112 to 80 tokens per frame.
Length-grouped batching now counts about 1,000 characters per image (at most 8 images) so image records are
batched together. Worst-case batches (16 rows of 5.9k tokens, 165k vision patches) peak at 30 GiB,
so batch 16 is kept.

## Why Hard is low

Per-item predictions for v1 (zero-shot, step-1000, step-2000, step-5000, final) were dumped with batch 2,
with and without state truncation. Batch size moves 1-2 items against the suite numbers above
(zero-shot 40.5 vs 39.6, step-2000 42.3 vs 41.4, final 36.9 vs 36.0).

**What Hard is.** 111 single-decision rows from two authoring pipelines ("opus" 54, "sol" 57):
choice 67 / noul 38 / score 6, 2-6 options, mean chance 33.6%. Families: long_policy 19, multi_hop 18,
judge_hard 17, temporal_numeric 15, probability 10, trap 8, ambiguous 7, tradeoff 6, adversarial 6,
routing_hard 5. Items are rule application over insurance / HR / billing documents, date and money
arithmetic, "does this response satisfy every requirement", and states with planted misleading notes or
prompt injections. States reach 3.7k tokens: 41 rows are over 1k tokens, 8 over 3,072. The public file
has no `gold_probs` (Intern's evaluator expects 10 from private provenance), so Hard TVD cannot be computed here.

**Findings.**

* Hard is near chance for every checkpoint: accuracy 36-42 vs chance 33.6 (skill 0.04-0.13; Intern 52.25,
  skill 0.28). With n = 111 the standard error is about 4.6 points. Zero-shot to final gains 19 items and
  loses 23; step-2000 to final gains 6 and loses 12 (sign test p = 0.24). Fine-tuning reshuffles which
  items are right rather than teaching the task; 37 items are wrong at every checkpoint
  (long_policy 10, temporal_numeric 7, multi_hop 7).
* Long, rule-heavy items get worse with training. Accuracy on 1k-3k-token states: zero-shot 42,
  step-2000 39, final 27 (n = 33); "opus" items (long policy documents): 31 / 39 / 26, below their 32% chance;
  long_policy 32 -> 21, multi_hop 39 -> 22. Every v1 source has median state length <= 341 tokens and
  essentially no state over 1k tokens (400-row samples per source), so long-document reading is never trained.
* Truncation is not the cause: raising `max_state_tokens` to 16k changes 0-3 items per checkpoint.
* Fine-tuning adds answer biases. noul: gold "yes" 17/38; predicted "yes" zero-shot 16, step-2000 32, final 28.
  The most confident errors of final are "yes" on items whose answer is "no" (judge_hard, trap, eligibility
  checks; conf 0.82-0.96), i.e. the surface-plausible answer. The training noul prior is not yes-heavy
  (typed_decisions 49%, boolq 59%, yelp 38%, sms_spam 16% yes), so this is learned surface matching, not a
  label prior. Choice/score rows with >= 3 options: option A predicted 26 times vs 16 gold (zero-shot 17).
* Overconfidence without discrimination. Mean max-probability on wrong answers: zero-shot 0.52, step-5000 0.62
  (12 wrong answers above 0.9), final 0.54 at T = 1.477. Hard ECE for final is 0.29 / 0.21 / 0.14 at
  T = 1 / 1.477 / 2.748 (diagnostic only; zero-shot 0.13 / 0.13 / 0.05). The 2% holdout is easy,
  in-distribution classification, so its temperature is far below what Hard needs; with accuracy near chance,
  ECE is low only when predictions are nearly flat.
* No analogue in the mix. v1 is 164.6k records of short classification / MCQ; the only Jev-style source is
  typed_decisions (2,700 records, 1.6%, the only soft targets). Nothing trains multi-clause policy application,
  date arithmetic, instruction-following judgments or trusted-vs-untrusted evidence. Easy / Original /
  ToolACE / AG News improve because they match that mix; Hard does not.

| family | n | chance | zero-shot | step-2000 | final |
|---|---:|---:|---:|---:|---:|
| long_policy | 19 | 30 | 32 | 21 | 21 |
| multi_hop | 18 | 25 | 39 | 39 | 22 |
| judge_hard | 17 | 50 | 41 | 41 | 41 |
| temporal_numeric | 15 | 31 | 27 | 40 | 27 |
| probability | 10 | 38 | 70 | 70 | 60 |
| trap | 8 | 38 | 50 | 50 | 62 |
| ambiguous | 7 | 32 | 14 | 14 | 14 |
| tradeoff | 6 | 31 | 17 | 67 | 17 |
| adversarial | 6 | 33 | 67 | 67 | 67 |
| routing_hard | 5 | 20 | 80 | 60 | 100 |

**Fixes, by expected impact** (none tunes on Hard):

1. Long, rule-application training data with soft labels: policy / contract / billing documents of 1-4k
   tokens with multi-clause decisions, date and amount arithmetic, planted distractors and injections.
   v3 covers this only partly: jebadiah_synth and nanojev carry soft targets and mojev_mix has about 21% of
   states over 1k tokens, but kev_suites and jebadiah_synth are mostly short (median 62 / 286 tokens).
2. Select checkpoints on a non-test dev set that resembles Hard (e.g. a held-out slice of long kev_suites /
   mojev_mix rows), not on the last step; v1's Hard peaked at step 2000 of 5,042.
3. Fit temperature on a harder calibration set (kev_suites calibration split or a long-row holdout) instead
   of the easy 2% holdout.
4. Less drift from the base model: lower lr, one epoch, or down-weighting of the easy classification
   sources; zero-shot already scores 40.5 on Hard.
5. Soft-target or Brier loss and label smoothing on hard-label sources, to limit sharpening toward
   surface-plausible answers.
6. Minor: raise `max_state_tokens` to cover the longest Hard state (3.7k tokens); affects at most 3 items.

### v4: long-document decisions

`configs/repro/intern_0.8b_v4.yaml` = v3 + 30 long-document sources (`data/converters/longdoc.py`,
`longdoc2.py`, train splits only), 1 epoch, `max_state_tokens` 4096. No standalone GUI datasets.
States over 14k characters (~3.5k tokens) are skipped rather than cut, except patents (topic survives
truncation to 12k characters) and QASPER (evidence paragraphs always kept).

| group | sources | records |
|---|---|---:|
| v3 | text 184,597; Jev-format 151,439; multimodal 104,707 | 440,743 |
| rules / legal | ShARC 6,000, LegalBench rule tasks 2,443, consumer contracts 316, CUAD 3,000, MAUD 4,000, ECtHR 7,040, LEDGAR 3,000 | 25,799 |
| reading / multi-hop / time | QuALITY 158 (79 x2), MuSiQue 10,000, HotpotQA 6,000, 2Wiki 4,000, WikiHop 5,000, TimeQA 6,000, QASPER 638 (x2), ReClor 4,638 | 36,434 |
| numbers in reports / tables | TAT-QA 2,192, FinQA 5,965, DROP 5,000, TabFact 5,000 | 18,157 |
| verification / faithfulness | DocNLI 6,000, WiCE 3,097, HaluEval summaries 4,000 | 13,097 |
| judging / agent outcomes | PPE-IFEval 1,576, RewardBench 2,350, RewardBench 2 1,305, MT-Bench human (soft) 1,933, SWE-agent patches 6,000 | 13,164 |
| triage / classification | phishing 4,000, patents 3,000, hyperpartisan 495 | 7,495 |

554,889 records after overlap removal (615, all from v3's sources); 543,792 train + 11,097 holdout,
8,497 steps at batch 16 x 4. Long-doc is 20.6% of records; most of its states are 1-3.5k tokens.
The trainer fits temperature only on the holdout; `longdoc-dev` (20 validation splits, <= 300 rows each)
is evaluated per checkpoint after training for checkpoint selection and as a harder calibration check.
