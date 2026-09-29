"""Response quality: HelpSteer2 rubric scores and pairwise preferences (UltraFeedback, SHP, HH-RLHF)."""

from __future__ import annotations

from jevtrainer.data.base import DatasetSpec, choice_record, hf, register, rid, take
from jevtrainer.schema import Question, Record, Target

HS = {
    "helpfulness": "How helpful is the response to the prompt overall?",
    "correctness": "How correct and complete are the facts in the response?",
    "coherence": "How clear and self-consistent is the response?",
    "complexity": "How much expertise does the response require to write?",
    "verbosity": "How verbose is the response relative to what was asked?",
}
LEVELS = ["0 (very low)", "1", "2", "3", "4 (very high)"]
PAIR = {"response_a": "Response A is better", "response_b": "Response B is better"}


def helpsteer2(split, cap, rng):
    ds = take(hf("nvidia/HelpSteer2", split={"test": "validation"}.get(split, split)), cap, rng)
    for i, r in enumerate(ds):
        qs = {k: Question("score", v, list(LEVELS)) for k, v in HS.items()}
        ts = {k: Target(str(int(r[k]))) for k in HS}
        yield Record(rid("helpsteer2", split, i), {"prompt": r["prompt"][:3000], "response": r["response"][:4000]}, qs, ts, meta={"area": "preference"})


def _pair_record(name, i, prompt, a, b, a_better, rng):
    if rng.random() < 0.5:
        a, b, a_better = b, a, not a_better
    state = {"prompt": prompt[:2500], "response_a": a[:3000], "response_b": b[:3000]}
    return choice_record(rid(name, i), state, "Which response is better?", dict(PAIR), "response_a" if a_better else "response_b", area="preference")


def ultrafeedback(split, cap, rng):
    ds = take(hf("HuggingFaceH4/ultrafeedback_binarized", split={"train": "train_prefs", "test": "test_prefs"}[split]), cap, rng)
    for i, r in enumerate(ds):
        if r["score_chosen"] == r["score_rejected"]:
            continue
        yield _pair_record("ultrafeedback", f"{split}{i}", r["prompt"], r["chosen"][-1]["content"], r["rejected"][-1]["content"], True, rng)


def shp(split, cap, rng):
    ds = take(hf("stanfordnlp/SHP", split=split).filter(lambda r: r["score_ratio"] >= 2.0), cap, rng)
    for i, r in enumerate(ds):
        yield _pair_record("shp", f"{split}{i}", r["history"], r["human_ref_A"], r["human_ref_B"], r["labels"] == 1, rng)


def hh_rlhf(split, cap, rng):
    ds = take(hf("Anthropic/hh-rlhf", split=split), cap, rng)
    for i, r in enumerate(ds):
        cut = r["chosen"].rfind("\n\nAssistant:")
        if cut < 0 or r["rejected"][:cut] != r["chosen"][:cut]:
            continue
        prompt = r["chosen"][:cut].strip()
        a, b = r["chosen"][cut + 12:].strip(), r["rejected"][cut + 12:].strip()
        if a and b and a != b:
            yield _pair_record("hh", f"{split}{i}", prompt, a, b, True, rng)


register(DatasetSpec("helpsteer2", helpsteer2, ("train", "test"), "nvidia/HelpSteer2", "cc-by-4.0", "preference", description="five 0-4 rubric scores per response"))
register(DatasetSpec("ultrafeedback", ultrafeedback, ("train", "test"), "HuggingFaceH4/ultrafeedback_binarized", "mit", "preference"))
register(DatasetSpec("shp", shp, ("train", "validation", "test"), "stanfordnlp/SHP (score ratio >= 2)", "see upstream", "preference"))
register(DatasetSpec("hh_rlhf", hh_rlhf, ("train", "test"), "Anthropic/hh-rlhf", "mit", "preference"))
