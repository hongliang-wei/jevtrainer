"""Prompt safety: harmful vs benign (Intern-Decision WildJailbreak wording).

allenai/wildjailbreak is gated; these public sources cover the same decision.
"""

from __future__ import annotations

from jevtrainer.data.base import DatasetSpec, choice_record, hf, register, rid, take

ASK = "Classify whether this prompt is harmful or benign."
CRIT = {
    "harmful": "The prompt requests harmful, unsafe, or policy-violating content.",
    "benign": "The prompt is benign: it does not request harmful content.",
}


def jailbreak_classification(split, cap, rng):
    ds = take(hf("jackhhao/jailbreak-classification", split=split), cap, rng)
    for i, r in enumerate(ds):
        label = "harmful" if r["type"] == "jailbreak" else "benign"
        yield choice_record(rid("jbc", split, i), {"prompt": r["prompt"]}, ASK, dict(CRIT), label, area="safety")


def toxic_chat(split, cap, rng):
    ds = take(hf("lmsys/toxic-chat", "toxicchat0124", split=split), cap, rng)
    for i, r in enumerate(ds):
        harmful = int(r["toxicity"]) == 1 or int(r["jailbreaking"]) == 1
        yield choice_record(rid("toxicchat", split, i), {"prompt": r["user_input"]}, ASK, dict(CRIT), "harmful" if harmful else "benign", area="safety")


def wildjailbreak(split, cap, rng):
    """Needs HF access to the gated repo (set HF_TOKEN)."""
    ds = take(hf("allenai/wildjailbreak", "train", split="train", delimiter="\t", keep_default_na=False), cap, rng)
    for i, r in enumerate(ds):
        prompt = r.get("adversarial") or r.get("vanilla")
        label = "harmful" if "harmful" in r["data_type"] else "benign"
        yield choice_record(rid("wjb", i), {"prompt": prompt}, ASK, dict(CRIT), label, area="safety")


register(DatasetSpec("jailbreak_classification", jailbreak_classification, ("train", "test"), "jackhhao/jailbreak-classification", "apache-2.0", "safety"))
register(DatasetSpec("toxic_chat", toxic_chat, ("train", "test"), "lmsys/toxic-chat:toxicchat0124", "cc-by-nc-4.0", "safety"))
register(DatasetSpec("wildjailbreak", wildjailbreak, ("train",), "allenai/wildjailbreak (gated)", "odc-by", "safety"))
