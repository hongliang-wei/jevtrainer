"""Entailment and reading-comprehension yes/no sources."""

from __future__ import annotations

from jevtrainer.data.base import DatasetSpec, choice_record, hf, noul_record, register, rid, take

NLI_CRIT = {
    "entailment": "The hypothesis follows from the premise.",
    "neutral": "The premise neither supports nor contradicts the hypothesis.",
    "contradiction": "The premise contradicts the hypothesis.",
}
NLI_KEYS = list(NLI_CRIT)


def _nli(name, repo, config, split_map):
    def build(split, cap, rng):
        ds = take(hf(repo, config, split=split_map.get(split, split)), cap, rng)
        for i, r in enumerate(ds):
            if r["label"] not in (0, 1, 2):
                continue
            yield choice_record(
                rid(name, split, i), {"premise": r["premise"], "hypothesis": r["hypothesis"]},
                "How does the premise relate to the hypothesis?", dict(NLI_CRIT), NLI_KEYS[r["label"]], area="nli",
            )

    return build


def boolq(split, cap, rng):
    ds = take(hf("google/boolq", split=split), cap, rng)
    for i, r in enumerate(ds):
        yield noul_record(rid("boolq", split, i), {"passage": r["passage"]}, r["question"].rstrip("?") + "?", bool(r["answer"]),
                          "The passage says yes", "The passage says no", area="reading")


for spec in [
    DatasetSpec("mnli", _nli("mnli", "nyu-mll/multi_nli", None, {"test": "validation_matched"}), ("train", "test"), "nyu-mll/multi_nli", "mixed (cc-by-sa, mit, other)", "nli"),
    DatasetSpec("anli", _nli("anli", "facebook/anli", None, {"train": "train_r3", "test": "test_r3"}), ("train", "test"), "facebook/anli (round 3)", "cc-by-nc-4.0", "nli"),
    DatasetSpec("snli", _nli("snli", "stanfordnlp/snli", None, {}), ("train", "test"), "stanfordnlp/snli", "cc-by-sa-4.0", "nli"),
    DatasetSpec("boolq", boolq, ("train", "validation"), "google/boolq", "cc-by-sa-3.0", "reading"),
]:
    register(spec)
