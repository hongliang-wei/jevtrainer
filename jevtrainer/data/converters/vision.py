"""Multimodal sources. Images are written once to the cache and referenced by path."""

from __future__ import annotations

import re

from jevtrainer.data.base import DatasetSpec, cache_dir, hf, mcq_record, register, rid, take


def _save(img, name: str, key: str) -> str:
    d = cache_dir() / "images" / name
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{key}.png"
    if not p.exists():
        img.convert("RGB").save(p)
    return str(p)


def scienceqa(split, cap, rng):
    ds = hf("derek-thomas/ScienceQA", split=split)
    ds = take(ds.filter(lambda r: r["image"] is not None), cap, rng)
    for i, r in enumerate(ds):
        key = rid("scienceqa", split, i)
        state = {"question": r["question"], "context": r["hint"]} if r["hint"] else {"question": r["question"]}
        rec = mcq_record(key, state, "Which option correctly answers the question about the image?", r["choices"], int(r["answer"]), area="vision")
        if rec:
            rec.images = [_save(r["image"], "scienceqa", key)]
            yield rec


_CHOICE_LINE = re.compile(r"^\s*([A-H])[\.\)]\s*(.+)$")


def _lettered(turn):
    """'A. foo' lines, answer is a letter."""
    lines = turn["user"].splitlines()
    opts = [(m.group(1), m.group(2).strip()) for m in map(_CHOICE_LINE.match, lines) if m]
    ans = re.search(r"\b([A-H])\b", turn["assistant"])
    letters = [l for l, _ in opts]
    if len(opts) < 2 or not ans or ans.group(1) not in letters:
        return None
    question = "\n".join(l for l in lines if not _CHOICE_LINE.match(l) and not l.lower().startswith(("choices", "answer with")))
    return question.strip(), [t for _, t in opts], letters.index(ans.group(1))


def _inline_options(turn):
    """'Options: a, b, c.' line, answer is the option text."""
    m = re.search(r"^Options:\s*(.+?)\.?\s*$", turn["user"], re.M)
    if not m:
        return None
    options = [o.strip() for o in m.group(1).split(",") if o.strip()]
    answer = turn["assistant"].strip().rstrip(".").lower()
    lower = [o.lower() for o in options]
    if len(options) < 2 or answer not in lower:
        return None
    question = turn["user"].splitlines()[0].strip()
    return question, options, lower.index(answer)


def _cauldron(subset):
    def build(split, cap, rng):
        ds = take(hf("HuggingFaceM4/the_cauldron", subset, split="train"), cap, rng)
        for i, r in enumerate(ds):
            if len(r["images"]) != 1:
                continue
            parsed = _lettered(r["texts"][0]) or _inline_options(r["texts"][0])
            if not parsed:
                continue
            question, options, gold = parsed
            key = rid("cauldron", subset, i)
            rec = mcq_record(key, {"question": question}, "Which option correctly answers the question about the image?", options, gold, area="vision")
            if rec:
                rec.images = [_save(r["images"][0], f"cauldron_{subset}", key)]
                yield rec

    return build


register(DatasetSpec("scienceqa_img", scienceqa, ("train", "validation", "test"), "derek-thomas/ScienceQA (image subset)", "cc-by-nc-sa-4.0", "vision", multimodal=True))
for sub in ("ai2d", "aokvqa", "scienceqa", "iconqa", "tqa"):
    register(DatasetSpec(f"cauldron_{sub}", _cauldron(sub), ("train",), f"HuggingFaceM4/the_cauldron:{sub}", "see subset", "vision", multimodal=True,
                         description="multiple-choice subsets parsed from The Cauldron chat turns"))
