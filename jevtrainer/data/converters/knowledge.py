"""Knowledge and commonsense multiple choice."""

from __future__ import annotations

import re

from jevtrainer.data.base import DatasetSpec, hf, mcq_record, register, rid, take

ASK = "Which option correctly answers the question?"


def _labels_choices(r):
    return list(r["choices"]["label"]), list(r["choices"]["text"])


def arc(split, cap, rng):
    for cfg in ("ARC-Challenge", "ARC-Easy"):
        ds = take(hf("allenai/ai2_arc", cfg, split=split), cap, rng)
        for i, r in enumerate(ds):
            labels, texts = _labels_choices(r)
            if r["answerKey"] in labels:
                rec = mcq_record(rid("arc", cfg, split, i), {"question": r["question"]}, ASK, texts, labels.index(r["answerKey"]), area="knowledge")
                if rec:
                    yield rec


def openbookqa(split, cap, rng):
    ds = take(hf("allenai/openbookqa", "main", split=split), cap, rng)
    for i, r in enumerate(ds):
        labels, texts = _labels_choices(r)
        rec = mcq_record(rid("obqa", split, i), {"question": r["question_stem"]}, ASK, texts, labels.index(r["answerKey"]), area="knowledge")
        if rec:
            yield rec


def csqa(split, cap, rng):
    ds = take(hf("tau/commonsense_qa", split=split), cap, rng)
    for i, r in enumerate(ds):
        labels, texts = _labels_choices(r)
        if r["answerKey"] in labels:
            rec = mcq_record(rid("csqa", split, i), {"question": r["question"]}, ASK, texts, labels.index(r["answerKey"]), area="commonsense")
            if rec:
                yield rec


def _mmlu(name, split_map):
    def build(split, cap, rng):
        ds = take(hf("cais/mmlu", "all", split=split_map[split]), cap, rng)
        for i, r in enumerate(ds):
            state = {"subject": r.get("subject", "").replace("_", " "), "question": r["question"]} if r.get("subject") else {"question": r["question"]}
            rec = mcq_record(rid(name, split, i), state, ASK, r["choices"], int(r["answer"]), area="knowledge")
            if rec:
                yield rec

    return build


def mmlu_pro(split, cap, rng):
    ds = take(hf("TIGER-Lab/MMLU-Pro", split="test"), cap, rng)
    for i, r in enumerate(ds):
        rec = mcq_record(rid("mmlu_pro", i), {"subject": r["category"], "question": r["question"]}, ASK, r["options"], int(r["answer_index"]), area="knowledge")
        if rec:
            yield rec


def hellaswag(split, cap, rng):
    ds = take(hf("Rowan/hellaswag", split={"train": "train", "test": "validation"}[split]), cap, rng)
    for i, r in enumerate(ds):
        ctx = re.sub(r"\[[^\]]*\]", "", r["ctx"]).strip()
        rec = mcq_record(rid("hellaswag", split, i), {"activity": r["activity_label"], "context": ctx}, "Which continuation is the most sensible?", r["endings"], int(r["label"]), area="commonsense")
        if rec:
            yield rec


def winogrande(split, cap, rng):
    ds = take(hf("allenai/winogrande", "winogrande_xl", split={"train": "train", "test": "validation"}[split]), cap, rng)
    for i, r in enumerate(ds):
        rec = mcq_record(rid("winogrande", split, i), {"sentence": r["sentence"]}, "Which option best fills the blank (_)?", [r["option1"], r["option2"]], int(r["answer"]) - 1, area="commonsense")
        if rec:
            yield rec


def gsm8k_mc(split, cap, rng):
    """Pick the final answer of a math word problem from 4 numbers, without working."""
    ds = take(hf("openai/gsm8k", "main", split=split), cap, rng)
    for i, r in enumerate(ds):
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        try:
            g = float(gold)
        except ValueError:
            continue
        cands = {gold}
        while len(cands) < 4:
            delta = rng.choice([1, 2, 3, 5, 10]) * rng.choice([-1, 1])
            v = g + delta if rng.random() < 0.6 else g * rng.choice([2, 0.5, 1.5])
            if v >= 0:
                cands.add(str(int(v)) if float(v).is_integer() else f"{v:g}")
        opts = sorted(cands, key=lambda x: float(x))
        rec = mcq_record(rid("gsm8k", split, i), {"problem": r["question"]}, "What is the final numeric answer?", opts, opts.index(gold), area="math")
        if rec:
            yield rec


for spec in [
    DatasetSpec("arc", arc, ("train", "validation", "test"), "allenai/ai2_arc (Challenge+Easy)", "cc-by-sa-4.0", "knowledge"),
    DatasetSpec("openbookqa", openbookqa, ("train", "validation", "test"), "allenai/openbookqa", "apache-2.0", "knowledge"),
    DatasetSpec("csqa", csqa, ("train", "validation"), "tau/commonsense_qa", "mit", "commonsense"),
    DatasetSpec("mmlu_aux", _mmlu("mmlu_aux", {"train": "auxiliary_train"}), ("train",), "cais/mmlu:auxiliary_train", "mit", "knowledge",
                description="MMLU auxiliary train (RACE, ARC, OBQA, MCTest mix)"),
    DatasetSpec("mmlu", _mmlu("mmlu", {"test": "test", "validation": "validation"}), ("test", "validation"), "cais/mmlu", "mit", "knowledge", eval_only=True),
    DatasetSpec("mmlu_pro", mmlu_pro, ("test",), "TIGER-Lab/MMLU-Pro", "mit", "knowledge", eval_only=True),
    DatasetSpec("hellaswag", hellaswag, ("train", "test"), "Rowan/hellaswag", "mit", "commonsense"),
    DatasetSpec("winogrande", winogrande, ("train", "test"), "allenai/winogrande:xl", "cc-by", "commonsense"),
    DatasetSpec("gsm8k_mc", gsm8k_mc, ("train", "test"), "openai/gsm8k", "mit", "math", description="final answer from 4 generated numbers"),
]:
    register(spec)
