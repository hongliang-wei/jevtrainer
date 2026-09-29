"""More multiple-choice and reasoning sources, plus eval-only reasoning suites (BBH, MuSR, CLadder, CRUXEval)."""

from __future__ import annotations

import ast
import json
import re

from jevtrainer.data.base import DatasetSpec, hf, mcq_record, multiple_choice, noul_record, register, rid, take

PARQUET = "refs/convert/parquet"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _sciq_opts(r):
    return [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]


def _sciq(split, cap, rng):
    ds = take(hf("allenai/sciq", split=split), cap, rng)
    for i, r in enumerate(ds):
        opts = _sciq_opts(r)
        order = list(range(4))
        rng.shuffle(order)
        state = {"question": r["question"], **({"support": r["support"]} if r["support"] else {})}
        rec = mcq_record(rid("sciq", split, i), state, "Which option correctly answers the science question?", [opts[j] for j in order], order.index(0), area="knowledge")
        if rec:
            yield rec


def _race(split, cap, rng):
    ds = take(hf("ehovy/race", "all", split=split), cap, rng)
    for i, r in enumerate(ds):
        rec = mcq_record(rid("race", split, i), {"article": r["article"], "question": r["question"]}, "Which option answers the question about the article?",
                         r["options"], LETTERS.index(r["answer"]), area="reading")
        if rec:
            yield rec


def _medqa(split, cap, rng):
    ds = take(hf("GBaker/MedQA-USMLE-4-options", split=split), cap, rng)
    for i, r in enumerate(ds):
        keys = sorted(r["options"])
        rec = mcq_record(rid("medqa", split, i), {"case": r["question"]}, "Which option is the best answer?", [r["options"][k] for k in keys], keys.index(r["answer_idx"]), area="medical")
        if rec:
            yield rec


def _truthfulqa(split, cap, rng):
    ds = hf("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    for i, r in enumerate(ds):
        t = r["mc1_targets"]
        opts, gold = list(t["choices"]), list(t["labels"]).index(1)
        order = list(range(len(opts)))
        rng.shuffle(order)
        rec = mcq_record(rid("truthfulqa", i), {"question": r["question"]}, "Which answer is true?", [opts[j] for j in order], order.index(gold), area="knowledge")
        if rec:
            yield rec


def _strategyqa(split, cap, rng):
    ds = take(hf("ChilleD/StrategyQA", split=split), cap, rng)
    for i, r in enumerate(ds):
        yield noul_record(rid("strategyqa", split, i), {"question": r["question"]}, "Is the answer to the question yes?", bool(r["answer"]), area="reasoning")


def _pubmedqa(split, cap, rng):
    ds = hf("qiaojin/PubMedQA", "pqa_labeled", split="train")
    for i, r in enumerate(ds):
        state = {"question": r["question"], "abstract": " ".join(r["context"]["contexts"])}
        crit = {"yes": "The study supports yes", "no": "The study supports no", "maybe": "The evidence is inconclusive"}
        from jevtrainer.data.base import choice_record

        yield choice_record(rid("pubmedqa", i), state, "What answer does the abstract support?", crit, r["final_decision"], area="medical")


def _social_iqa(split, cap, rng):
    ds = take(hf("allenai/social_i_qa", split=split, revision=PARQUET), cap, rng)
    for i, r in enumerate(ds):
        rec = mcq_record(rid("siqa", split, i), {"context": r["context"], "question": r["question"]}, "Which answer fits best?",
                         [r["answerA"], r["answerB"], r["answerC"]], int(r["label"]) - 1, area="commonsense")
        if rec:
            yield rec


def _logiqa(split, cap, rng):
    ds = take(hf("lucasmccabe/logiqa", split=split, revision=PARQUET), cap, rng)
    for i, r in enumerate(ds):
        rec = mcq_record(rid("logiqa", split, i), {"passage": r["context"], "question": r["query"]}, "Which option follows logically?",
                         r["options"], int(r["correct_option"]), area="reasoning")
        if rec:
            yield rec


BBH_TASKS = ["boolean_expressions", "causal_judgement", "date_understanding", "disambiguation_qa", "formal_fallacies", "geometric_shapes",
             "hyperbaton", "logical_deduction_five_objects", "logical_deduction_seven_objects", "logical_deduction_three_objects",
             "movie_recommendation", "navigate", "penguins_in_a_table", "reasoning_about_colored_objects", "ruin_names",
             "salient_translation_error_detection", "snarks", "sports_understanding", "temporal_sequences",
             "tracking_shuffled_objects_five_objects", "tracking_shuffled_objects_seven_objects", "tracking_shuffled_objects_three_objects", "web_of_lies"]
_OPT = re.compile(r"^\(([A-R])\)\s*(.+)$", re.M)


def _bbh(split, cap, rng):
    for task in BBH_TASKS:
        for i, r in enumerate(hf("lukaemon/bbh", task, split="test")):
            text, target = r["input"], r["target"].strip()
            opts = _OPT.findall(text)
            if opts:
                letters = [l for l, _ in opts]
                gold = target.strip("()")
                if gold not in letters:
                    continue
                question = text.split("\nOptions:")[0].strip()
                rec = mcq_record(rid("bbh", task, i), {"task": task, "problem": question}, "Which option is correct?", [t for _, t in opts], letters.index(gold), area="reasoning")
            elif target in ("True", "False", "Yes", "No", "valid", "invalid"):
                yes = target in ("True", "Yes", "valid")
                rec = noul_record(rid("bbh", task, i), {"task": task, "problem": text}, "Is the answer True / Yes / valid?", yes, area="reasoning")
            else:
                continue
            if rec:
                yield rec


def _musr(split, cap, rng):
    for sub in ("murder_mysteries", "object_placements", "team_allocation"):
        for i, r in enumerate(hf("TAUR-Lab/MuSR", split=sub)):
            choices = ast.literal_eval(r["choices"]) if isinstance(r["choices"], str) else r["choices"]
            rec = mcq_record(rid("musr", sub, i), {"story": r["narrative"]}, r["question"], choices, int(r["answer_index"]), area="reasoning")
            if rec:
                yield rec


def _cladder(split, cap, rng):
    ds = take(hf("causalnlp/CLadder", split="full_v1.5_default"), cap, rng)
    for i, r in enumerate(ds):
        yield noul_record(rid("cladder", i), {"problem": r["prompt"]}, "Is the answer to the causal question yes?", r["label"] == "yes", area="reasoning")


def _cruxeval(split, cap, rng):
    ds = list(hf("cruxeval-org/cruxeval", split="test"))
    outputs = [r["output"] for r in ds]
    for i, r in enumerate(ds):
        k = rng.randint(2, 9)
        pool = [o for o in dict.fromkeys(outputs) if o != r["output"] and type(_safe(o)) is type(_safe(r["output"]))]
        opts = rng.sample(pool, min(k, len(pool))) + [r["output"]]
        rng.shuffle(opts)
        rec = mcq_record(rid("cruxeval", i), {"code": r["code"], "call": f"f({r['input']})"}, "What does this call return (without running it)?",
                         opts, opts.index(r["output"]), area="code")
        if rec:
            yield rec


def _safe(s):
    try:
        return ast.literal_eval(s)
    except Exception:
        return s


register(DatasetSpec("sciq", _sciq, ("train", "validation", "test"), "allenai/sciq", "cc-by-nc-3.0", "knowledge"))
register(DatasetSpec("piqa", multiple_choice("piqa", "baber/piqa", None, lambda r: {"goal": r["goal"]}, lambda r: [r["sol1"], r["sol2"]],
                                             lambda r: int(r["label"]), "Which solution achieves the goal?", {"test": "validation"}, area="commonsense"),
                     ("train", "test"), "baber/piqa", "afl-3.0", "commonsense"))
register(DatasetSpec("qasc", multiple_choice("qasc", "allenai/qasc", None, lambda r: {"question": r["question"]}, lambda r: list(r["choices"]["text"]),
                                             lambda r: list(r["choices"]["label"]).index(r["answerKey"]), split_map={"test": "validation"}, area="knowledge"),
                     ("train", "test"), "allenai/qasc", "cc-by-4.0", "knowledge"))
register(DatasetSpec("race", _race, ("train", "validation", "test"), "ehovy/race:all", "other", "reading"))
register(DatasetSpec("medmcqa", multiple_choice("medmcqa", "openlifescienceai/medmcqa", None, lambda r: {"subject": r["subject_name"], "question": r["question"]},
                                                lambda r: [r["opa"], r["opb"], r["opc"], r["opd"]], lambda r: int(r["cop"]), split_map={"test": "validation"},
                                                area="medical"), ("train", "test"), "openlifescienceai/medmcqa", "apache-2.0", "medical"))
register(DatasetSpec("medqa", _medqa, ("train", "test"), "GBaker/MedQA-USMLE-4-options", "cc-by-4.0", "medical"))
register(DatasetSpec("truthfulqa", _truthfulqa, ("test",), "truthfulqa/truthful_qa:multiple_choice (mc1)", "apache-2.0", "knowledge", eval_only=True))
register(DatasetSpec("strategyqa", _strategyqa, ("train", "test"), "ChilleD/StrategyQA", "mit", "reasoning"))
register(DatasetSpec("pubmedqa", _pubmedqa, ("test",), "qiaojin/PubMedQA:pqa_labeled", "mit", "medical", eval_only=True))
register(DatasetSpec("social_iqa", _social_iqa, ("train", "validation"), "allenai/social_i_qa (parquet)", "cc-by-4.0", "commonsense"))
register(DatasetSpec("logiqa", _logiqa, ("train", "validation", "test"), "lucasmccabe/logiqa (parquet)", "unknown", "reasoning"))
register(DatasetSpec("bbh", _bbh, ("test",), "lukaemon/bbh (23 tasks)", "mit", "reasoning", eval_only=True))
register(DatasetSpec("musr", _musr, ("test",), "TAUR-Lab/MuSR", "cc-by-4.0", "reasoning", eval_only=True))
register(DatasetSpec("cladder", _cladder, ("test",), "causalnlp/CLadder:full_v1.5_default", "mit", "reasoning", eval_only=True))
register(DatasetSpec("cruxeval", _cruxeval, ("test",), "cruxeval-org/cruxeval", "mit", "code", eval_only=True,
                     description="pick the output of a Python call from 3-10 candidates (distractors: other samples' outputs of the same type)"))
