"""Sources that need extra work before they become records.

* GPQA / HLE : gated; accept the terms on the Hub and set HF_TOKEN. HLE keeps text-only multiple-choice items.
* BFCL       : .json files are really JSON Lines with per-category columns, so `datasets` cannot load them;
               read each file and join `possible_answer/`. One noul per supplied function ("call it?").
* S1-mini    : the question texts live in the generator's SCHEMA, not in the rows.
* NanoJev    : the question/candidate rows are under unified/hard/, not benchmark/.
* HoVer      : the original release has only evidence titles; Dzeniks/hover carries the evidence text.
"""

from __future__ import annotations

import json
import re

from jevtrainer.data.base import DatasetSpec, hf, mcq_record, register, rid, take
from jevtrainer.schema import Question, Record, Target


def _lines(repo: str, path: str):
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(repo, path, repo_type="dataset"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# ---- GPQA / HLE -------------------------------------------------------------------------
def gpqa(config):
    def build(split, cap, rng):
        for i, r in enumerate(hf("Idavidrein/gpqa", config, split="train")):
            opts = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
            order = list(range(4))
            rng.shuffle(order)
            rec = mcq_record(rid("gpqa", config, i), {"domain": r["Subdomain"], "question": r["Question"]}, "Which option correctly answers the question?",
                             [opts[j].strip() for j in order], order.index(0), area="knowledge")
            if rec:
                yield rec

    return build


_HLE_OPT = re.compile(r"^\s*([A-Z])[\.\)]\s+(.+?)\s*$")


def hle(split, cap, rng):
    for r in hf("cais/hle", split="test"):
        if r["answer_type"] != "multipleChoice" or r["image"]:
            continue
        text = r["question"]
        head, sep, tail = text.partition("Answer Choices:")
        lines = tail.splitlines() if sep else text.splitlines()
        opts = [(m.group(1), m.group(2)) for m in map(_HLE_OPT.match, lines) if m]
        letters = [l for l, _ in opts]
        gold = r["answer"].strip().rstrip(".")[:1]
        if len(opts) < 2 or gold not in letters:
            continue
        rec = mcq_record(r["id"], {"subject": r["raw_subject"], "question": (head if sep else text).strip()}, "Which answer choice is correct?",
                         [t for _, t in opts], letters.index(gold), area="knowledge")
        if rec:
            yield rec


# ---- BFCL ---------------------------------------------------------------------------------
BFCL_CATS = ["simple", "multiple", "parallel", "parallel_multiple", "live_simple", "live_multiple", "live_parallel",
             "live_parallel_multiple", "irrelevance", "live_irrelevance"]


def bfcl(split, cap, rng):
    repo = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
    for cat in BFCL_CATS:
        answers = {}
        if "irrelevance" not in cat:
            for a in _lines(repo, f"possible_answer/BFCL_v3_{cat}.json"):
                answers[a["id"]] = {name for call in a["ground_truth"] for name in call}
        for r in _lines(repo, f"BFCL_v3_{cat}.json"):
            funcs = r["function"] if isinstance(r["function"], list) else [r["function"]]
            if not funcs or len(funcs) > 40:
                continue
            turn = r["question"][0] if isinstance(r["question"][0], list) else r["question"]
            request = "\n".join(m["content"] for m in turn if m.get("role") == "user")
            gold = answers.get(r["id"], set()) if "irrelevance" not in cat else set()
            if "irrelevance" not in cat and not gold:
                continue
            qs, ts = {}, {}
            for j, f in enumerate(funcs):
                key = f"call_{j}"
                qs[key] = Question("noul", f"Should the function `{f['name']}` be called to handle this request?",
                                   {"true": "Yes, calling it is part of handling the request", "false": "No"})
                ts[key] = Target("yes" if f["name"] in gold else "no")
            state = {"request": request, "functions": [{"name": f["name"], "description": f.get("description", "")} for f in funcs]}
            yield Record(r["id"], state, qs, ts, meta={"area": "tools", "category": cat})


# ---- System One Mini -------------------------------------------------------------------------
S1_SCHEMA = {
    "task_complete": ("Has the task been successfully completed?", ["false", "true"]),
    "should_retry": ("Should the same failed action be retried without changing it?", ["false", "true"]),
    "failure_source": ("What is the source of the original reproduced failure?", ["code", "environment", "test", "dependency", "unknown"]),
    "next_action": ("What action should the agent take next under the diagnostic policy?", ["inspect_code", "rerun_tests", "inspect_environment", "revert_change", "escalate"]),
    "evidence_sufficient": ("Does exactly one controlled intervention identify the original failure source?", ["false", "true"]),
}
S1_POLICY = ("Controlled trials each start from the same failing snapshot and change only the named component. "
             "Exactly one repair clearing the original failure identifies its source; zero or multiple clearing repairs mean unknown. "
             "Task complete means latest checks pass AND acceptance was confirmed. Retry unchanged only if incomplete, "
             "the temporary blocker has cleared, and no unchanged retry already failed. Next action: completed => escalate (handoff); "
             "otherwise if retry is warranted or latest checks pass => rerun_tests; "
             "otherwise a code cause with rollback authorized => revert_change; code or test => inspect_code; "
             "environment or dependency => inspect_environment; unknown => escalate.")


def s1_mini(split, cap, rng):
    rows = list(_lines("DavidHatley/system-one-mini-data", f"data/{split}.jsonl"))
    rng.shuffle(rows)
    for r in rows[:cap]:
        qs, ts = {}, {}
        for name, (prompt, opts) in S1_SCHEMA.items():
            if opts == ["false", "true"]:
                qs[name] = Question("noul", prompt)
            else:
                qs[name] = Question("choice", prompt, {o: o.replace("_", " ") for o in opts})
            ts[name] = Target(qs[name].canonical_label(str(r[name])))
        yield Record(r["id"], {"policy": S1_POLICY, "report": r["state"]}, qs, ts, meta={"area": "agents"})


# ---- NanoJev -----------------------------------------------------------------------------------
def nanojev(split, cap, rng):
    fname = {"train": "train", "validation": "dev", "test": "test", "ood": "ood"}[split]
    rows = list(_lines("C-Tianyu/NanoJev-Data", f"unified/hard/{fname}.jsonl"))
    rng.shuffle(rows)
    for r in rows[:cap]:
        targets = {}
        gold = r.get("gold") or r.get("targets") or {}
        teacher = (r.get("teacher") or {}).get("native_probs") or {}
        expert = (r.get("expert") or {}).get("policy_probs")
        for name in r["questions"]:
            g = gold.get(name) if isinstance(gold, dict) else None
            label = g.get("label") if isinstance(g, dict) else g
            probs = teacher.get(name) or (expert if len(r["questions"]) == 1 else None)
            if label is None and probs:
                label = max(probs, key=probs.get)
            if label is not None:
                targets[name] = {"label": label, "probabilities": probs}
        if not targets:
            continue
        distilled = "typesafe" in json.dumps(r.get("teacher") or {})
        try:
            yield Record.from_dict({"id": r["id"], "state": r["state"], "questions": r["questions"], "targets": targets,
                                    "meta": {"area": "games", "task": (r.get("metadata") or {}).get("task"), "jev_distilled": distilled}}).validate()
        except (ValueError, KeyError, TypeError):
            continue


# ---- HoVer ---------------------------------------------------------------------------------------
def hover(split, cap, rng):
    labels = {}
    for sp in ("train", "validation"):
        for r in hf("bdsaglam/hover", split=sp):
            labels[r["claim"]] = (r["label"], r["num_hops"])
    ds = take(hf("Dzeniks/hover", split=split), cap, rng)
    for i, r in enumerate(ds):
        if r["claim"] not in labels:
            continue
        label, hops = labels[r["claim"]]
        rec = Record(rid("hover", split, i), {"claim": r["claim"], "evidence": r["evidence"][:6000]},
                     {"decision": Question("noul", "Do the evidence paragraphs support the claim?",
                                           {"true": "Supported by the evidence", "false": "Not supported"})},
                     {"decision": Target("yes" if int(label) == 1 else "no")}, meta={"area": "factcheck", "hops": hops})
        yield rec


register(DatasetSpec("gpqa_diamond", gpqa("gpqa_diamond"), ("test",), "Idavidrein/gpqa:gpqa_diamond (gated)", "cc-by-4.0", "knowledge", eval_only=True))
register(DatasetSpec("gpqa_main", gpqa("gpqa_main"), ("test",), "Idavidrein/gpqa:gpqa_main (gated)", "cc-by-4.0", "knowledge", eval_only=True))
register(DatasetSpec("hle", hle, ("test",), "cais/hle (gated; text-only multiple choice)", "mit", "knowledge", eval_only=True))
register(DatasetSpec("bfcl", bfcl, ("test",), "gorilla-llm/Berkeley-Function-Calling-Leaderboard v3", "apache-2.0", "tools", eval_only=True,
                     description="one noul per supplied function; score case_exact"))
register(DatasetSpec("s1_mini", s1_mini, ("train", "validation", "calibration", "test"), "DavidHatley/system-one-mini-data", "apache-2.0", "agents",
                     description="5 typed questions per synthetic diagnostic report; questions from the generator SCHEMA"))
register(DatasetSpec("nanojev", nanojev, ("train", "validation", "test", "ood"), "C-Tianyu/NanoJev-Data:unified/hard", "see upstream", "games",
                     tags=["partly-jev-distilled"], description="Maze / Snake / ViZDoom action choices"))
register(DatasetSpec("hover", hover, ("train", "test"), "Dzeniks/hover evidence + bdsaglam/hover labels", "cc-by-sa-4.0", "factcheck"))
