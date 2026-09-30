"""Long-document rule application, reading, multi-hop, numeric and judging decisions.

ShARC, QuALITY, ReClor, MuSiQue, TAT-QA, FinQA, TimeQA, PPE-IFEval and LegalBench rule tasks.
States longer than MAX_CHARS (about 3.5k Qwen tokens) are skipped rather than cut, since answers
can depend on any part of the document.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import defaultdict

from jevtrainer.data.base import DatasetSpec, choice_record, hf, mcq_record, noul_record, register, rid, take
from jevtrainer.schema import Question, Record, Target

PARQUET = "refs/convert/parquet"
MAX_CHARS = 14_000
NONE = "The document does not say"


def _list(x) -> list:
    """Some parquet exports store lists as their Python repr."""
    return ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else list(x)


def hash_split(key: str) -> str:
    """80/10/10 train/validation/test for sources that only ship one split."""
    h = int(hashlib.sha1(key.encode()).hexdigest(), 16) % 10
    return "test" if h == 0 else "validation" if h == 1 else "train"


def _multi(prefix: str, items: list[tuple[str, list[str], int]], rng) -> tuple[dict, dict]:
    """Several free-text MCQs on one state; options shuffled, neutral keys."""
    qs, ts = {}, {}
    for k, (ask, opts, gold) in enumerate(items):
        order = list(range(len(opts)))
        rng.shuffle(order)
        name = f"{prefix}{k + 1}"
        qs[name] = Question("choice", ask, {f"opt_{j + 1}": opts[o] for j, o in enumerate(order)})
        ts[name] = Target(f"opt_{order.index(gold) + 1}")
    return qs, ts


# ---- numbers ------------------------------------------------------------------------
_NUM = re.compile(r"^\(?-?\$?\s*-?[\d,]*\.?\d+\)?\s*%?$")


def _num(x) -> float | None:
    s = str(x).strip()
    if not _NUM.match(s):
        return None
    neg = s.startswith("(") or "-" in s
    v = float(re.sub(r"[^\d.]", "", s) or "nan")
    return -v if neg else v


def _decimals(x) -> int:
    s = str(x).strip().rstrip("%)")
    return min(len(s.split(".")[1]), 4) if "." in s else 0


def numeric_options(gold, related: list[float], rng, n: int, unit: str = "") -> tuple[list[str], int] | None:
    """Gold value plus close distractors: related quantities from the same document first, then
    relative / off-by-one / scale perturbations, all formatted like the gold answer."""
    g, d = _num(gold), _decimals(gold)
    if g is None:
        return None
    fmt = lambda x: f"{x:,.{d}f}{unit}"
    pert = [g * f for f in (0.9, 1.1, 0.95, 1.05, 0.8, 1.2, 0.5, 2.0)]
    rng.shuffle(pert)
    steps = [g + s * 10 ** -d for s in (1, -1, 2, -2)] if abs(g) < 50 * 10 ** -d else []
    cands = [x for x in related if x != g] + steps[:1] + pert + [g * 10, g / 10, -g] + steps[1:]
    opts = [fmt(g)]
    for x in cands:
        if fmt(x) not in opts:
            opts.append(fmt(x))
        if len(opts) == n:
            break
    if len(opts) < n:
        return None
    gold_text = opts[0]
    rng.shuffle(opts)
    return opts, opts.index(gold_text)


# ---- ShARC: rule text + scenario + follow-up history -> answer ------------------------------
SHARC_LABELS = ["yes", "no", "irrelevant", "more"]
SHARC = {
    "yes": "Yes: the rule applied to what the user has said gives yes",
    "no": "No: the rule applied to what the user has said gives no",
    "irrelevant": "The rule text does not address this question",
    "more": "Undecided: the user must first answer a further question about a condition in the rule",
}


def sharc(split, cap, rng):
    for i, r in enumerate(take(hf("tasksource/sharc", split=split), cap, rng)):
        state = {"rule_text": r["snippet"], "user_scenario": r["scenario"] or "(none given)", "question": r["question"],
                 "follow_up_answers": r["history"] or "(none yet)"}
        yield choice_record(rid("sharc", split, i), state, "Apply the rule text to the user's situation. What is the answer to the user's question?",
                            dict(SHARC), SHARC_LABELS[r["label"]], area="rules")


# ---- QuALITY: long articles, up to 5 questions per record -------------------------------------
def quality(split, cap, rng):
    by_article = defaultdict(list)
    for r in hf("emozilla/quality", split=split):
        if len(r["article"]) <= MAX_CHARS:
            by_article[r["article"]].append(r)
    n = 0
    for article, rows in by_article.items():
        for j in range(0, len(rows), 5):
            items = [(r["question"], [str(o).strip() for o in _list(r["options"])], int(r["answer"])) for r in rows[j:j + 5]]
            items = [it for it in items if len(set(it[1])) == len(it[1])]
            if not items or n >= cap:
                continue
            qs, ts = _multi("q", items, rng)
            n += 1
            yield Record(rid("quality", split, article[:300], j), {"article": article}, qs, ts, meta={"area": "reading"})


def reclor(split, cap, rng):
    for i, r in enumerate(take(hf("tasksource/reclor", split=split), cap, rng)):
        rec = mcq_record(rid("reclor", split, i), {"argument": r["context"], "question": r["question"]}, "Which option best answers the question about the argument?",
                         _list(r["answers"]), int(r["label"]), area="reasoning")
        if rec:
            yield rec


# ---- MuSiQue (full): answerable? + which entity answers the multi-hop question -------------------
def musique(split, cap, rng):
    f = {"train": "musique_full_v1.0_train.jsonl", "validation": "musique_full_v1.0_dev.jsonl"}[split]
    for i, r in enumerate(take(hf("bdsaglam/musique", split="train", data_files={"train": f}), cap, rng)):
        paras = r["paragraphs"]
        state = {"question": r["question"], "paragraphs": [f"[{p['idx'] + 1}] {p['title']}: {p['paragraph_text']}" for p in paras]}
        qs = {"answerable": Question("noul", "Do the paragraphs contain every fact needed to answer the question (all hops)?")}
        ts = {"answerable": Target("yes" if r["answerable"] else "no")}
        if r["answerable"]:
            bad = {a.lower() for a in [r["answer"], *(r["answer_aliases"] or [])]}
            hops = [d["answer"] for d in r["question_decomposition"][:-1]]
            titles = [p["title"] for p in paras if not p["is_supporting"]]
            rng.shuffle(titles)
            opts, n = [r["answer"]], rng.randint(4, 5)
            for c in hops + titles:
                if len(opts) < n and c.lower() not in bad and c not in opts:
                    opts.append(c)
            if len(opts) >= 3:
                q, t = _multi("answer", [("Which option answers the question?", opts, 0)], rng)
                qs["answer"], ts["answer"] = q["answer1"], t["answer1"]
        yield Record(rid("musique", split, i), state, qs, ts, meta={"area": "multi_hop"})


# ---- TAT-QA / FinQA: arithmetic over tables + text as numeric MCQ ---------------------------------
TATQA_FILES = {"train": "tatqa_dataset_train.json", "validation": "tatqa_dataset_dev.json", "test": "tatqa_dataset_test_gold.json"}
SCALE = {"": "", "thousand": " thousand", "million": " million", "billion": " billion", "percent": "%"}


def _table(rows) -> str:
    return "\n".join(" | ".join(str(c).strip() for c in row) for row in rows)


def tatqa(split, cap, rng):
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download("next-tat/TAT-QA", TATQA_FILES[split], repo_type="dataset"), encoding="utf-8") as fh:
        docs = json.load(fh)
    rng.shuffle(docs)
    for i, d in enumerate(docs[:cap]):
        qs_ = [q for q in d["questions"] if q.get("answer_type") == "arithmetic" and _num(q["answer"]) is not None and q.get("scale", "") in SCALE]
        items = []
        for q in qs_:
            related = [_num(o["answer"]) for o in qs_ if o is not q and o.get("scale") == q["scale"]]
            got = numeric_options(q["answer"], related, rng, rng.randint(3, 5), SCALE[q["scale"]])
            if got:
                items.append((q["question"], *got))
        if not items:
            continue
        text = "\n".join(p["text"] for p in sorted(d["paragraphs"], key=lambda p: p["order"]))
        qs, ts = {}, {}
        for k, (ask, opts, gold) in enumerate(items[:5]):
            qs[f"q{k + 1}"] = Question("choice", ask, {f"opt_{j + 1}": o for j, o in enumerate(opts)})
            ts[f"q{k + 1}"] = Target(f"opt_{gold + 1}")
        yield Record(rid("tatqa", split, i), {"table": _table(d["table"]["table"]), "report_text": text}, qs, ts, meta={"area": "finance"})


def finqa(split, cap, rng):
    for i, r in enumerate(take(hf("dreamerdeo/finqa", split=split, revision=PARQUET), cap, rng)):
        ans = str(r["answer"]).strip()
        got = numeric_options(ans.rstrip("%"), [], rng, rng.randint(3, 5), "%" if ans.endswith("%") else "")
        if not got:
            continue
        state = {"text_before": " ".join(_list(r["pre_text"])), "table": _table(_list(r["table"])), "text_after": " ".join(_list(r["post_text"])),
                 "question": r["question"]}
        rec = mcq_record(rid("finqa", split, i), state, "Which value answers the question (computed from the report)?", *got, area="finance")
        if rec:
            yield rec


# ---- TimeQA: time-scoped facts; distractors are the same page's answers for other periods -----------
def timeqa(split, cap, rng):
    rows = [r for r in hf("hugosousa/TimeQA", split=split, revision=PARQUET) if len(r["context"]) <= MAX_CHARS]
    by_page = defaultdict(set)
    for r in rows:
        by_page[r["idx"].split("#")[0]].update(t for t in r["targets"] if t)
    rng.shuffle(rows)
    n = 0
    for r in rows:
        gold_set = {t for t in r["targets"] if t}
        pool = sorted(by_page[r["idx"].split("#")[0]] - gold_set)
        if len(pool) < 2 or n >= cap:
            continue
        gold = rng.choice(sorted(gold_set)) if gold_set else NONE
        opts = rng.sample(pool, min(len(pool), rng.randint(2, 4))) + ([gold] if gold_set else [])
        rng.shuffle(opts)
        opts.append(NONE)
        rec = mcq_record(rid("timeqa", split, r["idx"]), {"document": r["context"], "question": r["question"]},
                         "Which option answers the question for the time period asked?", opts, opts.index(gold), area="temporal", level=r["level"])
        if rec:
            n += 1
            yield rec


# ---- PPE-IFEval: does a response satisfy every verifiable instruction? ---------------------------
def _constraint(iid: str, kw: dict) -> str:
    args = ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v!r}" for k, v in sorted(kw.items()) if v not in (None, [], ""))
    return iid.replace(":", " / ").replace("_", " ") + (f" ({args})" if args else "")


def ppe_ifeval(split, cap, rng):
    n = 0
    for r in hf("lmarena-ai/PPE-IFEval-Best-of-K", split="train"):
        if hash_split(r["question_id"]) != split or n >= cap:
            continue
        verdicts = [list(s["inst_level_strict_acc"]) for s in r["score_data"]]
        passed = [j for j, v in enumerate(verdicts) if all(v)]
        failed = [j for j, v in enumerate(verdicts) if not all(v)]
        m = min(len(passed), len(failed), 2) or 1
        picks = rng.sample(passed, min(m, len(passed))) + rng.sample(failed, min(m, len(failed)))
        cons = [_constraint(i, k) for i, k in zip(r["instruction_id_list"], r["kwargs"])]
        for j in picks:
            qs = {"all_met": Question("noul", "Does the response follow every instruction in the request (checked strictly)?")}
            ts = {"all_met": Target("yes" if all(verdicts[j]) else "no")}
            for c, (desc, ok) in enumerate(zip(cons, verdicts[j])):
                qs[f"c{c + 1}"] = Question("noul", f"Does the response satisfy this instruction from the request: {desc}?")
                ts[f"c{c + 1}"] = Target("yes" if ok else "no")
            n += 1
            yield Record(rid("ppe_ifeval", r["question_id"], j), {"request": r["ifeval_prompt"], "response": r[f"response_{j + 1}"]}, qs, ts,
                         meta={"area": "judge", "model": r["model_name"]})


# ---- LegalBench rule-application tasks (their test split is the data; re-split 80/10/10) ------------
YN = lambda a: {"yes": True, "no": False}.get(str(a).strip().lower())
LB_NOUL = {  # task: (state fields, instructions)
    "hearsay": ({"evidence": "text"}, "Is this evidence hearsay (an out-of-court statement offered to prove the truth of the matter asserted)?"),
    "personal_jurisdiction": ({"facts": "text"}, "Does the court where the suit was filed have personal jurisdiction over the defendant?"),
    "jcrew_blocker": ({"provision": "text"}, "Does this loan-agreement provision contain a J.Crew blocker (restricting transfers of material IP to unrestricted subsidiaries)?"),
    "telemarketing_sales_rule": ({"facts": "text"}, "Under the Telemarketing Sales Rule, is the answer to the question in the facts yes?"),
    "contract_qa": ({"clause": "text", "question": "question"}, "Is the answer to the question about the clause yes?"),
    **{f"diversity_{k}": ({"facts": "text"}, "Is there federal diversity jurisdiction (complete diversity of citizenship and over $75,000 in controversy against a defendant)?")
       for k in range(1, 7)},
}
SUPPLY = {
    "verification": "verification of product supply chains to evaluate and address risks of human trafficking and slavery",
    "audits": "audits of suppliers to evaluate compliance with company standards on trafficking and slavery",
    "certification": "requiring direct suppliers to certify that materials comply with slavery and trafficking laws",
    "accountability": "internal accountability standards and procedures for employees or contractors who fail to meet standards on slavery and trafficking",
    "training": "training on human trafficking and slavery for employees and managers responsible for supply-chain management",
}
INSURANCE = {"A": "covered", "B": "not_covered", "C": "ambiguous"}


def legalbench_rules(split, cap, rng):
    rows = []
    for task, (fields, ask) in LB_NOUL.items():
        for r in hf("nguha/legalbench", task, split="test"):
            y = YN(r["answer"])
            if y is not None and hash_split(task + r["text"]) == split:
                rows.append(noul_record(rid("lb", task, r["index"]), {k: r[v] for k, v in fields.items()}, ask, y, area="legal", task=task))
    for r in hf("nguha/legalbench", "sara_entailment", split="test"):
        if hash_split("sara" + r["text"]) == split:
            rows.append(noul_record(rid("lb", "sara", r["index"]), {"statute": r["statute"].strip(), "facts": r["description"], "claim": r["question"]},
                                    "Given the statute and the facts, is the claim true?", r["answer"] == "Entailment", area="legal", task="sara_entailment"))
    for r in hf("nguha/legalbench", "insurance_policy_interpretation", split="test"):
        if r["answer"] in INSURANCE and hash_split("ins" + r["claim"]) == split:
            rows.append(choice_record(rid("lb", "insurance", r["index"]), {"policy": r["policy"], "claim": r["claim"]}, "Does the policy cover the claim?",
                                      {"covered": "Yes, the claim is covered", "not_covered": "No, the claim is not covered",
                                       "ambiguous": "It is ambiguous whether the policy covers the claim"}, INSURANCE[r["answer"]], area="legal", task="insurance"))
    for r in hf("nguha/legalbench", "ucc_v_common_law", split="test"):
        if hash_split("ucc" + r["contract"]) == split:
            rows.append(choice_record(rid("lb", "ucc", r["index"]), {"contract": r["contract"]}, "Which body of law governs this contract?",
                                      {"UCC": "The UCC (primarily a sale of goods)", "Common Law": "The common law (services, real estate, or other non-goods)"},
                                      r["answer"].strip(), area="legal", task="ucc_v_common_law"))
    rows.extend(_supply_chain(split))
    rng.shuffle(rows)
    yield from rows[:cap]


def _supply_chain(split):
    by_text = defaultdict(dict)
    for kind in ("disclosed", "best_practice"):
        for topic in SUPPLY:
            for r in hf("nguha/legalbench", f"supply_chain_disclosure_{kind}_{topic}", split="test"):
                if YN(r["answer"]) is not None:
                    by_text[r["text"]][(kind, topic)] = YN(r["answer"])
    for text, labels in by_text.items():
        if hash_split("supply" + text) != split or len(text) > MAX_CHARS:
            continue
        qs, ts = {}, {}
        for (kind, topic), y in sorted(labels.items()):
            ask = (f"Does the statement disclose to what extent the company engages in {SUPPLY[topic]}?" if kind == "disclosed"
                   else f"Does the statement show the company meets best practice for {SUPPLY[topic]}?")
            qs[f"{kind}_{topic}"], ts[f"{kind}_{topic}"] = Question("noul", ask), Target("yes" if y else "no")
        yield Record(rid("lb", "supply", text[:300]), {"supply_chain_statement": text}, qs, ts, meta={"area": "legal", "task": "supply_chain_disclosure"})


def consumer_contracts(split, cap, rng):
    for r in hf("nguha/legalbench", "consumer_contracts_qa", split="test"):
        y = YN(r["answer"])
        if y is not None and hash_split(r["contract"][:500] + r["question"]) == split:
            yield noul_record(rid("lb", "ccqa", r["index"]), {"terms_of_service": r["contract"], "question": r["question"]},
                              "According to these terms, is the answer to the user's question yes?", y, area="legal")


S3 = ("train", "validation", "test")
register(DatasetSpec("sharc", sharc, S3, "tasksource/sharc", "cc-by-sa-3.0", "rules", description="rule text + scenario + follow-ups -> yes / no / irrelevant / need more"))
register(DatasetSpec("quality", quality, ("train", "validation"), "emozilla/quality (articles <= 14k chars)", "cc-by-4.0", "reading",
                     description="long-article MCQ, up to 5 questions per record"))
register(DatasetSpec("reclor", reclor, ("train", "validation"), "tasksource/reclor", "reclor (research only, non-commercial)", "reasoning", tags=["nc"]))
register(DatasetSpec("musique", musique, ("train", "validation"), "bdsaglam/musique (full v1.0)", "cc-by-4.0", "multi_hop",
                     description="20 paragraphs; answerable noul + entity MCQ with intermediate-hop distractors"))
register(DatasetSpec("tatqa", tatqa, S3, "next-tat/TAT-QA", "cc-by-4.0", "finance", description="arithmetic questions as numeric MCQ; distractors from sibling answers"))
register(DatasetSpec("finqa", finqa, S3, "dreamerdeo/finqa (parquet)", "mit", "finance", description="numeric MCQ with close perturbed distractors"))
register(DatasetSpec("timeqa", timeqa, S3, "hugosousa/TimeQA (parquet, context <= 14k chars)", "bsd-3-clause-clear", "temporal",
                     description="time-scoped facts; distractors are other periods' answers; unanswerable -> 'does not say'"))
register(DatasetSpec("ppe_ifeval", ppe_ifeval, S3, "lmarena-ai/PPE-IFEval-Best-of-K (split by prompt)", "unknown (lmarena PPE)", "judge",
                     description="all-instructions-met + per-instruction noul, labels from the IFEval strict verifier"))
register(DatasetSpec("legalbench_rules", legalbench_rules, S3, "nguha/legalbench (rule tasks, test split re-split 80/10/10)", "cc-by-4.0 / mit (per task)", "legal",
                     description="hearsay, diversity, personal jurisdiction, J.Crew, TSR, contract QA, SARA, insurance, UCC, supply-chain disclosure"))
register(DatasetSpec("legalbench_consumer_contracts", consumer_contracts, S3, "nguha/legalbench:consumer_contracts_qa (re-split 80/10/10)", "cc-by-nc-4.0", "legal",
                     tags=["nc"]))
