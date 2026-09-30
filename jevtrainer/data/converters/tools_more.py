"""More tool-use decisions: Hermes function calling, When2Call, and legal / RAG judgment sets."""

from __future__ import annotations

import json
import re

from jevtrainer.data.base import DatasetSpec, choice_record, hf, mcq_record, noul_record, register, rid, take
from jevtrainer.data.converters.tools import ASK, _tool_record
from jevtrainer.schema import Question, Record, Target


def hermes_tools(split, cap, rng):
    ds = hf("NousResearch/hermes-function-calling-v1", "func_calling_singleturn", split="train")
    rows = []
    for r in ds:
        try:
            tools = json.loads(r["tools"]) if isinstance(r["tools"], str) else r["tools"]
        except ValueError:
            continue
        catalog = {t["function"]["name"]: t["function"].get("description", "") for t in tools if "function" in t}
        conv = r["conversations"]
        user = next((c["value"] for c in conv if c["from"] == "human"), None)
        call = next((c["value"] for c in conv if c["from"] == "gpt"), "")
        m = re.search(r'"name":\s*"([^"]+)"', call)
        if user and m and m.group(1) in catalog:
            rows.append((user, catalog, m.group(1)))
    pool = sorted({(n, d) for _, c, _ in rows for n, d in c.items()})
    for i, (req, catalog, gold) in enumerate(rows[:cap]):
        yield _tool_record("hermes", i, req, catalog, gold, pool, rng, k_range=(max(3, len(catalog)), max(len(catalog), 8)))


W2C = {
    "direct": "Answer directly; no tool is needed.",
    "tool_call": "Call one of the available tools.",
    "request_for_info": "Ask the user for missing information before calling a tool.",
    "cannot_answer": "Decline: no tool can do this and it cannot be answered directly.",
}


def when2call(split, cap, rng):
    ds = hf("nvidia/When2Call", "test", split="mcq")
    for i, r in enumerate(ds):
        tools = [json.loads(t).get("name", "") if t.strip().startswith("{") else t for t in (r["tools"] or [])]
        state = {"request": r["question"], "tools": tools}
        if r["correct_answer"] in W2C:
            yield choice_record(rid("when2call", i), state, "What should the assistant do next?", dict(W2C), r["correct_answer"], area="tools")


UNFAIR = ["limitation_of_liability", "unilateral_termination", "unilateral_change", "content_removal", "contract_by_using",
          "choice_of_law", "jurisdiction", "arbitration"]


def unfair_tos(split, cap, rng):
    ds = take(hf("coastalcph/lex_glue", "unfair_tos", split=split), cap, rng)
    for i, r in enumerate(ds):
        qs = {k: Question("noul", f"Is this terms-of-service clause potentially unfair because of {k.replace('_', ' ')}?") for k in UNFAIR}
        ts = {k: Target("yes" if j in r["labels"] else "no") for j, k in enumerate(UNFAIR)}
        yield Record(rid("unfair_tos", split, i), {"clause": r["text"]}, qs, ts, meta={"area": "legal"})


def case_hold(split, cap, rng):
    ds = take(hf("coastalcph/lex_glue", "case_hold", split=split), cap, rng)
    for i, r in enumerate(ds):
        rec = mcq_record(rid("casehold", split, i), {"citing_context": r["context"]}, "Which holding does the cited case stand for?", r["endings"], int(r["label"]), area="legal")
        if rec:
            yield rec


def ragtruth(split, cap, rng):
    ds = take(hf("wandb/RAGTruth-processed", split=split), cap, rng)
    for i, r in enumerate(ds):
        labels = r["hallucination_labels_processed"] or {}
        halluc = any(int(v) > 0 for v in labels.values()) if isinstance(labels, dict) else bool(r["hallucination_labels"] not in ("[]", []))
        state = {"task": r["query"], "source": r["context"][:5000], "answer": r["output"][:2500]}
        yield noul_record(rid("ragtruth", split, i), state, "Does the answer contain anything the source does not support?", halluc,
                          "It adds or contradicts facts", "Everything is supported by the source", area="rag")


register(DatasetSpec("hermes_tools", hermes_tools, ("train",), "NousResearch/hermes-function-calling-v1:func_calling_singleturn", "apache-2.0", "tools"))
register(DatasetSpec("when2call", when2call, ("test",), "nvidia/When2Call:test/mcq", "cc-by-4.0", "tools", eval_only=True))
register(DatasetSpec("unfair_tos", unfair_tos, ("train", "validation", "test"), "coastalcph/lex_glue:unfair_tos", "cc-by-4.0", "legal",
                     description="8 unfairness noul questions per clause"))
register(DatasetSpec("case_hold", case_hold, ("train", "validation", "test"), "coastalcph/lex_glue:case_hold", "cc-by-4.0", "legal"))
register(DatasetSpec("ragtruth", ragtruth, ("train", "test"), "wandb/RAGTruth-processed", "mit", "rag"))
