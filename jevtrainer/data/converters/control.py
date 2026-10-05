"""Agent control-plane decisions: which model tier, whether a shell command may run, and whether a
tool response should stay in context.

Labels come from public matrices and annotations. The question never names a concrete model.
"""

from __future__ import annotations

import csv
import json
import re
import sys

from jevtrainer.data.base import DatasetSpec, choice_record, hf, hf_file, register, rid

ROUTE = {
    "fast": "A fast, cheap model",
    "mid": "A mid-cost model",
    "strong": "A strong, expensive model",
}
ROUTE_Q = "Which tier of model should answer this request?"
SHELL = {
    "allow": "Allow the command to run",
    "ask": "Ask the user to confirm before running",
    "deny": "Block the command",
}
SHELL_Q = "Before this command runs in the current session, what should the agent do?"
COMPACT = {
    "keep": "Keep the tool response",
    "truncate": "Keep only the useful lines and drop the rest",
    "drop": "Drop the tool response",
}
COMPACT_Q = "When compacting the context, what should happen to this tool response?"

# SPROUT columns that hold a judge score. Ordered only for a stable walk; tier comes from the name.
SPROUT_MODELS = (
    "wxai-llama-3-2-1b-instruct",
    "wxai-granite-3-2b-instruct-8k-max-tokens",
    "wxai-llama-3-2-3b-instruct",
    "wxai-granite-3-8b-instruct-8k-max-tokens",
    "wxai-llama-3-1-8b-instruct",
    "openai-gpt-4o-mini",
    "wxai-mixtral-8x7b-instruct-v01",
    "aws-titan-text-premier-v1",
    "wxai-llama-3-1-70b-instruct",
    "wxai-llama-3-3-70b-instruct",
    "openai-gpt-4o",
    "aws-claude-3-5-sonnet-v1",
    "wxai-llama-3-405b-instruct",
)
_OK = 0.8
_CHEAP = ("gpt-4o-mini", "gpt-3.5", "claude-instant", "claude-3-haiku", "gemini-1.5-flash", "gemini-flash")
_FRONTIER = ("gpt-4o", "gpt-4", "claude-3.5", "claude-3-5", "claude-3-opus", "claude-3-sonnet", "claude-2", "o1-preview", "o1-mini")
_B = re.compile(r"(\d+(?:\.\d+)?)\s*b")
_MX = re.compile(r"(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*b")


def model_tier(name: str) -> str | None:
    """fast / mid / strong from a public model id, by parameter count or known API price."""
    n = name.lower().replace("_", "-")
    if any(x in n for x in _CHEAP):
        return "fast"
    if any(x in n for x in _FRONTIER):
        return "strong"
    mx = _MX.search(n)
    if mx:
        total = int(mx.group(1)) * float(mx.group(2))
        return "mid" if total < 70 else "strong"
    m = _B.search(n)
    if not m:
        return None
    b = float(m.group(1))
    if b < 8:
        return "fast"
    if b < 40:
        return "mid"
    return "strong"


def correctness(cell) -> float | None:
    """SPROUT stores a judge blob; the score is correctness_score, else the outer score."""
    if isinstance(cell, str):
        try:
            cell = json.loads(cell)
        except json.JSONDecodeError:
            return None
    if not isinstance(cell, dict):
        return None
    jr = cell.get("judge_response")
    if isinstance(jr, str):
        try:
            jr = json.loads(jr)
        except json.JSONDecodeError:
            jr = None
    if isinstance(jr, dict) and jr.get("correctness_score") is not None:
        return float(jr["correctness_score"])
    if cell.get("score") is not None and not isinstance(cell["score"], dict):
        try:
            return float(cell["score"])
        except (TypeError, ValueError):
            return None
    return None


def cheapest_success(scores: dict[str, float | None]) -> str:
    """Tier of the cheapest model that scored at least 0.8. Strong when none did."""
    hit = set()
    for name, score in scores.items():
        tier = model_tier(name)
        if tier and score is not None and score >= _OK:
            hit.add(tier)
    for tier in ("fast", "mid", "strong"):
        if tier in hit:
            return tier
    return "strong"


def compact_label(n_kept: int, n_lines: int) -> str | None:
    if n_lines <= 0:
        return None
    frac = n_kept / n_lines
    if frac >= 0.85:
        return "keep"
    if frac <= 0.05:
        return "drop"
    return "truncate"


def _clip(text, n: int = 8000) -> str:
    if isinstance(text, list):
        parts = []
        for m in text:
            if isinstance(m, dict):
                parts.append(str(m.get("content", "")))
            else:
                parts.append(str(m))
        text = "\n".join(parts)
    elif isinstance(text, dict):
        text = json.dumps(text, ensure_ascii=False)
    return str(text).strip()[:n]


def _route(id: str, request: str, tier: str, **meta):
    return choice_record(id, {"request": _clip(request)}, ROUTE_Q, dict(ROUTE), tier, area="agents", **meta)


def _sprout(split, cap):
    ds = hf("CARROT-LLM-Routing/SPROUT", split=split)
    n = 0
    for r in ds:
        if n >= cap:
            return
        scores = {m: correctness(r.get(m)) for m in SPROUT_MODELS}
        rec = _route(rid("router", "sprout", r["key"]), r["prompt"], cheapest_success(scores), source_set="sprout")
        if rec and rec.state["request"]:
            n += 1
            yield rec


def _embedllm(split, cap):
    filename = {"train": "train.csv", "validation": "val.csv", "test": "test.csv"}[split]
    path = hf_file("RZ412/EmbedLLM", filename)
    grouped: dict[str, dict] = {}
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            slot = grouped.get(row["prompt_id"])
            if slot is None:
                slot = {"prompt": row["prompt"], "scores": {}}
                grouped[row["prompt_id"]] = slot
            try:
                score = float(row["label"])
            except (TypeError, ValueError):
                continue
            slot["scores"][row["model_name"]] = score
    n = 0
    for pid, slot in grouped.items():
        if n >= cap:
            return
        rec = _route(rid("router", "embedllm", pid), slot["prompt"], cheapest_success(slot["scores"]), source_set="embedllm")
        if rec and rec.state["request"]:
            n += 1
            yield rec


def _arena(cap):
    ds = hf("lmarena-ai/arena-human-preference-55k", split="train")
    n = 0
    for i, r in enumerate(ds):
        if n >= cap:
            return
        if r.get("winner_tie"):
            continue
        ta, tb = model_tier(str(r["model_a"])), model_tier(str(r["model_b"]))
        if not ta or not tb or ta == tb:
            continue
        winner = ta if r.get("winner_model_a") else tb if r.get("winner_model_b") else None
        if winner is None:
            continue
        # The cheaper tier won: use it. The more expensive tier won: the cheap one was not enough.
        rec = _route(rid("router", "arena", i), r["prompt"], winner, source_set="arena")
        if rec and rec.state["request"]:
            n += 1
            yield rec


def router_tier(split, cap, rng):
    """Train is SPROUT + EmbedLLM + Arena. Validation and test stay the held-out routing splits."""
    n = 0
    for rec in _sprout(split, cap):
        n += 1
        yield rec
    if n >= cap:
        return
    for rec in _embedllm(split, cap - n):
        n += 1
        yield rec
    if split == "train" and n < cap:
        yield from _arena(cap - n)


def shell_gate(split, cap, rng):
    ds = hf("tomngdev/shell-safety-v1.1", split=split)
    n = 0
    for i, r in enumerate(ds):
        if n >= cap:
            return
        label = str(r["label"]).strip().lower()
        if label not in SHELL or not r.get("command"):
            continue
        state = {"command": _clip(r["command"], 4000), "session": _clip(r.get("session_context"), 4000)}
        n += 1
        yield choice_record(rid("shell", split, i), state, SHELL_Q, dict(SHELL), label, area="agents", category=str(r.get("category") or ""))


def context_compact(split, cap, rng):
    path = hf_file("ayanami-kitasan/swe-pruner-pro-training-corpus", "training_corpus_22k.jsonl")
    n = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if n >= cap:
                return
            r = json.loads(line)
            kept = r.get("kept_frags") or []
            label = compact_label(len(kept), int(r.get("total_lines") or 0))
            if label is None or not r.get("tool_response"):
                continue
            call = r.get("tool_call") or {}
            state = {
                "tool": str(call.get("name") or ""),
                "call": _clip(call.get("arguments"), 2000),
                "tool_response": _clip(r["tool_response"], 12000),
            }
            n += 1
            yield choice_record(rid("compact", i), state, COMPACT_Q, dict(COMPACT), label, area="agents",
                                source_id=f"{r.get('instance_id')}:{r.get('step_idx')}")


register(DatasetSpec(
    "router_tier", router_tier, ("train", "validation", "test"),
    "CARROT-LLM-Routing/SPROUT + RZ412/EmbedLLM + lmarena-ai/arena-human-preference-55k",
    "see upstream", "agents",
    description="cheapest model tier (fast/mid/strong) that answered the request; Arena only in train"))
register(DatasetSpec(
    "shell_gate", shell_gate, ("train", "validation", "test"),
    "tomngdev/shell-safety-v1.1", "mit", "agents",
    description="allow, ask, or deny a shell command given the session"))
register(DatasetSpec(
    "context_compact", context_compact, ("train",),
    "ayanami-kitasan/swe-pruner-pro-training-corpus", "apache-2.0", "agents",
    description="keep, truncate, or drop a tool response from kept line spans"))
