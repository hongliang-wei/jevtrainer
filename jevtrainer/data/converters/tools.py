"""Tool selection: which single tool from a catalog should handle a request."""

from __future__ import annotations

import json
import re

from jevtrainer.data.base import DatasetSpec, choice_record, hf, register, rid

ASK = "Which single tool from the catalog should be called for this request?"


def _tool_record(name, i, request, catalog: dict[str, str], gold: str, pool: list[tuple[str, str]], rng, k_range=(3, 8)):
    tools = dict(catalog)
    k = rng.randint(*k_range)
    for n, d in rng.sample(pool, min(len(pool), 3 * k)):
        if len(tools) >= k:
            break
        tools.setdefault(n, d)
    items = list(tools.items())
    rng.shuffle(items)
    state = {"request": request, "tools": [{"name": n, "description": d} for n, d in items]}
    return choice_record(rid(name, i), state, ASK, dict(items), gold, area="tools")


def _toolace_rows(rng):
    ds = hf("Team-ACE/ToolACE", split="train")
    for r in ds:
        i = r["system"].find("you can invoke:")
        if i < 0:
            continue
        try:
            funcs, _ = json.JSONDecoder().raw_decode(r["system"][i + len("you can invoke:"):].lstrip())
        except ValueError:
            continue
        catalog = {f["name"]: f.get("description", "") for f in funcs if isinstance(f, dict) and "name" in f}
        conv = r["conversations"]
        if len(conv) < 2 or conv[0]["from"] != "user" or conv[1]["from"] != "assistant":
            continue
        m = re.match(r"\[\s*([^(\]]+?)\s*\(", conv[1]["value"].strip())
        gold = m.group(1).strip() if m else None
        if gold in catalog:
            yield conv[0]["value"], catalog, gold


def toolace(split, cap, rng):
    rows = list(_toolace_rows(rng))
    pool = sorted({(n, d) for _, c, _ in rows for n, d in c.items()})
    rng.shuffle(rows)
    for i, (req, catalog, gold) in enumerate(rows[:cap]):
        yield _tool_record("toolace", i, req, catalog, gold, pool, rng)


def glaive_tools(split, cap, rng):
    ds = hf("glaiveai/glaive-function-calling-v2", split="train")
    rx = re.compile(r'\{\s*"name":\s*"([^"]+)",\s*"description":\s*"([^"]*)"', re.S)
    rows = []
    for r in ds.select(range(min(len(ds), 4 * cap))):
        funcs = dict(rx.findall(r["system"]))
        m = re.search(r"USER:\s*(.*?)\n\n\nASSISTANT:\s*<functioncall>\s*(\{.*?\})", r["chat"], re.S)
        if not funcs or not m:
            continue
        g = re.search(r'"name":\s*"([^"]+)"', m.group(2))
        if g and g.group(1) in funcs:
            rows.append((m.group(1).strip(), funcs, g.group(1)))
    pool = sorted({(n, d) for _, c, _ in rows for n, d in c.items()})
    for i, (req, catalog, gold) in enumerate(rows[:cap]):
        yield _tool_record("glaive", i, req, catalog, gold, pool, rng)


register(DatasetSpec("toolace", toolace, ("train",), "Team-ACE/ToolACE", "apache-2.0", "tools",
                     description="first-turn tool choice; Intern-Decision toolace_test rows are removed by overlap check"))
register(DatasetSpec("glaive_tools", glaive_tools, ("train",), "glaiveai/glaive-function-calling-v2", "apache-2.0", "tools"))
