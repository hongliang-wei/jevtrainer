"""Web agent next-element selection (Mind2Web, text candidates)."""

from __future__ import annotations

import json

from jevtrainer.data.base import DatasetSpec, choice_record, hf, register, rid


def _elem(c: dict) -> str:
    try:
        attrs = json.loads(c.get("attributes") or "{}")
    except ValueError:
        attrs = {}
    bits = [c.get("tag", "")]
    for k in ("aria_label", "aria-label", "title", "alt", "placeholder", "name", "value", "type", "role", "id", "class"):
        if attrs.get(k):
            bits.append(f"{k}={str(attrs[k])[:40]}")
    return " ".join(bits)[:160]


def mind2web(split, cap, rng):
    ds = hf("osunlp/Mind2Web", split="train")
    n = 0
    for r in ds:
        prev = []
        for a, rep in zip(r["actions"], r["action_reprs"]):
            if a["pos_candidates"] and n < cap:
                pos = _elem(a["pos_candidates"][0])
                negs = [x for x in dict.fromkeys(_elem(c) for c in a["neg_candidates"]) if x != pos]
                opts = [pos] + rng.sample(negs, min(len(negs), rng.randint(3, 9)))
                rng.shuffle(opts)
                op = a["operation"]["op"]
                val = a["operation"].get("value") or ""
                state = {"website": f"{r['website']} ({r['domain']})", "task": r["confirmed_task"],
                         "actions_so_far": prev[-4:], "next_operation": op + (f" {val!r}" if val else "")}
                crit = {f"element_{i + 1}": o for i, o in enumerate(opts)}
                yield choice_record(rid("m2w", r["annotation_id"], len(prev)), state, "Which page element should the next operation target?",
                                    crit, f"element_{opts.index(pos) + 1}", area="agents")
                n += 1
            prev.append(rep)
        if n >= cap:
            return


register(DatasetSpec("mind2web", mind2web, ("train",), "osunlp/Mind2Web", "cc-by-4.0", "agents"))
