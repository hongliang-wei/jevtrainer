"""Decision-level metrics. `outputs` are dicts from eval.runner.collect:
{"logits", "k", "target", "soft", "type", "id", "name"}.
"""

from __future__ import annotations

from collections import defaultdict

import torch

from jevtrainer.predict import probs, temperature_for


def _probs(outputs, temps) -> list[torch.Tensor]:
    return [probs(o["logits"], o["k"], temperature_for(temps, o["type"])) for o in outputs]


def ece(conf: list[float], correct: list[bool], bins: int = 10) -> float:
    total, err = len(conf), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if (lo < c <= hi) or (b == 0 and c == 0)]
        if idx:
            acc = sum(correct[i] for i in idx) / len(idx)
            avg = sum(conf[i] for i in idx) / len(idx)
            err += len(idx) / total * abs(acc - avg)
    return err


def macro_f1(pred: list[int], gold: list[int]) -> float:
    classes = sorted(set(gold) | set(pred))
    f1s = []
    for c in classes:
        tp = sum(p == c and g == c for p, g in zip(pred, gold))
        fp = sum(p == c and g != c for p, g in zip(pred, gold))
        fn = sum(p != c and g == c for p, g in zip(pred, gold))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(f1s) / len(f1s) if f1s else 0.0


def summarize(outputs: list[dict], temps=None, metric: str = "accuracy") -> dict:
    rows = [o for o in outputs if o["target"] is not None]
    if not rows:
        return soft_only(outputs, temps)
    ps = _probs(rows, temps)
    pred = [int(p.argmax()) for p in ps]
    gold = [o["target"] for o in rows]
    correct = [a == b for a, b in zip(pred, gold)]
    conf = [float(p.max()) for p in ps]
    brier = sum(float(((p - torch.nn.functional.one_hot(torch.tensor(g), p.numel()).float()) ** 2).sum()) for p, g in zip(ps, gold)) / len(rows)
    nll = -sum(float(torch.log(p[g].clamp_min(1e-12))) for p, g in zip(ps, gold)) / len(rows)
    acc = sum(correct) / len(rows)
    chance = sum(1 / o["k"] for o in rows) / len(rows)
    out = {
        "n": len(rows),
        "accuracy": acc,
        "skill": max(0.0, (acc - chance) / (1 - chance)) if chance < 1 else 0.0,
        "chance": chance,
        "ece": ece(conf, correct),
        "brier": brier,
        "nll": nll,
    }
    if metric == "macro_f1":
        by_q = defaultdict(lambda: ([], []))
        for o, p, g in zip(rows, pred, gold):
            by_q[o["name"]][0].append(p)
            by_q[o["name"]][1].append(g)
        out["macro_f1"] = sum(macro_f1(*v) for v in by_q.values()) / len(by_q)
    if metric == "case_exact":
        cases = defaultdict(list)
        for o, c in zip(rows, correct):
            cases[o["id"]].append(c)
        out["case_exact"] = sum(all(v) for v in cases.values()) / len(cases)
    soft = [(p, o["soft"]) for p, o in zip(ps, rows) if o.get("soft") is not None]
    if soft:
        out["tvd"] = sum(0.5 * float((p - s).abs().sum()) for p, s in soft) / len(soft)
        out["soft_accuracy"] = sum(float(s[int(p.argmax())]) for p, s in soft) / len(soft)
    out["score"] = out.get(metric, acc)
    return out


def soft_only(outputs: list[dict], temps=None) -> dict:
    """Known-distribution items: compare predicted and gold distributions (no sampled outcome)."""
    rows = [o for o in outputs if o.get("soft") is not None]
    if not rows:
        return {"n": 0}
    ps = _probs(rows, temps)
    tvd = [0.5 * float((p - o["soft"]).abs().sum()) for p, o in zip(ps, rows)]
    # expected multiclass Brier under the gold distribution: sum_y q(y) * ||p - e_y||^2
    ebrier = [float((o["soft"] * ((p[None, :] - torch.eye(p.numel())) ** 2).sum(1)).sum()) for p, o in zip(ps, rows)]
    conf = [float(p.max()) for p in ps]
    exp_correct = [float(o["soft"][int(p.argmax())]) for p, o in zip(ps, rows)]
    bins = [sum(1 for c in conf if lo / 10 < c <= (lo + 1) / 10) for lo in range(10)]
    e = sum(
        abs(sum(x for c, x in zip(conf, exp_correct) if lo / 10 < c <= (lo + 1) / 10) - sum(c for c in conf if lo / 10 < c <= (lo + 1) / 10)) / len(rows)
        for lo in range(10) if bins[lo]
    )
    return {"n": len(rows), "tvd": sum(tvd) / len(rows), "expected_brier": sum(ebrier) / len(rows), "expected_ece": e,
            "score": 1 - sum(tvd) / len(rows)}
