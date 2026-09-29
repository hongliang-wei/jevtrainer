"""Turn option logits into typed answers: choice / score / noul."""

from __future__ import annotations

import torch

from jevtrainer.schema import Question


def temperature_for(temps: dict[str, float] | float | None, qtype: str) -> float:
    if temps is None:
        return 1.0
    if isinstance(temps, (int, float)):
        return float(temps)
    return float(temps.get(qtype, temps.get("default", 1.0)))


def probs(logits: torch.Tensor, k: int, t: float = 1.0) -> torch.Tensor:
    return torch.softmax(logits[:k].float() / t, -1)


def confidence(p: torch.Tensor) -> float:
    k = p.numel()
    return float((p.max() - 1 / k) / (1 - 1 / k))


def answer(q: Question, p: torch.Tensor) -> dict:
    labels = q.labels()
    dist = {l: float(v) for l, v in zip(labels, p.tolist())}
    if q.type == "noul":
        return {"type": "noul", "noul": dist["yes"], "probabilities": dist}
    if q.type == "score":
        levels = torch.arange(len(labels), dtype=p.dtype, device=p.device)
        legend = dict(zip(labels, q.descriptions()))
        return {"type": "score", "score": float((levels * p).sum()), "probabilities": dist, "confidence": confidence(p), "legend": legend}
    return {"type": "choice", "choice": labels[int(p.argmax())], "probabilities": dist, "confidence": confidence(p)}
