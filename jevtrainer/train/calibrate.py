"""Temperature scaling: pick T minimising held-out NLL on a log grid. Argmax never changes."""

from __future__ import annotations

import torch


def pad(logits: list[torch.Tensor], ks: list[int]) -> torch.Tensor:
    z = torch.full((len(logits), max(ks)), float("-inf"))
    for i, (l, k) in enumerate(zip(logits, ks)):
        z[i, :k] = l[:k].float().cpu()
    return z


def fit_temperature(logits: list[torch.Tensor], ks: list[int], targets: list[int], lo=0.25, hi=8.0, steps=161) -> float:
    if not logits:
        return 1.0
    z, y = pad(logits, ks), torch.tensor(targets)
    grid = torch.logspace(torch.log10(torch.tensor(lo)), torch.log10(torch.tensor(hi)), steps)
    nll = torch.stack([-torch.log_softmax(z / t, -1).gather(1, y[:, None]).mean() for t in grid])
    return float(grid[int(nll.argmin())])


def fit(outputs: list[dict], per_type: bool = False) -> dict[str, float]:
    """outputs: dicts with logits, k, target, type (see eval.runner.collect)."""
    rows = [o for o in outputs if o["target"] is not None]
    temps = {"default": fit_temperature([o["logits"] for o in rows], [o["k"] for o in rows], [o["target"] for o in rows])}
    if per_type:
        for qt in ("choice", "score", "noul"):
            sub = [o for o in rows if o["type"] == qt]
            if len(sub) >= 50:
                temps[qt] = fit_temperature([o["logits"] for o in sub], [o["k"] for o in sub], [o["target"] for o in sub])
    return temps
