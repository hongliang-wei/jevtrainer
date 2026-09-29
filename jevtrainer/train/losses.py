"""Per-question losses. Each takes the option logits of one question and returns a scalar.

`target` is the gold index; `soft` optionally a probability vector over the options.
Logits may carry one extra trailing entry (the marker readout's "rest of vocabulary"),
which only the cross-entropy losses use.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from jevtrainer.registry import LOSSES


def _target_dist(n: int, target: int | None, soft: torch.Tensor | None, device) -> torch.Tensor:
    if soft is not None:
        dist = torch.zeros(n, device=device)
        dist[: soft.numel()] = soft
        return dist
    return F.one_hot(torch.tensor(target, device=device), n).float()


@LOSSES.register("ce")
def ce(logits, k, target, soft=None, smoothing: float = 0.0, **_):
    logp = torch.log_softmax(logits.float(), -1)
    dist = _target_dist(logits.numel(), target, soft, logits.device)
    if smoothing:
        dist = dist * (1 - smoothing)
        dist[:k] += smoothing / k
    return -(dist * logp).sum()


@LOSSES.register("ce_smooth")
def ce_smooth(logits, k, target, soft=None, smoothing: float = 0.1, **_):
    return ce(logits, k, target, soft, smoothing=smoothing)


@LOSSES.register("brier")
def brier(logits, k, target, soft=None, **_):
    p = torch.softmax(logits[:k].float(), -1)
    return ((p - _target_dist(k, target, soft, logits.device)) ** 2).sum()


@LOSSES.register("ce_brier")
def ce_brier(logits, k, target, soft=None, **_):
    return ce(logits, k, target, soft) + brier(logits, k, target, soft)


def rps(logits, k, target, soft=None) -> torch.Tensor:
    """Ranked probability score for ordered (score) questions."""
    p = torch.softmax(logits[:k].float(), -1).cumsum(0)
    q = _target_dist(k, target, soft, logits.device).cumsum(0)
    return ((p - q) ** 2).sum() / max(k - 1, 1)
