"""Weighted mixture of datasets -> one shuffled training list, with leakage checks."""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from pathlib import Path

from jevtrainer.data.base import DatasetSpec, load
from jevtrainer.registry import DATASETS
from jevtrainer.schema import Record


@dataclass
class Source:
    name: str
    split: str = "train"
    max_samples: int | None = None
    weight: float = 1.0


def _norm(text: str) -> str:
    return hashlib.sha1(re.sub(r"\W+", " ", text.lower()).strip().encode()).hexdigest()


def _image_digest(ref: str) -> str:
    try:
        return hashlib.sha1(Path(ref).read_bytes()).hexdigest()
    except OSError:
        return ref


def state_keys(r: Record) -> set[str]:
    """The whole state plus every long text field, so a test request inside a new tool catalog still matches.
    With images, the same text over a different picture is a different item."""
    if r.images:
        return {_norm(r.state_text() + "|" + "|".join(_image_digest(i) for i in r.images))}
    keys = {_norm(r.state_text())}
    if isinstance(r.state, dict):
        for v in r.state.values():
            if isinstance(v, str) and len(v) >= 40:
                keys.add(_norm(v))
    return keys


def check_trainable(name: str) -> None:
    if name.endswith(".jsonl"):
        return
    spec: DatasetSpec = DATASETS.get(name)
    if spec.eval_only:
        raise ValueError(f"dataset '{name}' is eval_only and cannot be used for training")


def build_mixture(sources: list[Source], seed: int = 0, exclude: list[Record] | None = None) -> tuple[list[Record], dict]:
    """Returns (records, report). `exclude` records (the eval sets) are removed by normalised state."""
    banned = set().union(*(state_keys(r) for r in exclude)) if exclude else set()
    rng = random.Random(seed)
    records, report = [], {}
    for s in sources:
        check_trainable(s.name)
        rs = load(s.name, s.split, s.max_samples, seed)
        kept = [r for r in rs if not (state_keys(r) & banned)]
        removed = len(rs) - len(kept)
        if s.weight != 1.0 and kept:
            n = int(round(len(kept) * s.weight))
            kept = [kept[i % len(kept)] for i in range(n)] if n > len(kept) else rng.sample(kept, n)
        report[s.name] = {"loaded": len(rs), "overlap_removed": removed, "used": len(kept)}
        records.extend(kept)
    rng.shuffle(records)
    return records, report


def split_holdout(records: list[Record], frac: float, seed: int = 0) -> tuple[list[Record], list[Record]]:
    if frac <= 0:
        return records, []
    rng = random.Random(seed + 1)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    n = max(1, int(len(records) * frac))
    hold = set(idx[:n])
    return [r for i, r in enumerate(records) if i not in hold], [r for i, r in enumerate(records) if i in hold]
