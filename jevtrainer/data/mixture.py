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
    group_by: str | None = None  # meta key that splits the source into tasks (e.g. "question_type")
    per_group: int | None = None  # cap per task, so each task contributes about the same
    repeat_to: int | None = None  # a source (or task) smaller than this is repeated, at most 3x


def _norm(text: str) -> str:
    return hashlib.sha1(re.sub(r"\W+", " ", text.lower()).strip().encode()).hexdigest()


def _image_digest(ref: str) -> str:
    try:
        return hashlib.sha1(Path(ref).read_bytes()).hexdigest()
    except OSError:
        return ref


_YT = re.compile(r"^([A-Za-z0-9_-]{11})(?:_\d+){0,2}$")  # YouTube id, optionally followed by _start[_end]


def source_ids(r: Record) -> set[str]:
    """Identifiers of the source material (video / audio recording) behind a record.

    `meta["source_id"]` (a string or a list; converters write e.g. "yt:<YouTube id>" so the same video under another
    file path still matches). Without it, a media file kept under a YouTube-id-looking name yields "yt:<id>".
    """
    sid = r.meta.get("source_id")
    if sid:
        return {str(s) for s in (sid if isinstance(sid, (list, tuple)) else [sid])}
    out = set()
    for m in r.media:
        ref = m.get("path") or (m.get("frames") or [m.get("audio") or ""])[0]
        key = Path(ref).parent.name if m.get("frames") or Path(ref).stem in ("a", "i") else Path(ref).stem
        mo = _YT.match(key)
        if mo and not key.startswith("video_"):
            out.add("yt:" + mo.group(1))
    return out


def state_keys(r: Record) -> set[str]:
    """Content keys (see `_content_keys`) plus one key per source id, so the same video behind another question,
    path or split still counts as overlap."""
    return _content_keys(r) | {hashlib.sha1(("source|" + s).encode()).hexdigest() for s in source_ids(r)}


def _content_keys(r: Record) -> set[str]:
    """The whole state plus every long text field, so a test request inside a new tool catalog still matches.
    With images, the same text over a different picture is a different item."""
    if r.images:
        return {_norm(r.state_text() + "|" + "|".join(_image_digest(i) for i in r.images))}
    if r.media:  # same question over another clip is another item; files are identified by their path
        ref = "|".join(str(m.get("path") or (m.get("frames") or [""])[0]) for m in r.media)
        return {_norm(r.state_text() + "|" + ref)}
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


def repeat_up(rs: list[Record], n: int, max_factor: int = 3) -> list[Record]:
    """Repeat a small group cyclically up to n records (never more than max_factor copies)."""
    n = min(n, len(rs) * max_factor)
    return [rs[i % len(rs)] for i in range(n)]


def balance_tasks(rs: list[Record], key: str, per_group: int | None, repeat_to: int | None, rng: random.Random) -> tuple[list[Record], dict]:
    """Split by meta[key]; each task gets at most per_group records (random), small ones are repeated up to repeat_to."""
    groups: dict[str, list[Record]] = {}
    for r in rs:
        groups.setdefault(str(r.meta.get(key)), []).append(r)
    out, tasks = [], {}
    for g, items in sorted(groups.items()):
        if per_group and len(items) > per_group:
            items = rng.sample(items, per_group)
        elif repeat_to and len(items) < repeat_to:
            items = repeat_up(items, repeat_to)
        tasks[g] = len(items)
        out.extend(items)
    return out, tasks


def build_mixture(sources: list[Source], seed: int = 0, exclude: list[Record] | None = None) -> tuple[list[Record], dict]:
    """Returns (records, report). `exclude` records (the eval sets) are removed by normalised state."""
    banned = set().union(*(state_keys(r) for r in exclude)) if exclude else set()
    rng = random.Random(seed)
    records, report = [], {}
    for s in sources:
        check_trainable(s.name)
        rs = load(s.name, s.split, None if s.group_by else s.max_samples, seed)
        kept = [r for r in rs if not (state_keys(r) & banned)]
        removed = len(rs) - len(kept)
        if s.group_by:
            kept, tasks = balance_tasks(kept, s.group_by, s.per_group, s.repeat_to, rng)
            if s.max_samples and len(kept) > s.max_samples:
                kept = rng.sample(kept, s.max_samples)
            report[s.name + "#tasks"] = tasks
        elif s.repeat_to and 0 < len(kept) < s.repeat_to:
            kept = repeat_up(kept, s.repeat_to)
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
