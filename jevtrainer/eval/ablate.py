"""Modality ablations applied to the records of a benchmark at evaluation time.

    none           records as they are
    mute           no sound at all: videos lose their own sound track, standalone audio items are dropped
    black          every video frame (and image) is replaced by a black picture of the same size; sound is kept
    shuffle_audio  every sound (a video's track or a standalone audio item) is swapped for the sound of another record

The transform returns *new* records (the cached dataset files and the input records are not touched).
A record without the relevant modality passes through unchanged, so the same switch can run over a mixed suite.
"""

from __future__ import annotations

import copy
import random
import re
from pathlib import Path

from jevtrainer.schema import Record

ABLATIONS = ("none", "mute", "black", "shuffle_audio")
_DEFAULT_SIZE = (448, 252)
_AUDIO_TAG = re.compile(r"<audio:\d+>")


def _strip_audio_tags(state):
    if isinstance(state, str):
        return _AUDIO_TAG.sub("", state)
    if isinstance(state, dict):
        return {k: _strip_audio_tags(v) for k, v in state.items()}
    if isinstance(state, list):
        return [_strip_audio_tags(v) for v in state]
    return state


def _mute(r: Record) -> Record:
    media = []
    for m in r.media:
        if m["type"] == "audio":
            continue
        if m["type"] == "video":
            m = {k: v for k, v in m.items() if k != "audio"}
            m["mute"] = True  # video files decoded on the fly: media.py then skips their sound
        media.append(m)
    out = copy.copy(r)
    out.media = media
    out.state = _strip_audio_tags(r.state)
    out.meta = {**r.meta, "ablate": "mute"}
    return out


def _frame_size(ref: str, root: str | None) -> tuple[int, int]:
    try:
        from PIL import Image

        p = Path(ref)
        with Image.open(p if p.is_absolute() or not root else Path(root) / p) as im:
            return im.size
    except Exception:
        return _DEFAULT_SIZE


def black_frame(size: tuple[int, int], directory: str | Path) -> str:
    """Path of an all-black JPEG of `size` (written once, then reused)."""
    from PIL import Image

    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"black_{size[0]}x{size[1]}.jpg"
    if not p.exists():
        Image.new("RGB", size, (0, 0, 0)).save(p, quality=90)
    return str(p)


def _black(r: Record, directory: str | Path) -> Record:
    root = r.meta.get("media_root") or r.meta.get("image_root")
    media = []
    for m in r.media:
        if m["type"] == "video":
            m = dict(m)
            if m.get("frames"):
                black = black_frame(_frame_size(m["frames"][0], root), directory)
                m["frames"] = [black] * len(m["frames"])
            else:
                m["black"] = True  # decoded on the fly: media.py blacks the frames out
        elif m["type"] == "image":
            m = {**m, "path": black_frame(_frame_size(m["path"], root), directory)}
        media.append(m)
    out = copy.copy(r)
    out.media = media
    out.meta = {**r.meta, "ablate": "black"}
    return out


def _audio_slots(records: list[Record]) -> list[tuple[int, int]]:
    """(record index, media index) of every item that carries an explicit sound file."""
    slots = []
    for i, r in enumerate(records):
        for j, m in enumerate(r.media):
            if (m["type"] == "audio" and m.get("path")) or (m["type"] == "video" and m.get("audio")):
                slots.append((i, j))
    return slots


def _audio_ref(m: dict) -> str:
    return m["path"] if m["type"] == "audio" else m["audio"]


def _shuffle_audio(records: list[Record], seed: int) -> list[Record]:
    slots = _audio_slots(records)
    refs = {s: _audio_ref(records[s[0]].media[s[1]]) for s in slots}
    if len({*refs.values()}) < 2:
        return [_tag(r, "shuffle_audio") for r in records]  # nothing to swap with
    order = slots[:]
    random.Random(seed).shuffle(order)
    new: dict[tuple[int, int], str] = {}
    n = len(order)
    for k, s in enumerate(order):
        for step in range(1, n):  # next slot in the cycle that carries different sound
            cand = refs[order[(k + step) % n]]
            if cand != refs[s]:
                new[s] = cand
                break
    out = []
    for i, r in enumerate(records):
        mine = {j: new[(i, j)] for j in range(len(r.media)) if (i, j) in new}
        r2 = copy.copy(r)
        r2.media = [dict(m) for m in r.media]
        for j, ref in mine.items():
            if r2.media[j]["type"] == "audio":
                r2.media[j]["path"] = ref
            else:
                r2.media[j]["audio"] = ref
            r2.media[j].pop("start", None)
        r2.meta = {**r.meta, "ablate": "shuffle_audio"}
        out.append(r2)
    return out


def _tag(r: Record, mode: str) -> Record:
    out = copy.copy(r)
    out.meta = {**r.meta, "ablate": mode}
    return out


def ablate_records(records: list[Record], mode: str = "none", seed: int = 0, black_dir: str | Path | None = None) -> list[Record]:
    """Records with the modality removed / replaced as `mode` says (see module doc)."""
    if mode not in ABLATIONS:
        raise ValueError(f"ablate must be one of {ABLATIONS}, got {mode!r}")
    if mode == "none":
        return records
    if mode == "mute":
        return [_mute(r) if r.media else r for r in records]
    if mode == "black":
        if black_dir is None:
            from jevtrainer.data.base import cache_dir

            black_dir = cache_dir() / "ablate"
        return [_black(r, black_dir) if r.media else r for r in records]
    return _shuffle_audio(records, seed)
