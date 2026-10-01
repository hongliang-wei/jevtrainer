"""Video helpers on top of `avkit` for archives (zip / tar.gz) and for deterministic splits.

`avkit.convert_videos` fetches one repo file per clip. Many corpora ship as one big archive instead:
`convert_zip_videos` / `convert_tar_videos` pull the wanted members out one by one, cut frames + sound,
and delete the extracted member right away.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from jevtrainer.data import avkit


def hash_bucket(*parts, mod: int = 100) -> int:
    """Stable bucket in [0, mod) of an id; used to cut a fixed test split out of a one-split corpus."""
    return int(hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) % mod


def cached_item(name: str, key: str) -> dict | None:
    """Media item of a clip converted by an earlier run (None when absent)."""
    out = avkit.media_dir(name, key)
    files = sorted(out.glob("f*.jpg"))
    if len(files) < 2:
        return None
    dur = avkit.probe_cache(out)
    item = {"type": "video", "frames": [str(f) for f in files], "fps": len(files) / max(dur, 0.1), "duration": dur}
    if (out / "a.flac").exists() and (out / "a.flac").stat().st_size > 0:
        item["audio"] = str(out / "a.flac")
    return item


def _norm(wanted: dict) -> dict[str, tuple[str, dict]]:
    out = {}
    for member, v in wanted.items():
        key, kw = (v, {}) if isinstance(v, str) else v
        out[member] = (key, kw)
    return out


def _convert_file(src: Path, name: str, key: str, delete: bool, kw: dict) -> dict | None:
    try:
        item = avkit.video_item(src, name, key, **kw)
        if item:
            (avkit.media_dir(name, key) / "meta.txt").write_text(str(item["duration"]))
        return item
    finally:
        if delete:
            src.unlink(missing_ok=True)


def convert_zip_videos(archive: Path, name: str, wanted: dict, workers: int = 8, delete_raw: bool = True, **kw) -> dict[str, dict]:
    """`wanted`: {member name in the zip: key | (key, per-clip video_item kwargs)} -> {key: media item}."""
    import zipfile

    tmp = avkit.raw_dir(name) / "_x"
    tmp.mkdir(parents=True, exist_ok=True)
    wanted = _norm(wanted)

    def one(kv):
        member, (key, okw) = kv
        c = cached_item(name, key)
        if c:
            return key, c
        dst = tmp / f"{key}{Path(member).suffix}"
        try:
            with zipfile.ZipFile(archive) as z, z.open(member) as s, open(dst, "wb") as d:
                shutil.copyfileobj(s, d, 1 << 20)
        except Exception:
            dst.unlink(missing_ok=True)
            return key, None
        return key, _convert_file(dst, name, key, delete_raw, {**kw, **okw})

    with ThreadPoolExecutor(workers) as ex:
        return {k: v for k, v in ex.map(one, wanted.items()) if v}


def convert_tar_videos(archive: Path, name: str, wanted: dict, workers: int = 8, delete_raw: bool = True,
                       by_basename: bool = True, **kw) -> dict[str, dict]:
    """One streaming pass over a (compressed) tar; `wanted`: {basename (or full member name): key | (key, kwargs)}."""
    import tarfile

    tmp = avkit.raw_dir(name) / "_x"
    tmp.mkdir(parents=True, exist_ok=True)
    wanted = _norm(wanted)
    sem = threading.BoundedSemaphore(workers * 2)
    done: dict[str, dict] = {}
    futs = []

    def job(dst, key, okw):
        try:
            return key, _convert_file(dst, name, key, delete_raw, {**kw, **okw})
        finally:
            sem.release()

    with ThreadPoolExecutor(workers) as ex, tarfile.open(archive, "r|*") as t:
        for m in t:
            if not m.isfile():
                continue
            w = wanted.get(os.path.basename(m.name) if by_basename else m.name)
            if w is None:
                continue
            key, okw = w
            c = cached_item(name, key)
            if c:
                done[key] = c
                continue
            dst = tmp / f"{key}{Path(m.name).suffix}"
            src = t.extractfile(m)
            if src is None:
                continue
            with open(dst, "wb") as d:
                shutil.copyfileobj(src, d, 1 << 20)
            sem.acquire()
            futs.append(ex.submit(job, dst, key, okw))
        for f in futs:
            k, v = f.result()
            if v:
                done[k] = v
    return done


def balanced_take(rows: list, key, cap: int, rng) -> list:
    """At most cap rows, drawn round-robin over the classes key(row) so that rare classes are not drowned out."""
    from collections import defaultdict

    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    for g in groups.values():
        rng.shuffle(g)
    order = sorted(groups)
    rng.shuffle(order)
    out, i = [], 0
    while len(out) < cap and any(groups.values()):
        for k in order:
            if groups[k] and len(out) < cap:
                out.append(groups[k].pop())
        i += 1
    rng.shuffle(out)
    return out
