"""Helpers for general-video converters (action, procedure, temporal, quality, caption matching).

Builds on `avkit` (which stays untouched): local-file conversion, streaming conversion of tar / tar.gz archives that
may be only partly downloaded, rate-limit tolerant hub listings and a few record builders shared by the converters.
"""

from __future__ import annotations

import io
import os
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Iterator

from jevtrainer.data import avkit
from jevtrainer.data.base import mcq_record

WORKERS = int(os.environ.get("JEVTRAINER_AV_WORKERS", "8"))
Q_ACTION = "Which option best describes what happens in the video?"


# ---- hub access that survives the mirror's 429s --------------------------------------------------------
def retry(fn: Callable, tries: int = 6, wait: float = 8.0):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if i == tries - 1 or not any(s in str(e) for s in ("429", "Too Many", "timed out", "Timeout", "Connection")):
                raise
            time.sleep(wait * (i + 1))


def listing(repo: str, prefix: str = "") -> set[str]:
    from huggingface_hub import HfApi

    return {f for f in retry(lambda: HfApi().list_repo_files(repo, repo_type="dataset")) if f.startswith(prefix)}


def get_file(repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download

    return retry(lambda: hf_hub_download(repo, filename, repo_type="dataset"))


def read_json(repo: str, filename: str):
    import json

    with open(get_file(repo, filename), encoding="utf8") as f:
        return json.load(f)


def read_jsonl(repo: str, filename: str) -> list:
    import json

    with open(get_file(repo, filename), encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_csv(repo: str, filename: str, **kw):
    import pandas as pd

    return pd.read_csv(get_file(repo, filename), **kw)


def http_text(url: str, name: str, filename: str) -> str:
    """Small text file from the web (annotation files that live on GitHub), cached under raw/<name>."""
    import urllib.request

    p = avkit.raw_dir(name) / filename
    if not p.exists() or p.stat().st_size == 0:
        data = retry(lambda: urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "jevtrainer"}), timeout=60).read())
        p.write_bytes(data)
    return p.read_text(encoding="utf8")


def read_parquet(repo: str, filename: str):
    import pandas as pd

    return pd.read_parquet(get_file(repo, filename))


# ---- converting already-local files ---------------------------------------------------------------------
def cached_item(name: str, key: str) -> dict | None:
    """The media item of a clip converted by an earlier run, or None."""
    out = avkit.media_dir(name, key)
    files = sorted(out.glob("f*.jpg"))
    if len(files) < 2:
        return None
    dur = avkit.probe_cache(out)
    item = {"type": "video", "frames": [str(f) for f in files], "fps": len(files) / max(dur, 0.1), "duration": dur}
    if (out / "a.flac").exists() and (out / "a.flac").stat().st_size > 0:
        item["audio"] = str(out / "a.flac")
    return item


def finish(name: str, key: str, src: Path, **kw) -> dict | None:
    """video_item + remember the duration so that cached_item can rebuild the item later."""
    item = avkit.video_item(src, name, key, **kw)
    if item:
        (avkit.media_dir(name, key) / "meta.txt").write_text(str(item["duration"]))
    return item


def convert_local(name: str, jobs: dict[str, dict], workers: int = WORKERS, delete_raw: bool = True, **kw) -> dict[str, dict]:
    """`jobs`: {key: {"src": Path, "start": s, "end": s}} -> {key: media item}. Raw files are deleted when done
    (unless another job still needs the same source: sources are deleted after the last job that uses them)."""
    users: dict[Path, int] = {}
    for j in jobs.values():
        users[j["src"]] = users.get(j["src"], 0) + 1
    lock = threading.Lock()

    def one(kv):
        key, j = kv
        item = cached_item(name, key)
        try:
            if item is None and Path(j["src"]).exists():
                extra = {k: j[k] for k in ("start", "end") if j.get(k) is not None}
                item = finish(name, key, Path(j["src"]), **{**kw, **extra})
        except Exception:  # noqa: BLE001
            item = None
        finally:
            with lock:
                users[j["src"]] -= 1
                last = users[j["src"]] == 0
            if delete_raw and last:
                Path(j["src"]).unlink(missing_ok=True)
        return key, item

    with ThreadPoolExecutor(workers) as ex:
        return {k: v for k, v in ex.map(one, jobs.items()) if v}


def convert_hub(repo: str, name: str, files: dict[str, str], workers: int = WORKERS, **kw) -> dict[str, dict]:
    """Like avkit.convert_videos but retries rate limits and keeps durations for the cache."""

    def one(kv):
        key, fn = kv
        item = cached_item(name, key)
        if item:
            return key, item
        src = None
        for attempt in range(4):
            src = avkit.fetch_file(repo, fn, name)
            if src is not None:
                break
            time.sleep(10 * (attempt + 1))  # mirror rate limit
        if src is None:
            return key, None
        try:
            return key, finish(name, key, src, **kw)
        except Exception:  # noqa: BLE001
            return key, None
        finally:
            Path(src).unlink(missing_ok=True)

    def safe(kv):
        try:
            return one(kv)
        except Exception:  # noqa: BLE001
            return kv[0], None

    with ThreadPoolExecutor(workers) as ex:
        return {k: v for k, v in ex.map(safe, files.items()) if v}


def fetch_many(repo: str, name: str, files: dict[str, str], workers: int = 6) -> dict[str, Path]:
    """Download `{key: file in repo}` in parallel into raw/<name> (retrying rate limits); returns the keys that arrived."""

    def one(kv):
        key, fn = kv
        for attempt in range(4):
            src = avkit.fetch_file(repo, fn, name)
            if src is not None:
                return key, src
            time.sleep(10 * (attempt + 1))
        return key, None

    with ThreadPoolExecutor(workers) as ex:
        return {k: v for k, v in ex.map(one, files.items()) if v}


# ---- archives: concatenated / partly downloaded tar(.gz) streams ----------------------------------------
class ConcatReader(io.RawIOBase):
    """Read several files back to back (the pieces of a split archive); `drop` deletes a piece once it is consumed."""

    def __init__(self, paths: list[Path], drop: bool = False):
        self.paths, self.drop, self.i, self.f = [Path(p) for p in paths], drop, 0, None

    def readable(self):
        return True

    def readinto(self, b):
        while True:
            if self.f is None:
                if self.i >= len(self.paths):
                    return 0
                self.f = open(self.paths[self.i], "rb")
            n = self.f.readinto(b)
            if n:
                return n
            self.f.close()
            self.f = None
            if self.drop:
                self.paths[self.i].unlink(missing_ok=True)
            self.i += 1


def tar_members(paths: list[Path], keep: Callable[[str], bool], mode: str = "r|gz", drop: bool = False,
                done: Callable[[], bool] | None = None) -> Iterator[tuple[str, bytes]]:
    """(member name, bytes) of the members of a (possibly truncated) tar stream for which keep(name) holds."""
    import zlib

    reader = io.BufferedReader(ConcatReader(paths, drop), 1 << 22)
    try:
        with tarfile.open(fileobj=reader, mode=mode) as tf:
            for m in tf:
                if m.isfile() and keep(m.name):
                    f = tf.extractfile(m)
                    if f is not None:
                        yield m.name, f.read()
    except (EOFError, tarfile.ReadError, zlib.error, OSError):
        return  # truncated download: what was read so far is kept


def convert_stream(name: str, stream: Iterable[tuple[str, str, bytes]], workers: int = WORKERS, **kw) -> dict[str, dict]:
    """stream yields (key, file name, bytes); each is written to a scratch file, converted and deleted."""
    tmp = avkit.raw_dir(f"{name}_tmp")
    sem = threading.Semaphore(workers * 2)
    out: dict[str, dict] = {}

    def one(key, fn, data):
        p = tmp / f"{key}{Path(fn).suffix or '.mp4'}"
        try:
            p.write_bytes(data)
            item = cached_item(name, key) or finish(name, key, p, **kw)
            if item:
                out[key] = item
        except Exception:  # noqa: BLE001
            pass
        finally:
            p.unlink(missing_ok=True)
            sem.release()

    with ThreadPoolExecutor(workers) as ex:
        for key, fn, data in stream:
            sem.acquire()
            ex.submit(one, key, fn, data)
    return out


# ---- record builders ------------------------------------------------------------------------------------
def video_state(question: str, **extra) -> dict:
    return {"clip": "<video:1>", "question": question, **extra}


def option_record(id: str, media: dict, question: str, gold: str, others: list[str], rng, instructions: str = Q_ACTION,
                  n_opts: int = 4, state_extra: dict | None = None, area: str = "video", **meta):
    """One video, one question; options = `gold` + (n_opts - 1) of `others`, shuffled. Returns a Record or None."""
    pool = [o for o in dict.fromkeys(others) if o != gold]
    if not pool:
        return None
    options = [gold, *rng.sample(pool, min(n_opts - 1, len(pool)))]
    order = list(range(len(options)))
    rng.shuffle(order)
    rec = mcq_record(id, video_state(question, **(state_extra or {})), instructions, [options[i] for i in order], order.index(0),
                     area=area, **meta)
    if rec:
        rec.media = [media]
    return rec
