"""Helpers for the business-oriented video sets (AI-video detection, screen recordings, safety, short video).

Builds on `avkit` / `vidkit` (both untouched). Two things the other helpers do not cover:

* `stream_tar_videos`: read a tar of a hub repo *over HTTP* (no local copy of the 1 GB shard), convert the first
  `quota` accepted videos and stop; a dropped connection restarts the stream.
* `stream_rar`: pipe the first N MB of a rar archive through `bsdtar -x` (rar cannot be listed remotely, but it
  extracts file by file in order, so a prefix of the archive is a usable sample).

`deadline(minutes)` gives a wall-clock limit so that a slow mirror yields a partial set instead of hanging.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from jevtrainer.data import avkit, vidkit

WORKERS = int(os.environ.get("JEVTRAINER_AV_WORKERS", "8"))
BUDGET_MIN = float(os.environ.get("JEVTRAINER_DL_MINUTES", "35"))


def deadline(minutes: float | None = None) -> float:
    return time.time() + 60 * (BUDGET_MIN if minutes is None else minutes)


def stream_tar_videos(repo: str, filename: str, name: str, accept, quota: int, workers: int = 2, retries: int = 4,
                      stop_at: float | None = None, **kw) -> dict[str, tuple]:
    """Convert the first `quota` accepted videos of one tar in a hub repo; the tar is never stored.

    `accept(stem, sidecar_bytes | None, member_name) -> (key, extra) | None`; a `<stem>.json` member that comes
    before its video is passed as sidecar. Returns `{key: (extra, media item)}`.
    """
    import tarfile

    from huggingface_hub import HfFileSystem

    tmp = avkit.raw_dir(name) / ("tar_" + avkit.rid_(repo, filename))
    tmp.mkdir(parents=True, exist_ok=True)
    done: dict[str, tuple] = {}
    taken: dict[str, object] = {}

    def work(key, extra, path):
        try:
            item = vidkit.cached_item(name, key) or vidkit.finish(name, key, path, **kw)
        except Exception:  # noqa: BLE001
            item = None
        finally:
            path.unlink(missing_ok=True)
        if item:
            done[key] = (extra, item)

    for attempt in range(retries):
        try:
            side: dict[str, bytes] = {}
            with HfFileSystem().open(f"datasets/{repo}/{filename}", "rb") as f, tarfile.open(fileobj=f, mode="r|*") as t, \
                    ThreadPoolExecutor(workers) as ex:
                futs = []
                for m in t:
                    if not m.isfile():
                        continue
                    if stop_at and time.time() > stop_at:
                        break
                    stem, ext = os.path.splitext(os.path.basename(m.name))
                    ext = ext.lower()
                    if ext == ".json":
                        side[stem] = t.extractfile(m).read()
                        continue
                    if ext not in (".mp4", ".mov", ".avi", ".webm", ".mkv"):
                        continue
                    if len(taken) >= quota:
                        break
                    acc = accept(stem, side.pop(stem, None), m.name)
                    if not acc or acc[0] in taken:
                        continue
                    key, extra = acc
                    taken[key] = extra
                    if vidkit.cached_item(name, key):
                        done[key] = (extra, vidkit.cached_item(name, key))
                        continue
                    dst = tmp / f"{key}{ext}"
                    dst.write_bytes(t.extractfile(m).read())
                    futs.append(ex.submit(work, key, extra, dst))
                    while sum(not x.done() for x in futs) > 3 * workers:
                        time.sleep(0.05)
                for x in futs:
                    x.result()
            break
        except Exception as e:  # noqa: BLE001  dropped connection: restart, what is converted is reused
            print(f"[bizkit] {filename}: {type(e).__name__}: {str(e)[:100]}; retry {attempt + 1}", flush=True)
            time.sleep(6)
            taken = {k: v for k, v in taken.items() if k in done}
    shutil.rmtree(tmp, ignore_errors=True)
    return done


def stream_rar(repo: str, filename: str, name: str, max_mb: int, dest: str, stop_at: float | None = None,
               max_files: int | None = None) -> Path:
    """Pipe the first `max_mb` MB (or until `max_files` files exist) of an archive through `bsdtar -x` into raw/<name>/<dest>.

    Files appear in archive order; the newest one is cut off, so callers drop the last-modified file.
    """
    from huggingface_hub import HfFileSystem

    out = avkit.raw_dir(name) / dest
    out.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen([shutil.which("bsdtar") or "bsdtar", "-xf", "-", "-C", str(out)],
                         stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    n = 0
    try:
        with HfFileSystem().open(f"datasets/{repo}/{filename}", "rb") as f:
            while n < max_mb << 20 and not (stop_at and time.time() > stop_at):
                b = f.read(1 << 20)
                if not b:
                    break
                p.stdin.write(b)
                n += len(b)
                if max_files and (n >> 20) % 4 == 0 and sum(1 for x in out.rglob("*") if x.is_file()) > max_files:
                    break
    except (BrokenPipeError, OSError):
        pass
    finally:
        try:
            p.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)
        p.kill()
    return out


def complete_files(root: Path, exts=(".mp4", ".mov", ".avi", ".webm", ".mkv")) -> list[Path]:
    """Video files below root except the most recently written one (probably truncated)."""
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: p.stat().st_mtime)
    return files[:-1]
