"""Helpers for the evaluation-benchmark converters (`converters/av_bench.py`, `converters/video_bench.py`).

Builds on `avkit` / `vidkit` (untouched): reading single members out of big zip files on the hub with range requests
(no full download), a seek-based frame cutter for hour-long videos, and the record builders every benchmark shares.

Frames: the converters write 8 frames per clip (16 for the long-video sets); `jevtrainer.media` keeps
`min(len(frames), MediaOptions.video_frames)` of them, so a 16-frame item is only used at 16 frames when the evaluation
config sets `readout_options: {video_frames: 16}` (see configs/eval/video_bench.yaml).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from jevtrainer.data import avkit, vidkit
from jevtrainer.data.base import mcq_record

WORKERS = min(8, int(os.environ.get("JEVTRAINER_AV_WORKERS", "8")))
LONG_FRAMES = 16
_OPT = re.compile(r"^\s*\(?([A-Za-z])[\.\):]\s+")
Q_VIDEO = "Which option correctly answers the question about the video?"
Q_AV = "Which option correctly answers the question about the video and its sound?"
Q_AUDIO = "Which option correctly answers the question about the audio?"


# ---- option text ----------------------------------------------------------------------------------------
def strip_letter(opt: str) -> str:
    """'A. text' / '(B) text' -> 'text'."""
    return _OPT.sub("", str(opt), count=1).strip()


def letter_index(ans, n: int | None = None) -> int | None:
    """'B' / '(B)' / 'B. text' -> 1 (None when it is not a letter within n options)."""
    m = re.match(r"^\s*\(?([A-Za-z])(?:[\.\)\s]|$)", str(ans))
    if not m:
        return None
    i = ord(m.group(1).upper()) - 65
    return i if n is None or 0 <= i < n else None


def text_index(ans: str, options: list[str]) -> int | None:
    """Index of the option whose text equals `ans` (ignoring the 'A. ' prefix, case and spacing)."""
    norm = lambda s: re.sub(r"\s+", " ", strip_letter(s)).strip().lower()  # noqa: E731
    target = norm(ans)
    hits = [i for i, o in enumerate(options) if norm(o) == target]
    return hits[0] if len(hits) == 1 else None


def mcq(id: str, state, question: str, options: list[str], gold: int | None, media: list[dict], area: str, **meta):
    """mcq_record with media attached (None when the item is unusable)."""
    if gold is None or not media or any(m is None for m in media):
        return None
    meta = {k: v for k, v in meta.items() if v is not None}
    rec = mcq_record(id, state, question, options, gold, area=area, **meta)
    if rec:
        rec.media = list(media)
    return rec


def youtube_source(video_id: str | None) -> str | None:
    return f"yt:{video_id}" if video_id and re.match(r"^[A-Za-z0-9_-]{11}$", str(video_id)) else None


# ---- zip members on the hub -------------------------------------------------------------------------------
_local = threading.local()


def _zip(repo: str, filename: str):
    """A ZipFile over a hub file (range requests); one handle per thread and file."""
    import zipfile

    from huggingface_hub import HfFileSystem

    handles = _local.__dict__.setdefault("zips", {})
    key = (repo, filename)
    if key not in handles:
        handles[key] = zipfile.ZipFile(vidkit.retry(lambda: HfFileSystem().open(f"datasets/{repo}/{filename}", "rb")))
    return handles[key]


def zip_names(repo: str, filename: str) -> list[str]:
    return _zip(repo, filename).namelist()


def extract_member(repo: str, filename: str, member: str, dest: Path) -> Path | None:
    """Copy one zip member to `dest` (retrying; the thread's handle is dropped and reopened after an error)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            with _zip(repo, filename).open(member) as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out, 1 << 20)
            return dest
        except Exception:  # noqa: BLE001
            _local.__dict__.setdefault("zips", {}).pop((repo, filename), None)
            dest.unlink(missing_ok=True)
            import time

            time.sleep(5 * (attempt + 1))
    return None


# ---- hour-long videos: cut frames by seeking ---------------------------------------------------------------
def seek_video_item(src: Path, name: str, key: str, frames: int = LONG_FRAMES, max_side: int = 448, audio_s: float = 30.0,
                    keep_audio: bool = True) -> dict | None:
    """`frames` JPEGs at the centres of equal time slices (one seek each: fast on long files) + the first `audio_s` s of sound."""
    info = avkit.probe(src)
    if not info["has_video"] or info["duration"] <= 0.2:
        return None
    dur = info["duration"]
    out = avkit.media_dir(name, key)
    scale = f"scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'"
    ff = avkit.ffmpeg_exe()
    for i in range(frames):
        t = (i + 0.5) * dur / frames
        dst = out / f"f{i:02d}.jpg"
        subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-nostdin", "-ss", f"{t:.3f}", "-i", str(src), "-frames:v", "1",
                        "-vf", scale, "-q:v", "3", "-y", str(dst)], capture_output=True)
        if not dst.exists() and i:  # seeking past the last decodable frame: repeat the previous one
            shutil.copy(out / f"f{i - 1:02d}.jpg", dst)
    files = sorted(out.glob("f*.jpg"))
    if len(files) < 2:
        return None
    item = {"type": "video", "frames": [str(f) for f in files], "fps": len(files) / dur, "duration": round(dur, 3)}
    if keep_audio and info["has_audio"]:
        a = out / "a.flac"
        subprocess.run([ff, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
                        "-t", f"{audio_s:.1f}", "-y", str(a)], capture_output=True)
        if a.exists() and a.stat().st_size > 0:
            item["audio"] = str(a)
    (out / "meta.txt").write_text(str(item["duration"]))
    return item


def convert_zip_members(name: str, jobs: dict[str, dict], workers: int = WORKERS, frames: int = 8, long: bool = False,
                        **kw) -> dict[str, dict]:
    """`jobs`: {key: {"repo", "zip", "member", ["start", "end"]}} -> {key: media item}.

    Each member is read out of its zip on the hub, turned into frames + sound and deleted again. Keys converted by an
    earlier run are reused. `long=True` cuts the frames by seeking (use for videos longer than a few minutes).
    """
    def one(kv):
        key, j = kv
        item = vidkit.cached_item(name, key)
        if item and len(item["frames"]) >= frames:
            return key, item
        dest = avkit.raw_dir(name) / f"{key}{Path(j['member']).suffix or '.bin'}"
        src = extract_member(j["repo"], j["zip"], j["member"], dest)
        if src is None:
            return key, None
        try:
            if long:
                return key, seek_video_item(src, name, key, frames, **kw)
            extra = {k: j[k] for k in ("start", "end") if j.get(k) is not None}
            return key, vidkit.finish(name, key, src, frames=frames, **{**kw, **extra})
        except Exception:  # noqa: BLE001
            return key, None
        finally:
            src.unlink(missing_ok=True)

    with ThreadPoolExecutor(workers) as ex:
        return {k: v for k, v in ex.map(one, jobs.items()) if v}


def convert_long_local(name: str, jobs: dict[str, Path], frames: int = LONG_FRAMES, workers: int = WORKERS, delete_raw: bool = True,
                       **kw) -> dict[str, dict]:
    """Local long videos {key: path} -> media items (seek-based frames)."""
    def one(kv):
        key, src = kv
        item = vidkit.cached_item(name, key)
        if item and len(item["frames"]) >= frames:
            return key, item
        try:
            return key, seek_video_item(Path(src), name, key, frames, **kw)
        except Exception:  # noqa: BLE001
            return key, None
        finally:
            if delete_raw:
                Path(src).unlink(missing_ok=True)

    with ThreadPoolExecutor(workers) as ex:
        return {k: v for k, v in ex.map(one, jobs.items()) if v}


def image_frames_item(paths: list[Path], name: str, key: str, n: int = 8, fps: float = 3.0) -> dict | None:
    """A 'video' made of already extracted frame images (e.g. TVQA frame folders): n evenly spaced, resized JPEGs."""
    from PIL import Image

    if len(paths) < 2:
        return None
    pick = [paths[int((i + 0.5) * len(paths) / n)] for i in range(n)] if len(paths) > n else list(paths)
    out = avkit.media_dir(name, key)
    files = []
    for i, p in enumerate(pick):
        dst = out / f"f{i:02d}.jpg"
        im = Image.open(p).convert("RGB")
        s = 448 / max(im.size)
        if s < 1:
            im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))))
        im.save(dst, quality=90)
        files.append(str(dst))
    dur = len(paths) / fps
    (out / "meta.txt").write_text(str(dur))
    return {"type": "video", "frames": files, "fps": len(files) / dur, "duration": round(dur, 3)}

# ---- parquet shards with embedded media ------------------------------------------------------------------
def parquet_media(repo: str, files: list[str], name: str, handle, shards: int = 2, inner: int = 4, batch: int = 32,
                  workers_note: str = "") -> list:
    """Download every shard, call `handle(row)` (row = dict with all columns, binary ones as bytes) for each row and
    collect the non-None results; a shard is deleted once it is done. `handle` runs in `inner` threads per shard."""
    import pyarrow.parquet as pq

    def shard(fn):
        src = None
        for attempt in range(4):
            src = avkit.fetch_file(repo, fn, name)
            if src is not None:
                break
            import time

            time.sleep(10 * (attempt + 1))
        if src is None:
            return []
        out = []
        try:
            with ThreadPoolExecutor(inner) as ex:
                for b in pq.ParquetFile(str(src)).iter_batches(batch_size=batch):
                    out.extend(r for r in ex.map(_safe(handle), b.to_pylist()) if r is not None)
        finally:
            if not os.environ.get("JEVTRAINER_KEEP_RAW"):
                Path(src).unlink(missing_ok=True)
        return out

    with ThreadPoolExecutor(max(1, min(shards, len(files)))) as ex:
        return [r for part in ex.map(shard, files) for r in part]


def _safe(fn):
    def wrapped(x):
        try:
            return fn(x)
        except Exception:  # noqa: BLE001
            return None

    return wrapped


def blob(v) -> bytes | None:
    """Bytes of a HF Audio / Image struct ({'bytes', 'path'}) or of a plain binary column."""
    if isinstance(v, dict):
        v = v.get("bytes")
    return bytes(v) if v else None


def convert_hub_long(repo: str, name: str, files: dict[str, str], frames: int = LONG_FRAMES, workers: int = WORKERS,
                     **kw) -> dict[str, dict]:
    """Long videos that are single files of a hub repo: {key: file} -> media items (fetch, seek-cut, delete)."""
    def one(kv):
        key, fn = kv
        item = vidkit.cached_item(name, key)
        if item and len(item["frames"]) >= frames:
            return key, item
        src = None
        for attempt in range(4):
            src = avkit.fetch_file(repo, fn, name)
            if src is not None:
                break
            import time

            time.sleep(10 * (attempt + 1))
        if src is None:
            return key, None
        try:
            return key, seek_video_item(Path(src), name, key, frames, **kw)
        except Exception:  # noqa: BLE001
            return key, None
        finally:
            Path(src).unlink(missing_ok=True)

    with ThreadPoolExecutor(workers) as ex:
        return {k: v for k, v in ex.map(one, files.items()) if v}
