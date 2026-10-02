"""Helpers shared by audio / video converters: fetch raw files, cut frames and sound with ffmpeg.

Layout (all under `$JEVTRAINER_CACHE`, default ~/.cache/jevtrainer):

    raw/<dataset>/...             downloaded originals (delete after conversion with `drop_raw`)
    media/<dataset>/<key>/        f00.jpg f01.jpg ... a.flac (16 kHz mono)  <- what records point to

A converter calls `video_item(...)` / `audio_item(...)` and puts the returned dict in `Record.media`.
ffmpeg comes from the system or from the `imageio-ffmpeg` wheel.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from jevtrainer.data.base import cache_dir


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def raw_dir(name: str) -> Path:
    p = Path(os.environ.get("JEVTRAINER_RAW", cache_dir() / "raw")) / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def media_dir(name: str, key: str) -> Path:
    p = cache_dir() / "media" / name / key
    p.mkdir(parents=True, exist_ok=True)
    return p


def drop_raw(name: str) -> None:
    shutil.rmtree(raw_dir(name), ignore_errors=True)


# ---- fetching -------------------------------------------------------------------------
def fetch(repo: str, name: str | None = None, allow: list[str] | None = None, ignore: list[str] | None = None,
          repo_type: str = "dataset", workers: int = 16) -> Path:
    """snapshot_download into raw/<name> (resumable; a second call only checks what is missing)."""
    from huggingface_hub import snapshot_download

    dest = raw_dir(name or repo.replace("/", "__"))
    snapshot_download(repo, repo_type=repo_type, local_dir=str(dest), allow_patterns=allow, ignore_patterns=ignore,
                      max_workers=workers)
    return dest


def fetch_file(repo: str, filename: str, name: str, repo_type: str = "dataset") -> Path | None:
    """Download one file into raw/<name>/<filename> (None when it is missing or fails)."""
    from huggingface_hub import hf_hub_download

    try:
        return Path(hf_hub_download(repo, filename, repo_type=repo_type, local_dir=str(raw_dir(name))))
    except Exception:
        return None


def convert_videos(repo: str, name: str, items: dict[str, str], workers: int = 8, delete_raw: bool = True, **kw) -> dict[str, dict]:
    """Fetch `{key: file in repo}` one by one, turn each into a media item and delete the raw file.

    Runs `workers` fetch+ffmpeg jobs in parallel; returns `{key: media item}` for the ones that worked.
    """
    from concurrent.futures import ThreadPoolExecutor

    def one(kv):
        key, fn = kv
        out = media_dir(name, key)
        if (out / "f01.jpg").exists():  # converted by an earlier run
            files = sorted(out.glob("f*.jpg"))
            dur = probe_cache(out)
            item = {"type": "video", "frames": [str(f) for f in files], "fps": len(files) / max(dur, 0.1), "duration": dur}
            if (out / "a.flac").exists():
                item["audio"] = str(out / "a.flac")
            return key, item
        src = fetch_file(repo, fn, name)
        if src is None:
            return key, None
        try:
            item = video_item(src, name, key, **kw)
            if item:
                (out / "meta.txt").write_text(str(item["duration"]))
        finally:
            if delete_raw:
                src.unlink(missing_ok=True)
        return key, item

    with ThreadPoolExecutor(workers) as ex:
        return {k: v for k, v in ex.map(one, items.items()) if v}


def probe_cache(out: Path) -> float:
    try:
        return float((out / "meta.txt").read_text())
    except Exception:
        return 10.0


def unzip(archive: Path, dest: Path, members: list[str] | None = None) -> Path:
    import zipfile

    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(dest, members=members)
    return dest


def untar(archive: Path, dest: Path) -> Path:
    import tarfile

    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as t:
        t.extractall(dest)
    return dest


# ---- ffmpeg ---------------------------------------------------------------------------------
def _run(args: list[str]) -> str:
    p = subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", *args],
                       capture_output=True, text=True)
    return p.stderr


def probe(path: Path) -> dict:
    """duration (s), has_audio, has_video from `ffmpeg -i`."""
    p = subprocess.run([ffmpeg_exe(), "-hide_banner", "-nostdin", "-i", str(path)], capture_output=True, text=True)
    err = p.stderr
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
    dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else 0.0
    return {"duration": dur, "has_audio": " Audio:" in err, "has_video": " Video:" in err}


def video_item(src: Path, name: str, key: str, frames: int = 8, max_side: int = 448, audio_s: float = 30.0,
               start: float | None = None, end: float | None = None, keep_audio: bool = True) -> dict | None:
    """Cut `frames` uniformly spaced JPEGs and the sound track (16 kHz mono FLAC) out of a video segment.

    Returns the `Record.media` item, or None when the file cannot be decoded.
    """
    info = probe(src)
    if not info["has_video"]:
        return None
    t0 = start or 0.0
    t1 = end if end is not None else info["duration"]
    dur = (t1 - t0) if t1 else info["duration"]
    if dur <= 0.2:
        return None
    out = media_dir(name, key)
    pat = out / "f%02d.jpg"
    seek = ["-ss", f"{t0:.3f}"] if start else []
    vf = f"fps={frames / dur:.6f},scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'"
    args = [*seek, "-t", f"{dur:.3f}", "-i", str(src), "-vf", vf, "-frames:v", str(frames), "-q:v", "3", "-y", str(pat)]
    use_audio = keep_audio and info["has_audio"]
    if use_audio:
        args += ["-vn", "-ac", "1", "-ar", "16000", "-t", f"{min(dur, audio_s):.3f}", "-y", str(out / "a.flac")]
    _run(args)
    files = sorted(out.glob("f*.jpg"))
    if not files:
        return None
    item = {"type": "video", "frames": [str(f) for f in files], "fps": len(files) / dur, "duration": round(dur, 3)}
    if use_audio and (out / "a.flac").exists() and (out / "a.flac").stat().st_size > 0:
        item["audio"] = str(out / "a.flac")
    return item


def audio_item(src: Path, name: str, key: str, max_s: float = 30.0, start: float | None = None,
               end: float | None = None) -> dict | None:
    """16 kHz mono FLAC of an audio (or video) file segment, at most `max_s` seconds."""
    out = media_dir(name, key) / "a.flac"
    if not out.exists():
        t = max_s if end is None else min(max_s, end - (start or 0.0))
        seek = ["-ss", f"{start:.3f}"] if start else []
        _run([*seek, "-t", f"{t:.3f}", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-y", str(out)])
    if not out.exists() or out.stat().st_size == 0:
        return None
    return {"type": "audio", "path": str(out)}


def audio_from_array(array, sr: int, name: str, key: str, max_s: float = 30.0) -> dict | None:
    """Write a decoded waveform (HF `Audio` feature) as 16 kHz mono FLAC."""
    import numpy as np
    import soundfile as sf

    x = np.asarray(array, dtype="float32")
    if x.ndim > 1:
        x = x.mean(-1 if x.shape[-1] <= 8 else 0)
    if sr != 16000:
        import librosa

        x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    x = x[: int(max_s * 16000)]
    if len(x) < 1600:
        return None
    out = media_dir(name, key) / "a.flac"
    sf.write(str(out), x, 16000, format="FLAC")
    return {"type": "audio", "path": str(out)}


def image_item(path_or_pil, name: str, key: str) -> dict:
    from PIL import Image

    out = media_dir(name, key) / "i.jpg"
    if not out.exists():
        im = path_or_pil if hasattr(path_or_pil, "save") else Image.open(path_or_pil)
        im.convert("RGB").save(out, quality=92)
    return {"type": "image", "path": str(out)}


# ---- audio carried inside parquet shards (HF `Audio` columns) -----------------------------------------
def audio_from_bytes(data: bytes, name: str, key: str, max_s: float = 30.0) -> dict | None:
    """16 kHz mono FLAC from encoded audio bytes (wav / flac / ogg / mp3 ...), at most `max_s` seconds."""
    out = media_dir(name, key) / "a.flac"
    if out.exists() and out.stat().st_size > 0:
        return {"type": "audio", "path": str(out)}
    if not data:
        return None
    args = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-i", "pipe:0", "-vn", "-t", f"{max_s:.3f}",
            "-ac", "1", "-ar", "16000", "-y", str(out)]
    subprocess.run(args, input=data, capture_output=True)
    if not out.exists() or out.stat().st_size < 200:  # container that cannot be piped: go through a file
        out.unlink(missing_ok=True)
        import tempfile

        with tempfile.NamedTemporaryFile(dir=raw_dir("_tmp"), suffix=".bin", delete=False) as f:
            f.write(data)
        try:
            return audio_item(Path(f.name), name, key, max_s)
        finally:
            Path(f.name).unlink(missing_ok=True)
    return {"type": "audio", "path": str(out)}


def parquet_files(repo: str, prefix: str = "", suffix: str = ".parquet") -> list[str]:
    """Sorted parquet file names of a hub dataset repo that start with `prefix`."""
    import json
    import time

    from huggingface_hub import HfApi

    cache = cache_dir() / "listings" / (repo.replace("/", "__") + ".json")  # the mirror's rate limit hits listings hardest
    names = None
    if cache.exists():
        try:
            names = json.loads(cache.read_text())
        except Exception:
            names = None
    if names is None:
        for attempt in range(10):  # the hub mirror rate-limits listings now and then
            try:
                names = HfApi().list_repo_files(repo, repo_type="dataset")
                break
            except Exception:
                try:  # the repo-info endpoint is not throttled like the recursive tree one
                    names = [s.rfilename for s in HfApi().dataset_info(repo).siblings]
                    break
                except Exception:
                    pass
                if attempt == 9:
                    raise
                time.sleep(15 * (attempt + 1))
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(names))
        tmp.replace(cache)
    return sorted(f for f in names if f.startswith(prefix) and f.endswith(suffix))


def parquet_rows(repo: str, files: list[str], columns: list[str], name: str, workers: int = 4) -> list[dict]:
    """Rows of the given (non-audio) columns of parquet shards. The shards are downloaded into raw/<name> (4 at a time) and
    stay there so that `parquet_audio` can decode the picked rows without fetching them again.

    Every row gets `_file` (its path in the repo), `_i` (its index in that file) and `_rg` (its row group).
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    import pyarrow.parquet as pq

    def one(fn):
        src = None
        for attempt in range(4):
            src = fetch_file(repo, fn, name)
            if src is not None:
                break
            time.sleep(15 * (attempt + 1))
        if src is None:
            return []
        pf = pq.ParquetFile(str(src))
        groups = [k for k in range(pf.num_row_groups) for _ in range(pf.metadata.row_group(k).num_rows)]
        rows = pf.read(columns=columns).to_pylist() if columns else [{} for _ in groups]
        for i, r in enumerate(rows):
            r["_file"], r["_i"], r["_rg"] = fn, i, groups[i]
        return rows

    with ThreadPoolExecutor(max(1, min(workers, 8))) as ex:
        return [r for rows in ex.map(one, files) for r in rows]


def parquet_audio(repo: str, rows: list[dict], name: str, key_of, audio_col: str = "audio", max_s: float = 30.0,
                  shards: int = 4, decoders: int = 2, remote: bool = False) -> list[dict | None]:
    """Media items (aligned with `rows`) for rows found by `parquet_rows`.

    Each shard that holds a wanted row is downloaded, its wanted rows decoded to FLAC, and the shard deleted again.
    With `remote=True` only the row groups that hold a wanted row are fetched (range requests; use it for shards that are
    much bigger than the part you need, e.g. when taking 200 clips out of every one of 100 languages).
    """
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor

    import pyarrow.parquet as pq

    by_file: dict[str, dict[int, int]] = defaultdict(dict)  # file -> {row index in file: position in `rows`}
    for pos, r in enumerate(rows):
        by_file[r["_file"]][r["_i"]] = pos
    out: list[dict | None] = [None] * len(rows)

    def decode(job):
        pos, blob = job
        if isinstance(blob, dict):
            blob = blob.get("bytes")
        try:
            return pos, audio_from_bytes(blob, name, key_of(rows[pos]), max_s)
        except Exception:
            return pos, None

    def shard_remote(item):
        from huggingface_hub import HfFileSystem

        fn, want = item
        for attempt in range(3):
            try:
                with HfFileSystem().open(f"datasets/{repo}/{fn}") as f:
                    pf = pq.ParquetFile(f)
                    offset = 0
                    with ThreadPoolExecutor(decoders) as ex:
                        for g in range(pf.num_row_groups):
                            n = pf.metadata.row_group(g).num_rows
                            hit = [i for i in want if offset <= i < offset + n]
                            if hit:
                                col = pf.read_row_group(g, columns=[audio_col]).column(0).to_pylist()
                                for pos, m in ex.map(decode, [(want[i], col[i - offset]) for i in hit]):
                                    out[pos] = m
                            offset += n
                return
            except Exception:
                import time

                time.sleep(10 * (attempt + 1))

    def shard(item):
        fn, want = item
        src = fetch_file(repo, fn, name)
        if src is None:
            return
        try:
            offset = 0
            with ThreadPoolExecutor(decoders) as ex:
                for batch in pq.ParquetFile(str(src)).iter_batches(batch_size=64, columns=[audio_col]):
                    col = batch.column(0).to_pylist()
                    jobs = [(want[offset + j], col[j]) for j in range(len(col)) if offset + j in want]
                    for pos, m in ex.map(decode, jobs):
                        out[pos] = m
                    offset += len(col)
        finally:
            if not os.environ.get("JEVTRAINER_KEEP_RAW"):  # set it while several splits share the same shards
                src.unlink(missing_ok=True)

    with ThreadPoolExecutor(max(1, min(shards, 4))) as ex:
        list(ex.map(shard_remote if remote else shard, by_file.items()))
    return out


def balanced(rows: list, group_of, cap: int, rng) -> list:
    """At most `cap` rows, drawn round-robin over groups so that classes are as even as possible."""
    groups: dict = {}
    for r in rows:
        groups.setdefault(group_of(r), []).append(r)
    for g in groups.values():
        rng.shuffle(g)
    keys = sorted(groups, key=str)
    picked = []
    while len(picked) < cap and any(groups.values()):
        rng.shuffle(keys)
        for k in keys:
            if groups[k] and len(picked) < cap:
                picked.append(groups[k].pop())
    return picked


def hash_pct(key: str, pct: int = 10) -> bool:
    """True for a fixed `pct`% of keys (stable across runs): used to cut a test split out of single-split corpora."""
    import hashlib

    return int(hashlib.sha1(key.encode()).hexdigest(), 16) % 100 < pct


def humanize(label: str) -> str:
    """'dog_bark' / 'Dog-Bark' -> 'dog bark'."""
    return re.sub(r"[_\-]+", " ", str(label)).strip().lower()


def audio_state(question: str, **extra) -> dict:
    return {"clip": "<audio:1>", "question": question, **extra}


AUDIO_Q = "Which option best answers the question about the audio clip?"


def audio_mcq(id: str, media: dict, question: str, gold: str, pool: list[str], rng, n_opts: int = 4, instructions: str = AUDIO_Q,
              state_extra: dict | None = None, **meta):
    """One audio clip, one question; the options are `gold` plus n_opts-1 random other strings of `pool`."""
    from jevtrainer.data.base import mcq_record

    others = sorted(set(pool) - {gold})
    options = [gold, *rng.sample(others, min(n_opts - 1, len(others)))]
    order = list(range(len(options)))
    rng.shuffle(order)
    rec = mcq_record(id, audio_state(question, **(state_extra or {})), instructions, [options[i] for i in order], order.index(0),
                     area="audio", **meta)
    if rec:
        rec.media = [media]
    return rec


def release(name: str) -> None:
    """Delete the downloaded shards of a dataset once its records are written (set JEVTRAINER_KEEP_RAW=1 to keep them while
    several splits of the dataset are being prepared one after the other; delete them by hand afterwards)."""
    if not os.environ.get("JEVTRAINER_KEEP_RAW"):
        drop_raw(name)


def use_remote(rows: list[dict], picked: list[dict]) -> bool:
    """True when the picked rows sit in less than half of the row groups: then range requests beat downloading shards."""
    total = {(r["_file"], r["_rg"]) for r in rows}
    hit = {(r["_file"], r["_rg"]) for r in picked}
    return len(hit) < 0.5 * len(total)


def clip_choice(name: str, split: str, cap: int, rng, repo: str, files: list[str], columns: list[str], label_of, question: str,
                key_of, pool: list[str] | None = None, n_opts: int = 4, max_s: float = 30.0, keep=None, meta_of=None,
                balance: bool = True, audio_col: str = "audio", instructions: str = AUDIO_Q, group_of=None):
    """Generic 'what is in this clip' builder over parquet shards: pick (balanced) rows, decode them, yield records.

    `label_of(row)` gives the answer text (None = skip row); distractors are drawn from `pool` (default: all answers seen).
    `pool` may also be a function row -> list. Only shards that hold a picked row are downloaded (and deleted afterwards).
    """
    rows = [r for r in parquet_rows(repo, files, columns, name) if (keep is None or keep(r))]
    for r in rows:
        r["_y"] = label_of(r)
    rows = [r for r in rows if r["_y"]]
    default_pool = sorted({r["_y"] for r in rows})
    rng.shuffle(rows)
    picked = balanced(rows, group_of or (lambda r: r["_y"]), cap, rng) if balance else rows[:cap]
    media = parquet_audio(repo, picked, name, lambda r: rid_(name, split, key_of(r)), audio_col, max_s,
                          remote=False)
    for r, m in zip(picked, media):
        if not m:
            continue
        p = pool(r) if callable(pool) else (pool or default_pool)
        rec = audio_mcq(rid_(name, split, key_of(r)), m, question, r["_y"], p, rng, n_opts, instructions,
                        **(meta_of(r) if meta_of else {}))
        if rec:
            yield rec
    release(name)


def rid_(*parts) -> str:
    from jevtrainer.data.base import rid

    return rid(*parts)


def audio_choice(id: str, media: dict, question: str, criteria: dict[str, str], label: str, instructions: str = AUDIO_Q,
                 state_extra: dict | None = None, **meta):
    """One audio clip, one question, a fixed label set (`criteria`: label -> description of what that label means)."""
    from jevtrainer.data.base import choice_record

    rec = choice_record(id, audio_state(question, **(state_extra or {})), instructions, dict(criteria), label, area="audio", **meta)
    rec.media = [media]
    return rec


def audio_noul(id: str, media: dict, question: str, yes: bool, true: str = "", false: str = "", state_extra: dict | None = None,
               **meta):
    """One audio clip, one yes/no question."""
    from jevtrainer.data.base import noul_record

    rec = noul_record(id, audio_state(question, **(state_extra or {})), question, yes, true, false, area="audio", **meta)
    rec.media = [media]
    return rec


def tar_members(path: Path, suffixes: tuple[str, ...] = (".wav", ".flac", ".mp3", ".ogg")):
    """Stream (member name, bytes) of the audio files of a .tar / .tar.gz without extracting it."""
    import tarfile

    with tarfile.open(path, "r|*") as t:
        for m in t:
            if m.isfile() and m.name.lower().endswith(suffixes):
                yield m.name, t.extractfile(m).read()


def clip_label(name: str, split: str, cap: int, rng, repo: str, files: list[str], columns: list[str], label_of, question: str,
                criteria: dict[str, str], key_of, max_s: float = 30.0, keep=None, meta_of=None, balance: bool = True,
                audio_col: str = "audio", instructions: str = AUDIO_Q, remote: bool | None = None, rows: list[dict] | None = None):
    """Like `clip_choice` but with a fixed label set (`criteria`: label -> meaning) or, when `criteria` has exactly the keys
    {"true", "false"}, a yes/no question (`label_of` then returns True / False). `label_of(row)` = None skips a row."""
    if rows is None:
        rows = [r for r in parquet_rows(repo, files, columns, name) if (keep is None or keep(r))]
    for r in rows:
        r["_y"] = label_of(r)
    rows = [r for r in rows if r["_y"] is not None]
    rng.shuffle(rows)
    picked = balanced(rows, lambda r: r["_y"], cap, rng) if balance else rows[:cap]
    media = parquet_audio(repo, picked, name, lambda r: rid_(name, split, key_of(r)), audio_col, max_s,
                          remote=bool(remote))
    yes_no = set(criteria) == {"true", "false"}
    for r, m in zip(picked, media):
        if not m:
            continue
        meta = meta_of(r) if meta_of else {}
        if yes_no:
            yield audio_noul(rid_(name, split, key_of(r)), m, question, bool(r["_y"]), criteria["true"], criteria["false"], **meta)
        else:
            yield audio_choice(rid_(name, split, key_of(r)), m, question, criteria, r["_y"], instructions, **meta)
    release(name)


def parquet_head(repo: str, fn: str, name: str, n_rows: int, columns: list[str] | None = None) -> list[dict]:
    """The first rows (whole row groups, at least `n_rows`) of a big parquet shard, fetched with ONE streaming range request.

    Row groups are stored one after another, so the first k of them are a contiguous byte prefix of the file. The prefix and
    the footer are written into a sparse file that pyarrow can open; nothing else of the shard is downloaded.
    """
    import urllib.request

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(repo, fn, repo_type="dataset")
    tail = 4 << 20

    def get(a: int, b: int):
        req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
        return urllib.request.urlopen(req, timeout=120)

    head = urllib.request.Request(url, method="HEAD")
    size = int(urllib.request.urlopen(head, timeout=120).headers["Content-Length"])
    path = raw_dir(name) / "_head" / (fn.replace("/", "__") + f".{os.getpid()}.{id(fn)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "wb") as f:
            f.truncate(size)
            a = max(0, size - tail)
            f.seek(a)
            f.write(get(a, size - 1).read())
        md = pq.ParquetFile(str(path)).metadata
        end, rows, k = 0, 0, 0
        while k < md.num_row_groups and rows < n_rows:
            rg = md.row_group(k)
            for c in range(rg.num_columns):
                col = rg.column(c)
                start = col.dictionary_page_offset if col.has_dictionary_page and col.dictionary_page_offset else col.data_page_offset
                end = max(end, start + col.total_compressed_size)
            rows += rg.num_rows
            k += 1
        with open(path, "r+b") as f, get(0, min(end, size) - 1) as resp:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        pf = pq.ParquetFile(str(path))
        out: list[dict] = []
        for g in range(k):
            out += pf.read_row_group(g, columns=columns).to_pylist()
        for i, r in enumerate(out):
            r["_file"], r["_i"] = fn, i
        return out
    finally:
        path.unlink(missing_ok=True)
