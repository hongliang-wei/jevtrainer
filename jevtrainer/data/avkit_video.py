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

    import zlib

    with ThreadPoolExecutor(workers) as ex, tarfile.open(archive, "r|*") as t:
        dst = None
        try:  # a deliberately truncated download ends in the middle of a member: keep what was read
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
                dst = None
        except (EOFError, tarfile.ReadError, zlib.error, OSError):
            if dst is not None:
                dst.unlink(missing_ok=True)
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


# ---- big files: several connections on one file ----------------------------------------------------------
def fetch_ranged(repo: str, filename: str, name: str, conns: int = 8, chunk: int = 16 << 20, repo_type: str = "dataset",
                 max_bytes: int | None = None) -> Path | None:
    """Download one big hub file with `conns` parallel range requests (resumable; falls back to a plain download).

    `max_bytes`: keep only the first `max_bytes` of the file (a truncated .tar.gz still streams, see `convert_tar_videos`;
    the clips of the shipped tars are in random order, so the prefix is a random subset).

    The mirror serves a single connection at a few hundred KB/s, so one multi-GB archive is only practical this way.
    Chunks already on disk (listed in `<file>.done`) are skipped after an interruption.
    """
    import math
    import time
    import urllib.request

    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    try:
        url = hf_hub_url(repo, filename, repo_type=repo_type)
        size = int(get_hf_file_metadata(url).size)
        if max_bytes:
            size = min(size, int(max_bytes))
    except Exception:
        return avkit.fetch_file(repo, filename, name, repo_type)
    dest = avkit.raw_dir(name) / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == size:
        return dest
    part, donef = dest.with_name(dest.name + ".part"), dest.with_name(dest.name + ".done")
    n = math.ceil(size / chunk)
    done = {int(x) for x in donef.read_text().split()} if donef.exists() and part.exists() and part.stat().st_size == size else set()
    if not done:
        with open(part, "wb") as f:
            f.truncate(size)
        donef.write_text("")
    lock = threading.Lock()

    def grab(i: int) -> bool:
        s, e = i * chunk, min(size, (i + 1) * chunk) - 1
        for attempt in range(8):
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={s}-{e}", "User-Agent": "jevtrainer"})
                with urllib.request.urlopen(req, timeout=90) as r:
                    if r.status != 206 and not (r.status == 200 and n == 1):
                        raise IOError(f"status {r.status}")
                    buf = r.read()
                if len(buf) != e - s + 1:
                    raise IOError("short read")
                with lock:
                    with open(part, "r+b") as f:
                        f.seek(s)
                        f.write(buf)
                    with open(donef, "a") as f:
                        f.write(f"{i}\n")
                return True
            except Exception:
                time.sleep(min(30, 2 ** attempt))
        return False

    todo = [i for i in range(n) if i not in done]
    with ThreadPoolExecutor(max(1, min(conns, 8))) as ex:
        ok = all(ex.map(grab, todo))
    if not ok:
        return None
    part.replace(dest)
    donef.unlink(missing_ok=True)
    return dest


def convert_parquet_videos(path: Path, name: str, id_col: str, video_col: str, keys: dict[str, str] | None = None,
                           workers: int = 8, delete_raw: bool = True, **kw) -> int:
    """Cut frames + sound out of the mp4 bytes of a parquet column; `keys` maps row id -> clip key (None: id itself).

    Rows whose clip is already converted are skipped. Returns the number of newly converted clips.
    """
    import pyarrow.parquet as pq

    tmp = avkit.raw_dir(name) / "_x"
    tmp.mkdir(parents=True, exist_ok=True)
    n = 0

    def one(item):
        rid_, data = item
        key = rid_ if keys is None else keys.get(rid_)
        if key is None or cached_item(name, key):
            return 0
        dst = tmp / f"{key}.mp4"
        dst.write_bytes(data)
        return int(_convert_file(dst, name, key, delete_raw, kw) is not None)

    pf = pq.ParquetFile(str(path))
    with ThreadPoolExecutor(workers) as ex:
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=[id_col, video_col])
            ids, vids = tbl.column(id_col).to_pylist(), tbl.column(video_col).to_pylist()
            del tbl
            n += sum(ex.map(one, zip(ids, vids)))
    return n

# ---- members of a remote zip, read with range requests (no need to download the whole archive) -----------------
class _RangeFile:
    """Read-only seekable file over HTTP range requests (1 MB block cache); enough for `zipfile` to read the directory."""

    def __init__(self, url: str, size: int, block: int = 1 << 20):
        self.url, self.size, self.block, self.pos, self.cache = url, size, block, 0, {}

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _block(self, i: int) -> bytes:
        if i not in self.cache:
            s = i * self.block
            self.cache[i] = http_range(self.url, s, min(self.size, s + self.block) - 1)
        return self.cache[i]

    def read(self, n: int = -1) -> bytes:
        end = self.size if n is None or n < 0 else min(self.size, self.pos + n)
        out = []
        while self.pos < end:
            i = self.pos // self.block
            b = self._block(i)
            lo = self.pos - i * self.block
            take = b[lo:lo + (end - self.pos)]
            out.append(take)
            self.pos += len(take)
        return b"".join(out)


def http_range(url: str, start: int, end: int, retries: int = 8) -> bytes:
    """Bytes [start, end] (inclusive) of a URL, retried with backoff."""
    import time
    import urllib.request

    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "jevtrainer"})
            with urllib.request.urlopen(req, timeout=90) as r:
                buf = r.read()
            if len(buf) == end - start + 1:
                return buf
        except Exception:
            pass
        time.sleep(min(30, 2 ** attempt))
    raise IOError(f"range request failed: {url} {start}-{end}")


def remote_zip_index(repo: str, filename: str, repo_type: str = "dataset"):
    """(url, {member name: ZipInfo}) of a zip file on the hub, reading only its directory."""
    import zipfile

    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    url = hf_hub_url(repo, filename, repo_type=repo_type)
    size = int(get_hf_file_metadata(url).size)
    with zipfile.ZipFile(_RangeFile(url, size)) as z:
        return url, {i.filename: i for i in z.infolist() if not i.is_dir()}


def remote_zip_member(url: str, info) -> bytes:
    """Bytes of one member (stored or deflated) fetched with a single range request."""
    import struct
    import zlib

    head = http_range(url, info.header_offset, info.header_offset + 30 + 1024 + info.compress_size - 1)
    fn_len, extra_len = struct.unpack("<HH", head[26:30])
    data = head[30 + fn_len + extra_len:30 + fn_len + extra_len + info.compress_size]
    return data if info.compress_type == 0 else zlib.decompress(data, -15)


def convert_remote_zip_videos(repo: str, filename: str, name: str, wanted: dict, workers: int = 8, delete_raw: bool = True,
                              **kw) -> dict[str, dict]:
    """Like `convert_zip_videos` but each member is fetched on its own: `wanted` {member: key | (key, kwargs)}."""
    url, index = remote_zip_index(repo, filename)
    tmp = avkit.raw_dir(name) / "_x"
    tmp.mkdir(parents=True, exist_ok=True)

    def one(kv):
        member, (key, okw) = kv
        c = cached_item(name, key)
        if c:
            return key, c
        info = index.get(member)
        if info is None:
            return key, None
        try:
            dst = tmp / f"{key}{Path(member).suffix}"
            dst.write_bytes(remote_zip_member(url, info))
        except Exception:
            return key, None
        return key, _convert_file(dst, name, key, delete_raw, {**kw, **okw})

    with ThreadPoolExecutor(workers) as ex:
        return {k: v for k, v in ex.map(one, _norm(wanted).items()) if v}