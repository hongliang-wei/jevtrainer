"""Clips out of big zip archives on the hub without downloading the archives.

`bizkit.RemoteZip` reads the central directory of a zip with a few range requests and then one range request per member,
so a 16 GB archive of videos is only touched where the wanted clips are. The mirror answers each request slowly (seconds
under load), so `convert_zip_clips` first merges members that lie close together in the archive into one range request
(`gap` bytes of slack, at most `group_mb` per request) and cuts them apart locally; every member then goes to a scratch
file, is cut into frames + sound by `vidkit.finish` and deleted right away.
"""

from __future__ import annotations

import os
import struct
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from jevtrainer.data import avkit, vidkit
from jevtrainer.data.bizkit import RemoteZip

WORKERS = min(6, int(os.environ.get("JEVTRAINER_AV_WORKERS", "6")))
GAP = int(float(os.environ.get("JEVTRAINER_ZIP_GAP_MB", "3")) * (1 << 20))
GROUP = int(float(os.environ.get("JEVTRAINER_ZIP_GROUP_MB", "40")) * (1 << 20))
_INDEX: dict[tuple[str, str], RemoteZip] = {}
_LOCK = threading.Lock()


def open_zip(repo: str, filename: str) -> RemoteZip:
    """RemoteZip of one hub zip (index read once per process, retried: the mirror answers 429 under load)."""
    with _LOCK:
        z = _INDEX.get((repo, filename))
    if z is None:
        for attempt in range(6):
            try:
                z = RemoteZip(repo, filename)
                break
            except Exception:  # noqa: BLE001
                time.sleep(10 * (attempt + 1))
        else:
            raise OSError(f"cannot open {repo}/{filename}")
        with _LOCK:
            _INDEX[(repo, filename)] = z
    return z


def read_group(z: RemoteZip, members: list[str]) -> dict[str, bytes]:
    """Bytes of several members of one zip fetched with a single range request (they should lie close together)."""
    spans = {m: z.index[m] for m in members}
    start = min(o for o, _, _ in spans.values())
    end = max(o + s + 30 + 600 for o, s, _ in spans.values())
    buf = z.rf._get(start, end).content
    out = {}
    for m, (off, size, ctype) in spans.items():
        loc = buf[off - start:]
        n_len, x_len = struct.unpack("<HH", loc[26:30])
        if 30 + n_len + x_len + size > len(loc):  # header longer than guessed: read this member on its own
            out[m] = z.read(m)
            continue
        data = loc[30 + n_len + x_len:30 + n_len + x_len + size]
        out[m] = zlib.decompress(data, -15) if ctype == 8 else data
    return out


def _groups(z: RemoteZip, members: list[str]) -> list[list[str]]:
    """Members sorted by position in the archive, merged while the next one starts within GAP and the group stays < GROUP."""
    ms = sorted(members, key=lambda m: z.index[m][0])
    groups: list[list[str]] = []
    for m in ms:
        off, size, _ = z.index[m]
        if groups:
            g = groups[-1]
            go, gs, _ = z.index[g[-1]]
            if off - (go + gs) <= GAP and off + size - z.index[g[0]][0] <= GROUP:
                g.append(m)
                continue
        groups.append([m])
    return groups


def convert_zip_clips(name: str, jobs: dict[str, tuple[RemoteZip, str, dict]], workers: int = WORKERS, **kw) -> dict[str, dict]:
    """`jobs`: {key: (zip, member name, per-clip video_item kwargs)} -> {key: media item} for the clips that worked."""
    tmp = avkit.raw_dir(f"{name}_zip")
    todo: dict[int, tuple[RemoteZip, dict[str, list[tuple[str, dict]]]]] = {}
    done: dict[str, dict] = {}
    for key, (z, member, okw) in jobs.items():
        item = vidkit.cached_item(name, key)
        if item:
            done[key] = item
        else:
            todo.setdefault(id(z), (z, {}))[1].setdefault(member, []).append((key, okw))

    batches = []
    for z, by_member in todo.values():
        for g in _groups(z, list(by_member)):
            batches.append((z, g, by_member))

    def one(batch):
        z, members, by_member = batch
        data = {}
        for attempt in range(3):
            try:
                data = read_group(z, members)
                break
            except Exception:  # noqa: BLE001
                time.sleep(5 * (attempt + 1))
        out = {}
        for member in members:
            if member not in data:
                continue
            for key, okw in by_member[member]:
                dst = tmp / f"{key}{Path(member).suffix or '.mp4'}"
                try:
                    dst.write_bytes(data[member])
                    item = vidkit.finish(name, key, dst, **{**kw, **okw})
                    if item:
                        out[key] = item
                except Exception:  # noqa: BLE001
                    pass
                finally:
                    dst.unlink(missing_ok=True)
        return out

    with ThreadPoolExecutor(workers) as ex:
        for out in ex.map(one, batches):
            done.update(out)
    return done
