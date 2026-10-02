"""Builders for audio corpora that ship as plain files (wav / mp3 / flac in an archive or a hub folder).

`avkit` covers parquet shards with audio columns; this module covers the rest: the converter lists its clips as
rows (`dict`s that carry the local file in `_path`), and `file_label` / `file_score` pick a class-balanced sample,
convert the picked files to 16 kHz mono FLAC (at most `max_s` seconds, 6 ffmpeg jobs in parallel) and build the records.
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from jevtrainer.data import avkit
from jevtrainer.data.avkit import AUDIO_Q, audio_choice, audio_noul, audio_state, balanced, rid_
from jevtrainer.data.base import rid
from jevtrainer.schema import Question, Record, Target

WORKERS = 6


def convert_files(name: str, rows: list[dict], key_of, max_s: float = 30.0, start_of=None, workers: int = WORKERS) -> list[dict | None]:
    """Media items (aligned with `rows`) of the files in `row["_path"]`."""

    def one(r):
        try:
            start = start_of(r) if start_of else None
            return avkit.audio_item(Path(r["_path"]), name, key_of(r), max_s, start=start)
        except Exception:
            return None

    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, rows))


def file_label(name: str, split: str, cap: int, rng: random.Random, rows: list[dict], label_of, question: str,
               criteria: dict[str, str], key_of, meta_of=None, balance: bool = True, max_s: float = 30.0,
               instructions: str = AUDIO_Q, start_of=None, group_of=None):
    """One record per picked row: a fixed label set (`criteria`: label -> meaning) or, when `criteria` has exactly the keys
    {"true", "false"}, a yes/no question (`label_of` then returns True / False). `label_of(row)` = None skips a row."""
    rows = [r for r in rows if (r.setdefault("_y", label_of(r))) is not None]
    rng.shuffle(rows)
    picked = balanced(rows, group_of or (lambda r: r["_y"]), cap, rng) if balance else rows[:cap]
    media = convert_files(name, picked, lambda r: rid_(name, split, key_of(r)), max_s, start_of)
    yes_no = set(criteria) == {"true", "false"}
    for r, m in zip(picked, media):
        if not m:
            continue
        meta = meta_of(r) if meta_of else {}
        i = rid_(name, split, key_of(r))
        if yes_no:
            yield audio_noul(i, m, question, bool(r["_y"]), criteria["true"], criteria["false"], **meta)
        else:
            yield audio_choice(i, m, question, criteria, r["_y"], instructions, **meta)


def audio_score(id: str, media: dict, question: str, levels: list[str], level: int, instructions: str | None = None, **meta) -> Record:
    """One audio clip, one graded question (`levels`: ordered level descriptions, `level`: index of the right one)."""
    rec = Record(id, audio_state(question), {"score": Question("score", instructions or question, list(levels))},
                 {"score": Target(str(level))}, meta={"area": "audio", **meta})
    rec.media = [media]
    return rec


def file_score(name: str, split: str, cap: int, rng: random.Random, rows: list[dict], level_of, question: str,
               levels: list[str], key_of, meta_of=None, max_s: float = 30.0, instructions: str | None = None, start_of=None):
    """Like `file_label` for a graded answer: `level_of(row)` is the index into `levels` (None skips the row); the sample is
    balanced over the levels."""
    rows = [r for r in rows if (r.setdefault("_y", level_of(r))) is not None]
    rng.shuffle(rows)
    picked = balanced(rows, lambda r: r["_y"], cap, rng)
    media = convert_files(name, picked, lambda r: rid_(name, split, key_of(r)), max_s, start_of)
    for r, m in zip(picked, media):
        if m:
            yield audio_score(rid_(name, split, key_of(r)), m, question, levels, r["_y"], instructions,
                              **(meta_of(r) if meta_of else {}))


def clip_score(name: str, split: str, cap: int, rng: random.Random, repo: str, files: list[str], columns: list[str], level_of,
               question: str, levels: list[str], key_of, max_s: float = 30.0, keep=None, meta_of=None, instructions: str | None = None,
               media_name: str | None = None, remote: bool | None = None, rows: list[dict] | None = None):
    """Like `avkit.clip_label` for a graded answer over parquet shards: `level_of(row)` is the index into `levels` (None skips the
    row), the sample is balanced over the levels. Several tasks on the same clips can share their FLAC files with `media_name`."""
    media_name = media_name or name
    if rows is None:
        rows = [r for r in avkit.parquet_rows(repo, files, columns, media_name) if (keep is None or keep(r))]
    for r in rows:
        r["_y"] = level_of(r)
    rows = [r for r in rows if r["_y"] is not None]
    rng.shuffle(rows)
    picked = balanced(rows, lambda r: r["_y"], cap, rng)
    media = avkit.parquet_audio(repo, picked, media_name, lambda r: rid_(media_name, split, key_of(r)), "audio", max_s, remote=bool(remote))
    for r, m in zip(picked, media):
        if m:
            yield audio_score(rid_(name, split, key_of(r)), m, question, levels, r["_y"], instructions, **(meta_of(r) if meta_of else {}))
    avkit.release(media_name)


def remote_rows(repo: str, files: list[str], columns: list[str], workers: int = 4) -> list[dict]:
    """Rows of the given (non-audio) columns of parquet shards, read with range requests: only the footer and those column chunks
    travel, the audio stays on the hub. Every row gets `_file`, `_i` (index in the file) and `_rg` (row group)."""
    import time

    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    def one(fn):
        for attempt in range(5):
            try:
                with HfFileSystem().open(f"datasets/{repo}/{fn}") as f:
                    pf = pq.ParquetFile(f)
                    groups = [k for k in range(pf.num_row_groups) for _ in range(pf.metadata.row_group(k).num_rows)]
                    rows = pf.read(columns=columns).to_pylist()
                for i, r in enumerate(rows):
                    r["_file"], r["_i"], r["_rg"] = fn, i, groups[i]
                return rows
            except Exception:
                time.sleep(10 * (attempt + 1))
        return []

    with ThreadPoolExecutor(max(1, min(workers, 6))) as ex:
        return [r for rows in ex.map(one, files) for r in rows]


def few_groups(rows: list[dict], rng: random.Random, want: int) -> list[dict]:
    """Rows of randomly chosen row groups, until at least `want` rows are kept (so that only those groups are downloaded)."""
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["_file"], r["_rg"]), []).append(r)
    keys = sorted(groups)
    rng.shuffle(keys)
    out: list[dict] = []
    for k in keys:
        if len(out) >= want:
            break
        out += groups[k]
    return out


def pick_rows(repo: str, files: list[str], columns: list[str], rng: random.Random, want: int, keep=None) -> list[dict]:
    """Candidate rows for a converter that wants `want` clips out of big audio shards: labels are read remotely, then a random
    set of row groups holding about `want` rows is kept. Pass the result as `rows=` (and `remote=True`) to
    `avkit.clip_label` / `clip_score`."""
    rows = [r for r in remote_rows(repo, files, columns) if keep is None or keep(r)]
    return few_groups(rows, rng, want)


def bucket(x: float, edges: list[float]) -> int:
    """Index of the level of `x`: number of `edges` that are <= x (edges ascending)."""
    return sum(x >= e for e in edges)


__all__ = ["file_label", "file_score", "audio_score", "convert_files", "bucket", "rid", "balanced", "audio_state"]
