"""DatasetSpec and loading. A dataset is one `register(DatasetSpec(...))` call.

`build(split, cap, seed)` yields `Record`s. Results are cached as JSONL under
``$JEVTRAINER_CACHE`` (default ``~/.cache/jevtrainer``), so conversion runs once.
A path ending in ``.jsonl`` can be used anywhere a dataset name is accepted.
"""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Iterator

from jevtrainer.registry import DATASETS
from jevtrainer.schema import Question, Record, Target, read_jsonl, write_jsonl


def cache_dir() -> Path:
    return Path(os.environ.get("JEVTRAINER_CACHE", Path.home() / ".cache" / "jevtrainer"))


@dataclass
class DatasetSpec:
    name: str
    build: Callable[[str, int, random.Random], Iterable[Record]]  # (split, cap, rng) -> records
    splits: tuple[str, ...] = ("train", "test")
    source: str = ""  # upstream hub id or URL
    license: str = "unknown"
    area: str = "misc"
    eval_only: bool = False  # never allowed in a training mixture
    multimodal: bool = False
    description: str = ""
    version: str = "1"  # bump to invalidate the cache after changing a converter
    tags: list[str] = field(default_factory=list)


def register(spec: DatasetSpec) -> DatasetSpec:
    DATASETS.add(spec.name, spec)
    return spec


def load(name: str, split: str = "train", max_samples: int | None = None, seed: int = 0, cap: int = 100_000) -> list[Record]:
    """Records of one dataset split, subsampled to max_samples (deterministic by seed)."""
    if name.endswith(".jsonl") or Path(name).is_file():
        records = list(read_jsonl(name))
    else:
        spec: DatasetSpec = DATASETS.get(name)
        if split not in spec.splits:
            raise ValueError(f"dataset '{name}' has no split '{split}' (has {spec.splits})")
        path = cache_dir() / "records" / name / f"{split}-cap{cap}-v{spec.version}.jsonl"
        if not path.exists():
            rng = random.Random(f"{name}:{split}")
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            n = write_jsonl((r.validate() for r in spec.build(split, cap, rng)), tmp)
            if n == 0:
                raise ValueError(f"dataset '{name}' split '{split}' produced no records")
            tmp.replace(path)
        records = list(read_jsonl(path))
        for r in records:
            r.meta.setdefault("source", name)
    if max_samples is not None and len(records) > max_samples:
        records = random.Random(seed).sample(records, max_samples)
    return records


# ---- helpers used by converters ---------------------------------------------------
def hf(repo: str, config: str | None = None, split: str = "train", **kw):
    from datasets import load_dataset

    return load_dataset(repo, config, split=split, **kw) if config else load_dataset(repo, split=split, **kw)


def take(ds, cap: int, rng: random.Random):
    """Shuffle a HF dataset deterministically and keep at most cap rows."""
    n = len(ds)
    if n > cap:
        ds = ds.shuffle(seed=rng.randint(0, 2**31)).select(range(cap))
    return ds


def rid(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


def choice_record(
    id: str,
    state,
    instructions: str,
    criteria: dict[str, str],
    label: str,
    name: str = "decision",
    **meta,
) -> Record:
    return Record(id, state, {name: Question("choice", instructions, criteria)}, {name: Target(label)}, meta=meta)


def mcq_record(id: str, state, instructions: str, options: list[str], gold: int, **meta) -> Record | None:
    """Multiple choice with free-text options; keys are neutral opt_1.. so they carry no answer."""
    options = [str(o).strip() for o in options]
    if len(set(options)) != len(options) or not 0 <= gold < len(options) or len(options) < 2:
        return None
    crit = {f"opt_{i + 1}": o for i, o in enumerate(options)}
    return choice_record(id, state, instructions, crit, f"opt_{gold + 1}", **meta)


def noul_record(id: str, state, instructions: str, yes: bool, true: str = "", false: str = "", **meta) -> Record:
    crit = {"true": true, "false": false} if (true or false) else None
    return Record(id, state, {"decision": Question("noul", instructions, crit)}, {"decision": Target("yes" if yes else "no")}, meta=meta)


def classification(
    name: str, repo: str, config: str | None, state: Callable[[dict], object], label: Callable[[dict], str | None],
    instructions: str, criteria: dict[str, str], split_map: dict[str, str] | None = None, revision: str | None = None,
    area: str = "misc", keep: Callable[[dict], bool] | None = None,
):
    """Build function for 'one text -> one of fixed labels' datasets."""

    def build(split, cap, rng):
        ds = hf(repo, config, split=(split_map or {}).get(split, split), **({"revision": revision} if revision else {}))
        if keep is not None:
            ds = ds.filter(keep)
        for i, r in enumerate(take(ds, cap, rng)):
            y = label(r)
            if y is None or y not in criteria:
                continue
            yield choice_record(rid(name, split, i), state(r), instructions, dict(criteria), y, area=area)

    return build


def multiple_choice(
    name: str, repo: str, config: str | None, state: Callable[[dict], object], options: Callable[[dict], list[str]],
    gold: Callable[[dict], int], instructions: str = "Which option correctly answers the question?",
    split_map: dict[str, str] | None = None, revision: str | None = None, area: str = "knowledge",
):
    def build(split, cap, rng):
        ds = hf(repo, config, split=(split_map or {}).get(split, split), **({"revision": revision} if revision else {}))
        for i, r in enumerate(take(ds, cap, rng)):
            try:
                rec = mcq_record(rid(name, split, i), state(r), instructions, options(r), gold(r), area=area)
            except (ValueError, KeyError, IndexError, TypeError):
                continue
            if rec:
                yield rec

    return build


def label_names(repo: str, config: str | None, column: str, split: str = "train", revision: str | None = None) -> list[str]:
    ds = hf(repo, config, split=split, **({"revision": revision} if revision else {}))
    return ds.features[column].names


def iter_limit(it: Iterable, cap: int) -> Iterator:
    for i, x in enumerate(it):
        if i >= cap:
            return
        yield x
