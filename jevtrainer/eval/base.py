"""BenchmarkSpec: a dataset split plus a headline metric. One `bench(...)` call per benchmark."""

from __future__ import annotations

from dataclasses import dataclass

from jevtrainer.data.base import load
from jevtrainer.registry import BENCHMARKS
from jevtrainer.schema import Record


@dataclass
class BenchmarkSpec:
    name: str
    dataset: str
    split: str = "test"
    metric: str = "accuracy"  # accuracy | macro_f1 | case_exact
    area: str = "misc"
    suite: str = ""  # group name, e.g. "intern-accuracy-v1"
    max_samples: int | None = None  # default subsample for big test sets
    description: str = ""


def bench(name: str, dataset: str, split: str = "test", **kw) -> BenchmarkSpec:
    spec = BenchmarkSpec(name, dataset, split, **kw)
    BENCHMARKS.add(name, spec)
    return spec


def expand(names: list[str]) -> list[str]:
    """Allow suite names (e.g. 'intern-accuracy-v1') alongside benchmark names."""
    out = []
    for n in names:
        if n in BENCHMARKS:
            out.append(n)
        else:
            members = [k for k, s in BENCHMARKS.items() if s.suite == n]
            if not members:
                BENCHMARKS.get(n)  # raises with suggestions
            out.extend(members)
    return list(dict.fromkeys(out))


def benchmark_records(name: str, max_samples: int | None = None) -> list[Record]:
    recs = []
    for n in expand([name]):
        spec: BenchmarkSpec = BENCHMARKS.get(n)
        recs.extend(load(spec.dataset, spec.split, max_samples or spec.max_samples, seed=0))
    return recs
