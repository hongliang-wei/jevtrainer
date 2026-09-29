"""Typed-decision suites already in the record format: the Intern-Decision accuracy-v1
bundle (JevBench easy/original/hard, typed-decisions, ToolACE, AG News, WildJailbreak)
and the known-distribution calibration pilot.

Fetch once with `jt data fetch intern` (sparse clone of internlm/Intern-Decision), or
point $JEVTRAINER_INTERN_BENCH at an existing `benchmarks/` directory.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from jevtrainer.data.base import DatasetSpec, cache_dir, register
from jevtrainer.schema import Question, Record, Target

INTERN_REPO = "https://github.com/internlm/Intern-Decision.git"


def intern_root() -> Path:
    env = os.environ.get("JEVTRAINER_INTERN_BENCH")
    return Path(env) if env else cache_dir() / "intern-decision" / "benchmarks"


def fetch_intern() -> Path:
    dest = cache_dir() / "intern-decision"
    if not (dest / "benchmarks" / "accuracy-v1" / "manifest.json").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", INTERN_REPO, str(dest)], check=True)
        subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", "benchmarks"], check=True)
    return dest / "benchmarks"


def _jevbench_row(d: dict) -> Record:
    q = d["question"]
    question = Question(q["type"], q.get("instructions", ""), q.get("criteria"))
    expected = d["expected"]
    if q["type"] == "score" and expected not in question.labels():
        expected = str(list(question.criteria).index(expected)) if expected in question.criteria else str(expected)
    probs = d.get("gold_probs")
    target = Target(question.canonical_label(expected), {question.canonical_label(k): float(v) for k, v in probs.items()} if isinstance(probs, dict) else None)
    meta = {k: d[k] for k in ("family", "group", "split") if k in d}
    return Record(str(d["id"]), d["state"], {"decision": question}, {"decision": target}, meta=meta)


def _file_loader(rel: str, jevbench: bool = False):
    def build(split, cap, rng):
        path = intern_root() / rel
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run `jt data fetch intern` or set JEVTRAINER_INTERN_BENCH")
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    yield _jevbench_row(d) if jevbench else Record.from_dict(d)

    return build


INTERN = [
    ("jevbench_easy", "accuracy-v1/jevbench/easy.jsonl", True, "MIT"),
    ("jevbench_original", "accuracy-v1/jevbench/original.jsonl", True, "MIT"),
    ("jevbench_hard", "accuracy-v1/jevbench/hard.jsonl", True, "MIT"),
    ("typed_decisions_test", "accuracy-v1/typed_decisions/test.jsonl", False, "see upstream"),
    ("toolace_test", "accuracy-v1/toolace/test.jsonl", False, "apache-2.0"),
    ("agnews_test", "accuracy-v1/agnews/test.jsonl", False, "unknown"),
    ("wildjailbreak_test", "accuracy-v1/wildjailbreak/test.jsonl", False, "see upstream (gated)"),
]
for name, rel, jb, lic in INTERN:
    register(DatasetSpec(f"intern/{name}", _file_loader(rel, jb), ("test",), f"Intern-Decision benchmarks/{rel}", lic, "typed", eval_only=True))

def _pilot(split, cap, rng):
    root = intern_root() / "known-distribution-pilot-v1"
    refs = {}
    with open(root / "references.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            refs[d["id"]] = d
    for r in _file_loader("known-distribution-pilot-v1/inputs.jsonl")(split, cap, rng):
        ref = refs[r.id]
        r.targets = {ref["field"]: Target(None, {str(k): float(v) for k, v in ref["gold_probs"].items()})}
        r.meta.update(family=ref.get("family"), category=ref.get("category"))
        yield r


register(DatasetSpec("intern/known_distribution_pilot", _pilot, ("test",), "Intern-Decision benchmarks/known-distribution-pilot-v1",
                     "apache-2.0", "calibration", eval_only=True, description="soft targets only; scored by TVD / expected Brier"))
