"""The one record format every dataset, benchmark and readout speaks.

A record is a state (text or JSON), optional images, and named typed questions:

* ``choice``: ``criteria`` maps option key -> description (2..255 options)
* ``score``:  ``criteria`` is an ordered list of level descriptions (2..10); labels are "0".."n-1"
* ``noul``:   yes/no; optional ``criteria`` {"true": ..., "false": ...}; labels are "no", "yes"

Targets hold a hard ``label`` and/or soft ``probs`` over the question's labels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

QTYPES = ("choice", "score", "noul")
NOUL_LABELS = ["no", "yes"]
_NOUL_ALIASES = {"true": "yes", "false": "no", "1": "yes", "0": "no", True: "yes", False: "no"}


@dataclass
class Question:
    type: str
    instructions: str
    criteria: dict[str, str] | list[str] | None = None

    def labels(self) -> list[str]:
        if self.type == "choice":
            return list(self.criteria)
        if self.type == "score":
            return [str(i) for i in range(len(self.criteria))]
        return list(NOUL_LABELS)

    def descriptions(self) -> list[str]:
        if self.type == "choice":
            return [str(v) if v is not None else "" for v in self.criteria.values()]
        if self.type == "score":
            return [str(v) for v in self.criteria]
        crit = self.criteria or {}
        return [str(crit.get("false") or "no"), str(crit.get("true") or "yes")]

    def options(self) -> list[tuple[str, str]]:
        return list(zip(self.labels(), self.descriptions()))

    def canonical_label(self, label: Any) -> str:
        if self.type == "noul":
            key = label.lower() if isinstance(label, str) else label
            return _NOUL_ALIASES.get(key, key)
        return str(label)

    def validate(self, name: str) -> None:
        if self.type not in QTYPES:
            raise ValueError(f"question '{name}': type must be one of {QTYPES}, got {self.type!r}")
        if not isinstance(self.instructions, str):
            raise ValueError(f"question '{name}': instructions must be text, got {type(self.instructions).__name__}")
        n = len(self.labels())
        if self.type == "choice" and not (isinstance(self.criteria, dict) and 2 <= n <= 255):
            raise ValueError(f"question '{name}': choice needs a dict of 2..255 options, got {n}")
        if self.type == "score" and not (isinstance(self.criteria, list) and 2 <= n <= 10):
            raise ValueError(f"question '{name}': score needs a list of 2..10 levels, got {n}")

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if self.criteria is not None:
            d["criteria"] = self.criteria
        return d


@dataclass
class Target:
    label: str | None = None
    probs: dict[str, float] | None = None

    def to_dict(self) -> dict:
        d: dict[str, Any] = {}
        if self.label is not None:
            d["label"] = self.label
        if self.probs is not None:
            d["probs"] = self.probs
        return d


@dataclass
class Record:
    id: str
    state: Any
    questions: dict[str, Question]
    targets: dict[str, Target] = field(default_factory=dict)
    images: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "Record":
        questions = {k: Question(q["type"], q.get("instructions") or "", q.get("criteria")) for k, q in d["questions"].items()}
        targets = {}
        for k, t in (d.get("targets") or {}).items():
            q = questions[k]
            label = t.get("label") if isinstance(t, dict) else t
            probs = (t.get("probs") or t.get("probabilities")) if isinstance(t, dict) else None
            targets[k] = Target(
                q.canonical_label(label) if label is not None else None,
                {q.canonical_label(a): float(p) for a, p in probs.items()} if probs else None,
            )
        meta = dict(d.get("meta") or {})
        for extra in ("category", "family", "split", "group"):
            if extra in d and extra not in meta:
                meta[extra] = d[extra]
        return cls(str(d["id"]), d.get("state", ""), questions, targets, list(d.get("images") or []), meta)

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "id": self.id,
            "state": self.state,
            "questions": {k: q.to_dict() for k, q in self.questions.items()},
        }
        if self.targets:
            d["targets"] = {k: t.to_dict() for k, t in self.targets.items()}
        if self.images:
            d["images"] = self.images
        if self.meta:
            d["meta"] = self.meta
        return d

    def validate(self) -> "Record":
        if not self.questions:
            raise ValueError(f"record {self.id}: no questions")
        for name, q in self.questions.items():
            q.validate(name)
        for name, t in self.targets.items():
            if name not in self.questions:
                raise ValueError(f"record {self.id}: target '{name}' has no question")
            labels = self.questions[name].labels()
            if t.label is not None and t.label not in labels:
                raise ValueError(f"record {self.id}: target '{name}'={t.label!r} not in {labels}")
        return self

    def state_text(self) -> str:
        return self.state if isinstance(self.state, str) else json.dumps(self.state, ensure_ascii=False, indent=2)

    def target_index(self, name: str) -> int | None:
        t = self.targets.get(name)
        if t is None or t.label is None:
            return None
        return self.questions[name].labels().index(t.label)


def read_jsonl(path: str | Path) -> Iterator[Record]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield Record.from_dict(json.loads(line))


def write_jsonl(records: Iterable[Record], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n
