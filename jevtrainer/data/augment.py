"""Training-time augmentation. Every function is pure: (record, rng) -> new record.

* shuffle   : reorder choice options so position carries no information
* none      : with p_none remove the gold option and make "none of the above" correct;
              with p_distract add "none of the above" as a wrong option
* fit       : subsample options (always keeping gold) to what the readout supports
* pack      : merge several single-question records into one multi-question record
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass

from jevtrainer.schema import Question, Record, Target

NONE_KEY = "none_of_the_above"
NONE_DESC = "None of the listed options is correct or supported by the state."


@dataclass
class AugmentConfig:
    shuffle: bool = True
    p_none: float = 0.05
    p_distract: float = 0.1
    p_pack: float = 0.0
    pack_max: int = 4


def shuffle(r: Record, rng: random.Random) -> Record:
    for q in r.questions.values():
        if q.type == "choice":
            items = list(q.criteria.items())
            rng.shuffle(items)
            q.criteria = dict(items)
    return r


def none_option(r: Record, rng: random.Random, p_none: float, p_distract: float) -> Record:
    for name, q in r.questions.items():
        t = r.targets.get(name)
        if q.type != "choice" or t is None or t.label is None or NONE_KEY in q.criteria or t.probs:
            continue
        u = rng.random()
        if u < p_none and len(q.criteria) >= 3:
            crit = {k: v for k, v in q.criteria.items() if k != t.label}
            crit[NONE_KEY] = NONE_DESC
            q.criteria, r.targets[name] = crit, Target(NONE_KEY)
        elif u < p_none + p_distract:
            q.criteria[NONE_KEY] = NONE_DESC
    return r


def fit_options(r: Record, max_options: int, rng: random.Random) -> Record:
    for name, q in r.questions.items():
        if q.type != "choice" or len(q.criteria) <= max_options:
            continue
        t = r.targets.get(name)
        gold = t.label if t else None
        others = [k for k in q.criteria if k != gold]
        keep = set(rng.sample(others, max_options - (1 if gold else 0)))
        if gold:
            keep.add(gold)
        q.criteria = {k: v for k, v in q.criteria.items() if k in keep}
        if t and t.probs:
            t.probs = None
    return r


def pack(records: list[Record], rng: random.Random) -> Record:
    """Several independent records -> one record whose state lists them as item_1..item_n."""
    state, questions, targets, images = {}, {}, {}, []
    for i, r in enumerate(records, 1):
        key = f"item_{i}"
        state[key] = r.state
        for name, q in r.questions.items():
            qn = f"{key}_{name}"
            questions[qn] = Question(q.type, f"About {key}: {q.instructions}", copy.deepcopy(q.criteria))
            if name in r.targets:
                targets[qn] = r.targets[name]
        images += r.images
    meta = dict(records[0].meta, packed=len(records))
    return Record("+".join(r.id for r in records), state, questions, targets, images, meta)


def augment(records: list[Record], cfg: AugmentConfig, rng: random.Random) -> list[Record]:
    out = []
    i = 0
    while i < len(records):
        r = copy.deepcopy(records[i])
        if cfg.p_pack and rng.random() < cfg.p_pack and not r.images:
            n = rng.randint(2, cfg.pack_max)
            group = [copy.deepcopy(x) for x in records[i : i + n] if not x.images]
            if len(group) >= 2:
                r = pack(group, rng)
                i += len(group) - 1
        if cfg.shuffle:
            r = shuffle(r, rng)
        r = none_option(r, rng, cfg.p_none, cfg.p_distract)
        out.append(r)
        i += 1
    return out
