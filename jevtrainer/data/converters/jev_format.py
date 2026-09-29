"""Community datasets already built for Jev-like models (typed questions over a state).

Some are distilled from TypeSafe's hosted Jev; those carry the tag "jev-distilled" and should be
checked against TypeSafe's terms before commercial use.
"""

from __future__ import annotations

import ast
import json

from jevtrainer.data.base import DatasetSpec, hf, mcq_record, register, rid, take
from jevtrainer.schema import Question, Record, Target


def _obj(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return ast.literal_eval(v)
    return v


def jebadiah_synth(split, cap, rng):
    ds = take(hf("frontier-infra/jebadiah-synth-v2", split="train"), cap, rng)
    for r in ds:
        questions, label, target = _obj(r["questions"]), _obj(r["label"]) or {}, _obj(r["target"]) or {}
        targets = {}
        for k, q in questions.items():
            if k not in label:
                continue
            probs = target.get(k)
            if q["type"] == "noul" and isinstance(probs, (int, float)):
                probs = {"yes": float(probs), "no": 1 - float(probs)}
            targets[k] = {"label": label[k], "probabilities": probs if isinstance(probs, dict) else None}
        rec = Record.from_dict({"id": r["id"], "state": _obj(r["state"]), "questions": questions, "targets": targets, "meta": {"area": "typed", "subset": r["subset"]}})
        try:
            yield rec.validate()
        except ValueError:
            continue


def _single(name, repo, split_map, state_key="state", area="typed"):
    def build(split, cap, rng):
        ds = take(hf(repo, split=split_map.get(split, split)), cap, rng)
        for i, r in enumerate(ds):
            opts = _obj(r["options"])
            if r.get("ordered") in (True, "True") and 2 <= len(opts) <= 10:
                rec = Record(rid(name, split, i), r[state_key], {"decision": Question("score", r["question"], [str(o) for o in opts])},
                             {"decision": Target(str(int(r["answer_index"])))}, meta={"area": area, "task": r.get("task") or r.get("family")})
            else:
                rec = mcq_record(rid(name, split, i), r[state_key], r["question"], [str(o) for o in opts], int(r["answer_index"]), area=area)
            if rec:
                yield rec

    return build


register(DatasetSpec("jebadiah_synth", jebadiah_synth, ("train",), "frontier-infra/jebadiah-synth-v2", "apache-2.0", "typed",
                     description="14.7k typed questions over 3.2k synthetic states; teacher probabilities from open models"))
register(DatasetSpec("pngwn_typed_v2", _single("pngwn_typed_v2", "pngwn/typed-decisions-v2-system-one", {"validation": "val"}), ("train", "validation", "test"),
                     "pngwn/typed-decisions-v2-system-one", "cc-by-nc-4.0", "typed", tags=["non-commercial"]))
register(DatasetSpec("pngwn_system_one", _single("pngwn_system_one", "pngwn/system-one-decisions", {"validation": "val"}), ("train", "validation", "test"),
                     "pngwn/system-one-decisions", "cc-by-nc-4.0", "typed", tags=["non-commercial"]))
register(DatasetSpec("this_that_complex", _single("this_that_complex", "limberc/this-that-complex-decisions", {}), ("test",),
                     "limberc/this-that-complex-decisions", "see upstream", "typed", eval_only=True))
register(DatasetSpec("this_that_spatial", _single("this_that_spatial", "limberc/this-that-spatial-bench", {}), ("test",),
                     "limberc/this-that-spatial-bench", "see upstream", "spatial", eval_only=True))
