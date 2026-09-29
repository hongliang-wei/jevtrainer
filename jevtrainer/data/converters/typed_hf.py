"""LocalLLaMA/typed-decisions: multi-question workflow states with teacher probabilities.

The test split is the 400-case suite used by Intern-Decision (also available as
`intern/typed_decisions_test`); only the train split is trainable.
"""

from __future__ import annotations

import json

from jevtrainer.data.base import DatasetSpec, hf, register, take
from jevtrainer.schema import Record


def typed_decisions(split, cap, rng):
    ds = take(hf("LocalLLaMA/typed-decisions", "all", split=split), cap, rng)
    for r in ds:
        state = json.loads(r["state"]) if isinstance(r["state"], str) else r["state"]
        questions = json.loads(r["questions"]) if isinstance(r["questions"], str) else r["questions"]
        gold = json.loads(r["gold"]) if isinstance(r["gold"], str) else r["gold"]
        targets = {k: {"label": g.get("label"), "probabilities": g.get("probabilities")} for k, g in gold.items() if k in questions}
        yield Record.from_dict(
            {"id": r["id"], "state": state, "questions": questions, "targets": targets, "meta": {"workflow": r.get("workflow"), "area": "typed"}}
        )


register(DatasetSpec("typed_decisions", typed_decisions, ("train",), "LocalLLaMA/typed-decisions:all", "see upstream", "typed",
                     description="4 business workflows, 5 typed questions per state, teacher probability targets"))
register(DatasetSpec("typed_decisions_hf_test", typed_decisions, ("test",), "LocalLLaMA/typed-decisions:all", "see upstream", "typed", eval_only=True))
