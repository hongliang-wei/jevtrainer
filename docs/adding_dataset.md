# Adding a dataset

A dataset is a function that yields `Record`s plus one `register(DatasetSpec(...))` call. Put it in
any file under `jevtrainer/data/converters/`; files there are imported automatically.

```python
from jevtrainer.data.base import DatasetSpec, choice_record, hf, register, rid, take

CRIT = {"refund": "The customer wants money back.", "exchange": "The customer wants another item."}

def returns(split, cap, rng):
    ds = take(hf("my-org/returns", split=split), cap, rng)   # deterministic shuffle + cap
    for i, r in enumerate(ds):
        yield choice_record(rid("returns", split, i), {"message": r["text"]},
                            "What does the customer want?", CRIT, r["label"], area="support")

register(DatasetSpec("returns", returns, ("train", "test"), source="my-org/returns", license="cc-by-4.0", area="support"))
```

Then use it: `dataset: returns` or `dataset: returns:test`.

Rules:

* **Labels are keys of `criteria`.** Descriptions are what the model reads; keys are what targets use.
  Use neutral keys (`mcq_record` gives `opt_1`, `opt_2`, …) when a key would leak the answer.
* Multi-question records are encouraged: several questions on one `state`. Build them with `Record` directly.
* `noul` targets are `yes` / `no` (`true` / `false` are accepted). `score` targets are level indices `"0"`, `"1"`, ….
* Soft targets: `Target(label, probs={label: p, ...})`; the CE loss then trains on the distribution.
* Images: put file paths (or URLs) in `record.images`; `vision.py` shows how to save PIL images to the cache.
* Mark test-only sources `eval_only=True` so they can never be trained on.
* After changing a converter, bump `version=` so the cache is rebuilt.

A plain JSONL file in the record format works without any code: `dataset: ./my_data.jsonl`.

## Adding a benchmark

```python
from jevtrainer.eval.base import bench
bench("returns", "returns", "test", metric="macro_f1", suite="support", area="support")
```

in any file under `jevtrainer/eval/benchmarks/`. `metric` is `accuracy`, `macro_f1` or `case_exact`;
every benchmark also reports skill, ECE, Brier and NLL. Naming a `suite` lets configs run the group:
`eval_dataset: support`.
