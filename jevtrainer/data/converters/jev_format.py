"""Community datasets already built for Jev-like models (typed questions over a state).

Some are distilled from TypeSafe's hosted Jev; those carry the tag "jev-distilled" and should be
checked against TypeSafe's terms before commercial use.
"""

from __future__ import annotations

import ast
import json

from jevtrainer.data.base import DatasetSpec, cache_dir, hf, mcq_record, register, rid, take
from jevtrainer.schema import Question, Record, Target


def _obj(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return ast.literal_eval(v)
    return v


def _jsonl_files(repo: str, prefix: str):
    """Rows of every JSONL file in a dataset repo (for repos whose files have differing columns)."""
    from huggingface_hub import HfApi, hf_hub_download

    for f in sorted(HfApi().list_repo_files(repo, repo_type="dataset")):
        if f.startswith(prefix) and f.endswith(".jsonl"):
            with open(hf_hub_download(repo, f, repo_type="dataset"), encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        yield json.loads(line)


def jebadiah_synth(split, cap, rng):
    rows = list(_jsonl_files("frontier-infra/jebadiah-synth-v2", "data/"))
    rng.shuffle(rows)
    for r in rows[:cap]:
        questions, label, target = _obj(r["questions"]), _obj(r["label"]) or {}, _obj(r["target"]) or {}
        targets = {}
        for k, q in questions.items():
            if k not in label:
                continue
            probs = target.get(k)
            if q["type"] == "noul" and isinstance(probs, (int, float)):
                probs = {"yes": float(probs), "no": 1 - float(probs)}
            targets[k] = {"label": label[k], "probabilities": probs if isinstance(probs, dict) else None}
        rec = Record.from_dict({"id": r["id"], "state": _obj(r["state"]), "questions": questions, "targets": targets, "meta": {"area": "typed", "subset": r.get("subset")}})
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


def kev_suites(split, cap, rng):
    """Kev's frozen suites: questions carry their own `label`. Train = every suite's train.jsonl; test = test.jsonl."""
    from huggingface_hub import HfApi, hf_hub_download

    fname = {"train": "train.jsonl", "test": "test.jsonl", "validation": "calibration.jsonl"}[split]
    files = [f for f in HfApi().list_repo_files("jaredpalmer/kev-suites", repo_type="dataset") if f.endswith("/" + fname) and not f.startswith("diagnostics")]
    rows, seen = [], set()
    for f in files:
        with open(hf_hub_download("jaredpalmer/kev-suites", f, repo_type="dataset"), encoding="utf-8") as fh:
            for j, line in enumerate(fh):
                d = json.loads(line)
                key = json.dumps(d.get("state"), sort_keys=True, ensure_ascii=False)[:2000]
                if key in seen:
                    continue
                seen.add(key)
                qs, ts = {}, {}
                for name, q in d["questions"].items():
                    q = dict(q)
                    label = q.pop("label", None)
                    q.pop("src", None)
                    if isinstance(q.get("instructions"), dict):  # {"question": ..., "focus": ...}
                        q["instructions"] = " ".join(str(v) for v in q["instructions"].values() if v)
                    qs[name] = q
                    if label is not None:
                        ts[name] = {"label": label if not isinstance(label, bool) else ("yes" if label else "no"), "probabilities": q.pop("target", None)}
                rows.append({"id": f"kev/{f.rsplit('/', 1)[0]}/{j}", "state": d["state"], "questions": qs, "targets": ts, "meta": {"area": "typed", "suite": f.rsplit("/", 1)[0]}})
    rng.shuffle(rows)
    for d in rows[:cap]:
        try:
            yield Record.from_dict(d).validate()
        except (ValueError, KeyError, TypeError):
            continue


def mojev_mix(split, cap, rng):
    from huggingface_hub import hf_hub_download

    p = hf_hub_download("MoLeMo-Lab/mojev-mix", f"{split}.jsonl", repo_type="dataset")
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    rng.shuffle(rows)
    for i, r in enumerate(rows[:cap]):
        qs, ts = {}, {}
        for name, opts in r["options"].items():
            opts = [str(o) for o in opts]
            gold = str(r["labels"].get(name))
            if gold not in opts or len(set(opts)) != len(opts) or len(opts) < 2:
                continue
            if opts == ["no", "yes"]:
                qs[name], ts[name] = Question("noul", "Answer the yes/no question in the context."), Target(gold)
            else:
                qs[name] = Question("choice", "Answer the question in the context.", {o: "" for o in opts})
                ts[name] = Target(gold)
        if qs:
            yield Record(rid("mojev", split, i), r["context"], qs, ts, meta={"area": "typed", "source": r.get("source")})


def onejev(split, cap, rng):
    """OneJev-Data: one typed question per row, often with images (saved to the cache)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    shards = sorted(f for f in HfApi().list_repo_files("OmniJev/OneJev-Data", repo_type="dataset") if f.startswith("data/train-"))
    rng.shuffle(shards)
    img_dir = cache_dir() / "images" / "onejev"
    img_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for shard in shards:
        table = pq.read_table(hf_hub_download("OmniJev/OneJev-Data", shard, repo_type="dataset"))
        for r in table.to_pylist():
            if n >= cap:
                return
            q = _obj(r["question"])
            target = _obj(r["target"]) or {}
            paths = []
            for k, im in enumerate(r["images"] or []):
                p = img_dir / f"{rid(r['id'])}_{k}.png"
                if not p.exists() and im and im.get("bytes"):
                    p.write_bytes(im["bytes"])
                paths.append(str(p))
            label = max(target, key=target.get) if target else None
            d = {"id": r["id"], "state": _obj(r["state"]), "questions": {"decision": q}, "images": paths,
                 "targets": {"decision": {"label": label, "probabilities": target}} if label else {},
                 "meta": {"area": "typed", "source": r["source"], "license": r["license"], "task": r["task"]}}
            try:
                yield Record.from_dict(d).validate()
                n += 1
            except (ValueError, KeyError, TypeError):
                continue


register(DatasetSpec("kev_suites", kev_suites, ("train", "validation", "test"), "jaredpalmer/kev-suites (all public suites)", "see per suite", "typed", version="2",
                     description="Kev's frozen decision suites; states deduplicated across suites"))
register(DatasetSpec("mojev_mix", mojev_mix, ("train", "validation", "test"), "MoLeMo-Lab/mojev-mix", "see upstream", "typed"))
register(DatasetSpec("onejev", onejev, ("train",), "OmniJev/OneJev-Data", "per source (licenses.csv); some research-only / non-commercial", "typed",
                     multimodal=True, tags=["mixed-license"], description="~97k multimodal typed decisions with teacher probabilities"))
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
