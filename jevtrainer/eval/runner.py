"""Run a model over records / benchmarks and write results.json + results.md."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from jevtrainer.eval.ablate import ablate_records
from jevtrainer.eval.base import BenchmarkSpec, benchmark_records, expand
from jevtrainer.eval.metrics import summarize
from jevtrainer.registry import BENCHMARKS
from jevtrainer.train.batching import Collator, DecisionModel, to_device


# record meta fields the per-benchmark breakdown is reported over (when a converter sets them)
BREAKDOWN_KEYS = ("category", "task_type", "domain", "duration", "modality", "difficulty", "question_type", "task",
                  "sub_category", "audio_class")


def breakdown(outs: list[dict]) -> dict:
    """{meta key: {value: {n, accuracy}}} over the decisions that have a gold label (only keys with >= 2 values)."""
    table: dict = {}
    for o in outs:
        if o["target"] is None:
            continue
        hit = int(o["logits"].argmax()) == o["target"]
        for k, v in (o.get("meta") or {}).items():
            for val in v if isinstance(v, list) else [v]:
                c = table.setdefault(k, {}).setdefault(str(val), [0, 0])
                c[0] += 1
                c[1] += hit
    return {k: {v: {"n": n, "accuracy": h / n} for v, (n, h) in sorted(vals.items())}
            for k, vals in table.items() if len(vals) >= 2}


@torch.no_grad()
def collect(b, records, batch_size: int = 16, max_length: int = 8192) -> list[dict]:
    """Option logits for every question of every record (records that do not fit are skipped)."""
    net = DecisionModel(b.model, b.readout)
    was = net.training
    net.eval()
    device = next(b.model.parameters()).device
    loader = DataLoader(records, batch_size=batch_size, collate_fn=Collator(b.readout, max_length), shuffle=False)
    outs = []
    for batch in loader:
        if batch is None:
            continue
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = net(to_device(batch, device))
        for z, (_, rd), t in zip(logits, batch["reads"], batch["targets"]):
            r = batch["records"][rd.record]
            outs.append(
                {"logits": z[: t["k"]].float().cpu(), "k": t["k"], "target": t["index"], "soft": t["soft"],
                 "type": t["type"], "id": r.id, "name": rd.name, "group": r.meta.get("category") or r.meta.get("family"),
                 "meta": {k: r.meta[k] for k in BREAKDOWN_KEYS if r.meta.get(k) is not None}}
            )
    net.train(was)
    return outs


def run_benchmarks(b, names: list[str], out_dir: str | Path | None = None, max_samples: int | None = None, batch_size: int = 16,
                   ablate: str = "none") -> dict:
    """`ablate`: none | mute | black | shuffle_audio (jevtrainer/eval/ablate.py); applied to the records, not to the cache."""
    results = {}
    for name in expand(names):
        spec: BenchmarkSpec = BENCHMARKS.get(name)
        recs = ablate_records(benchmark_records(name, max_samples), ablate)
        outs = collect(b, recs, batch_size)
        res = summarize(outs, b.temperature, spec.metric)
        res["ablate"] = ablate
        if res.get("n"):
            res["breakdown"] = breakdown(outs)
        expected = sum(1 for r in recs for t in r.targets.values() if t.label is not None)
        res["skipped_decisions"] = expected - res.get("n", 0) if expected else 0
        if expected and res.get("n"):
            # skipped decisions (too long / too many options) count as wrong
            res["coverage"] = res["n"] / expected
            if spec.metric == "accuracy":
                res["score"] = res["accuracy"] * res["coverage"]
        results[name] = res
        print(json.dumps({"benchmark": name, **{k: round(v, 4) if isinstance(v, float) else v for k, v in res.items() if k != "breakdown"}}))
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        (out / "results.md").write_text(to_markdown(results, ablate), encoding="utf-8")
        (out / "meta.json").write_text(json.dumps({"ablate": ablate, "benchmarks": list(results)}, indent=2), encoding="utf-8")
    return results


def to_markdown(results: dict, ablate: str = "none") -> str:
    def f(v, scale=1.0, digits=2):
        return "" if v is None else f"{scale * v:.{digits}f}"

    lines = [f"ablate: {ablate}", "", "| benchmark | n | score | accuracy | skill | ECE | Brier | TVD |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in results.items():
        if not r.get("n"):
            lines.append(f"| {name} | 0 | | | | | | |")
            continue
        lines.append(
            f"| {name} | {r['n']} | {f(r['score'], 100)} | {f(r.get('accuracy'), 100)} | {f(r.get('skill'), 100)} "
            f"| {f(r.get('ece', r.get('expected_ece')), 1, 4)} | {f(r.get('brier', r.get('expected_brier')), 1, 4)} | {f(r.get('tvd'), 1, 4)} |"
        )
    scored = [r["score"] for r in results.values() if r.get("n")]
    if scored:
        lines.append(f"| **average** | | {100 * sum(scored) / len(scored):.2f} | | | | |")
    for name, r in results.items():  # accuracy per task type / domain / duration ... when the benchmark carries them
        for key, vals in (r.get("breakdown") or {}).items():
            lines += ["", f"{name} by {key}:", "", f"| {key} | n | accuracy |", "|---|---:|---:|"]
            lines += [f"| {v} | {c['n']} | {100 * c['accuracy']:.2f} |" for v, c in vals.items()]
    return "\n".join(lines) + "\n"
