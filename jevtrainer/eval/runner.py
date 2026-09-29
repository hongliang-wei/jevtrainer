"""Run a model over records / benchmarks and write results.json + results.md."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from jevtrainer.eval.base import BenchmarkSpec, benchmark_records, expand
from jevtrainer.eval.metrics import summarize
from jevtrainer.registry import BENCHMARKS
from jevtrainer.train.batching import Collator, DecisionModel, to_device


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
                 "type": t["type"], "id": r.id, "name": rd.name, "group": r.meta.get("category") or r.meta.get("family")}
            )
    net.train(was)
    return outs


def run_benchmarks(b, names: list[str], out_dir: str | Path | None = None, max_samples: int | None = None, batch_size: int = 16) -> dict:
    results = {}
    for name in expand(names):
        spec: BenchmarkSpec = BENCHMARKS.get(name)
        recs = benchmark_records(name, max_samples)
        outs = collect(b, recs, batch_size)
        res = summarize(outs, b.temperature, spec.metric)
        expected = sum(1 for r in recs for t in r.targets.values() if t.label is not None)
        res["skipped_decisions"] = expected - res.get("n", 0) if expected else 0
        if expected and res.get("n"):
            # skipped decisions (too long / too many options) count as wrong
            res["coverage"] = res["n"] / expected
            if spec.metric == "accuracy":
                res["score"] = res["accuracy"] * res["coverage"]
        results[name] = res
        print(json.dumps({"benchmark": name, **{k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()}}))
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        (out / "results.md").write_text(to_markdown(results), encoding="utf-8")
    return results


def to_markdown(results: dict) -> str:
    def f(v, scale=1.0, digits=2):
        return "" if v is None else f"{scale * v:.{digits}f}"

    lines = ["| benchmark | n | score | accuracy | skill | ECE | Brier | TVD |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
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
    return "\n".join(lines) + "\n"
