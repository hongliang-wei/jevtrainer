"""Push a run directory's records to Weights & Biases.

    python scripts/wandb_upload.py runs/repro/intern_0.8b_v1 [--project jevtrainer]

Logs checkpoint benchmark evals (step-*/eval/results.json) against `ckpt_step` and
the final metrics.json. If the run trained with report_to=wandb (wandb_id.txt exists)
it appends to that run; otherwise it creates one and also uploads train_log.jsonl.
Safe to re-run: the run id is reused.
"""

import argparse
import json
import os
import re
from pathlib import Path

import wandb
import yaml

from jevtrainer.train.trainer import flatten

ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "jevtrainer"))
args = ap.parse_args()
run_dir: Path = args.run

id_file = run_dir / "wandb_id.txt"
live = id_file.exists()
run_id = id_file.read_text().strip() if live else re.sub(r"[^\w-]", "-", str(run_dir)).strip("-")
cfg = run_dir / "config.yaml"
run = wandb.init(
    project=args.project,
    id=run_id,
    name=run_dir.name,
    resume="allow",
    config=yaml.safe_load(cfg.read_text(encoding="utf-8")) if cfg.exists() else None,
)

if not live and (run_dir / "train_log.jsonl").exists():
    for line in (run_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        wandb.log({k: v for k, v in r.items() if k != "step"}, step=r["step"])

wandb.define_metric("ckpt_step")
wandb.define_metric("ckpt/*", step_metric="ckpt_step")
for ck in sorted(run_dir.glob("step-*"), key=lambda p: int(p.name.split("-")[1])):
    res = ck / "eval" / "results.json"
    if res.exists():
        m = flatten(json.loads(res.read_text(encoding="utf-8")), "ckpt")
        wandb.log({"ckpt_step": int(ck.name.split("-")[1]), **m})

if (run_dir / "metrics.json").exists():
    final = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    run.summary.update(flatten(final, "final"))
for table in ("intern_table.md",):
    if (run_dir / table).exists():
        wandb.save(str(run_dir / table), base_path=str(run_dir), policy="now")
run.finish()
print(f"uploaded {run_dir} -> {run.url or run_id}")
