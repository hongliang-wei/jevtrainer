"""Supervised trainer driven entirely by TrainConfig.

`Trainer.loss(batch)` is the only method a new training method (e.g. RL) needs to
override; data, optimisation, saving, calibration and evaluation stay the same.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from jevtrainer.config import TrainConfig
from jevtrainer.data.mixture import Source, build_mixture, split_holdout
from jevtrainer.model.load import Bundle, build, count, prepare_finetune, save, trainable_parameters
from jevtrainer.registry import LOSSES, TRAINERS
from jevtrainer.train import losses as L
from jevtrainer.train.batching import Collator, DecisionModel, LengthGroupedBatches, to_device
from jevtrainer.train.calibrate import fit


def flatten(d, prefix: str = "") -> dict[str, float]:
    """Nested metrics -> {"a/b/c": number} for experiment trackers."""
    if isinstance(d, bool):
        return {}
    if isinstance(d, (int, float)):
        return {prefix: d}
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            out.update(flatten(v, f"{prefix}/{k}" if prefix else str(k)))
        return out
    return {}


@TRAINERS.register("sft")
class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.out = Path(cfg.output_dir)
        self.loss_fn = LOSSES.get(cfg.loss)

    # ---- pieces a subclass may override -----------------------------------------
    def loss(self, logits: list[torch.Tensor], batch: dict) -> torch.Tensor:
        total = []
        for z, t in zip(logits, batch["targets"]):
            if t["index"] is None and t["soft"] is None:
                continue
            soft = t["soft"].to(z.device) if t["soft"] is not None else None
            l = self.loss_fn(z, t["k"], t["index"], soft)
            if self.cfg.score_rps_weight and t["type"] == "score":
                l = l + self.cfg.score_rps_weight * L.rps(z, t["k"], t["index"], soft)
            total.append(l)
        return torch.stack(total).mean() if total else logits[0].sum() * 0

    # ---- setup -------------------------------------------------------------------
    def load_model(self) -> Bundle:
        c = self.cfg
        dtype = "fp32" if c.finetune == "full" else c.dtype
        b = build(c.model, c.readout, c.family, dtype, c.readout_options, c.max_state_tokens)
        prepare_finetune(b, c.finetune, c.lora.model_dump(), c.freeze_vision, c.grad_ckpt)
        return b

    def load_data(self):
        from jevtrainer.eval.base import benchmark_records

        c = self.cfg
        exclude = []
        if c.exclude_eval_overlap:
            for name in dict.fromkeys(c.eval_dataset + c.exclude):
                exclude.extend(benchmark_records(name))
        sources = [Source(s.name, s.split, s.max_samples or c.max_samples, s.weight) for s in c.dataset]
        records, report = build_mixture(sources, c.seed, exclude)
        train, hold = split_holdout(records, c.holdout, c.seed)
        return train, hold, report

    def dry_run(self) -> dict:
        train, hold, report = self.load_data()
        b = self.load_model()
        info = {
            "config": self.cfg.model_dump(),
            "data": report,
            "train_records": len(train),
            "holdout_records": len(hold),
            "trainable_params": count(trainable_parameters(b)),
            "total_params": count(b.model.parameters()) + count(b.readout.parameters()),
            "steps": self.num_steps(len(train)),
        }
        collate = Collator(b.readout, self.cfg.max_length, self.cfg.augment.to_config(), self.cfg.seed)
        sample = collate(train[: min(4, len(train))])
        if sample is not None:
            info["sample_row_tokens"] = [int(x) for x in sample["attention_mask"].sum(1)]
            info["sample_text"] = b.tok.decode(sample["input_ids"][0][: int(sample["attention_mask"][0].sum())])
        return info

    def num_steps(self, n_train: int) -> int:
        c = self.cfg
        per_epoch = math.ceil(n_train / (c.batch_size * c.grad_accum))
        return c.max_steps or max(1, int(per_epoch * c.epochs))

    # ---- main loop -----------------------------------------------------------------
    def run(self) -> dict:
        from accelerate import Accelerator

        c = self.cfg
        torch.manual_seed(c.seed)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "config.yaml").write_text(yaml.safe_dump(c.model_dump(), allow_unicode=True, sort_keys=False), encoding="utf-8")
        train, hold, report = self.load_data()
        (self.out / "data_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not train:
            raise ValueError(f"no training records left after holdout={c.holdout}; see {self.out / 'data_report.json'}")

        acc = Accelerator(
            mixed_precision="no" if c.dtype == "fp32" else c.dtype,
            gradient_accumulation_steps=c.grad_accum,
            log_with=None if c.report_to == "none" else c.report_to,
            project_dir=str(self.out),
        )
        b = self.load_model()
        net = DecisionModel(b.model, b.readout)
        head = [p for p in b.readout.parameters() if p.requires_grad]
        body = [p for p in b.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            [{"params": body, "lr": c.lr}, {"params": head, "lr": c.head_lr}], weight_decay=c.weight_decay, betas=(0.9, 0.98)
        )
        steps = self.num_steps(len(train))
        warm = int(steps * c.warmup_ratio)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps))))
        )
        workers = c.num_workers if os.name != "nt" else 0
        collate = Collator(b.readout, c.max_length, c.augment.to_config(), c.seed)
        if c.group_by_length:
            loader = DataLoader(train, batch_sampler=LengthGroupedBatches(train, c.batch_size, c.seed), collate_fn=collate, num_workers=workers)
        else:
            loader = DataLoader(train, batch_size=c.batch_size, shuffle=True, collate_fn=collate, num_workers=workers, drop_last=True)
        net, opt, sched = acc.prepare(net, opt, sched)
        if c.report_to != "none":
            acc.init_trackers(
                os.environ.get("WANDB_PROJECT", "jevtrainer"),
                config=c.model_dump(),
                init_kwargs={"wandb": {"name": self.out.name, "dir": os.environ.get("WANDB_DIR")}} if c.report_to == "wandb" else {},
            )
            if c.report_to == "wandb" and acc.is_main_process:
                (self.out / "wandb_id.txt").write_text(acc.get_tracker("wandb", unwrap=True).id, encoding="utf-8")
        acc.print(f"train={len(train)} holdout={len(hold)} steps={steps} trainable={count(body) + count(head):,}")

        log = open(self.out / "train_log.jsonl", "a", encoding="utf-8")
        step, t0, running = 0, time.time(), []
        net.train()
        while step < steps:
            for batch in loader:
                if batch is None:
                    continue
                with acc.accumulate(net):
                    logits = net(to_device(batch, acc.device))
                    loss = self.loss(logits, batch)
                    acc.backward(loss)
                    if acc.sync_gradients:
                        acc.clip_grad_norm_(body + head, c.max_grad_norm)
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                running.append(float(loss.detach()))
                if not acc.sync_gradients:
                    continue
                step += 1
                if step % c.log_steps == 0 or step == steps:
                    rec = {"step": step, "loss": sum(running) / len(running), "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1)}
                    running = []
                    acc.print(json.dumps(rec))
                    log.write(json.dumps(rec) + "\n")
                    log.flush()
                    if c.report_to != "none":
                        acc.log(rec, step=step)
                if c.save_steps and step % c.save_steps == 0:
                    self.save(b, acc, self.out / f"step-{step}")
                if c.eval_every and step % c.eval_every == 0 and hold:
                    val = self.validate(b, hold)
                    acc.print(json.dumps({"step": step, "holdout": val}))
                    if c.report_to != "none":
                        acc.log(flatten(val, "holdout"), step=step)
                    net.train()
                if step >= steps:
                    break
        log.close()

        result = {"steps": step, "train_records": len(train)}
        if hold:
            outs = self.collect(b, hold)
            from jevtrainer.eval.metrics import summarize

            result["holdout_uncalibrated"] = summarize(outs)
            if c.calibrate:
                temps = fit(outs, c.calibrate_per_type)
                (self.out / "calibration.json").write_text(json.dumps({"temperature": temps, "fit_on": "holdout", "n": len(outs)}, indent=2), encoding="utf-8")
                b.temperature = temps
                result["temperature"] = temps
                result["holdout"] = summarize(outs, temps)
        self.save(b, acc, self.out)
        if c.eval_dataset:
            from jevtrainer.eval.runner import run_benchmarks

            result["benchmarks"] = run_benchmarks(b, c.eval_dataset, self.out / "eval", c.eval_max_samples)
        (self.out / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        if c.report_to != "none":
            acc.log(flatten(result, "final"), step=step)
        acc.end_training()
        return result

    def collect(self, b: Bundle, records):
        from jevtrainer.eval.runner import collect

        return collect(b, records, batch_size=self.cfg.batch_size, max_length=self.cfg.max_length)

    def validate(self, b: Bundle, hold) -> dict:
        from jevtrainer.eval.metrics import summarize

        return summarize(self.collect(b, hold[:512]), b.temperature)

    def save(self, b: Bundle, acc, path: Path) -> None:
        if not acc.is_main_process:
            return
        c = self.cfg
        meta = {
            "model": c.model,
            "readout": c.readout,
            "family": c.family,
            "readout_options": c.readout_options,
            "max_state_tokens": c.max_state_tokens,
            "max_length": c.max_length,
            "finetune": c.finetune,
        }
        save(b, path, meta)
        if b.temperature and path != self.out:
            (path / "calibration.json").write_text(json.dumps({"temperature": b.temperature}), encoding="utf-8")
