"""Records -> padded batches with targets, shared by training and evaluation."""

from __future__ import annotations

import random

import torch
from torch import nn

from jevtrainer.data.augment import AugmentConfig, augment, fit_options
from jevtrainer.readouts import Readout
from jevtrainer.schema import Record


class DecisionModel(nn.Module):
    """Backbone + readout as one module: forward(batch) -> list of option-logit tensors."""

    def __init__(self, model: nn.Module, readout: Readout):
        super().__init__()
        self.model, self.readout = model, readout

    def forward(self, batch: dict) -> list[torch.Tensor]:
        return self.readout(self.model, batch)


class Collator:
    def __init__(self, readout: Readout, max_length: int, augment_cfg: AugmentConfig | None = None, seed: int = 0):
        self.readout, self.max_length, self.aug = readout, max_length, augment_cfg
        self.rng = random.Random(seed)

    def __call__(self, records: list[Record]) -> dict | None:
        if self.aug is not None:
            records = augment(records, self.aug, self.rng)
        rows, used = [], []
        for r in records:
            if self.aug is not None:  # never change an evaluation task; unfit records are skipped instead
                r = fit_options(r, self.readout.max_options, self.rng)
            try:
                rr = self.readout.encode(r, self.rng if self.aug is not None else None)
            except ValueError:
                continue
            if any(len(x.input_ids) > self.max_length for x in rr):
                continue
            for x in rr:
                for rd in x.reads:
                    rd.record = len(used)
            rows.extend(rr)
            used.append(r)
        if not rows:
            return None
        batch = self.readout.collate(rows)
        batch["records"] = used
        batch["targets"] = [_target(used[rd.record], rd.name) for _, rd in batch["reads"]]
        return batch


def _target(r: Record, name: str) -> dict:
    q = r.questions[name]
    labels = q.labels()
    t = r.targets.get(name)
    soft = None
    if t is not None and t.probs:
        soft = torch.tensor([t.probs.get(l, 0.0) for l in labels], dtype=torch.float)
        soft = soft / soft.sum() if soft.sum() > 0 else None
    return {
        "type": q.type,
        "k": len(labels),
        "index": labels.index(t.label) if t is not None and t.label is not None else None,
        "soft": soft,
    }


IMAGE_CHARS = 1000  # rough text-equivalent of one image; many small frames count as at most 8


def approx_length(r: Record) -> int:
    text = len(r.state_text()) + sum(len(q.instructions) + sum(map(len, q.descriptions())) for q in r.questions.values())
    return text + IMAGE_CHARS * min(len(r.images), 8)


class LengthGroupedBatches:
    """Shuffled batches of records with similar length (less padding). Re-shuffles every epoch."""

    def __init__(self, records: list[Record], batch_size: int, seed: int = 0, mega: int = 64):
        self.lengths = [approx_length(r) for r in records]
        self.batch_size, self.mega, self.rng = batch_size, mega, random.Random(seed)

    def __iter__(self):
        idx = list(range(len(self.lengths)))
        self.rng.shuffle(idx)
        size = self.batch_size * self.mega
        batches = []
        for i in range(0, len(idx), size):
            chunk = sorted(idx[i : i + size], key=self.lengths.__getitem__)
            batches += [chunk[j : j + self.batch_size] for j in range(0, len(chunk) - self.batch_size + 1, self.batch_size)]
        self.rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return len(self.lengths) // self.batch_size


def to_device(batch: dict, device) -> dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
