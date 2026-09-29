"""Intern-Decision style: one prompt per record, an assistant JSON skeleton with one
``<decision>`` placeholder per question. The LM-head logits at the position right
before each placeholder, restricted to the question's option symbols (A, B, ...),
are that question's distribution. All questions share one forward pass.
"""

from __future__ import annotations

import json
import string

import torch

from jevtrainer.readouts.base import Read, Readout, option_text, record_images
from jevtrainer.registry import READOUTS
from jevtrainer.schema import Record

SYMBOLS = string.ascii_uppercase + string.ascii_lowercase + string.digits
SYSTEM = (
    "You are a decision model. Read the state and the decision schema, then fill every field of the "
    "JSON object with the symbol of exactly one option."
)


@READOUTS.register("marker")
class MarkerReadout(Readout):
    name = "marker"
    max_options = len(SYMBOLS)
    special_tokens = ["<decision>"]

    def build(self, family, model) -> None:
        ids = []
        for s in SYMBOLS:
            t = self.tok.encode(s, add_special_tokens=False)
            if len(t) != 1:
                raise ValueError(f"symbol {s!r} is not a single token for this tokenizer")
            ids.append(t[0])
        self.symbol_ids = ids
        if family.lm_head(model) is None:
            raise ValueError("marker readout needs a model with an LM head")

    def messages(self, record: Record, n_images: int) -> list[dict]:
        lines = ["## State", self.truncate_state(record), "", "## Decision schema"]
        for name, q in record.questions.items():
            lines.append(f"### {name} ({q.type})")
            lines.append(self.clean(q.instructions))
            for sym, (label, desc) in zip(SYMBOLS, q.options()):
                lines.append(f"{sym} = {self.clean(option_text(label, desc, 'choice'))}")
            lines.append("")
        user = "\n".join(lines)
        skeleton = json.dumps({k: "<decision>" for k in record.questions}, indent=4, ensure_ascii=False)
        content = [{"type": "image"}] * n_images + [{"type": "text", "text": user}] if n_images else user
        return [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": content},
            {"role": "assistant", "content": skeleton},
        ]

    def encode(self, record: Record, rng=None):
        for name, q in record.questions.items():
            if len(q.labels()) > self.max_options:
                raise ValueError(f"{record.id}/{name}: marker supports at most {self.max_options} options")
        images = record_images(record) if record.images else None
        text = self.family.chat_text(self.tok, self.processor, self.messages(record, len(images or [])), False)
        row = self.tokenize(text, images)
        pos = self.positions(row.input_ids, "<decision>")
        if len(pos) != len(record.questions):
            raise ValueError(f"{record.id}: expected {len(record.questions)} markers, found {len(pos)} (truncated?)")
        for p, (name, q) in zip(pos, record.questions.items()):
            row.reads.append(Read(0, name, {"pos": p - 1, "k": len(q.labels())}))
        return [row]

    def score(self, model, hidden, batch):
        reads = batch["reads"]
        rows = torch.tensor([r for r, _ in reads], device=hidden.device)
        pos = torch.tensor([rd.info["pos"] for _, rd in reads], device=hidden.device)
        logits = self.family.lm_head(model)(hidden[rows, pos]).float()  # [N, V], only N positions
        sym = torch.tensor(self.symbol_ids, device=hidden.device)
        full_vocab = self.cfg.options.get("full_vocab_loss", False) and self.training
        out = []
        for i, (_, rd) in enumerate(reads):
            ids = sym[: rd.info["k"]]
            cand = logits[i, ids]
            if full_vocab:
                rest = logits[i].clone()
                rest[ids] = float("-inf")
                cand = torch.cat([cand, torch.logsumexp(rest, 0, keepdim=True)])
            out.append(cand)
        return out
