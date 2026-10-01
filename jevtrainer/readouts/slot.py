"""Bosun v3.1 style: 256 reserved slot tokens ``<|decision_000|>`` ... One prompt per
question; its options are shuffled onto slots 0..K-1, and the last token's hidden
state is scored against the slot output embeddings. The shuffle means a slot id
carries no meaning of its own, only "the option presented in this slot".
"""

from __future__ import annotations

import hashlib
import json
import random

import torch
from torch import nn

from jevtrainer.readouts.base import NEUTRAL_KEY, Read, Readout
from jevtrainer.registry import READOUTS
from jevtrainer.schema import Record

N_SLOTS = 256
SYSTEM = "Choose exactly one supplied decision token. Reply with that token only."


@READOUTS.register("slot")
class SlotReadout(Readout):
    name = "slot"
    max_options = N_SLOTS - 1
    special_tokens = [f"<|decision_{i:03d}|>" for i in range(N_SLOTS)]

    def build(self, family, model) -> None:
        if not hasattr(self, "slot_out"):
            self.slot_out = nn.Parameter(self.special.weight.detach().clone())

    def order(self, record: Record, name: str, k: int, rng) -> list[int]:
        """slot index for each option (in label order)."""
        slots = list(range(k))
        if rng is not None:
            rng.shuffle(slots)
        else:
            seed = int(hashlib.sha256(f"{record.id}:{name}".encode()).hexdigest()[:16], 16)
            random.Random(seed).shuffle(slots)
        return slots

    def encode(self, record: Record, rng=None):
        images = self.images(record)
        state, plan = self.state_for(record)
        rows = []
        for name, q in record.questions.items():
            opts = q.options()
            slots = self.order(record, name, len(opts), rng)
            presented = sorted(range(len(opts)), key=lambda i: slots[i])
            criteria = [
                {"t": f"<|decision_{slots[i]:03d}|>", "n": "" if NEUTRAL_KEY.fullmatch(opts[i][0]) else self.clean(opts[i][0]), "d": self.clean(opts[i][1])}
                for i in presented
            ]
            question = {"type": q.type, "instructions": self.clean(q.instructions), "criteria": criteria}
            user = json.dumps({"state": state, "question": question}, ensure_ascii=False)
            content = [{"type": "image"}] * len(images) + [{"type": "text", "text": user}] if images else user
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
            row = self.tokenize(self.family.chat_text(self.tok, self.processor, msgs, True), images, plan=plan)
            row.reads.append(Read(0, name, {"pos": len(row.input_ids) - 1, "slots": slots}))
            rows.append(row)
        return rows

    def score(self, model, hidden, batch):
        out = []
        for row, rd in batch["reads"]:
            h = hidden[row, rd.info["pos"]].float()
            w = self.slot_out[torch.tensor(rd.info["slots"], device=h.device)].float()
            out.append(w @ h)
        return out
