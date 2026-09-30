"""Kev style pointer head. One row per question (state + that question), so it also
works on hybrid models whose linear-attention layers ignore block masks:

    <state> state <q> instructions (<opt> option </opt>)* <decide>

logit(option) = W_k h(</opt>) . W_q h(<decide>) / sqrt(d_head)
"""

from __future__ import annotations

import math

import torch
from torch import nn

from jevtrainer.readouts.base import Read, Readout, option_text
from jevtrainer.registry import READOUTS
from jevtrainer.schema import Record


@READOUTS.register("pointer")
class PointerReadout(Readout):
    name = "pointer"
    max_options = 255
    special_tokens = ["<state>", "<q>", "<opt>", "</opt>", "<decide>"]

    def build(self, family, model) -> None:
        if not hasattr(self, "q_proj"):
            d, dp = family.hidden_size(model), int(self.cfg.options.get("head_dim", 256))
            self.q_proj, self.k_proj = nn.Linear(d, dp), nn.Linear(d, dp)
            self.scale = 1 / math.sqrt(dp)

    def encode(self, record: Record, rng=None):
        images = self.images(record)
        prefix = self.tok.bos_token or ""
        if images:
            prefix += self.family.image_placeholder(self.processor) * len(images)
        head = f"{prefix}<state>{self.truncate_state(record)}"
        rows = []
        for name, q in record.questions.items():
            opts = "".join(f"<opt>{self.clean(option_text(l, d, q.type))}</opt>" for l, d in q.options())
            text = f"{head}<q>{self.clean(q.instructions)}{opts}<decide>"
            row = self.tokenize(text, images, add_special_tokens=not self.family.causal)
            ends = self.positions(row.input_ids, "</opt>")
            decide = self.positions(row.input_ids, "<decide>")
            if len(ends) != len(q.labels()) or len(decide) != 1:
                raise ValueError(f"{record.id}/{name}: option markers were lost; state or option text contains them?")
            row.reads.append(Read(0, name, {"opts": ends, "decide": decide[0]}))
            rows.append(row)
        return rows

    def score(self, model, hidden, batch):
        out = []
        for row, rd in batch["reads"]:
            h = hidden[row].float()
            qv = self.q_proj(h[rd.info["decide"]])
            kv = self.k_proj(h[rd.info["opts"]])
            out.append(kv @ qv * self.scale)
        return out
