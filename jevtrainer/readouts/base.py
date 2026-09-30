"""Readout: how a record becomes token rows, and how option logits are read back.

Every readout returns, for every question it reads, a 1-D tensor of logits in the
question's own label order (``Question.labels()``). Everything downstream (losses,
calibration, prediction, serving) only sees those tensors.

Special tokens (``<decision>``, ``<|decision_007|>``, ``<decide>``, ...) are added to
the tokenizer only. The model's vocabulary is never resized: at forward time the
positions holding a special id get embeddings from a small table owned by the
readout, which is trainable under LoRA and full fine-tuning alike.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from jevtrainer.schema import Record


@dataclass
class Read:
    """One question read from one row."""

    record: int  # index of the record in the batch
    name: str  # question name
    info: dict[str, Any]  # readout-specific positions / token ids


@dataclass
class Row:
    input_ids: list[int]
    reads: list[Read]
    pixel_values: torch.Tensor | None = None
    image_grid_thw: torch.Tensor | None = None
    mm_token_type_ids: list[int] | None = None


@dataclass
class ReadoutConfig:
    max_state_tokens: int = 2048
    max_length: int = 4096
    options: dict[str, Any] = field(default_factory=dict)


class Readout(nn.Module):
    name = "base"
    max_options = 255
    special_tokens: list[str] = []

    def __init__(self, cfg: ReadoutConfig | None = None):
        super().__init__()
        self.cfg = cfg or ReadoutConfig()
        self.special = None  # nn.Embedding, created in setup()
        self.first_special_id = None

    # ---- setup ---------------------------------------------------------------
    def setup(self, family, model: nn.Module, tok, processor=None) -> None:
        """Attach tokenizer and model facts. Must be called before encode/forward."""
        self.family, self.tok, self.processor = family, tok, processor
        missing = [t for t in self.special_tokens if t not in tok.get_vocab()]
        if missing:
            tok.add_special_tokens({"additional_special_tokens": missing}, replace_extra_special_tokens=False)
        ids = tok.convert_tokens_to_ids(self.special_tokens)
        self.special_ids = dict(zip(self.special_tokens, ids))
        self.first_special_id = min(ids) if ids else None
        if ids and (max(ids) - min(ids) + 1 != len(ids) or max(ids) != len(tok) - 1):
            raise ValueError("readout special tokens must be the last contiguous ids of the tokenizer")
        if self.special is None and ids:
            emb = family.input_embeddings(model).weight
            d = emb.shape[1]
            self.special = nn.Embedding(len(ids), d)
            with torch.no_grad():
                sample = emb[torch.randint(0, emb.shape[0], (4096,))].float()
                self.special.weight.copy_(sample.mean(0) + sample.std(0) * torch.randn(len(ids), d) * 0.5)
        self.build(family, model)

    def build(self, family, model: nn.Module) -> None:
        """Create readout-owned parameters that depend on the hidden size."""

    # ---- encoding ------------------------------------------------------------
    def encode(self, record: Record, rng=None) -> list[Row]:
        raise NotImplementedError

    def clean(self, text: str) -> str:
        """Stop user text from forging readout tokens."""
        for t in self.special_tokens:
            if t in text:
                text = text.replace(t, t.replace("<", "‹"))
        return text

    def truncate_state(self, record: Record) -> str:
        text = self.clean(record.state_text())
        ids = self.tok(text, add_special_tokens=False).input_ids
        if len(ids) > self.cfg.max_state_tokens:
            text = self.tok.decode(ids[: self.cfg.max_state_tokens]) + " …"
        return text

    def tokenize(self, text: str, images: list | None = None, add_special_tokens: bool = False) -> Row:
        if images:
            if self.processor is None:
                raise ValueError("record has images but the model has no processor")
            out = self.processor(text=[text], images=images, return_tensors="pt")
            row = Row(out["input_ids"][0].tolist(), [])
            row.pixel_values = out.get("pixel_values")
            row.image_grid_thw = out.get("image_grid_thw")
            if out.get("mm_token_type_ids") is not None:
                row.mm_token_type_ids = out["mm_token_type_ids"][0].tolist()
            return row
        return Row(self.tok(text, add_special_tokens=add_special_tokens).input_ids, [])

    def images(self, record: Record) -> list | None:
        """Loaded images, within the `image_pixel_budget` option (total pixels per record)."""
        return record_images(record, self.cfg.options.get("image_pixel_budget")) if record.images else None

    def positions(self, ids: list[int], token: str) -> list[int]:
        tid = self.special_ids[token]
        return [i for i, t in enumerate(ids) if t == tid]

    # ---- batching ------------------------------------------------------------
    def collate(self, rows: list[Row]) -> dict:
        pad = self.tok.pad_token_id
        n = max(len(r.input_ids) for r in rows)
        ids = torch.full((len(rows), n), pad, dtype=torch.long)
        mask = torch.zeros((len(rows), n), dtype=torch.long)
        mm = torch.zeros((len(rows), n), dtype=torch.long)
        reads = []
        for i, r in enumerate(rows):
            ids[i, : len(r.input_ids)] = torch.tensor(r.input_ids)
            mask[i, : len(r.input_ids)] = 1
            if r.mm_token_type_ids:
                mm[i, : len(r.mm_token_type_ids)] = torch.tensor(r.mm_token_type_ids)
            reads.extend((i, rd) for rd in r.reads)
        batch = {"input_ids": ids, "attention_mask": mask, "reads": reads}
        pv = [r.pixel_values for r in rows if r.pixel_values is not None]
        if pv:
            batch["pixel_values"] = torch.cat(pv)
            batch["image_grid_thw"] = torch.cat([r.image_grid_thw for r in rows if r.image_grid_thw is not None])
            batch["mm_token_type_ids"] = mm
        return batch

    # ---- forward -------------------------------------------------------------
    def embed(self, model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
        emb = self.family.input_embeddings(model)
        if self.special is None:
            return emb(input_ids)
        is_sp = input_ids >= self.first_special_id
        x = emb(input_ids.masked_fill(is_sp, 0))
        if is_sp.any():
            sp = self.special(input_ids[is_sp] - self.first_special_id).to(x.dtype)
            x = x.masked_scatter(is_sp.unsqueeze(-1), sp)
        return x

    def hidden(self, model: nn.Module, batch: dict) -> torch.Tensor:
        backbone = self.family.backbone(model)
        kw = self.family.forward_kwargs(model, batch)
        if self.family.causal:
            kw["use_cache"] = False
        out = backbone(inputs_embeds=self.embed(model, batch["input_ids"]), attention_mask=batch["attention_mask"], **kw)
        return out.last_hidden_state

    def forward(self, model: nn.Module, batch: dict) -> list[torch.Tensor]:
        """Logits per read, in the order of batch['reads']."""
        return self.score(model, self.hidden(model, batch), batch)

    def score(self, model: nn.Module, hidden: torch.Tensor, batch: dict) -> list[torch.Tensor]:
        raise NotImplementedError

    # ---- persistence ----------------------------------------------------------
    def state(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.state_dict().items()}


def load_image(ref: str, root: str | None = None):
    from PIL import Image

    if ref.startswith(("http://", "https://")):
        import urllib.request

        with urllib.request.urlopen(ref, timeout=30) as r:
            return Image.open(io.BytesIO(r.read())).convert("RGB")
    p = Path(ref)
    if not p.is_absolute() and root:
        p = Path(root) / p
    return Image.open(p).convert("RGB")


def record_images(record: Record, pixel_budget: int | None = None) -> list:
    """The record's images; if their total area exceeds pixel_budget, all are shrunk by the same factor."""
    from PIL import Image

    root = record.meta.get("image_root")
    images = [load_image(ref, root) for ref in record.images]
    total = sum(im.width * im.height for im in images)
    if pixel_budget and total > pixel_budget:
        s = (pixel_budget / total) ** 0.5
        images = [im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BICUBIC) for im in images]
    return images


NEUTRAL_KEY = re.compile(r"(opt|option|element|choice)_\d+")


def option_text(label: str, desc: str, qtype: str) -> str:
    """What the model reads for one option. Neutral keys (opt_3) carry no meaning, so only the text is shown."""
    if qtype == "score" or (desc and NEUTRAL_KEY.fullmatch(label)):
        return desc
    if not desc or desc == label:
        return label
    return f"{label}: {desc}"
