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

from jevtrainer.media import MediaOptions, load_media
from jevtrainer.media import inline as inline_media
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
    mm_token_type_ids: list[int] | None = None
    extra: dict[str, torch.Tensor] = field(default_factory=dict)  # processor outputs per media item (pixel_values, ...)

    @property
    def pixel_values(self):
        return self.extra.get("pixel_values")

    @property
    def image_grid_thw(self):
        return self.extra.get("image_grid_thw")


SEQUENCE_KEYS = ("input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids")


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
            emb = family.input_embeddings(model)
            n, d = emb.weight.shape
            self.special = nn.Embedding(len(ids), d)
            with torch.no_grad():  # through the module: some embeddings scale their output (Gemma: x sqrt(hidden))
                idx = torch.randint(0, n, (4096,))
                sample = emb(idx.to(emb.weight.device)).float().cpu()
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

    def tokenize(self, text: str, images: list | None = None, add_special_tokens: bool = False, plan=None) -> Row:
        """Text -> Row. With `images` (legacy) or a media `plan` (see `state_for`) the processor expands placeholders."""
        if plan is not None:
            if self.processor is None:
                raise ValueError("record has media but the model has no processor")
            media, order = plan
            out = self.family.process_media(self.processor, text, media, order, self.media_opts)
            return self._row_from(out)
        if images:
            if self.processor is None:
                raise ValueError("record has images but the model has no processor")
            return self._row_from(self.processor(text=[text], images=images, return_tensors="pt"))
        return Row(self.tok(text, add_special_tokens=add_special_tokens).input_ids, [])

    @staticmethod
    def _row_from(out) -> Row:
        row = Row(out["input_ids"][0].tolist(), [])
        if out.get("mm_token_type_ids") is not None:
            row.mm_token_type_ids = out["mm_token_type_ids"][0].tolist()
        row.extra = {k: v for k, v in out.items() if k not in SEQUENCE_KEYS and isinstance(v, torch.Tensor)}
        return row

    def images(self, record: Record) -> list | None:
        """Loaded images, within the `image_pixel_budget` option (total pixels per record)."""
        return record_images(record, self.cfg.options.get("image_pixel_budget")) if record.images else None

    @property
    def media_opts(self) -> MediaOptions:
        if getattr(self, "_media_opts", None) is None:
            self._media_opts = MediaOptions.from_options(self.cfg.options)
        return self._media_opts

    def state_for(self, record: Record) -> tuple[str, tuple | None]:
        """The (truncated) state text and, for records with `media`, the plan to hand to `tokenize`.

        Tags ``<image:N>`` / ``<video:N>`` / ``<audio:N>`` in the state become the family's placeholders at the
        same place; media the state does not mention is placed in front.
        """
        text = self.truncate_state(record)
        if not record.media:
            return text, None
        if record.images:
            raise ValueError(f"{record.id}: use either `images` or `media`, not both")
        cache = getattr(self, "_media_cache", None)
        if cache is not None and cache[0] is record:
            media = cache[1]
        else:
            media = load_media(record, self.media_opts)
            self._media_cache = (record, media)
        text, order = inline_media(text, media, lambda kind, has_audio: self.family.media_placeholder(kind, has_audio, self.processor))
        return text, (media, order)

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
        keys = list(dict.fromkeys(k for r in rows for k in r.extra))
        for k in keys:
            batch[k] = cat_padded([r.extra[k] for r in rows if k in r.extra])
        if keys or any(r.mm_token_type_ids for r in rows):
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
        embeds = self.family.merge_multimodal(model, batch, self.embed(model, batch["input_ids"]))
        args = {"inputs_embeds": embeds, "attention_mask": batch["attention_mask"]}
        args.update(self.family.forward_kwargs(model, batch))  # a family may replace the attention mask
        if self.family.causal:
            args["use_cache"] = False
        return backbone(**args).last_hidden_state

    def forward(self, model: nn.Module, batch: dict) -> list[torch.Tensor]:
        """Logits per read, in the order of batch['reads']."""
        return self.score(model, self.hidden(model, batch), batch)

    def score(self, model: nn.Module, hidden: torch.Tensor, batch: dict) -> list[torch.Tensor]:
        raise NotImplementedError

    # ---- persistence ----------------------------------------------------------
    def state(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.state_dict().items()}


def cat_padded(parts: list[torch.Tensor]) -> torch.Tensor:
    """Concatenate along dim 0, zero-padding trailing dims that differ (audio features of different lengths)."""
    if len(parts) == 1:
        return parts[0]
    nd = parts[0].dim()
    if nd < 2 or all(p.shape[1:] == parts[0].shape[1:] for p in parts):
        return torch.cat(parts)
    shape = [max(p.shape[d] for p in parts) for d in range(1, nd)]
    out = []
    for p in parts:
        pad = []
        for d in range(nd - 1, 0, -1):
            pad += [0, shape[d - 1] - p.shape[d]]
        out.append(torch.nn.functional.pad(p, pad))
    return torch.cat(out)


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
