"""Bidirectional encoders (ModernBERT, mmBERT, DeBERTa). Usable with slot and pointer readouts."""

from __future__ import annotations

import torch
from torch import nn

from jevtrainer.model.families.base import ModelFamily
from jevtrainer.registry import FAMILIES


@FAMILIES.register("encoder")
class EncoderFamily(ModelFamily):
    name = "encoder"
    model_types = ("modernbert", "bert", "roberta", "xlm-roberta", "deberta", "deberta-v2")
    causal = False

    def load_model(self, path: str, dtype: torch.dtype, attn_implementation: str | None = None) -> nn.Module:
        from transformers import AutoModel

        kw = {"dtype": dtype}
        if attn_implementation:
            kw["attn_implementation"] = attn_implementation
        return AutoModel.from_pretrained(path, **kw)

    def backbone(self, model: nn.Module) -> nn.Module:
        return self.base(model)

    def lm_head(self, model: nn.Module) -> None:
        return None
