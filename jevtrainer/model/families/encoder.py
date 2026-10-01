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

    def load_model(self, path: str, dtype: torch.dtype, attn_implementation: str | None = None, quantize: str = "none",
                   device_map: str | dict | None = None, max_memory: dict | None = None) -> nn.Module:
        from transformers import AutoModel

        from jevtrainer.model.families.base import quantization_kwargs

        kw = {"dtype": dtype, **quantization_kwargs(quantize, dtype)}
        if attn_implementation:
            kw["attn_implementation"] = attn_implementation
        if device_map is not None:
            kw["device_map"] = device_map
            if max_memory:
                kw["max_memory"] = max_memory
        return AutoModel.from_pretrained(path, **kw)

    def backbone(self, model: nn.Module) -> nn.Module:
        return self.base(model)

    def lm_head(self, model: nn.Module) -> None:
        return None
