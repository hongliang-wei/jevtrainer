"""Qwen2.5 / Qwen3 / Qwen3.5 / Qwen3.6 / Qwen3.8, text and VL."""

from __future__ import annotations

import torch

from jevtrainer.model.families.base import ModelFamily
from jevtrainer.registry import FAMILIES


@FAMILIES.register("qwen")
class QwenFamily(ModelFamily):
    name = "qwen"
    model_types = (
        "qwen2", "qwen2_moe", "qwen2_vl", "qwen2_5_vl", "qwen3", "qwen3_moe", "qwen3_next",
        "qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe",
    )

    def image_placeholder(self, processor) -> str:
        return "<|vision_start|><|image_pad|><|vision_end|>"

    def forward_kwargs(self, model, batch: dict) -> dict:
        kw = super().forward_kwargs(model, batch)
        backbone = self.backbone(model)
        if not hasattr(backbone, "get_rope_index"):
            return kw
        # We feed inputs_embeds, so the model cannot build M-RoPE positions itself; compute them here.
        backbone.rope_deltas = None
        mask = batch["attention_mask"]
        if batch.get("image_grid_thw") is not None:
            pos, _ = backbone.get_rope_index(
                batch["input_ids"],
                mm_token_type_ids=batch["mm_token_type_ids"],
                image_grid_thw=batch["image_grid_thw"],
                attention_mask=mask,
            )
        else:
            pos = (mask.long().cumsum(-1) - 1).clamp(min=0)
            pos = pos.unsqueeze(0).expand(3, -1, -1)
        kw["position_ids"] = pos
        return kw
