"""Qwen2.5-Omni and Qwen3-Omni: text, image, video and audio in one sequence (the "thinker" only).

The talker / code2wav speech-output parts are never loaded. A video's own sound track goes in with it
(`use_audio_in_video`): the processor interleaves video and audio tokens in 2-second chunks and the model's
TMRoPE positions put both on one time axis, so a record can ask "what is being said while this happens".

We feed `inputs_embeds`, so the thinker cannot build its multimodal inputs itself: `merge_multimodal` encodes
images / videos / audio and writes them over the placeholder positions, `forward_kwargs` computes the 3-axis
positions and, for Qwen3-Omni, the DeepStack features the vision tower adds to the first decoder layers.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import AutoConfig, PretrainedConfig

from jevtrainer.model.families.base import ModelFamily, quantization_kwargs
from jevtrainer.registry import FAMILIES


@FAMILIES.register("omni")
class QwenOmniFamily(ModelFamily):
    name = "omni"
    model_types = ("qwen2_5_omni", "qwen2_5_omni_thinker", "qwen3_omni_moe", "qwen3_omni_moe_thinker")
    thinker_classes = {
        "qwen2_5_omni": "Qwen2_5OmniThinkerForConditionalGeneration",
        "qwen3_omni_moe": "Qwen3OmniMoeThinkerForConditionalGeneration",
    }

    # ---- loading -------------------------------------------------------------
    def is_multimodal(self, config: PretrainedConfig) -> bool:
        return True

    def load_model(self, path, dtype, attn_implementation=None, quantize="none", device_map=None, max_memory=None):
        import transformers

        config = AutoConfig.from_pretrained(path)
        kind = str(config.model_type).replace("_thinker", "")
        thinker_cfg = getattr(config, "thinker_config", config)
        cls = getattr(transformers, self.thinker_classes[kind])
        kw = {"dtype": dtype, "config": thinker_cfg, **quantization_kwargs(quantize, dtype)}
        if attn_implementation:
            kw["attn_implementation"] = attn_implementation
        if device_map is not None:
            kw["device_map"] = device_map
            if max_memory:
                kw["max_memory"] = max_memory
        return cls.from_pretrained(path, **kw)

    # ---- structure -------------------------------------------------------------
    def backbone(self, model: nn.Module) -> nn.Module:
        return self.base(model).model  # the text model; positions and features come from us

    def hidden_size(self, model: nn.Module) -> int:
        return int(self.base(model).config.text_config.hidden_size)

    def lm_head(self, model: nn.Module):
        return self.base(model).lm_head

    # ---- prompting -------------------------------------------------------------
    def chat_text(self, tok, processor, messages, add_generation_prompt):
        # not processor.apply_chat_template: it checks for Qwen's speech-output system prompt and expects list content
        template = None if getattr(tok, "chat_template", None) else getattr(processor, "chat_template", None)
        return tok.apply_chat_template(messages, chat_template=template, tokenize=False,
                                       add_generation_prompt=add_generation_prompt)

    def image_placeholder(self, processor) -> str:
        return processor.vision_bos_token + processor.image_token + processor.vision_eos_token

    def media_placeholder(self, kind, has_audio, processor):
        if kind == "image":
            return self.image_placeholder(processor)
        if kind == "video":  # the processor swaps this for the interleaved video+audio sequence
            return processor.vision_bos_token + processor.video_token + processor.vision_eos_token
        return processor.audio_bos_token + processor.audio_token + processor.audio_eos_token

    def process_media(self, processor, text, media, order, opts):
        import numpy as np

        from jevtrainer.media import processor_inputs

        ins = processor_inputs(media, order, silent_video_audio=opts.use_audio_in_video)
        kw = {}
        if ins["images"]:
            kw["images"] = ins["images"]
        use_audio_in_video = False
        if ins["videos"]:
            kw["videos"] = [np.stack([np.asarray(f) for f in v.frames]) for v in ins["videos"]]
            kw["fps"] = float(ins["videos"][0].fps)  # seconds per temporal grid = temporal_patch_size / fps
            use_audio_in_video = bool(opts.use_audio_in_video)
            kw["use_audio_in_video"] = use_audio_in_video
        if ins["audios"]:
            kw["audio"] = ins["audios"]
        out = processor(text=[text], return_tensors="pt", **kw)
        out["use_audio_in_video"] = torch.tensor([int(use_audio_in_video)])
        return out

    # ---- forward ------------------------------------------------------------------
    def merge_multimodal(self, model, batch, embeds):
        th = self.base(model)
        cfg, ids = th.config, batch["input_ids"]

        def put(embeds, token_id, feats, what):
            mask = (ids == token_id).unsqueeze(-1)
            feats = feats.to(embeds.device, embeds.dtype)
            if int(mask.sum()) * embeds.shape[-1] != feats.numel():
                raise ValueError(f"{what}: {int(mask.sum())} placeholders but {feats.numel() // embeds.shape[-1]} feature rows")
            return embeds.masked_scatter(mask.expand_as(embeds), feats), mask.squeeze(-1)

        extra: dict = {}
        if batch.get("input_features") is not None:
            f = th.get_audio_features(batch["input_features"], feature_attention_mask=batch["feature_attention_mask"],
                                      return_dict=True).last_hidden_state
            embeds, _ = put(embeds, cfg.audio_token_id, f, "audio")
        deep, masks = {}, {}
        if batch.get("pixel_values") is not None:
            out = th.get_image_features(batch["pixel_values"], batch["image_grid_thw"], return_dict=True)
            embeds, masks["image"] = put(embeds, cfg.image_token_id, out.pooler_output, "image")
            deep["image"] = getattr(out, "deepstack_features", None)
        if batch.get("pixel_values_videos") is not None:
            out = th.get_video_features(batch["pixel_values_videos"], batch["video_grid_thw"], return_dict=True)
            embeds, masks["video"] = put(embeds, cfg.video_token_id, out.pooler_output, "video")
            deep["video"] = getattr(out, "deepstack_features", None)
        if masks and all(v is not None for v in deep.values()):
            pos = torch.stack(list(masks.values())).any(0)
            if pos.any():
                layers = []
                for i in range(len(next(iter(deep.values())))):
                    joint = embeds.new_zeros(int(pos.sum()), embeds.shape[-1])
                    for k, m in masks.items():
                        joint[m[pos]] = deep[k][i].to(joint.dtype)
                    layers.append(joint)
                extra = {"visual_pos_masks": pos, "deepstack_visual_embeds": layers}
        batch["_mm_extra"] = extra
        return embeds

    def forward_kwargs(self, model, batch: dict) -> dict:
        th = self.base(model)
        mask = batch["attention_mask"]
        flag = batch.get("use_audio_in_video")
        pos, _ = th.get_rope_index(
            batch["input_ids"],
            batch.get("image_grid_thw"),
            batch.get("video_grid_thw"),
            mask,
            bool(flag.any()) if flag is not None else False,
            batch["feature_attention_mask"].sum(-1) if batch.get("feature_attention_mask") is not None else None,
            batch.get("video_second_per_grid"),
        )
        return {"position_ids": pos, **batch.get("_mm_extra", {})}
