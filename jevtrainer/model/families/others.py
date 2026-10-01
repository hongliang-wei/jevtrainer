"""Families that only need a name and a model_type match; GenericFamily does the rest.

Tested: gemma3 (text). Untested but expected to work through GenericFamily: the rest.
"""

from __future__ import annotations

from jevtrainer.model.families.base import ModelFamily, _text_config
from jevtrainer.registry import FAMILIES


@FAMILIES.register("gemma")
class GemmaFamily(ModelFamily):
    """Gemma 2 / 3 / 3n / 4.

    Gemma 4 E-series (and 3n) add per-layer embeddings looked up from token ids. Given only inputs_embeds,
    the multimodal wrapper recovers the ids by comparing against the whole vocabulary, which fails on the
    readout's own token embeddings and costs seq x vocab x hidden memory, so these models run the
    language model directly with per-layer inputs computed here (text only).
    """

    name = "gemma"
    model_types = ("gemma", "gemma2", "gemma3", "gemma3_text", "gemma3n", "gemma3n_text", "gemma4", "gemma4_text",
                   "gemma4_unified")

    def image_placeholder(self, processor) -> str:
        return getattr(processor, "boi_token", "<start_of_image>")

    def _per_layer(self, model) -> bool:
        return bool(getattr(_text_config(self.base(model).config), "hidden_size_per_layer_input", None))

    def _gemma4(self, model) -> bool:
        return str(getattr(self.base(model).config, "model_type", "")).startswith("gemma4")

    def _wrapper(self, model):
        """Gemma4Model (towers + language model), None for a text-only checkpoint."""
        inner = getattr(self.base(model), "model", None)
        return inner if inner is not None and hasattr(inner, "language_model") else None

    def backbone(self, model):
        inner = super().backbone(model)
        if (self._per_layer(model) or self._gemma4(model)) and hasattr(inner, "language_model"):
            return inner.language_model  # Gemma 4: we merge image / video / audio features ourselves (merge_multimodal)
        return inner

    # ---- image / video / audio (Gemma 4) -----------------------------------------------
    def media_placeholder(self, kind, has_audio, processor):
        if kind == "image":
            return processor.image_token
        if kind == "video":  # frames, then the clip's own sound
            return processor.video_token + (processor.audio_token if has_audio else "")
        return processor.audio_token

    def process_media(self, processor, text, media, order, opts):
        from transformers.video_utils import VideoMetadata

        from jevtrainer.media import processor_inputs

        ins = processor_inputs(media, order)
        kw = {}
        if ins["images"]:
            kw["images"] = ins["images"]
        if ins["videos"]:
            processor.video_processor.max_soft_tokens = opts.frame_tokens
            kw["videos"] = [v.frames for v in ins["videos"]]
            kw["video_metadata"] = [
                VideoMetadata(total_num_frames=len(v.frames), fps=v.fps, duration=v.duration,
                              frames_indices=list(range(len(v.frames))))
                for v in ins["videos"]
            ]
            kw["do_sample_frames"] = False
        if ins["audios"]:
            kw["audio"] = ins["audios"]
        return processor(text=[text], return_tensors="pt", **kw)

    def merge_multimodal(self, model, batch, embeds):
        keys = ("pixel_values", "pixel_values_videos", "input_features")
        if not self._gemma4(model) or not any(batch.get(k) is not None for k in keys):
            return embeds
        w = self._wrapper(model)
        cfg, ids = w.config, batch["input_ids"]

        def put(embeds, token_id, feats, what):
            mask = (ids == token_id).unsqueeze(-1)
            feats = feats.to(embeds.device, embeds.dtype)
            if int(mask.sum()) * embeds.shape[-1] != feats.numel():
                raise ValueError(f"{what}: {int(mask.sum())} placeholders but {feats.numel() // embeds.shape[-1]} feature rows")
            return embeds.masked_scatter(mask.expand_as(embeds), feats)

        if batch.get("pixel_values") is not None:
            f = w.get_image_features(batch["pixel_values"], batch.get("image_position_ids"), return_dict=True).pooler_output
            embeds = put(embeds, cfg.image_token_id, f, "image")
        if batch.get("pixel_values_videos") is not None:
            f = w.get_video_features(batch["pixel_values_videos"], batch.get("video_position_ids"), return_dict=True).pooler_output
            embeds = put(embeds, cfg.video_token_id, f, "video")
        if batch.get("input_features") is not None:
            out = w.get_audio_features(batch["input_features"], batch["input_features_mask"], return_dict=True)
            embeds = put(embeds, cfg.audio_token_id, out.pooler_output[out.attention_mask], "audio")
        return embeds

    def forward_kwargs(self, model, batch: dict) -> dict:
        if not self._gemma4(model):
            if not self._per_layer(model):
                return super().forward_kwargs(model, batch)
            if batch.get("pixel_values") is not None:
                raise NotImplementedError("images are not supported for Gemma 3n")
        kw: dict = {}
        lm = self.backbone(model)
        ids = batch["input_ids"]
        if self._per_layer(model):
            w = self._wrapper(model)
            pad = lm.config.pad_token_id
            ids = ids.masked_fill(ids >= lm.config.vocab_size_per_layer_input, pad)  # readout tokens
            if w is not None:  # image / video / audio placeholders carry no token identity
                for t in ("image_token_id", "video_token_id", "audio_token_id"):
                    ids = ids.masked_fill(batch["input_ids"] == getattr(w.config, t), pad)
            kw["per_layer_inputs"] = lm.get_per_layer_inputs(ids, None)
        if self._gemma4(model) and batch.get("mm_token_type_ids") is not None and batch["mm_token_type_ids"].any() and \
                getattr(_text_config(self.base(model).config), "use_bidirectional_attention", None) == "vision":
            # the larger Gemma 4 models attend bidirectionally inside an image / video frame
            from transformers.models.gemma4.modeling_gemma4 import create_causal_mask_mapping

            B, T = ids.shape
            dummy = torch.empty(B, T, 1, dtype=next(lm.parameters()).dtype, device=ids.device)
            pos = torch.arange(T, device=ids.device).unsqueeze(0)
            kw["attention_mask"] = create_causal_mask_mapping(
                self.base(model).config, dummy, batch["attention_mask"], None, pos,
                mm_token_type_ids=batch["mm_token_type_ids"], is_training=True)
        return kw

    def chat_text(self, tok, processor, messages: list[dict], add_generation_prompt: bool) -> str:
        if getattr(tok, "chat_template", None) or (processor is not None and getattr(processor, "chat_template", None)):
            return super().chat_text(tok, processor, messages, add_generation_prompt)
        # base checkpoints ship without a template; use the instruction-tuned turn format
        vocab = tok.get_vocab()
        if "<|turn>" in vocab:
            start, end, sys_role = "<|turn>", "<turn|>", "system"
        elif "<start_of_turn>" in vocab:
            start, end, sys_role = "<start_of_turn>", "<end_of_turn>", None
        else:
            return super().chat_text(tok, processor, messages, add_generation_prompt)
        out, pending = [tok.bos_token or ""], ""
        for m in messages:
            text = m["content"] if isinstance(m["content"], str) else "".join(c.get("text", "") for c in m["content"])
            role = {"assistant": "model"}.get(m["role"], m["role"])
            if role == "system" and sys_role is None:
                pending = text + "\n\n"
                continue
            out.append(f"{start}{role}\n{pending if role == 'user' else ''}{text}{end}\n")
            if role == "user":
                pending = ""
        if add_generation_prompt:
            out.append(f"{start}model\n")
        return "".join(out)


@FAMILIES.register("llama")
class LlamaFamily(ModelFamily):
    name = "llama"
    model_types = ("llama", "llama4", "llama4_text", "mllama")


@FAMILIES.register("mistral")
class MistralFamily(ModelFamily):
    name = "mistral"
    model_types = ("mistral", "mistral3", "ministral", "mixtral", "pixtral")


@FAMILIES.register("internvl")
class InternVLFamily(ModelFamily):
    name = "internvl"
    model_types = ("internvl", "internvl_chat", "interns1")

    def image_placeholder(self, processor) -> str:
        return getattr(processor, "image_token", "<IMG_CONTEXT>")
