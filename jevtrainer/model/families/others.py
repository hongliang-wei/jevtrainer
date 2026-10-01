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

    def backbone(self, model):
        inner = super().backbone(model)
        if self._per_layer(model) and hasattr(inner, "language_model"):
            return inner.language_model
        return inner

    def forward_kwargs(self, model, batch: dict) -> dict:
        if not self._per_layer(model):
            return super().forward_kwargs(model, batch)
        if batch.get("pixel_values") is not None:
            raise NotImplementedError("images are not supported yet for Gemma models with per-layer embeddings (E-series)")
        lm = self.backbone(model)
        cfg = lm.config
        ids = batch["input_ids"]
        ids = ids.masked_fill(ids >= cfg.vocab_size_per_layer_input, cfg.pad_token_id)  # readout tokens
        return {"per_layer_inputs": lm.get_per_layer_inputs(ids, None)}

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
