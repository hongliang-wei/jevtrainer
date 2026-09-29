"""ModelFamily: the only place that knows how a given architecture is loaded and wired.

Readouts never mention a model name. They ask the family for the tokenizer, the
backbone that returns last hidden states, the LM head, LoRA targets and so on.
`GenericFamily` infers all of that from the checkpoint, so a new model usually
needs no code; add a subclass only when inference gets something wrong.
"""

from __future__ import annotations

import re
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, PretrainedConfig

from jevtrainer.registry import FAMILIES

VISION_PAT = re.compile(r"(^|\.)(visual|vision_tower|vision_model|vision|image_encoder|multi_modal_projector|mm_projector|merger)(\.|$)")
LINEAR_ATTN_PAT = re.compile(r"DeltaNet|Mamba|GatedDelta|LinearAttention|RWKV|Recurrent", re.I)
LORA_NAME_PAT = re.compile(r"proj|^fc\d?$|dense|^(query|key|value)$|^w[qkvo123]$|^W(qkv|o|i)$")


class ModelFamily:
    """Default behaviour; subclasses override single methods."""

    name = "generic"
    model_types: tuple[str, ...] = ()
    causal = True

    @classmethod
    def match(cls, config: PretrainedConfig) -> bool:
        return getattr(config, "model_type", None) in cls.model_types

    # ---- loading -------------------------------------------------------
    def is_multimodal(self, config: PretrainedConfig) -> bool:
        return getattr(config, "vision_config", None) is not None

    def load_model(self, path: str, dtype: torch.dtype, attn_implementation: str | None = None) -> nn.Module:
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        cls = AutoModelForImageTextToText if self.is_multimodal(config) else AutoModelForCausalLM
        kw: dict[str, Any] = {"dtype": dtype, "trust_remote_code": True}
        if attn_implementation:
            kw["attn_implementation"] = attn_implementation
        return cls.from_pretrained(path, **kw)

    def load_processor(self, path: str, multimodal: bool):
        """Return (tokenizer, processor_or_None)."""
        from transformers import AutoProcessor, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        processor = None
        if multimodal:
            try:
                processor = AutoProcessor.from_pretrained(path, trust_remote_code=True)
                tok = processor.tokenizer
            except Exception:
                processor = None
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token or tok.unk_token
        return tok, processor

    # ---- structure -------------------------------------------------------
    def base(self, model: nn.Module) -> nn.Module:
        return model.get_base_model() if hasattr(model, "get_base_model") else model

    def backbone(self, model: nn.Module) -> nn.Module:
        """Module whose forward(inputs_embeds=...) returns `.last_hidden_state` (post final norm)."""
        base = self.base(model)
        return getattr(base, "model", None) or getattr(base, base.base_model_prefix)

    def lm_head(self, model: nn.Module) -> nn.Module | None:
        return self.base(model).get_output_embeddings()

    def input_embeddings(self, model: nn.Module) -> nn.Embedding:
        return self.base(model).get_input_embeddings()

    def hidden_size(self, model: nn.Module) -> int:
        return self.input_embeddings(model).embedding_dim

    def has_linear_attention(self, model: nn.Module) -> bool:
        return any(LINEAR_ATTN_PAT.search(type(m).__name__) for m in self.base(model).modules())

    def vision_modules(self, model: nn.Module) -> list[nn.Module]:
        found = []
        for name, mod in self.base(model).named_modules():
            if VISION_PAT.search(name) and not any(name.startswith(p + ".") for p, _ in found):
                found.append((name, mod))
        return [m for _, m in found]

    def lora_target_regex(self, model: nn.Module) -> str:
        """Regex over full module names: every Linear in the language model except the LM head."""
        leaves = set()
        for name, mod in self.base(model).named_modules():
            if isinstance(mod, nn.Linear) and not VISION_PAT.search(name) and not name.endswith("lm_head"):
                leaf = name.rsplit(".", 1)[-1]
                if LORA_NAME_PAT.search(leaf):
                    leaves.add(leaf)
        if not leaves:
            raise ValueError("could not infer LoRA targets; set lora.targets in the config")
        return r"^(?!.*(visual|vision|lm_head)).*\.(" + "|".join(sorted(leaves)) + r")$"

    # ---- prompting -------------------------------------------------------
    def chat_text(self, tok, processor, messages: list[dict], add_generation_prompt: bool) -> str:
        renderer = processor if (processor is not None and _has_images(messages)) else tok
        if getattr(renderer, "chat_template", None) or getattr(tok, "chat_template", None):
            return renderer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False
            )
        return _plain_chat(messages, add_generation_prompt)

    def image_placeholder(self, processor) -> str:
        return getattr(processor, "image_token", "<image>")

    def forward_kwargs(self, model: nn.Module, batch: dict) -> dict:
        """Extra kwargs for backbone forward (position ids, pixel values, ...)."""
        kw = {}
        for k in ("pixel_values", "image_grid_thw", "image_sizes", "pixel_attention_mask"):
            if batch.get(k) is not None:
                kw[k] = batch[k]
        return kw


def _has_images(messages: list[dict]) -> bool:
    return any(isinstance(m["content"], list) and any(c.get("type") == "image" for c in m["content"]) for m in messages)


def _plain_chat(messages: list[dict], add_generation_prompt: bool) -> str:
    parts = []
    for m in messages:
        content = m["content"] if isinstance(m["content"], str) else "".join(c.get("text", "") for c in m["content"])
        parts.append(f"{m['role']}: {content}\n")
    if add_generation_prompt:
        parts.append("assistant: ")
    return "".join(parts)


@FAMILIES.register("generic")
class GenericFamily(ModelFamily):
    name = "generic"

    @classmethod
    def match(cls, config: PretrainedConfig) -> bool:
        return True


def resolve_family(model_path: str, name: str = "auto") -> ModelFamily:
    if name != "auto":
        return FAMILIES.get(name)()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    for fam_name, cls in FAMILIES.items():
        if fam_name != "generic" and cls.match(config):
            return cls()
    return GenericFamily()
