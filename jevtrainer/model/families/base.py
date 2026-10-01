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

# non-language towers: frozen by default, never LoRA targets
VISION_PAT = re.compile(
    r"(^|\.)(visual|vision_tower|vision_model|vision|image_encoder|multi_modal_projector|mm_projector|merger"
    r"|audio_tower|audio_model|embed_vision|embed_audio)(\.|$)"
)
LINEAR_ATTN_PAT = re.compile(r"DeltaNet|Mamba|GatedDelta|LinearAttention|RWKV|Recurrent", re.I)
LORA_NAME_PAT = re.compile(r"proj|^fc\d?$|dense|^(query|key|value)$|^w[qkvo123]$|^W(qkv|o|i)$")
# MoE routers and factorized-embedding projections (Nandi / Lumma) stay out of LoRA
LORA_SKIP_PAT = re.compile(r"(^|\.)router(\.|$)|(^|\.)(embedding_proj|lm_head_proj)$")


class OutputHead(nn.Module):
    """LM head as the CausalLM forward applies it: optional down-projection first, optional logit soft-capping."""

    def __init__(self, head: nn.Module, pre: nn.Module | None = None, softcap: float | None = None):
        super().__init__()
        self.head, self.pre, self.softcap = head, pre, softcap

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.head(self.pre(h) if self.pre is not None else h)
        return torch.tanh(z / self.softcap) * self.softcap if self.softcap else z


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

    def load_model(self, path: str, dtype: torch.dtype, attn_implementation: str | None = None, quantize: str = "none",
                   device_map: str | dict | None = None, max_memory: dict | None = None) -> nn.Module:
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        cls = AutoModelForImageTextToText if self.is_multimodal(config) else AutoModelForCausalLM
        kw: dict[str, Any] = {"dtype": dtype, "trust_remote_code": True, **quantization_kwargs(quantize, dtype)}
        if attn_implementation:
            kw["attn_implementation"] = attn_implementation
        if device_map is not None:
            kw["device_map"] = device_map
            if max_memory:
                kw["max_memory"] = max_memory
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
        base = self.base(model)
        head = base.get_output_embeddings()
        if head is None:
            return None
        pre = getattr(base, "lm_head_proj", None)
        cap = getattr(_text_config(base.config), "final_logit_softcapping", None)
        return OutputHead(head, pre, cap) if (pre is not None or cap) else head

    def input_embeddings(self, model: nn.Module) -> nn.Embedding:
        return self.base(model).get_input_embeddings()

    def hidden_size(self, model: nn.Module) -> int:
        """Width of the backbone's last hidden state (can differ from the embedding width when it is factorized)."""
        h = getattr(_text_config(self.base(model).config), "hidden_size", None)
        return int(h) if h else self.input_embeddings(model).embedding_dim

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
            if isinstance(mod, nn.Linear) and not VISION_PAT.search(name) and not LORA_SKIP_PAT.search(name) and not name.endswith("lm_head"):
                leaf = name.rsplit(".", 1)[-1]
                if LORA_NAME_PAT.search(leaf):
                    leaves.add(leaf)
        if not leaves:
            raise ValueError("could not infer LoRA targets; set lora.targets in the config")
        return r"^(?!.*(visual|vision|audio|lm_head|router\.)).*\.(" + "|".join(sorted(leaves)) + r")$"

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

    # ---- images / video / audio inside the state -----------------------------------
    def media_placeholder(self, kind: str, has_audio: bool, processor) -> str:
        """What stands for one media item in the text; the processor expands it to the right number of tokens."""
        if kind == "image":
            return self.image_placeholder(processor)
        tok = getattr(processor, f"{kind}_token", None)
        if not tok:
            raise ValueError(f"model family {self.name!r} has no {kind} input")
        return tok

    def process_media(self, processor, text: str, media, order, opts):
        """Run the processor on `text` (with placeholders in `order`) and the media; returns its BatchFeature."""
        from jevtrainer.media import processor_inputs

        ins = processor_inputs(media, order)
        kw: dict[str, Any] = {}
        if ins["images"]:
            kw["images"] = ins["images"]
        if ins["videos"]:
            kw["videos"] = [v.frames for v in ins["videos"]]
        if ins["audios"]:
            kw["audio"] = ins["audios"]
        return processor(text=[text], return_tensors="pt", **kw)

    def merge_multimodal(self, model: nn.Module, batch: dict, embeds: torch.Tensor) -> torch.Tensor:
        """Write image / video / audio features over their placeholder positions of `embeds`.

        The default does nothing: the backbone gets `pixel_values` & co. through `forward_kwargs` and merges itself.
        """
        return embeds

    def forward_kwargs(self, model: nn.Module, batch: dict) -> dict:
        """Extra kwargs for backbone forward (position ids, pixel values, ...)."""
        kw = {}
        for k in ("pixel_values", "image_grid_thw", "image_sizes", "pixel_attention_mask"):
            if batch.get(k) is not None:
                kw[k] = batch[k]
        return kw


def quantization_kwargs(quantize: str, dtype: torch.dtype) -> dict:
    """bitsandbytes loading for QLoRA. Only nn.Linear layers are quantized: fused MoE experts stay in `dtype`."""
    if quantize in (None, "none"):
        return {}
    from transformers import BitsAndBytesConfig

    skip = ["lm_head", "visual", "vision_tower", "audio_tower", "multi_modal_projector", "embed_vision", "embed_audio", "router"]
    if quantize == "4bit":
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=dtype, llm_int8_skip_modules=skip)
    elif quantize == "8bit":
        q = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=skip)
    else:
        raise ValueError(f"quantize must be none | 8bit | 4bit, got {quantize!r}")
    return {"quantization_config": q, "device_map": {"": torch.cuda.current_device() if torch.cuda.is_available() else "cpu"}}


def _text_config(config: PretrainedConfig) -> PretrainedConfig:
    return config.get_text_config() if hasattr(config, "get_text_config") else config


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
