"""Families that only need a name and a model_type match; GenericFamily does the rest.

Tested: gemma3 (text). Untested but expected to work through GenericFamily: the rest.
"""

from __future__ import annotations

from jevtrainer.model.families.base import ModelFamily
from jevtrainer.registry import FAMILIES


@FAMILIES.register("gemma")
class GemmaFamily(ModelFamily):
    name = "gemma"
    model_types = ("gemma", "gemma2", "gemma3", "gemma3_text", "gemma3n", "gemma4", "gemma4_text")

    def image_placeholder(self, processor) -> str:
        return getattr(processor, "boi_token", "<start_of_image>")


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
