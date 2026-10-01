"""Build (model, readout) from a config, apply LoRA or full fine-tuning, save and reload."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from jevtrainer.model.families import ModelFamily, resolve_family
from jevtrainer.readouts import Readout, ReadoutConfig
from jevtrainer.registry import READOUTS

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


@dataclass
class Bundle:
    model: nn.Module
    readout: Readout
    family: ModelFamily
    tok: object
    processor: object | None
    temperature: dict | None = None


def build(
    model_path: str,
    readout: str,
    family: str = "auto",
    dtype: str = "bf16",
    readout_options: dict | None = None,
    max_state_tokens: int = 2048,
    attn_implementation: str | None = None,
    tokenizer_path: str | None = None,
    quantize: str = "none",
    device_map: str | dict | None = None,
    max_memory: dict | None = None,
) -> Bundle:
    from transformers import AutoConfig

    fam = resolve_family(model_path, family)
    model = fam.load_model(model_path, DTYPES[dtype], attn_implementation, quantize, device_map, max_memory)
    multimodal = fam.is_multimodal(AutoConfig.from_pretrained(model_path, trust_remote_code=True))
    tok, processor = fam.load_processor(tokenizer_path or model_path, multimodal)
    ro = READOUTS.get(readout)(ReadoutConfig(max_state_tokens=max_state_tokens, options=dict(readout_options or {})))
    ro.setup(fam, model, tok, processor)
    return Bundle(model, ro, fam, tok, processor)


def prepare_finetune(b: Bundle, finetune: str, lora: dict, freeze_vision: bool = True, grad_ckpt: bool = True) -> None:
    quantized = getattr(b.model, "is_loaded_in_4bit", False) or getattr(b.model, "is_loaded_in_8bit", False)
    if quantized and finetune != "lora":
        raise ValueError("quantized models train with finetune: lora only (QLoRA)")
    for m in b.family.vision_modules(b.model) if freeze_vision else []:
        m.requires_grad_(False)
    if finetune == "lora":
        # no prepare_model_for_kbit_training: it upcasts every non-quantized weight to fp32, which for MoE
        # models (fused experts are not quantized) doubles the largest part of the model
        from peft import LoraConfig, get_peft_model

        targets = lora.get("targets", "auto")
        cfg = LoraConfig(
            r=lora.get("r", 16),
            lora_alpha=lora.get("alpha", 2 * lora.get("r", 16)),
            lora_dropout=lora.get("dropout", 0.05),
            target_modules=b.family.lora_target_regex(b.model) if targets == "auto" else targets,
            bias="none",
        )
        b.model = get_peft_model(b.model, cfg)
    elif finetune == "full":
        vision = {id(p) for m in (b.family.vision_modules(b.model) if freeze_vision else []) for p in m.parameters()}
        for p in b.model.parameters():
            p.requires_grad_(id(p) not in vision)
    elif finetune != "frozen":
        raise ValueError(f"finetune must be lora | full | frozen, got {finetune!r}")
    if grad_ckpt and finetune != "frozen":
        b.family.base(b.model).gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    b.readout.float().requires_grad_(True)


def trainable_parameters(b: Bundle) -> list[nn.Parameter]:
    return [p for p in b.model.parameters() if p.requires_grad] + [p for p in b.readout.parameters() if p.requires_grad]


def count(params) -> int:
    return sum(p.numel() for p in params)


# ---- checkpoints -----------------------------------------------------------------
def save(b: Bundle, out: str | Path, meta: dict) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if hasattr(b.model, "peft_config"):
        b.model.save_pretrained(out / "adapter")
    else:
        b.model.save_pretrained(out / "model", safe_serialization=True)
    b.tok.save_pretrained(out / "tokenizer")
    if b.processor is not None:
        b.processor.save_pretrained(out / "tokenizer")
    save_file({k: v.contiguous() for k, v in b.readout.state().items()}, str(out / "readout.safetensors"))
    (out / "readout.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def load(ckpt: str | Path, dtype: str = "bf16", device: str | None = None) -> Bundle:
    ckpt = Path(ckpt)
    meta = json.loads((ckpt / "readout.json").read_text(encoding="utf-8"))
    full = (ckpt / "model").exists()
    b = build(
        str(ckpt / "model") if full else meta["model"],
        meta["readout"],
        meta.get("family", "auto"),
        dtype,
        meta.get("readout_options"),
        meta.get("max_state_tokens", 2048),
        tokenizer_path=str(ckpt / "tokenizer"),
        quantize=meta.get("quantize", "none"),
        device_map=meta.get("device_map"),
        max_memory=meta.get("max_memory"),
    )
    placed = meta.get("quantize", "none") != "none" or meta.get("device_map") is not None
    if (ckpt / "adapter").exists():
        from peft import PeftModel

        b.model = PeftModel.from_pretrained(b.model, str(ckpt / "adapter"))
        if not placed:  # cannot merge into quantized weights
            b.model = b.model.merge_and_unload()
    b.readout.load_state_dict(load_file(str(ckpt / "readout.safetensors")))
    b.readout.float()
    b.temperature = _temperature(ckpt)
    if device:
        if not placed:
            b.model.to(device)
        b.readout.to(device)
    b.model.eval()
    b.readout.eval()
    return b


def _temperature(ckpt: Path) -> dict[str, float]:
    p = ckpt / "calibration.json"
    if not p.exists():
        return {"default": 1.0}
    return json.loads(p.read_text(encoding="utf-8"))["temperature"]
