"""A new family must pass these with a random tiny config: load, LoRA targets, forward for its readouts."""

import pytest
import torch
from conftest import DEVICE, on_device

from jevtrainer.model.families import resolve_family
from jevtrainer.model.load import build, prepare_finetune
from jevtrainer.train.batching import Collator, DecisionModel, to_device


def _logits(b, record):
    on_device(b)
    return DecisionModel(b.model, b.readout)(to_device(Collator(b.readout, 4096)([record]), DEVICE))


def test_qwen35_introspection(tiny_vl):
    from transformers import AutoModelForImageTextToText

    fam = resolve_family(tiny_vl)
    assert fam.name == "qwen"
    m = AutoModelForImageTextToText.from_pretrained(tiny_vl)
    rx = fam.lora_target_regex(m)
    assert "in_proj_qkv" in rx and "q_proj" in rx and "visual" in rx
    assert fam.has_linear_attention(m)
    assert fam.vision_modules(m)


def _save_tiny(tmp_path, config_cls, model_cls, tokenizer_repo, **cfg):
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(tokenizer_repo)
    except Exception as e:  # offline CI
        pytest.skip(f"tokenizer unavailable: {e}")
    c = config_cls(vocab_size=len(tok) + 300, **cfg)
    torch.manual_seed(0)
    model_cls.from_config(c).save_pretrained(tmp_path)
    tok.save_pretrained(tmp_path)
    return str(tmp_path)


def test_llama_generic_path(tmp_path, record):
    from transformers import AutoModelForCausalLM, LlamaConfig

    path = _save_tiny(tmp_path, LlamaConfig, AutoModelForCausalLM, "Qwen/Qwen3.5-0.8B",
                      hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1)
    assert resolve_family(path).name == "llama"
    for readout in ("marker", "slot", "pointer"):
        b = build(path, readout, dtype="fp32")
        prepare_finetune(b, "lora", {"r": 4}, grad_ckpt=False)
        assert [x.numel() for x in _logits(b, record)] == [3, 2, 3]


def test_lfm2_generic_path(tmp_path, record):
    from transformers import AutoModelForCausalLM, Lfm2Config

    path = _save_tiny(tmp_path, Lfm2Config, AutoModelForCausalLM, "Qwen/Qwen3.5-0.8B",
                      hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=1, layer_types=["conv", "full_attention"])
    for readout in ("marker", "pointer"):
        b = build(path, readout, dtype="fp32")
        prepare_finetune(b, "lora", {"r": 4}, grad_ckpt=False)
        assert [x.numel() for x in _logits(b, record)] == [3, 2, 3]


def _tiny_gemma4(tmp_path, **extra):
    from transformers import AutoModelForCausalLM, Gemma4TextConfig

    kw = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
              head_dim=16, global_head_dim=16, layer_types=["sliding_attention", "full_attention"], sliding_window=32,
              final_logit_softcapping=30.0)
    return _save_tiny(tmp_path, Gemma4TextConfig, AutoModelForCausalLM, "Qwen/Qwen3.5-0.8B", **{**kw, **extra})


def test_gemma4_per_layer_inputs(tmp_path, record):
    from jevtrainer.model.families.base import OutputHead

    path = _tiny_gemma4(tmp_path, hidden_size_per_layer_input=16, vocab_size_per_layer_input=1000)
    assert resolve_family(path).name == "gemma"
    b = build(path, "marker", dtype="fp32")
    assert isinstance(b.family.lm_head(b.model), OutputHead)
    prepare_finetune(b, "lora", {"r": 4}, grad_ckpt=False)
    out = _logits(b, record)
    assert [x.numel() for x in out] == [3, 2, 3]
    out[0].sum().backward()


def test_gemma4_moe_lora_skips_router(tmp_path, record):
    import re

    path = _tiny_gemma4(tmp_path, hidden_size_per_layer_input=0, enable_moe_block=True, num_experts=4, top_k_experts=2,
                        moe_intermediate_size=32)
    b = build(path, "marker", dtype="fp32")
    rx = re.compile(b.family.lora_target_regex(b.model))
    names = [n for n, _ in b.model.named_modules()]
    assert not any(rx.match(n) for n in names if ".router" in n)
    assert any(rx.match(n) for n in names if n.endswith("q_proj"))
    prepare_finetune(b, "lora", {"r": 4}, grad_ckpt=False)
    assert [x.numel() for x in _logits(b, record)] == [3, 2, 3]


def test_encoder_family(tmp_path, record):
    from transformers import AutoModel, ModernBertConfig

    path = _save_tiny(tmp_path, ModernBertConfig, AutoModel, "answerdotai/ModernBERT-base",
                      hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2)
    fam = resolve_family(path)
    assert fam.name == "encoder" and not fam.causal
    for readout in ("slot", "pointer"):
        b = build(path, readout, dtype="fp32")
        prepare_finetune(b, "full", {}, grad_ckpt=False)
        assert [x.numel() for x in _logits(b, record)] == [3, 2, 3]
    with pytest.raises(ValueError, match="LM head"):
        build(path, "marker", dtype="fp32")
