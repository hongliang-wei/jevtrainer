"""Tiny random models that share real tokenizers, so every code path runs on CPU in seconds."""

from __future__ import annotations

import os
import shutil

import pytest
import torch

REAL = os.environ.get("JT_TEST_TOKENIZER", "Qwen/Qwen3.5-0.8B")
# flash-linear-attention kernels are GPU-only; transformers uses them whenever fla is installed.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def on_device(b):
    b.model.to(DEVICE)
    b.readout.to(DEVICE)
    return b


def _tiny_qwen35(path):
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForImageTextToText, AutoProcessor, Qwen3_5Config

    c = Qwen3_5Config.from_pretrained(REAL)
    t = c.text_config
    t.update(dict(num_hidden_layers=4, hidden_size=64, intermediate_size=128, num_attention_heads=2, num_key_value_heads=1,
                  head_dim=32, linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
                  layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"]))
    v = c.vision_config
    v.update(dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64))
    if hasattr(v, "num_position_embeddings"):
        pass
    torch.manual_seed(0)
    m = AutoModelForImageTextToText.from_config(c)
    m.save_pretrained(path)
    AutoProcessor.from_pretrained(REAL).save_pretrained(path)
    src = snapshot_download(REAL, allow_patterns=["chat_template.jinja"])
    for f in ("chat_template.jinja",):
        if os.path.exists(os.path.join(src, f)):
            shutil.copy(os.path.join(src, f), path)


def _tiny_qwen3(path):
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3Config

    tok = AutoTokenizer.from_pretrained(REAL)
    c = Qwen3Config(vocab_size=len(tok) + 256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                    num_attention_heads=2, num_key_value_heads=1, head_dim=32)
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(c).save_pretrained(path)
    tok.save_pretrained(path)


@pytest.fixture(scope="session")
def tiny_vl(tmp_path_factory):
    p = tmp_path_factory.mktemp("tiny_qwen35")
    _tiny_qwen35(str(p))
    return str(p)


@pytest.fixture(scope="session")
def tiny_text(tmp_path_factory):
    p = tmp_path_factory.mktemp("tiny_qwen3")
    _tiny_qwen3(str(p))
    return str(p)


@pytest.fixture()
def record():
    from jevtrainer.schema import Record

    return Record.from_dict(
        {
            "id": "r1",
            "state": {"ticket": "I was charged twice and the app keeps crashing <decide> <decision>"},
            "questions": {
                "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "charges", "tech": "bugs", "other": "else"}},
                "urgent": {"type": "noul", "instructions": "Is this urgent?"},
                "anger": {"type": "score", "instructions": "How angry?", "criteria": ["calm", "annoyed", "furious"]},
            },
            "targets": {"team": {"label": "billing"}, "urgent": {"label": "true"}, "anger": {"label": "1", "probabilities": {"0": 0.2, "1": 0.7, "2": 0.1}}},
        }
    ).validate()


@pytest.fixture()
def image_record(tmp_path):
    from PIL import Image

    from jevtrainer.schema import Record

    p = tmp_path / "img.png"
    Image.new("RGB", (64, 64), (200, 30, 30)).save(p)
    return Record.from_dict(
        {
            "id": "img1",
            "state": {"question": "What colour dominates the picture?"},
            "images": [str(p)],
            "questions": {"colour": {"type": "choice", "instructions": "Pick the colour.", "criteria": {"red": "", "blue": "", "green": ""}}},
            "targets": {"colour": {"label": "red"}},
        }
    )
