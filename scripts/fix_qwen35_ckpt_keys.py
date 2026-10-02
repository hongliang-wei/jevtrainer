"""Rewrite a Qwen3.5 checkpoint saved by transformers 5.5 so newer transformers (>= 5.17) can load it.

transformers 5.5 wrote Qwen3.5 (ForConditionalGeneration) weights as
    model.language_model.language_model.language_model.<x>   (text)
    model.language_model.visual.<x>                          (vision)
and read them back through the matching (buggy) conversion. Newer releases expect the plain layout
    model.language_model.<x> / model.visual.<x>
and otherwise report every weight as UNEXPECTED and run with random initialisation.

    python scripts/fix_qwen35_ckpt_keys.py runs/repro/intern_0.8b_v4 /root/autodl-tmp/ckpt_v4_fixed

Copies everything except `step-*` / `eval*` directories; `model/model.safetensors` gets the renamed keys.
The result also loads under transformers 5.5.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

RULES = [("model.language_model.language_model.language_model.", "model.language_model."),
         ("model.language_model.visual.", "model.visual.")]


def fix(k: str) -> str:
    for a, b in RULES:
        if k.startswith(a):
            return b + k[len(a):]
    return k


def main(src: str, dst: str) -> None:
    src, dst = Path(src), Path(dst)
    for p in src.iterdir():
        if p.name == "model" or p.name.startswith(("step-", "eval")):
            continue
        shutil.copytree(p, dst / p.name) if p.is_dir() else (dst.mkdir(parents=True, exist_ok=True), shutil.copy2(p, dst / p.name))
    (dst / "model").mkdir(parents=True, exist_ok=True)
    for p in (src / "model").iterdir():
        if p.suffix != ".safetensors":
            shutil.copy2(p, dst / "model" / p.name)
            continue
        with safe_open(str(p), "pt") as f:
            tensors = {fix(k): f.get_tensor(k) for k in f.keys()}
            meta = f.metadata() or {}
        meta.setdefault("format", "pt")
        save_file(tensors, str(dst / "model" / p.name), metadata=meta)
        print(f"{p.name}: {len(tensors)} tensors, e.g. {sorted(tensors)[0]}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
