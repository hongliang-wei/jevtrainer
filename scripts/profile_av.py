"""Where does a training step go? Times collation (CPU) and forward+backward (GPU) on cached records.

    python scripts/profile_av.py --model Qwen/Qwen2.5-Omni-3B --records <records.jsonl> [--bs 2]
"""

from __future__ import annotations

import argparse
import time

import torch

from jevtrainer.data.base import read_jsonl
from jevtrainer.model.load import build, prepare_finetune
from jevtrainer.train.batching import Collator, DecisionModel, to_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--audio-max-s", type=float, default=30.0)
    ap.add_argument("--no-ckpt", action="store_true")
    a = ap.parse_args()

    recs = [r for _, r in zip(range(a.n), read_jsonl(a.records))]
    b = build(a.model, "marker", "auto", "bf16", {"video_frames": a.frames, "audio_max_s": a.audio_max_s})
    prepare_finetune(b, "lora", {"r": 16}, True, not a.no_ckpt)
    dev = torch.device("cuda")
    b.model.to(dev)
    b.readout.to(dev)
    col = Collator(b.readout, 6144)
    net = DecisionModel(b.model, b.readout)
    net.train()

    t0 = time.time()
    rows = [col([r]) for r in recs]
    print(f"collate: {(time.time() - t0) / len(recs):.2f} s/record (one process)")
    print("tokens/record:", [int(x["attention_mask"].sum()) for x in rows if x is not None][:8])

    batch = to_device(col(recs[:a.bs]), dev)
    th = b.family.base(b.model) if hasattr(b.family, "base") else None
    print("attn:", getattr(b.model.config, "_attn_implementation", None), getattr(th, "config", None) and getattr(th.config, "_attn_implementation", None))
    for name, fn in (
        ("audio tower", lambda: th.get_audio_features(batch["input_features"], feature_attention_mask=batch["feature_attention_mask"], return_dict=True)),
        ("video tower", lambda: th.get_video_features(batch["pixel_values_videos"], batch["video_grid_thw"], return_dict=True)),
    ):
        if th is None:
            break
        for i in range(2):
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                fn()
            torch.cuda.synchronize()
        print(f"{name}: {time.time() - t0:.2f} s (micro-batch of {a.bs})")
    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.time()
        net(batch)
        torch.cuda.synchronize()
        print(f"full forward no-grad: {time.time() - t0:.2f} s")

    for step in range(3):
        batches = [col(recs[i:i + a.bs]) for i in range(0, len(recs), a.bs)]
        torch.cuda.synchronize()
        t0 = time.time()
        for batch in batches:
            logits = net(to_device(batch, dev))
            sum(x.float().logsumexp(-1).sum() for x in logits).backward()
        torch.cuda.synchronize()
        print(f"pass {step}: {(time.time() - t0) / len(batches):.2f} s per micro-batch of {a.bs} "
              f"peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GB")


if __name__ == "__main__":
    main()
