"""Smoke test: image / video(+sound) / audio records through a real model, forward + backward.

    python scripts/smoke_av.py --model google/gemma-4-E2B-it --readout marker
    python scripts/smoke_av.py --model Qwen/Qwen2.5-Omni-3B --readout marker --lora
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import torch

from jevtrainer.data.avkit import ffmpeg_exe
from jevtrainer.model.load import build, prepare_finetune
from jevtrainer.schema import Question, Record, Target
from jevtrainer.train.batching import Collator, DecisionModel, to_device


def make_clips(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    ff = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y"]
    v = root / "clip.mp4"
    if not v.exists():
        subprocess.run([*ff, "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=12:duration=6",
                        "-f", "lavfi", "-i", "sine=frequency=440:duration=6", "-shortest", str(v)], check=True)
    a = root / "tone.wav"
    if not a.exists():
        subprocess.run([*ff, "-f", "lavfi", "-i", "sine=frequency=880:duration=4", "-ar", "16000", str(a)], check=True)
    img = root / "img.jpg"
    if not img.exists():
        subprocess.run([*ff, "-f", "lavfi", "-i", "testsrc2=size=256x192:rate=1:duration=1", "-frames:v", "1", str(img)], check=True)
    return {"video": str(v), "audio": str(a), "image": str(img)}


def records(p: dict) -> list[Record]:
    q = lambda: {"d": Question("choice", "Which is it?", {"a": "alpha", "b": "beta", "c": "gamma"})}  # noqa: E731
    return [
        Record("v1", "Clip: <video:1>. What does it show?", q(), {"d": Target("a")}, media=[{"type": "video", "path": p["video"]}]),
        Record("a1", "Sound: <audio:1>. What is it?", q(), {"d": Target("b")}, media=[{"type": "audio", "path": p["audio"]}]),
        Record("av", "Video <video:1> and a separate sound <audio:1>.", q(), {"d": Target("c")},
               media=[{"type": "video", "path": p["video"]}, {"type": "audio", "path": p["audio"]}]),
        Record("i1", "Picture <image:1> then the video <video:1>.", q(), {"d": Target("a")},
               media=[{"type": "image", "path": p["image"]}, {"type": "video", "path": p["video"]}]),
        Record("t", "No media at all.", q(), {"d": Target("a")}),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--readout", default="marker")
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--frame-tokens", type=int, default=70)
    ap.add_argument("--audio-max-s", type=float, default=30.0)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--root", default="/tmp/avsmoke")
    args = ap.parse_args()

    clips = make_clips(Path(args.root))
    opts = {"video_frames": args.frames, "frame_tokens": args.frame_tokens, "audio_max_s": args.audio_max_s}
    t0 = time.time()
    b = build(args.model, args.readout, "auto", args.dtype, opts)
    print(f"loaded {args.model} as {b.family.name} in {time.time() - t0:.0f}s")
    prepare_finetune(b, "lora" if args.lora else "frozen", {"r": 8}, True, False)
    dev = torch.device("cuda")
    b.model.to(dev)
    b.readout.to(dev)
    col = Collator(b.readout, 8192)
    net = DecisionModel(b.model, b.readout)
    for rec in records(clips):
        batch = col([rec])
        if batch is None:
            print(rec.id, "SKIPPED (too long or error)")
            continue
        n = int(batch["attention_mask"].sum())
        extras = {k: tuple(v.shape) for k, v in batch.items() if isinstance(v, torch.Tensor) and k not in ("input_ids", "attention_mask")}
        t0 = time.time()
        net.train(args.lora)
        with torch.set_grad_enabled(args.lora):
            logits = net(to_device(batch, dev))
            if args.lora:
                loss = torch.nn.functional.cross_entropy(logits[0][None].float(), torch.tensor([0], device=dev))
                loss.backward()
        torch.cuda.synchronize()
        print(f"{rec.id}: tokens={n} extras={extras} logits={[round(float(x), 3) for x in logits[0].float().cpu()]} "
              f"{time.time() - t0:.1f}s peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GB")
    print("batch of 3 mixed records:")
    batch = col(records(clips)[:3] + records(clips)[3:])
    logits = net(to_device(batch, dev))
    print("OK", [tuple(x.shape) for x in logits])


if __name__ == "__main__":
    main()
