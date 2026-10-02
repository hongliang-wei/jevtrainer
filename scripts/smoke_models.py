"""Text (and optionally image) smoke test for a base model: build -> marker readout forward -> LoRA backward.

    python scripts/smoke_models.py --model Qwen/Qwen3.5-4B --image
    python scripts/smoke_models.py --model Qwen/Qwen3.8-27B --quantize 4bit
    python scripts/smoke_models.py --model google/gemma-4-26B-A4B-it --device-map auto --max-memory '{"0":"44GiB","cpu":"200GiB"}'

Prints per-record logits, forward / backward seconds, peak GPU memory, and one final `RESULT {json}` line.
Exit code 0 only when every step ran and every logit / gradient was finite.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import torch

from jevtrainer.model.load import build, prepare_finetune
from jevtrainer.schema import Record
from jevtrainer.train.batching import Collator, DecisionModel, to_device

TEXTS = [
    "I was charged twice and the app keeps crashing <decide> <decision>",
    "Hi team, could you please reset my password? I have been locked out for two days and the deadline is tomorrow. "
    "Nothing else is wrong with the account. <decide> <decision>",
    "Thanks, everything works great now and the new release is wonderful! <decide> <decision>",
]
QUESTIONS = {
    "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "charges", "tech": "bugs", "other": "else"}},
    "urgent": {"type": "noul", "instructions": "Is this urgent?"},
    "anger": {"type": "score", "instructions": "How angry?", "criteria": ["calm", "annoyed", "furious"]},
}
TARGETS = {
    "team": {"label": "billing"},
    "urgent": {"label": "true"},
    "anger": {"label": "1", "probabilities": {"0": 0.2, "1": 0.7, "2": 0.1}},
}


def make_records(image: str | None) -> list[Record]:
    recs = [Record.from_dict({"id": f"t{i}", "state": {"ticket": t}, "questions": QUESTIONS, "targets": TARGETS}).validate()
            for i, t in enumerate(TEXTS)]
    if image:
        recs.append(Record.from_dict({
            "id": "img", "state": {"ticket": "Screenshot <image:1> shows the error dialog. <decide> <decision>"},
            "questions": QUESTIONS, "targets": TARGETS, "media": [{"type": "image", "path": image}],
        }).validate())
    return recs


def make_image(path: Path) -> str:
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (336, 224), (30, 60, 120))
    d = ImageDraw.Draw(im)
    d.rectangle([40, 40, 200, 140], fill=(220, 40, 40))
    d.text((50, 160), "ERROR 500", fill=(255, 255, 255))
    im.save(path)
    return str(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--family", default="auto")
    ap.add_argument("--readout", default="marker")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--quantize", default="none", choices=["none", "8bit", "4bit"])
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--max-memory", default=None, help="JSON, e.g. '{\"0\": \"44GiB\", \"cpu\": \"200GiB\"}'")
    ap.add_argument("--image", action="store_true", help="also run a record with an image")
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--root", default="/tmp/smoke_models")
    args = ap.parse_args()

    res = {"model": args.model, "ok": False}
    dev = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    try:
        t0 = time.time()
        max_memory = json.loads(args.max_memory) if args.max_memory else None
        if max_memory:
            max_memory = {(int(k) if k.isdigit() else k): v for k, v in max_memory.items()}
        b = build(args.model, args.readout, args.family, args.dtype, quantize=args.quantize,
                  device_map=args.device_map, max_memory=max_memory)
        placed = args.quantize != "none" or args.device_map is not None
        if not placed:
            b.model.to(dev)
        b.readout.to(dev)
        res.update(family=b.family.name, load_s=round(time.time() - t0, 1),
                   params_b=round(sum(p.numel() for p in b.model.parameters()) / 1e9, 3))
        print(f"loaded {args.model} as {b.family.name} in {res['load_s']}s ({res['params_b']}B params)", flush=True)
        res["load_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)

        prepare_finetune(b, "lora", {"r": args.lora_r}, True, not args.no_grad_ckpt)
        net = DecisionModel(b.model, b.readout)
        col = Collator(b.readout, args.max_length)
        image = make_image(Path(args.root) / "img.png") if args.image else None
        recs = make_records(image)
        fwd, bwd, ok = [], [], True
        for rec in recs:
            batch = col([rec])
            if batch is None:
                print(rec.id, "SKIPPED by collator")
                ok = False
                continue
            batch = to_device(batch, dev)
            ntok = int(batch["attention_mask"].sum())
            net.train()
            torch.cuda.synchronize()
            t = time.time()
            logits = net(batch)
            torch.cuda.synchronize()
            tf = time.time() - t
            loss = sum(torch.nn.functional.cross_entropy(x.float()[None], torch.tensor([0], device=dev)) for x in logits)
            t = time.time()
            loss.backward()
            torch.cuda.synchronize()
            tb = time.time() - t
            finite = all(bool(torch.isfinite(x).all()) for x in logits) and bool(torch.isfinite(loss))
            grads = [p.grad for p in net.parameters() if p.requires_grad and p.grad is not None]
            gfin = bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads)
            gnorm = float(torch.sqrt(sum((g.float() ** 2).sum() for g in grads))) if grads else float("nan")
            net.zero_grad(set_to_none=True)
            ok &= finite and gfin
            fwd.append(tf)
            bwd.append(tb)
            print(f"{rec.id}: tokens={ntok} logits={[[round(float(v), 2) for v in x.float().cpu()] for x in logits]} "
                  f"loss={float(loss):.3f} gnorm={gnorm:.3g} finite={finite and gfin} fwd={tf:.2f}s bwd={tb:.2f}s "
                  f"peak={torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)
        res.update(ok=ok and bool(fwd), n_records=len(recs), fwd_s=round(sum(fwd[1:] or fwd) / len(fwd[1:] or fwd), 3),
                   bwd_s=round(sum(bwd[1:] or bwd) / len(bwd[1:] or bwd), 3),
                   peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2), image=args.image)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        res.update(ok=False, error=f"{type(e).__name__}: {str(e)[:300]}")
    print("RESULT " + json.dumps(res), flush=True)
    raise SystemExit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
