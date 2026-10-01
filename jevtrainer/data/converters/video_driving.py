"""Driving safety from dashcam video: Nexar collision prediction."""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from jevtrainer.data import avkit, vidkit
from jevtrainer.data.base import DatasetSpec, noul_record, register, rid

# The official repo `nexar-ai/nexar_collision_prediction` is gated (form + consent); this ungated copy has identical files
# and the same Nexar Open Data License (free use with attribution, research OK).
_NX = "akhil0790/nexar_collision_prediction"
YES = "a collision or near-collision with another vehicle, pedestrian or object is about to happen"
NO = "ordinary driving: no collision or near-collision is about to happen"


def nexar(split, cap, rng):
    """Nexar collision prediction: does the dashcam clip end right before a collision / near-miss?

    Test split = official test-public + test-private clips (~10 s that stop 0.5 / 1.0 / 1.5 s before the event for
    positives; labels by folder). Train videos are ~40 s long, so a 10 s window is cut to mimic the test clips:
    positives end 0.5-1.5 s before `time_of_event`, negatives use a random 10 s window. Balanced yes / no. Download
    ceiling JT_NEXAR_MAX (default 450 train / 300 test videos; ~11 MB each)."""
    import pandas as pd

    dirs = ["train"] if split == "train" else ["test-public", "test-private"]
    rows = []
    for d in dirs:
        for lab in ("positive", "negative"):
            df = vidkit.read_csv(_NX, f"{d}/{lab}/metadata.csv", dtype={"file_name": str})
            for r in df.to_dict("records"):
                rows.append({**r, "label": lab == "positive", "path": f"{d}/{lab}/{r['file_name']}", "key": f"{d}_{r['file_name'][:-4]}"})
    rows = avkit.balanced(rows, lambda r: r["label"], min(cap, int(os.environ.get("JT_NEXAR_MAX", 450 if split == "train" else 300))), rng)
    raw = vidkit.fetch_many(_NX, "nexar", {r["key"]: r["path"] for r in rows}, workers=6)
    plan = {}
    for r in rows:
        if r["key"] not in raw:
            continue
        src = raw[r["key"]]
        job = {"src": src}
        if split == "train":
            dur = avkit.probe(src)["duration"]
            if dur < 11:
                continue
            if r["label"]:
                end = float(r["time_of_event"]) - rng.choice([0.5, 1.0, 1.5])
                if end < 4:
                    continue
            else:
                end = rng.uniform(10, dur - 0.5)
            job.update(start=max(0.0, end - 10.0), end=end)
        plan[r["key"]] = job
    media = vidkit.convert_local("nexar", plan, frames=16, max_side=384, audio_s=12)
    for r in rows:
        m = media.get(r["key"])
        if not m:
            continue
        rec = noul_record(rid("nexar", split, r["key"]),
                          {"clip": "<video:1>", "question": "Dashcam clip. Will a collision or near-collision happen right after it ends?"},
                          "Does this dashcam clip end right before a collision or near-collision?", bool(r["label"]), YES, NO,
                          area="video", dataset="nexar", scene=str(r.get("scene")), weather=str(r.get("weather")))
        rec.media = [m]
        yield rec

register(DatasetSpec("nexar", nexar, ("train", "test"), _NX, "other (Nexar Open Data License)", "video", multimodal=True,
                     description="Nexar dashcam collision prediction: will the clip be followed by a collision or near-miss?"))
