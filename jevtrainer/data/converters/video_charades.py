"""Charades (Sigurdsson et al. 2016): everyday indoor activities filmed by crowd workers (480p copy of the videos).

Annotations come from the official ``Charades.zip`` (train / test csv, 157 action classes with time spans, scene,
script sentence); the videos are read member by member out of ``Charades_v1_480.zip`` of `tdat1465/star-charades`.
One video, one question; which of three question kinds a video gets is drawn at random (`meta["task"]`):

    action  an annotated action with its [start, end] span (>= 1.5 s, at most 20 s); the clip is that span and the options
            are its class description plus 3 classes that are *not* annotated anywhere in the video
    scene   the room type of the whole video (Kitchen, Bedroom, ...), distractors are other scene labels
    script  which of 4 sentences is the script the actor was asked to perform (the 3 others are scripts of other videos
            of the same split)

Splits are the official train / test lists (test videos never appear in train).
"""

from __future__ import annotations

import io
import urllib.request
import zipfile

from jevtrainer.data import avkit, vidkit
from jevtrainer.data.base import DatasetSpec, register, rid
from jevtrainer.data.zipkit import convert_zip_clips, open_zip

_ANN = "https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades.zip"
_VID = ("tdat1465/star-charades", "Charades_v1_480.zip")
_Q_ACT = "Which activity is the person doing in this video?"
_Q_SCENE = "In which kind of room was this video filmed?"
_INS = "Watch the video and choose the correct option."
_Q_SCRIPT = "Which sentence is the script that the person was acting out in this video?"


def _annotations(split: str):
    import pandas as pd

    p = avkit.raw_dir("charades") / "Charades.zip"
    if not p.exists() or p.stat().st_size < 1_000_000:
        data = vidkit.retry(lambda: urllib.request.urlopen(urllib.request.Request(_ANN, headers={"User-Agent": "jevtrainer"}), timeout=120).read())
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(p)
    with zipfile.ZipFile(p) as z:
        df = pd.read_csv(io.BytesIO(z.read(f"Charades/Charades_v1_{split}.csv")))
        classes = {}
        for line in z.read("Charades/Charades_v1_classes.txt").decode("utf8").splitlines():
            if line.strip():
                k, v = line.split(" ", 1)
                classes[k] = v.strip()
    return df.to_dict("records"), classes


def charades(split, cap, rng):
    rows, classes = _annotations("train" if split == "train" else "test")
    z = open_zip(*_VID)
    have = {n.rsplit("/", 1)[-1][:-4] for n in z.names() if n.endswith(".mp4")}
    rows = [r for r in rows if r["id"] in have and isinstance(r.get("scene"), str)]
    rng.shuffle(rows)
    scenes = sorted({r["scene"] for r in rows})
    scripts = [str(r["script"]).strip() for r in rows if isinstance(r.get("script"), str) and len(str(r["script"])) > 20]
    plan = []
    for r in rows:
        acts = []
        for a in str(r.get("actions") or "").split(";"):
            p = a.split()
            if len(p) == 3 and p[0] in classes and 1.5 <= float(p[2]) - float(p[1]) <= 20:
                acts.append((p[0], float(p[1]), float(p[2])))
        kind = rng.choices(["action", "scene", "script"], [3, 1, 1])[0]
        if kind == "action" and not acts:
            kind = "scene"
        if kind == "script" and not (isinstance(r.get("script"), str) and len(r["script"]) > 20):
            kind = "scene"
        plan.append((r, kind, rng.choice(acts) if kind == "action" else None))
        if len(plan) >= cap:
            break
    jobs = {}
    for r, kind, act in plan:
        key = r["id"] if act is None else f"{r['id']}_{act[0]}_{int(act[1] * 10)}"
        jobs[key] = (z, f"Charades_v1_480/{r['id']}.mp4", {} if act is None else {"start": act[1], "end": act[2]})
    media = convert_zip_clips("charades", jobs, frames=8, max_side=448, audio_s=20)
    for r, kind, act in plan:
        key = r["id"] if act is None else f"{r['id']}_{act[0]}_{int(act[1] * 10)}"
        m = media.get(key)
        if not m:
            continue
        if kind == "action":
            seen = {a.split()[0] for a in str(r["actions"]).split(";") if a.strip()}
            others = [v for k, v in classes.items() if k not in seen]
            rec = vidkit.option_record(rid("charades", split, key), m, _Q_ACT, classes[act[0]], others, rng, _INS,
                                       dataset="charades", task="action")
        elif kind == "scene":
            rec = vidkit.option_record(rid("charades", split, key), m, _Q_SCENE, r["scene"], scenes, rng, _INS,
                                       dataset="charades", task="scene")
        else:
            gold = str(r["script"]).strip()
            rec = vidkit.option_record(rid("charades", split, key), m, _Q_SCRIPT, gold, rng.sample(scripts, 12), rng,
                                       _INS, dataset="charades", task="script")
        if rec:
            yield rec


register(DatasetSpec("charades", charades, ("train", "test"), "https://prior.allenai.org/projects/charades (+ tdat1465/star-charades videos)",
                     "other (research, non-commercial)", "video", multimodal=True,
                     description="Charades: activity of an annotated span, room type, or the actor's script sentence (indoor home videos)"))
