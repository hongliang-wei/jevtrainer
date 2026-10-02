"""Procedural / instructional video understanding: which step is being performed (cooking, kitchen actions).

One clip (a labelled step segment cut out of a longer video), one question, four options: the true step description
plus three descriptions of steps taken from other videos (other recipes / other kitchens), so that the visual content
and sound decide, not the surrounding text.
"""

from __future__ import annotations

import re
from pathlib import Path

from jevtrainer.data import avkit, vidkit
from jevtrainer.data.base import DatasetSpec, register, rid

Q_STEP = "Which option describes the step being performed in the clip?"


# ---- YouCook2: cooking steps ------------------------------------------------------------------------------------
_YC = "VLM2Vec/YouCook2"


def youcook2(split, cap, rng):
    """YouCook2 (labelled `val` part of the dataset, the only part with public step sentences): each recipe step
    segment is a clip; the answer is its imperative sentence ("pick the ends off the verdalago"). Distractors are
    sentences of steps from other videos. The test split is a fixed 10% of the *videos* (hash of the YouTube id)."""
    df = vidkit.read_parquet(_YC, "data/val-00000-of-00001.parquet")
    rows = [r for r in df.to_dict("records") if isinstance(r["sentence"], str) and len(r["sentence"].split()) >= 3]
    rows = [r for r in rows if avkit.hash_pct(r["youtube_id"], 10) == (split == "test")]
    files = {}
    for f in vidkit.listing(_YC, "raw_videos/"):
        files[re.sub(r"\.[A-Za-z0-9]+$", "", f.split("/")[-1])] = f
    rows = [r for r in rows if r["id"] in files]
    rng.shuffle(rows)
    rows = rows[:cap]
    media = vidkit.convert_hub(_YC, "youcook2", {r["id"]: files[r["id"]] for r in rows}, frames=8, max_side=448, audio_s=30)
    pool = [r["sentence"].strip() for r in rows]
    for r in rows:
        m = media.get(r["id"])
        if not m:
            continue
        gold = r["sentence"].strip()
        others = [r2["sentence"].strip() for r2 in rng.sample(rows, min(40, len(rows))) if r2["youtube_id"] != r["youtube_id"]]
        rec = vidkit.option_record(rid("youcook2", split, r["id"]), m, "What cooking step is being performed in this clip?", gold,
                                   others, rng, Q_STEP, area="video", dataset="youcook2", recipe_type=str(r["recipe_type"]))
        if rec:
            yield rec


# ---- EPIC-KITCHENS-100: what is the person doing (egocentric kitchen actions, narrated) -------------------------------
_EK = "lightly-ai/epic-kitchens-100-clips"


def epic_kitchens(split, cap, rng):
    """EPIC-KITCHENS-100 clips (the extension videos, one short clip per narrated action; sound kept when present).
    The answer is the participant's narration ("open door", "wash plate"); distractors are narrations whose verb
    differs from the true one. Official train -> train, official validation -> val; the clips of the repo are cut
    from both. Balanced over verb classes."""
    sp = {"train": "EPIC_100_train", "val": "EPIC_100_validation"}[split]
    df = vidkit.read_csv(_EK, f"epic-kitchens-100-annotations/{sp}.csv")
    have = vidkit.listing(_EK, "clips/")
    rows = [r for r in df.to_dict("records") if f"clips/{r['participant_id']}/{r['narration_id']}.mp4" in have
            and isinstance(r["narration"], str) and len(r["narration"].split()) >= 2]
    rows = avkit.balanced(rows, lambda r: r["verb"], cap, rng)
    media = vidkit.convert_hub(_EK, "epic_kitchens", {r["narration_id"]: f"clips/{r['participant_id']}/{r['narration_id']}.mp4" for r in rows},
                               frames=8, max_side=448, audio_s=10)
    pool = [r for r in rows if r["narration_id"] in media]
    for r in pool:
        others = [o["narration"] for o in rng.sample(pool, min(40, len(pool))) if o["verb"] != r["verb"]]
        rec = vidkit.option_record(rid("epic_kitchens", split, r["narration_id"]), media[r["narration_id"]],
                                   "What is the person doing in this kitchen clip?", r["narration"].strip(), [o.strip() for o in others],
                                   rng, "Which option best describes the action the person is performing?", area="video",
                                   dataset="epic_kitchens", verb=r["verb"], noun=r["noun"])
        if rec:
            yield rec


# ---- COIN: instructional steps over 12 domains ------------------------------------------------------------------------
_COIN_VID = "ttyue/COIN_Dataset"
_COIN_ANN = "https://raw.githubusercontent.com/coin-dataset/annotations/master/COIN.json"


def coin(split, cap, rng):
    """COIN (180 tasks, 778 step labels, YouTube how-to videos): each annotated step segment is cut out of its video (with
    sound); the answer is the step label ("put on the hair extensions"); distractors are step labels of other tasks.
    Official training -> train, testing -> test. Whole videos are ~35 MB, so at most JT_COIN_VIDEOS (default 220 train /
    60 test) videos are fetched; every step segment of a fetched video becomes one clip."""
    import json
    import os

    db = json.loads(vidkit.http_text(_COIN_ANN, "coin", "COIN.json"))["database"]
    want = "training" if split == "train" else "testing"
    path = {Path(f).stem: f for f in vidkit.listing(_COIN_VID, "videos/") if f.endswith(".mp4")}
    vids = sorted(v for v, a in db.items() if a["subset"] == want and v in path and a.get("annotation"))
    rng.shuffle(vids)
    vids = vids[:min(int(os.environ.get("JT_COIN_VIDEOS", 220 if split == "train" else 60)), max(1, cap // 3))]
    raw = vidkit.fetch_many(_COIN_VID, "coin", {v: path[v] for v in vids}, workers=4)
    jobs, segs = {}, {}
    for v, src in raw.items():
        for a in db[v]["annotation"]:
            s, e = a["segment"]
            if e - s < 1.5 or e - s > 60:
                continue
            key = f"{v}_{a['id']}"
            jobs[key] = {"src": src, "start": float(s), "end": float(e)}
            segs[key] = (v, a["label"].strip(), db[v].get("class", ""))
    media = vidkit.convert_local("coin", jobs, frames=8, max_side=448, audio_s=30)
    labels = sorted({s[1] for s in segs.values()})
    keys = [k for k in media]
    rng.shuffle(keys)
    for k in keys[:cap]:
        v, gold, task = segs[k]
        others = [x for x in rng.sample(labels, min(60, len(labels))) if x != gold]
        rec = vidkit.option_record(rid("coin", split, k), media[k], "Which step of the task is being performed in this clip?", gold, others, rng,
                                   Q_STEP, area="video", dataset="coin", task=task)
        if rec:
            yield rec


register(DatasetSpec("coin", coin, ("train", "test"), "https://coin-dataset.github.io", "cc-by-nc-sa-4.0", "video", multimodal=True,
                     description="COIN instructional steps: pick the step label of a how-to video segment"))
register(DatasetSpec("epic_kitchens", epic_kitchens, ("train", "val"), _EK, "cc-by-nc-4.0", "video", multimodal=True,
                     description="EPIC-KITCHENS-100 egocentric clips: pick the narrated action"))
register(DatasetSpec("youcook2", youcook2, ("train", "test"), _YC, "other (research)", "video", multimodal=True,
                     description="YouCook2 step recognition: which recipe step sentence matches the cooking clip"))
