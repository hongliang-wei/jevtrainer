"""Video-caption matching: which sentence describes the clip (video and its sound track).

One video, one question, four caption options: one of the video's human captions plus three captions of other videos.
Distractors come from the same split, so the sentence style gives nothing away.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from jevtrainer.data import avkit, vidkit
from jevtrainer.data.base import DatasetSpec, register, rid

Q_CAP = "Which caption best describes the video?"


def _emit(name, split, rng, vids, captions, media, lang_q="Which sentence describes this video?", **meta):
    """vids: list of video keys; captions: key -> list[str]. Distractors: captions of other kept videos."""
    keys = [k for k in vids if k in media and captions.get(k)]
    pool = [(k, c) for k in keys for c in captions[k][:3]]
    for k in keys:
        gold = rng.choice(captions[k]).strip()
        others = [c.strip() for k2, c in rng.sample(pool, min(30, len(pool))) if k2 != k]
        rec = vidkit.option_record(rid(name, split, k), media[k], lang_q, gold, others, rng, Q_CAP, area="video", dataset=name, **meta)
        if rec:
            yield rec


# ---- MSR-VTT ------------------------------------------------------------------------------------------------------
_MSR = "VLM2Vec/MSR-VTT"


def msrvtt(split, cap, rng):
    """MSR-VTT (10k web videos, 20 captions each): official train / validate / test lists -> train / val / test."""
    data = vidkit.read_json(_MSR, "raw_data/MSRVTT_data.json")
    want = {"train": "train", "val": "validate", "test": "test"}[split]
    vids = [v["video_id"] for v in data["videos"] if v["split"] == want]
    caps = defaultdict(list)
    for s in data["sentences"]:
        caps[s["video_id"]].append(s["caption"])
    files = {Path(f).stem: f for f in vidkit.listing(_MSR) if f.endswith(".mp4")}
    vids = [v for v in vids if v in files]
    rng.shuffle(vids)
    vids = vids[:cap]
    media = vidkit.convert_hub(_MSR, "msrvtt", {v: files[v] for v in vids}, frames=8, max_side=448, audio_s=30)
    yield from _emit("msrvtt", split, rng, vids, caps, media)


# ---- VATEX (English, and Chinese for the validation videos) -------------------------------------------------------
_VX = "Blazewild/Vatex_reencoded"


def _vatex_files():
    return {Path(f).stem: f for f in vidkit.listing(_VX) if f.endswith(".mp4")}


def vatex(split, cap, rng):
    """VATEX English captions (10 per video; re-encoded small mirror of the videos). Official train / val lists."""
    data = vidkit.read_json(_VX, "VATEX_CAPTIONS.json")
    vid_of = {v["video_id"]: v for v in data["videos"]}
    caps = defaultdict(list)
    for a in data["annotations"]:
        if a["split"] == split:
            caps[a["video_id"]].append(a["caption"])
    files = _vatex_files()
    vids = [k for k, v in vid_of.items() if v["split"] == split and v["videoID"] in files and caps.get(k)]
    rng.shuffle(vids)
    vids = vids[:cap]
    media = vidkit.convert_hub(_VX, "vatex", {k: files[vid_of[k]["videoID"]] for k in vids}, frames=8, max_side=448, audio_s=12)
    yield from _emit("vatex", split, rng, vids, caps, media)


def vatex_zh(split, cap, rng):
    """VATEX Chinese captions (10 per video) for the 2.2k validation videos (the only part with public Chinese labels in
    the hub mirrors); options are Chinese sentences. 10% of the videos (hash of the YouTube id) form the test split."""
    df = vidkit.read_parquet("lmms-eval/VATEX_ZH", "vatex_val_zh/validation-00000-of-00001.parquet")
    files = _vatex_files()
    rows = [r for r in df.to_dict("records") if r["videoID"] in files and len(r["chCap"]) > 0
            and avkit.hash_pct(r["videoID"], 10) == (split == "test")]
    rng.shuffle(rows)
    rows = rows[:cap]
    media = vidkit.convert_hub(_VX, "vatex", {r["videoID"]: files[r["videoID"]] for r in rows}, frames=8, max_side=448, audio_s=12)
    caps = {r["videoID"]: [str(c) for c in r["chCap"]] for r in rows}
    yield from _emit("vatex_zh", split, rng, [r["videoID"] for r in rows], caps, media, lang_q="哪一句话最准确地描述了这段视频？")


register(DatasetSpec("msrvtt", msrvtt, ("train", "val", "test"), _MSR, "other (research)", "video", multimodal=True,
                     description="MSR-VTT caption matching: pick the caption of the video among 4"))
register(DatasetSpec("vatex", vatex, ("train", "val"), _VX, "cc-by-4.0", "video", multimodal=True,
                     description="VATEX English caption matching (4 options)"))
register(DatasetSpec("vatex_zh", vatex_zh, ("train", "test"), "lmms-eval/VATEX_ZH", "cc-by-4.0", "video", multimodal=True,
                     description="VATEX Chinese caption matching (4 Chinese sentences)"))
