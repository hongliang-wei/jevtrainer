"""Temporal understanding and retrieval inside a video: when does the described moment happen (candidate time ranges),
plus Perception Test multiple choice (evaluation only).

Moment localisation is cast as pointing at one of four candidate time ranges ("12.0 s - 19.5 s"): the true moment and three
windows of the same length placed elsewhere in the clip (no overlap with any annotated moment of the query). The video
is converted whole (16 frames + sound), so a time range can be read off the frame timeline.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from jevtrainer.data import avkit, vidkit
from jevtrainer.data.base import DatasetSpec, mcq_record, register, rid

Q_WHEN = "In which time range of the video does the described moment happen?"
FR, SIDE = 16, 336


def _iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def _overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def window_record(id, media, query, dur, gold, avoid, rng, **meta):
    """gold = (start, end) seconds; distractors: same-length windows that overlap none of `avoid`."""
    length = gold[1] - gold[0]
    if length < 1.0 or dur - length < 1.0:
        return None
    wins = [gold]
    for _ in range(400):
        s = rng.uniform(0, dur - length)
        w = (s, s + length)
        if any(_overlap(w, a) > 0 for a in avoid) or any(_iou(w, x) > 0.2 for x in wins):
            continue
        wins.append(w)
        if len(wins) == 4:
            break
    if len(wins) < 4:
        return None
    order = list(range(4))
    rng.shuffle(order)
    text = [f"{wins[i][0]:.1f} s - {wins[i][1]:.1f} s" for i in order]
    state = {"clip": "<video:1>", "moment": query, "video_length": f"{dur:.1f} s"}
    rec = mcq_record(id, state, Q_WHEN, text, order.index(0), area="video", **meta)
    if rec:
        rec.media = [media]
    return rec


def _cap_env(name: str, default: int, cap: int) -> int:
    """Downloads are the bottleneck for whole-video datasets: JT_<NAME>_MAX overrides the default ceiling."""
    return min(cap, int(os.environ.get(f"JT_{name.upper()}_MAX", default)))


# ---- Perception Test (evaluation) ---------------------------------------------------------------------------------
_PT = "advaitgupta/perception_test_mcq"


def perception_test(split, cap, rng):
    """Perception Test multiple-choice video QA, validation questions with public answers (1000 videos, one question
    each; sound kept). Evaluation only: never in a training mixture."""
    df = vidkit.read_csv(_PT, "data/metadata.csv")
    rows = []
    for r in df.to_dict("records"):
        try:
            opts = json.loads(r["options"])
        except Exception:  # noqa: BLE001
            continue
        ans = r.get("correct_answer")
        if isinstance(ans, str) and ans in opts and len(opts) >= 2:
            rows.append({**r, "opts": opts})
    rng.shuffle(rows)
    rows = rows[:cap]
    media = vidkit.convert_hub(_PT, "perception_test", {r["video_id"]: f"data/{r['file_name']}" for r in rows}, frames=FR, max_side=SIDE,
                               audio_s=40)
    for r in rows:
        m = media.get(r["video_id"])
        if not m:
            continue
        rec = mcq_record(rid("perception_test", r["video_id"], r["question"]), {"clip": "<video:1>", "question": r["question"]},
                         "Which option correctly answers the question about the video?", r["opts"], r["opts"].index(r["correct_answer"]),
                         area="video", dataset="perception_test")
        if rec:
            rec.media = [m]
            yield rec


# ---- Charades-STA ---------------------------------------------------------------------------------------------------
_CH_ANN = "VideoSearchR1/charades-stage1_data"
_CH_VID = "jinyoungkim/Charades"


def charades_sta(split, cap, rng):
    """Charades-STA (official train / test sentence annotations): one sentence per video, pick the time range that
    shows it among four. Whole videos (~30 s, with sound). Download ceiling JT_CHARADES_STA_MAX (default 1500 train /
    600 test videos)."""
    rows = vidkit.read_jsonl(_CH_ANN, f"raw_annotation/{'train' if split == 'train' else 'test'}.jsonl")
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["video"], []).append(r)
    vids = sorted(by)
    rng.shuffle(vids)
    vids = vids[:_cap_env("charades_sta", 1500 if split == "train" else 600, cap)]
    media = vidkit.convert_hub(_CH_VID, "charades_sta", {v: f"videos/{v}.mp4" for v in vids}, frames=FR, max_side=SIDE, audio_s=40)
    for v in vids:
        m = media.get(v)
        if not m:
            continue
        r = rng.choice(by[v])
        dur = m["duration"]
        s, e = r["time"]
        rec = window_record(rid("charades_sta", split, v), m, r["cog_desc"].strip(), dur, (s, min(e, dur)),
                            [(a["time"][0], a["time"][1]) for a in by[v]], rng, dataset="charades_sta")
        if rec:
            yield rec


# ---- QVHighlights ---------------------------------------------------------------------------------------------------
_QVH_VID = "ayushsdev/qvhighlights-videos"


def _merge(wins, gap=2.0):
    out = []
    for s, e in sorted(wins):
        if out and s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(w) for w in out]


def qvhighlights(split, cap, rng):
    """QVHighlights (official train / val; test has no public labels): a 150 s YouTube clip and a query; the gold range
    is the longest annotated relevant moment (adjacent 10 s pieces merged), distractors avoid all annotated pieces.
    Download ceiling JT_QVHIGHLIGHTS_MAX (default 1000 train / 400 val clips)."""
    url = f"https://raw.githubusercontent.com/jayleicn/moment_detr/main/data/highlight_{'train' if split == 'train' else 'val'}_release.jsonl"
    rows = [json.loads(x) for x in vidkit.http_text(url, "qvhighlights", f"{split}.jsonl").splitlines() if x.strip()]
    have = vidkit.listing(_QVH_VID)
    rows = [r for r in rows if any(f.endswith(f"/{r['vid']}.mp4") or f == f"{r['vid']}.mp4" for f in ()) or True]
    names = {Path(f).stem: f for f in have if f.endswith(".mp4")}
    rows = [r for r in rows if r["vid"] in names]
    seen, uniq = set(), []
    rng.shuffle(rows)
    for r in rows:  # one query per video
        if r["vid"] not in seen:
            seen.add(r["vid"])
            uniq.append(r)
    uniq = uniq[:_cap_env("qvhighlights", 1000 if split == "train" else 400, cap)]
    media = vidkit.convert_hub(_QVH_VID, "qvhighlights", {r["vid"]: names[r["vid"]] for r in uniq}, frames=FR, max_side=SIDE, audio_s=150)
    for r in uniq:
        m = media.get(r["vid"])
        if not m:
            continue
        wins = _merge(r["relevant_windows"])
        gold = max(wins, key=lambda w: w[1] - w[0])
        rec = window_record(rid("qvhighlights", split, r["qid"]), m, r["query"].strip(), min(float(r["duration"]), m["duration"]), gold,
                            [tuple(w) for w in r["relevant_windows"]], rng, dataset="qvhighlights")
        if rec:
            yield rec


# ---- ActivityNet Captions ----------------------------------------------------------------------------------------------
_AN = "friedrichor/ActivityNet_Captions"


def activitynet_captions(split, cap, rng):
    """ActivityNet Captions: one captioned segment per video, pick its time range among four. The video archive is a
    split tar (42 GB); only the first JT_ACTIVITYNET_PARTS parts (default 2, ~3.8k videos) are fetched. Videos longer than
    3 minutes are skipped so that the 120 s of sound cover the whole clip. Official train -> train, val_1 -> val."""
    ann = vidkit.read_json(_AN, "raw_data/train.json" if split == "train" else "raw_data/val_1.json")
    ok = {v: a for v, a in ann.items() if a["duration"] <= 180 and a["timestamps"]}
    nparts = int(os.environ.get("JT_ACTIVITYNET_PARTS", "2"))
    parts = [avkit.fetch_file(_AN, f"ActivityNet_Videos.tar.part-{i:03d}", "activitynet") for i in range(nparts)]
    parts = [p for p in parts if p]
    limit = _cap_env("activitynet", 2500 if split == "train" else 500, cap)
    n = [0]

    def keep(name):
        ok_ = Path(name).stem in ok and n[0] < limit
        n[0] += ok_
        return ok_

    stream = ((Path(nm).stem, nm, b) for nm, b in vidkit.tar_members(parts, keep, mode="r|", done=lambda: n[0] >= limit))
    media = vidkit.convert_stream("activitynet", stream, frames=FR, max_side=SIDE, audio_s=120)
    for v, m in media.items():
        a = ok[v]
        idx = [i for i, (s, e) in enumerate(a["timestamps"]) if e - s >= 3 and (e - s) <= 0.6 * a["duration"]]
        if not idx:
            continue
        i = rng.choice(idx)
        s, e = a["timestamps"][i]
        dur = min(a["duration"], m["duration"])
        rec = window_record(rid("activitynet_captions", split, v), m, a["sentences"][i].strip(), dur, (s, min(e, dur)),
                            [tuple(t) for t in a["timestamps"]], rng, dataset="activitynet_captions")
        if rec:
            yield rec


register(DatasetSpec("perception_test", perception_test, ("val",), _PT, "apache-2.0", "video", eval_only=True, multimodal=True,
                     description="Perception Test validation multiple-choice video QA (evaluation only)"))
register(DatasetSpec("charades_sta", charades_sta, ("train", "test"), _CH_ANN, "other (research)", "video", multimodal=True,
                     description="Charades-STA: pick the time range of a sentence in a ~30 s indoor video"))
register(DatasetSpec("qvhighlights", qvhighlights, ("train", "val"), _QVH_VID, "cc-by-nc-sa-4.0", "video", multimodal=True,
                     description="QVHighlights: pick the time range of a query moment in a 150 s clip"))
register(DatasetSpec("activitynet_captions", activitynet_captions, ("train", "val"), _AN, "other (research)", "video", multimodal=True,
                     description="ActivityNet Captions: pick the time range of a caption in an untrimmed video"))
