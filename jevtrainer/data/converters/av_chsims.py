"""CH-SIMS v1 (Yu et al. 2020): Chinese talking-head clips with separate sentiment labels per modality.

Source ``tamb2203579/CH-SIMS`` (apache-2.0): ``label.csv`` + ``Raw.zip`` (2,281 clips of 60 raw videos; clips are read straight
out of the 8.8 GB zip member by member). Official train / valid (= `val`) / test split from the csv's `mode` column.

    chsims            3-way sentiment of the speaker (negative / neutral / positive) from the annotated multimodal label
                      (`annotation` column); the transcript is part of the state.
    chsims_nonverbal  the same question without the transcript: the label is the average of the audio-only and the
                      vision-only annotations (label_A, label_V; <= -0.1 negative, >= 0.1 positive, else neutral), so the
                      answer has to come from voice and face.  (v2 of the corpus is `chsims2`.)
"""

from __future__ import annotations

import re

from jevtrainer.data.base import DatasetSpec, choice_record, hf_file, register, rid
from jevtrainer.data.zipkit import convert_zip_clips, open_zip

_REPO = "tamb2203579/CH-SIMS"
_MODE = {"train": "train", "val": "valid", "test": "test"}
LEVELS = {
    "negative": "负面：不满、生气、厌恶、悲伤或担忧",
    "neutral": "中性：没有明显的情感倾向",
    "positive": "正面：愉快、认同、赞赏或期待",
}
_Q = "观看并聆听这段视频中人物的说话，说话人表达的情感倾向是？"
_Q_NV = "只根据说话人的表情和语气（不看文字内容），说话人表达的情感倾向是？"


def _sign(x: float, t: float = 0.1) -> str:
    return "negative" if x <= -t else "positive" if x >= t else "neutral"


def _rows(split: str, cap: int, rng) -> list[tuple[dict, dict]]:
    import pandas as pd

    df = pd.read_csv(hf_file(_REPO, "label.csv"), header=None, dtype={0: str, 1: str}, encoding="utf-8",
                     names=["video_id", "clip_id", "text", "label", "label_T", "label_A", "label_V", "annotation", "mode"])
    df = df[df["mode"] == _MODE[split]]
    z = open_zip(_REPO, "Raw.zip")
    by = {}
    for n in z.names():
        m = re.search(r"(video_\d+)/(\d+)\.mp4$", n)
        if m:
            by[(m.group(1), m.group(2))] = n
    rows = [r for r in df.to_dict("records") if (r["video_id"], r["clip_id"]) in by]
    rng.shuffle(rows)
    rows = rows[:cap]
    jobs = {f"{r['video_id']}_{r['clip_id']}": (z, by[(r["video_id"], r["clip_id"])], {}) for r in rows}
    media = convert_zip_clips("chsims", jobs, frames=8, max_side=448, audio_s=20)
    return [(r, media[f"{r['video_id']}_{r['clip_id']}"]) for r in rows if f"{r['video_id']}_{r['clip_id']}" in media]


def chsims(split, cap, rng):
    for r, m in _rows(split, cap, rng):
        y = str(r["annotation"]).strip().lower()
        if y not in LEVELS:
            continue
        rec = choice_record(rid("chsims", split, r["video_id"], r["clip_id"]), {"clip": "<video:1>", "文本": str(r["text"]).strip()}, _Q,
                            dict(LEVELS), y, area="av", dataset="chsims", task="sentiment", raw_score=float(r["label"]))
        rec.media = [m]
        yield rec


def chsims_nonverbal(split, cap, rng):
    for r, m in _rows(split, cap, rng):
        y = _sign((float(r["label_A"]) + float(r["label_V"])) / 2)
        rec = choice_record(rid("chsims_nv", split, r["video_id"], r["clip_id"]), {"clip": "<video:1>"}, _Q_NV, dict(LEVELS), y,
                            area="av", dataset="chsims", task="nonverbal_sentiment", raw_a=float(r["label_A"]), raw_v=float(r["label_V"]))
        rec.media = [m]
        yield rec


register(DatasetSpec("chsims", chsims, ("train", "val", "test"), _REPO, "apache-2.0", "av", multimodal=True,
                     description="CH-SIMS v1: negative / neutral / positive sentiment of a Chinese speaker (video, voice, transcript)"))
register(DatasetSpec("chsims_nonverbal", chsims_nonverbal, ("train", "val", "test"), _REPO, "apache-2.0", "av", multimodal=True,
                     description="CH-SIMS v1: sentiment from voice and face only (average of the audio-only and vision-only labels)"))
