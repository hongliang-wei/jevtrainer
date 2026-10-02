"""CREMA-D video + sound: which of six emotions does the actor act? (audio-visual version of `crema_d`).

Source: `NoahMartinezXiang/2014_CREMA-D` (`CREMA-D.zip`, 7,442 flv clips of 91 actors reading 12 sentences, with sound).
The repo's own train / val / test folders cut clips at random, so the same actor sits in every split; the three folders are
merged here and re-split by *actor* (hash of the actor id: ~10 % test, ~10 % val, rest train). The zip is not downloaded
as a whole: every member is fetched with its own range request.
"""

from __future__ import annotations

import re
import threading

from jevtrainer.data import avkit, vidkit
from jevtrainer.data.avkit_video import cached_item, convert_remote_zip_videos, hash_bucket
from jevtrainer.data.base import DatasetSpec, choice_record, register, rid

_REPO = "NoahMartinezXiang/2014_CREMA-D"
_LOCK = threading.Lock()
EMOTIONS = {
    "angry": "angry: tense face and a hard, raised or clipped voice",
    "disgust": "disgust: wrinkled nose, sneering mouth and a revolted voice",
    "fear": "fear: wide eyes, tense face and a shaky or high voice",
    "happy": "happy: smiling face and a bright, lively voice",
    "neutral": "neutral: relaxed face and a flat, matter-of-fact voice",
    "sad": "sad: downcast face and a low, slow, heavy voice",
}


def _bucket(actor: str) -> str:
    b = hash_bucket("crema_video_actor", actor)
    return "test" if b < 10 else "val" if b < 20 else "train"


def _rows():
    from jevtrainer.data.avkit_video import remote_zip_index

    _, idx = remote_zip_index(_REPO, "CREMA-D.zip")
    rows = []
    for member in sorted(idx):
        m = re.search(r"/(\d{4})_([A-Z]{3})_[A-Z]{3}_[A-Z]{2}\.flv$", member)
        if not m:
            continue
        emo = member.split("/")[-2].lower()
        if emo in EMOTIONS:
            rows.append({"member": member, "actor": m.group(1), "sentence": m.group(2), "emotion": emo, "key": member.split("/")[-1][:-4],
                         "split": _bucket(m.group(1))})
    return rows


def crema_d_video(split, cap, rng):
    rows = [r for r in _rows() if r["split"] == split]
    rows = avkit.balanced(rows, lambda r: r["emotion"], cap, rng)
    todo = {r["member"]: r["key"] for r in rows if not cached_item("crema_d_video", r["key"])}
    if todo:
        with _LOCK:
            convert_remote_zip_videos(_REPO, "CREMA-D.zip", "crema_d_video", todo, workers=6, frames=6, max_side=384, audio_s=6)
    for r in rows:
        m = cached_item("crema_d_video", r["key"])
        if not m:
            continue
        rec = choice_record(rid("crema_d_video", split, r["key"]),
                            {"clip": "<video:1>", "question": "Which emotion is the actor expressing?"},
                            "Watch and listen to the actor reading a sentence. Which emotion is being acted?", dict(EMOTIONS), r["emotion"],
                            area="av", dataset="crema_d_video", actor=r["actor"])
        rec.media = [m]
        yield rec


register(DatasetSpec("crema_d_video", crema_d_video, ("train", "val", "test"), _REPO, "odbl-1.0", "av", multimodal=True,
                     description="CREMA-D audio-visual emotion (6 classes), actor-disjoint train / val / test"))
