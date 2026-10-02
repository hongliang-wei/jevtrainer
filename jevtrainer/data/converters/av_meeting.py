"""Meeting / conversation audio: who speaks and how many speak at once (AMI headset mics, VoxConverse broadcasts).

AMI  (edinburghcstr/ami, config `ihm` = individual headset mics, cc-by-4.0)  -- audio only
    Official splits: train = first 4 train shards, val = validation shard 0, test = test shards 0-1 (each split reads whole
    shards, so splits never share a meeting). Utterances of 1.5-10 s.
    ami_same_speaker  yes/no: do two utterances of one meeting come from the same participant? Each pooled utterance is the
                      anchor of one pair, positive or negative by a fixed coin (hash of the audio id). Positive partner:
                      another utterance of the same `speaker_id` in the same meeting; negative partner: an utterance of a
                      different `speaker_id` of the same meeting. Both clips are separate audio items (`<audio:1>`, `<audio:2>`).
    ami_gender        2-way choice: is the speaker male or female? The first letter of the AMI participant id (M... / F...)
                      is the annotated gender.

VoxConverse  (diarizers-community/voxconverse, cc-by-4.0)  -- audio only, news / debate recordings with diarization labels
    train = dev shards 0-3, val = dev shard 4, test = test shards 0-2. Every ~6 min row yields up to 5 random 30 s windows;
    labels are computed from the diarization segments clipped to the window.
    voxconverse_speakers  4-way choice: number of people speaking at least 1 s in the window (1, 2, 3, 4 or more)
    voxconverse_overlap   yes/no: two or more people talk at the same time. yes = at least 1.0 s of overlapped speech,
                          no = at most 0.1 s; windows in between are dropped.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict

from jevtrainer.data import avkit
from jevtrainer.data.avkit_video import fetch_ranged, hash_bucket
from jevtrainer.data.base import DatasetSpec, choice_record, noul_record, register, rid

_CONNS = int(os.environ.get("JEVTRAINER_AV_CONNS", "4"))
_WORKERS = int(os.environ.get("JEVTRAINER_AV_WORKERS", "8"))

# ---- AMI ---------------------------------------------------------------------------------------------------------------
_AMI = "edinburghcstr/ami"
_AMI_SHARDS = {
    "train": [f"ihm/train-{i:05d}-of-00042.parquet" for i in range(4)],
    "val": ["ihm/validation-00000-of-00005.parquet"],
    "test": [f"ihm/test-{i:05d}-of-00004.parquet" for i in range(2)],
}


def _ami_pool(split: str, n: int, rng) -> list[dict]:
    """Audio-decoded utterances {id, meeting, spk, media}; cached next to the clips as pool.json."""
    out_dir = avkit.media_dir("ami", f"_pool_{split}")
    cache = out_dir / f"pool_{n}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    import pyarrow.parquet as pq

    pool = []
    for fn in _AMI_SHARDS[split]:
        path = fetch_ranged(_AMI, fn, "ami", conns=_CONNS)
        if path is None:
            continue
        pf = pq.ParquetFile(str(path))
        meta = pf.read(columns=["meeting_id", "audio_id", "begin_time", "end_time", "speaker_id"]).to_pylist()
        for i, r in enumerate(meta):
            r["i"] = i
        cand = [r for r in meta if 1.5 <= r["end_time"] - r["begin_time"] <= 10.0]
        rng.shuffle(cand)
        want = {r["i"]: r for r in cand[: n // len(_AMI_SHARDS[split]) + 1]}
        offset = 0
        for batch in pf.iter_batches(batch_size=32, columns=["audio"]):
            for j, a in enumerate(batch.column(0).to_pylist()):
                r = want.get(offset + j)
                if r is None:
                    continue
                key = rid("ami", r["audio_id"])
                item = avkit.audio_from_bytes(a["bytes"], "ami", key, max_s=10.0)
                if item:
                    pool.append({"id": r["audio_id"], "meeting": r["meeting_id"], "spk": r["speaker_id"], "path": item["path"]})
            offset += batch.num_rows
        path.unlink(missing_ok=True)
    cache.write_text(json.dumps(pool))
    return pool


def ami_same_speaker(split, cap, rng):
    pool = _ami_pool(split, min(cap, {"train": 3000, "val": 300, "test": 600}[split]), rng)
    by_meeting: dict[str, list[dict]] = defaultdict(list)
    for u in pool:
        by_meeting[u["meeting"]].append(u)
    for a in pool:
        positive = hash_bucket("ami_pair", a["id"], 2) == 0
        same = [u for u in by_meeting[a["meeting"]] if u["spk"] == a["spk"] and u["id"] != a["id"]]
        diff = [u for u in by_meeting[a["meeting"]] if u["spk"] != a["spk"]]
        partners = same if positive else diff
        if not partners:
            continue
        b = rng.choice(partners)
        rec = noul_record(rid("ami_same_speaker", split, a["id"], b["id"]),
                          {"first": "<audio:1>", "second": "<audio:2>", "question": "Are both utterances spoken by the same person?"},
                          "Listen to the two utterances from the same meeting. Were they spoken by the same participant?", positive,
                          true="same speaker: the voice in both utterances belongs to one participant",
                          false="different speakers: two different participants spoke", area="audio", meeting=a["meeting"])
        rec.media = [{"type": "audio", "path": a["path"]}, {"type": "audio", "path": b["path"]}]
        yield rec


GENDER = {"male": "male: a man's voice", "female": "female: a woman's voice"}


def ami_gender(split, cap, rng):
    pool = _ami_pool(split, min(cap, {"train": 3000, "val": 300, "test": 600}[split]), rng)
    rng.shuffle(pool)
    for a in pool:
        g = {"M": "male", "F": "female"}.get(a["spk"][:1])
        if g is None:
            continue
        rec = choice_record(rid("ami_gender", split, a["id"]), {"clip": "<audio:1>", "question": "Is the speaker male or female?"},
                            "Listen to the utterance. Is the speaker male or female?", dict(GENDER), g, area="audio")
        rec.media = [{"type": "audio", "path": a["path"]}]
        yield rec


# ---- VoxConverse -----------------------------------------------------------------------------------------------------------
_VOX = "diarizers-community/voxconverse"
_VOX_SHARDS = {
    "train": [f"data/dev-{i:05d}-of-00005.parquet" for i in range(4)],
    "val": ["data/dev-00004-of-00005.parquet"],
    "test": [f"data/test-{i:05d}-of-00011.parquet" for i in range(3)],
}
_WINDOW = 30.0


def _window_stats(starts, ends, spks, t0: float) -> tuple[int, float]:
    """(#speakers talking >= 1 s, seconds with >= 2 speakers) inside [t0, t0 + 30] from diarization segments."""
    import numpy as np

    step = 0.05
    n = int(_WINDOW / step)
    act: dict[str, np.ndarray] = {}
    for s, e, k in zip(starts, ends, spks):
        lo, hi = max(s, t0) - t0, min(e, t0 + _WINDOW) - t0
        if hi <= lo:
            continue
        a = act.setdefault(k, np.zeros(n, dtype=bool))
        a[int(lo / step): int(np.ceil(hi / step))] = True
    count = sum(1 for a in act.values() if a.sum() * step >= 1.0)
    total = np.sum([a for a in act.values()], axis=0) if act else np.zeros(n)
    return count, float((total >= 2).sum() * step)


def _vox_windows(split: str, rng) -> list[dict]:
    """Cut 30 s windows out of the shards of a split (once; cached as windows.json)."""
    out_dir = avkit.media_dir("voxconverse", f"_win_{split}")
    cache = out_dir / "windows.json"
    if cache.exists():
        return json.loads(cache.read_text())
    import pyarrow.parquet as pq

    wins = []
    tmp = avkit.raw_dir("voxconverse") / "_x"
    tmp.mkdir(parents=True, exist_ok=True)
    for fn in _VOX_SHARDS[split]:
        path = fetch_ranged(_VOX, fn, "voxconverse", conns=_CONNS)
        if path is None:
            continue
        pf = pq.ParquetFile(str(path))
        for batch in pf.iter_batches(batch_size=2):
            for j, row in enumerate(batch.to_pylist()):
                wav = tmp / f"{os.getpid()}_{j}.wav"
                wav.write_bytes(row["audio"]["bytes"])
                dur = avkit.probe(wav)["duration"]
                for w in range(5):
                    if dur < _WINDOW + 1:
                        break
                    t0 = rng.uniform(0, dur - _WINDOW)
                    count, overlap = _window_stats(row["timestamps_start"], row["timestamps_end"], row["speakers"], t0)
                    key = rid("voxconverse", split, fn, batch.num_rows, j, w, round(dur, 2), round(t0, 2))
                    item = avkit.audio_item(wav, "voxconverse", key, max_s=_WINDOW, start=t0)
                    if item:
                        wins.append({"key": key, "path": item["path"], "count": count, "overlap": overlap})
                wav.unlink(missing_ok=True)
        path.unlink(missing_ok=True)
    cache.write_text(json.dumps(wins))
    return wins


COUNTS = {
    "one": "one person speaks (a single voice in the whole clip)",
    "two": "two people speak",
    "three": "three people speak",
    "four_plus": "four or more different people speak",
}


def voxconverse_speakers(split, cap, rng):
    wins = [w for w in _vox_windows(split, rng) if w["count"] >= 1]
    rng.shuffle(wins)
    for w in wins[:cap]:
        label = ["one", "two", "three", "four_plus"][min(w["count"], 4) - 1]
        rec = choice_record(rid("voxconverse_speakers", split, w["key"]),
                            {"clip": "<audio:1>", "question": "How many different people speak in this clip?"},
                            "Listen to the recording. How many different people speak in it?", dict(COUNTS), label, area="audio")
        rec.media = [{"type": "audio", "path": w["path"]}]
        yield rec


def voxconverse_overlap(split, cap, rng):
    wins = [w for w in _vox_windows(split, rng) if w["overlap"] >= 1.0 or (w["overlap"] <= 0.1 and w["count"] >= 2)]
    rng.shuffle(wins)
    for w in wins[:cap]:
        rec = noul_record(rid("voxconverse_overlap", split, w["key"]),
                          {"clip": "<audio:1>", "question": "Do two or more people talk at the same time somewhere in this clip?"},
                          "Listen to the recording. Is there overlapped speech, where at least two people talk at the same time?",
                          w["overlap"] >= 1.0, true="overlap: several voices at once for at least a second",
                          false="no overlap: speakers take turns", area="audio")
        rec.media = [{"type": "audio", "path": w["path"]}]
        yield rec


register(DatasetSpec("ami_same_speaker", ami_same_speaker, ("train", "val", "test"), _AMI, "cc-by-4.0", "audio", multimodal=True,
                     description="AMI meetings: do two utterances come from the same participant?"))
register(DatasetSpec("ami_gender", ami_gender, ("train", "val", "test"), _AMI, "cc-by-4.0", "audio", multimodal=True,
                     description="AMI meetings: male or female speaker (from the participant id)"))
register(DatasetSpec("voxconverse_speakers", voxconverse_speakers, ("train", "val", "test"), _VOX, "cc-by-4.0", "audio", multimodal=True,
                     description="VoxConverse: number of speakers in a 30 s window (from diarization)"))
register(DatasetSpec("voxconverse_overlap", voxconverse_overlap, ("train", "val", "test"), _VOX, "cc-by-4.0", "audio", multimodal=True,
                     description="VoxConverse: is there overlapped speech in a 30 s window (from diarization)"))
