"""LRS3 test set (Afouras et al. 2018): TED talk utterances as 96x96 grey mouth-region video plus 16 kHz speech.

Source ``mattymchen/lrs3-test`` (two parquet shards, 1,321 utterances: `video` = (frames, 96, 96) uint8 at 25 fps, `audio` =
16 kHz samples, `label` = the lower-case transcript). The hub copy has a single split, so the splits are cut by a hash of
the utterance (about 8 % test).

    lrs3_transcript   which of 4 transcripts is what the speaker says? The 3 wrong ones are transcripts of other
                      utterances with a similar word count (+-3), so the answer needs the lips and the sound, not the length.
"""

from __future__ import annotations

import numpy as np

from jevtrainer.data import avkit
from jevtrainer.data.avkit_video import fetch_ranged, hash_bucket
from jevtrainer.data.base import DatasetSpec, mcq_record, register, rid

_REPO = "mattymchen/lrs3-test"
_FILES = ["data/train-00000-of-00002-6d235766a16b101a.parquet", "data/train-00001-of-00002-a89b175a3b0aa629.parquet"]
_Q = "Watch the speaker's mouth and listen. Which sentence is being said?"
_INS = "Which option is the transcript of what the speaker says in the clip?"


def _item(video: np.ndarray, audio: np.ndarray, key: str, n_frames: int = 8) -> dict | None:
    from PIL import Image

    out = avkit.media_dir("lrs3", key)
    if not len(video) or len(audio) < 1600:
        return None
    dur = len(audio) / 16000
    idx = np.linspace(0, len(video) - 1, min(n_frames, len(video))).round().astype(int)
    frames = []
    for i, j in enumerate(idx, 1):
        p = out / f"f{i:02d}.jpg"
        Image.fromarray(np.asarray(video[j], dtype="uint8")).convert("RGB").save(p, quality=92)
        frames.append(str(p))
    item = {"type": "video", "frames": frames, "fps": len(frames) / dur, "duration": round(dur, 3)}
    a = avkit.audio_from_array(np.asarray(audio, dtype="float32") / 32768.0, 16000, "lrs3", key, max_s=30)
    if a:
        item["audio"] = a["path"]
    (out / "meta.txt").write_text(str(item["duration"]))
    return item


def lrs3_transcript(split, cap, rng):
    import pyarrow.parquet as pq

    paths = [fetch_ranged(_REPO, f, "lrs3") for f in _FILES]
    meta = []
    for fi, p in enumerate(paths):
        t = pq.read_table(str(p), columns=["idx", "label"]).to_pydict()
        meta += [(fi, i, str(l).strip()) for i, l in zip(t["idx"], t["label"])]
    texts = {(fi, i): l for fi, i, l in meta}
    words = {k: len(l.split()) for k, l in texts.items()}
    mine = [k for k in texts if (hash_bucket("lrs3", *k) < 8) == (split == "test") and words[k] >= 3]
    rng.shuffle(mine)
    mine = set(mine[:cap])
    media: dict[tuple, dict] = {}
    for fi, p in enumerate(paths):
        pf = pq.ParquetFile(str(p))
        for batch in pf.iter_batches(batch_size=8):
            for r in batch.to_pylist():
                k = (fi, r["idx"])
                if k not in mine:
                    continue
                it = avkit_cached(k) or _item(np.array(r["video"], dtype="uint8"), np.array(r["audio"]), f"{fi}_{r['idx']}")
                if it:
                    media[k] = it
    same = [k for k in texts if (hash_bucket("lrs3", *k) < 8) == (split == "test")]
    for k, m in media.items():
        near = [texts[o] for o in same if o != k and abs(words[o] - words[k]) <= 3 and texts[o] != texts[k]]
        if len(near) < 3:
            continue
        options = [texts[k], *rng.sample(near, 3)]
        rng.shuffle(options)
        rec = mcq_record(rid("lrs3", split, *k), {"clip": "<video:1>", "question": _Q}, _INS, options, options.index(texts[k]),
                         area="av", dataset="lrs3", task="transcript")
        if rec:
            rec.media = [m]
            yield rec


def avkit_cached(k):
    from jevtrainer.data.avkit_video import cached_item

    return cached_item("lrs3", f"{k[0]}_{k[1]}")


register(DatasetSpec("lrs3_transcript", lrs3_transcript, ("train", "test"), _REPO, "cc-by-nc-nd-4.0 (LRS3)", "av", multimodal=True,
                     description="LRS3 test utterances: pick the transcript of the speaker (96x96 mouth video + speech, 4 options)"))
