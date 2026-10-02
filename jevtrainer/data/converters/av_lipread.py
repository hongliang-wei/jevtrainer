"""LRS3 (Afouras et al. 2018): TED / TEDx talk utterances, cropped around the speaker's mouth, with their voice.

Source ``mattymchen/lrs3-test`` (the 1,321 utterances of the LRS3 test set, 96x96 grey mouth crops + 16 kHz audio + transcript
in two parquet shards). The hub copy has a single split, so a fixed hash of the utterance index cuts ~8 % off as `test`.

    lrs3_transcript   which of four sentences was spoken? The distractors are transcripts of other utterances with a similar
                      number of words, so the answer has to come from the lips and the voice, not from the length.
"""

from __future__ import annotations

import numpy as np

from jevtrainer.data import avkit
from jevtrainer.data.avkit_video import fetch_ranged, hash_bucket
from jevtrainer.data.base import DatasetSpec, mcq_record, register, rid

_REPO = "mattymchen/lrs3-test"
_FILES = ["data/train-00000-of-00002-6d235766a16b101a.parquet", "data/train-00001-of-00002-a89b175a3b0aa629.parquet"]
_Q = "Which sentence is being spoken in this clip?"
_INS = "Watch the speaker's mouth and listen to the voice. Which option is the spoken sentence?"
_FPS = 25.0


def _media(key: str, vid: np.ndarray, aud: np.ndarray) -> dict | None:
    from PIL import Image

    out = avkit.media_dir("lrs3", key)
    if (out / "f01.jpg").exists() and (out / "a.flac").exists():
        files = sorted(out.glob("f*.jpg"))
        dur = avkit.probe_cache(out)
        return {"type": "video", "frames": [str(f) for f in files], "fps": len(files) / max(dur, 0.1), "duration": dur,
                "audio": str(out / "a.flac")}
    t = len(vid)
    if t < 8:
        return None
    paths = []
    for i, j in enumerate(np.linspace(0, t - 1, 8).round().astype(int)):
        p = out / f"f{i + 1:02d}.jpg"
        Image.fromarray(vid[j].astype("uint8")).convert("RGB").resize((192, 192), Image.BICUBIC).save(p, quality=92)
        paths.append(str(p))
    dur = t / _FPS
    a = avkit.audio_from_array(aud.astype("float32") / 32768.0, 16000, "lrs3", key)
    if not a:
        return None
    (out / "meta.txt").write_text(str(dur))
    return {"type": "video", "frames": paths, "fps": 8 / dur, "duration": round(dur, 3), "audio": a["path"]}


def _utterances() -> list[tuple[str, str, str, int, int]]:
    """(local path, utterance key, transcript, row group, row in group) of every utterance, reading only the label column."""
    import pyarrow.parquet as pq

    rows = []
    for fn in _FILES:
        path = fetch_ranged(_REPO, fn, "lrs3_src")
        pf = pq.ParquetFile(str(path))
        for g in range(pf.num_row_groups):
            for i, (idx, lab) in enumerate(zip(*[pf.read_row_group(g, columns=[c]).column(0).to_pylist() for c in ("idx", "label")])):
                rows.append((str(path), f"{fn}:{idx}", str(lab).strip().lower(), g, i))
    return rows


def lrs3_transcript(split, cap, rng):
    import pyarrow.parquet as pq

    every = _utterances()
    utt = [u for u in every if (hash_bucket("lrs3", u[1]) < 8) == (split == "test") and len(u[2].split()) >= 2]
    rng.shuffle(utt)
    utt = utt[:cap]
    texts = [u[2] for u in every]
    by_group: dict[tuple[str, int], list[tuple]] = {}
    for u in utt:
        by_group.setdefault((u[0], u[3]), []).append(u)
    done = {}
    for (path, g), items in by_group.items():
        tbl = pq.ParquetFile(path).read_row_group(g, columns=["video", "audio"])
        for u in items:
            i = u[4]
            v = np.asarray(tbl.column("video")[i].values.flatten().flatten().to_numpy(zero_copy_only=False))
            a = np.asarray(tbl.column("audio")[i].values.to_numpy(zero_copy_only=False))
            frames = v.reshape(-1, 96, 96)
            m = _media(rid("lrs3", u[1]), frames, a)
            if m:
                done[u[1]] = m
        del tbl
    for u in utt:
        m = done.get(u[1])
        if not m:
            continue
        n = len(u[2].split())
        close = [t for t in texts if t != u[2] and abs(len(t.split()) - n) <= 3]
        if len(close) < 3:
            continue
        options = [u[2], *rng.sample(close, 3)]
        order = list(range(4))
        rng.shuffle(order)
        rec = mcq_record(rid("lrs3_transcript", split, u[1]), {"clip": "<video:1>", "question": _Q}, _INS, [options[i] for i in order],
                         order.index(0), area="av", dataset="lrs3", task="lipread_transcript")
        if rec:
            rec.media = [m]
            yield rec


register(DatasetSpec("lrs3_transcript", lrs3_transcript, ("train", "test"), _REPO, "cc-by-nc-nd-4.0 (LRS3 research licence)", "av",
                     multimodal=True, description="LRS3: which of four sentences does the speaker say (mouth crop video + voice)"))
