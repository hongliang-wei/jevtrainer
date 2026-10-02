"""Questions about text that appears in videos, and other business video sets that do not fit the other modules.

* ``m4_vitevqa`` - M4-ViteVQA (Zhao et al. 2022): questions about scene text in short videos. The hub copy keeps 10 frames per
  video (every 16th frame, `img_dir/<video>/<n>.jpg`); five of them (stride 32) are used. The dataset has open answers, so
  the record is a 4-way choice: the most frequent annotated answer against answers to *other* questions (answers of the
  same video's other questions first, then answers of other videos).
  train = task-1 split-1 train, val = its val, test = its test (never mixed into training).
"""

from __future__ import annotations

import io
import json
import os
import random
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from jevtrainer.data import avkit, bizkit, vidkit
from jevtrainer.data.base import DatasetSpec, mcq_record, register, rid

_VITE = "yankie123/tea-m4vitevqa"
_VITE_FILES = {"train": "t1s1train", "val": "t1s1val", "test": "t1s1test"}
_VITE_N = {"train": int(os.environ.get("JEVTRAINER_VITE_TRAIN", "600")), "val": 150, "test": int(os.environ.get("JEVTRAINER_VITE_TEST", "300"))}
_VITE_STRIDE = 2  # keep every 2nd of the 10 stored frames
_VITE_STEP_S = 32 / 30.0  # frames are 16 apart; stride 2 -> 32 frames, taken as 30 fps footage
_Q_VITE = "Which option correctly answers the question about the text or events in this video?"
_zip: dict = {}


def _vite_zip():
    if "z" not in _zip:
        _zip["z"] = bizkit.RemoteZip(_VITE, "img_dir.zip")
    return _zip["z"]


def _vite_video(video_id: str) -> dict | None:
    """Download the kept frames of one video and store them like a converted clip."""
    from PIL import Image

    cached = vidkit.cached_item("m4_vitevqa", video_id)
    if cached:
        return cached
    z = _vite_zip()
    names = sorted((n for n in z.index if n.startswith(f"img_dir/{video_id}/")), key=lambda n: int(n.rsplit("/", 1)[1].split(".")[0]))
    names = names[::_VITE_STRIDE]
    if len(names) < 2:
        return None
    out = avkit.media_dir("m4_vitevqa", video_id)
    files = []
    for i, n in enumerate(names, 1):
        im = Image.open(io.BytesIO(z.read(n))).convert("RGB")
        s = 640 / max(im.size)
        if s < 1:
            im = im.resize((round(im.width * s), round(im.height * s)))
        p = out / f"f{i:02d}.jpg"
        im.save(p, quality=88)
        files.append(str(p))
    dur = round(len(files) * _VITE_STEP_S, 3)
    (out / "meta.txt").write_text(str(dur))
    return {"type": "video", "frames": files, "fps": len(files) / dur, "duration": dur}


def m4_vitevqa(split, cap, rng):
    from huggingface_hub import hf_hub_download

    ann = zipfile.ZipFile(vidkit.retry(lambda: hf_hub_download(_VITE, "Annotations.zip", repo_type="dataset")))
    data = json.load(ann.open(f"Annotations/ViteVQA_0.0.2_{_VITE_FILES[split]}.json"))["data"]
    by_video: dict[str, list] = {}
    for r in data:
        if r.get("answers") and r.get("question"):
            by_video.setdefault(r["video_id"], []).append(r)
    pool = sorted({Counter(a.strip() for a in r["answers"]).most_common(1)[0][0] for rs in by_video.values() for r in rs})
    vids = sorted(by_video)
    rng.shuffle(vids)
    vids = vids[: min(cap, _VITE_N[split])]
    stop = bizkit.deadline()
    got: dict[str, dict] = {}

    def one(v):
        if time.time() > stop:
            return v, None
        try:
            return v, _vite_video(v)
        except Exception as e:  # noqa: BLE001
            print(f"[m4_vitevqa] {v}: {type(e).__name__}: {str(e)[:80]}", flush=True)
            return v, None

    with ThreadPoolExecutor(int(os.environ.get("JEVTRAINER_VITE_THREADS", "4"))) as ex:
        for v, item in ex.map(one, vids):
            if item:
                got[v] = item
    for v in vids:
        if v not in got:
            continue
        rs = by_video[v]
        r = rng.choice(rs)
        gold = Counter(a.strip() for a in r["answers"]).most_common(1)[0][0]
        same = sorted({Counter(a.strip() for a in x["answers"]).most_common(1)[0][0] for x in rs if x is not r} - {gold})
        near = same[:3]
        gw = len(gold.split())
        cand = [a for a in rng.sample(pool, min(len(pool), 300)) if a.lower() != gold.lower() and a not in near]
        cand.sort(key=lambda a: abs(len(a.split()) - gw) + 2 * (a.replace(",", "").replace(".", "").isdigit() != gold.replace(",", "").replace(".", "").isdigit()))
        options = [gold, *near, *cand][:4]
        if len(options) < 4:
            continue
        order = list(range(4))
        rng.shuffle(order)
        rec = mcq_record(rid("m4_vitevqa", split, r["question_id"]), {"clip": "<video:1>", "question": r["question"]}, _Q_VITE,
                         [options[i] for i in order], order.index(0), area="video", evidence=r.get("source1"), modality=r.get("source2"))
        if rec:
            rec.media = [got[v]]
            yield rec


register(DatasetSpec("m4_vitevqa", m4_vitevqa, ("train", "val", "test"), _VITE, "research use (M4-ViteVQA)", "video", multimodal=True,
                     description="M4-ViteVQA: questions about scene text in videos (5 frames per video); distractors are answers to other questions"))
