"""Video quality assessment on real user-generated videos (score task).

LSVQ (Ying et al. 2021): ~39k social-media videos with crowd-sourced mean opinion scores (MOS, 0-100).
"""

from __future__ import annotations

import os
from pathlib import Path

from jevtrainer.data import avkit, vidkit
from jevtrainer.data.base import DatasetSpec, register, rid
from jevtrainer.schema import Question, Record, Target

_LSVQ = "teowu/LSVQ-videos"
_LABELS = "https://raw.githubusercontent.com/VQAssessment/FAST-VQA-and-FasterVQA/dev/examplar_data_labels/"
LEVELS = [
    "bad: severe blur, noise, shake or compression artefacts throughout (MOS below 20)",
    "poor: clearly annoying quality problems (MOS 20 to 40)",
    "fair: visible but tolerable quality problems (MOS 40 to 60)",
    "good: only slight imperfections (MOS 60 to 80)",
    "excellent: sharp, clean and stable (MOS 80 to 100)",
]


def _level(mos: float) -> int:
    return min(4, max(0, int(mos // 20)))


def lsvq(split, cap, rng):
    """LSVQ: how good is the picture quality of the video? Five levels of the mean opinion score (0-100, 20 points per
    level). Official train labels -> train, LSVQ_test labels -> test. Videos come from the community tar.gz copy; only
    the first JT_LSVQ_TARS (default 6 for train, 2 for test) archives (~1 GB, ~800 videos each) are read, and the sample
    is balanced over the five levels."""
    name = "train_labels.txt" if split == "train" else "LSVQ/labels_test.txt"
    labels = {}
    for line in vidkit.http_text(_LABELS + name, "lsvq", name.replace("/", "_")).splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 4 and p[0].endswith(".mp4"):
            try:
                labels[p[0]] = float(p[3])
            except ValueError:
                pass
    tars = sorted(f for f in vidkit.listing(_LSVQ) if f.endswith(".tar.gz"))
    n_tars = int(os.environ.get("JT_LSVQ_TARS", "6" if split == "train" else "2"))
    lo = 0 if split == "train" else 6  # test videos are looked for in other archives than the train ones
    chosen = [t for t in tars if t.startswith(("yfcc", "ia-"))][lo:lo + n_tars] or tars[lo:lo + n_tars]
    by_level: dict[int, int] = {}
    quota = -(-cap // 5)
    media: dict[str, dict] = {}
    for tname in chosen:
        src = avkit.fetch_file(_LSVQ, tname, "lsvq")
        if src is None:
            continue

        def keep(member):
            key = "/".join(member.split("/")[-2:])
            mos = labels.get(key)
            if mos is None or by_level.get(_level(mos), 0) >= quota:
                return False
            by_level[_level(mos)] = by_level.get(_level(mos), 0) + 1
            return True

        stream = (("/".join(n.split("/")[-2:]), n, b) for n, b in vidkit.tar_members([src], keep))
        stream = ((k.replace("/", "__"), n, b) for k, n, b in stream)
        media.update(vidkit.convert_stream("lsvq", stream, frames=8, max_side=512, audio_s=12))
    ids = sorted(media)
    rng.shuffle(ids)
    for k in ids[:cap]:
        mos = labels["/".join(k.split("__"))]
        yield Record(rid("lsvq", split, k), {"clip": "<video:1>", "question": "Rate the technical picture quality of this video (not its content)."},
                     {"quality": Question("score", "How good is the technical quality of this video? Judge sharpness, noise, exposure, "
                                          "stability and compression artefacts.", list(LEVELS))},
                     {"quality": Target(str(_level(mos)))}, media=[media[k]], meta={"area": "video", "dataset": "lsvq", "mos": round(mos, 2)})


register(DatasetSpec("lsvq", lsvq, ("train", "test"), _LSVQ, "other (LIVE research licence)", "video", multimodal=True,
                     description="LSVQ: five-level quality score (from MOS) of user-generated videos"))
