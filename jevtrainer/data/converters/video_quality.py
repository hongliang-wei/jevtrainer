"""Video quality assessment on real user-generated videos (score task).

LSVQ (Ying et al. 2021): ~39k social-media videos with crowd-sourced mean opinion scores (MOS, 0-100).
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from jevtrainer.data import avkit, vidkit
from jevtrainer.data.avkit_video import cached_item, fetch_ranged
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


_LOCK = threading.Lock()


def _labels(split: str) -> dict[str, float]:
    name = "train_labels.txt" if split == "train" else "LSVQ/labels_test.txt"
    labels = {}
    for line in vidkit.http_text(_LABELS + name, "lsvq", name.replace("/", "_")).splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 4 and p[0].endswith(".mp4"):
            try:
                labels[p[0]] = float(p[3])
            except ValueError:
                pass
    return labels


def _manifests() -> dict[str, tuple[str, float]]:
    """{video key: (split, mos)} of the clips converted so far (one manifest per archive that was read completely)."""
    out = {}
    for f in (avkit.raw_dir("lsvq") / "manifests").glob("*.json"):
        out.update({k: tuple(v) for k, v in json.loads(f.read_text()).items()})
    return out


def _pass(cap: int) -> None:
    """Read the next archives: ONE pass converts the train and the test videos of an archive, then the archive is deleted.
    Sample quota per (split, level) is ceil(cap / 5); stops early when every quota is full."""
    labels = {"train": _labels("train"), "test": _labels("test")}
    mdir = avkit.raw_dir("lsvq") / "manifests"
    mdir.mkdir(exist_ok=True)
    quota = -(-cap // 5)
    counts = {(sp, lv): 0 for sp in labels for lv in range(5)}
    for sp, mos in _manifests().values():
        counts[(sp, _level(mos))] += 1
    tars = sorted(f for f in vidkit.listing(_LSVQ) if f.endswith(".tar.gz"))
    ia = [t for t in tars if t.startswith("ia-")]
    yf = [t for t in tars if t.startswith("yfcc-")]
    order = [t for pair in zip(ia, yf) for t in pair] + ia[len(yf):] + yf[len(ia):]  # interleaved: both sources early
    for tname in order[:int(os.environ.get("JT_LSVQ_TARS", "20"))]:
        if all(c >= quota for c in counts.values()):
            return
        mf = mdir / (tname + ".json")
        if mf.exists():
            continue
        src = fetch_ranged(_LSVQ, tname, "lsvq", conns=8)
        if src is None:
            continue
        picked: dict[str, tuple[str, float]] = {}

        def keep(member):
            key = "/".join(member.split("/")[-2:])
            for sp in ("train", "test"):
                mos = labels[sp].get(key)
                if mos is not None:
                    c = (sp, _level(mos))
                    if counts[c] >= quota:
                        return False
                    counts[c] += 1
                    picked[key] = (sp, mos)
                    return True
            return False

        stream = (("__".join(n.split("/")[-2:]), n, b) for n, b in vidkit.tar_members([src], keep))
        media = vidkit.convert_stream("lsvq", stream, frames=8, max_side=512, audio_s=12)
        mf.write_text(json.dumps({k: v for k, v in picked.items() if k.replace("/", "__") in media}))
        Path(src).unlink(missing_ok=True)


def lsvq(split, cap, rng):
    """LSVQ: how good is the picture quality of the video? Five levels of the mean opinion score (0-100, 20 points per
    level). Official train labels -> train, LSVQ_test labels -> test. Videos come from the community tar.gz copy (ia-* and
    yfcc-* archives of ~1-2 GB, ~800 videos each, train and test videos mixed): the first JT_LSVQ_TARS (default 20, interleaved
    over both sources, downloaded with 8 connections) archives are read in one pass each and deleted; the sample is balanced
    over the five levels."""
    with _LOCK:
        _pass(cap)
    rows = [(k, mos) for k, (sp, mos) in _manifests().items() if sp == split]
    rows = avkit.balanced(rows, lambda r: _level(r[1]), cap, rng)
    for k, mos in rows:
        key = k.replace("/", "__")
        m = cached_item("lsvq", key)
        if not m:
            continue
        yield Record(rid("lsvq", split, key), {"clip": "<video:1>", "question": "Rate the technical picture quality of this video (not its content)."},
                     {"quality": Question("score", "How good is the technical quality of this video? Judge sharpness, noise, exposure, "
                                          "stability and compression artefacts.", list(LEVELS))},
                     {"quality": Target(str(_level(mos)))}, media=[m], meta={"area": "video", "dataset": "lsvq", "mos": round(mos, 2)})


register(DatasetSpec("lsvq", lsvq, ("train", "test"), _LSVQ, "other (LIVE research licence)", "video", multimodal=True, version="2",
                     description="LSVQ: five-level quality score (from MOS) of user-generated videos"))
