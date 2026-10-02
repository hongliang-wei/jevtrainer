"""Violence detection in film / web videos with their sound track: XD-Violence (jherng/xd-violence, mit).

The repo stores 4,750 untrimmed videos (3,950 train / 800 test) whose file names carry the video-level labels
(`..._label_A.mp4` normal, `B1` fighting, `B2` shooting, `B4` riot, `B5` abuse, `B6` car accident, `G` explosion;
several codes joined by `-` when more than one kind occurs). Only the smaller files (<= 6 MB) are used, to keep the
download manageable; clips longer than 30 s are cut to their central 30 s (frames and sound come from the same window).
`test_videos/` is the test split; the official train videos are split into train (90 %) and val (10 %) by hash of the file name.

    xd_violence        yes/no: does this video contain violence? (yes = any code except A; class-balanced over the codes)
    xd_violence_type   7-way choice: normal / fighting / shooting / riot / abuse / car accident / explosion
                       (only videos with exactly one code, so the answer is unambiguous)
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from jevtrainer.data import avkit
from jevtrainer.data.avkit_video import balanced_take, cached_item, hash_bucket
from jevtrainer.data.base import DatasetSpec, choice_record, noul_record, register, rid

_REPO = "jherng/xd-violence"
_WORKERS = int(os.environ.get("JEVTRAINER_AV_WORKERS", "8"))
_MAX_BYTES = int(float(os.environ.get("JEVTRAINER_XD_MAX_MB", "6")) * 1e6)

CODES = {"A": "normal", "B1": "fighting", "B2": "shooting", "B4": "riot", "B5": "abuse", "B6": "car_accident", "G": "explosion"}
TYPES = {
    "normal": "normal: nothing violent happens (ordinary scenes, dialogue, music, everyday action)",
    "fighting": "fighting: people punching, kicking or struggling physically",
    "shooting": "shooting: guns fired, gunfights or people being shot",
    "riot": "riot: a crowd rioting, clashing with police, looting or violent protest",
    "abuse": "abuse: a person (often a child or a weaker person) being hit or mistreated",
    "car_accident": "car accident: vehicles crashing, colliding or skidding out of control",
    "explosion": "explosion: blasts, bombs or burning wreckage with loud detonation sound",
}


def _listing() -> list[dict]:
    """[{path, size, codes}] of the videos of the repo (cached in the media cache)."""
    cache = avkit.raw_dir("xd_violence") / "listing.json"
    if cache.exists():
        return json.loads(cache.read_text())
    from huggingface_hub import HfApi

    rows = []
    for s in HfApi().dataset_info(_REPO, files_metadata=True).siblings:
        m = re.search(r"label_([A-Z0-9-]+)\.mp4$", s.rfilename)
        if s.rfilename.startswith("data/video/") and m and s.size:
            rows.append({"path": s.rfilename, "size": s.size, "codes": [c for c in m.group(1).split("-") if c in CODES]})
    cache.write_text(json.dumps(rows))
    return rows


def _split_rows(split: str) -> list[dict]:
    out = []
    for r in _listing():
        if r["size"] > _MAX_BYTES or not r["codes"]:
            continue
        test = "/test_videos/" in r["path"]
        val = not test and hash_bucket("xd", r["path"]) < 10
        if (split == "test") == test and (split != "val" or val) and (split != "train" or not val):
            out.append({**r, "key": rid("xd", r["path"])[:12]})
    return out


def _convert(rows: list[dict]) -> None:
    def one(r):
        if cached_item("xd_violence", r["key"]):
            return
        src = avkit.fetch_file(_REPO, r["path"], "xd_violence")
        if src is None:
            return
        try:
            dur = avkit.probe(src)["duration"]
            kw = {}
            if dur > 30:
                kw = {"start": (dur - 30) / 2, "end": (dur + 30) / 2}
            item = avkit.video_item(src, "xd_violence", r["key"], frames=8, **kw)
            if item:
                (avkit.media_dir("xd_violence", r["key"]) / "meta.txt").write_text(str(item["duration"]))
        finally:
            src.unlink(missing_ok=True)

    with ThreadPoolExecutor(_WORKERS) as ex:
        list(ex.map(one, rows))


def _pick(split: str, cap: int, rng, single: bool) -> list[dict]:
    rows = [r for r in _split_rows(split) if len(r["codes"]) == 1 or not single]
    rng.shuffle(rows)
    per = {"train": 100, "val": 12, "test": 25}[split]
    group = lambda r: r["codes"][0] if len(r["codes"]) == 1 else "multi"
    rows = balanced_take(rows, group, min(cap, per * 8), rng)  # ~100 train videos per class: the mirror is slow
    return rows


def xd_violence(split, cap, rng):
    rows = _pick(split, cap, rng, single=False)
    _convert(rows)
    for r in rows:
        media = cached_item("xd_violence", r["key"])
        if not media:
            continue
        violent = r["codes"] != ["A"]
        rec = noul_record(rid("xd_violence", split, r["path"]), {"clip": "<video:1>", "question": "Does this video contain violence?"},
                          "Watch and listen to the video. Does it show or sound like real or staged violence (fights, shootings, riots, abuse, "
                          "car crashes or explosions)?", violent,
                          true="violence: at least one fight, shooting, riot, abuse, accident or explosion happens",
                          false="no violence: ordinary scenes without any violent event", area="safety",
                          codes="-".join(r["codes"]))
        rec.media = [media]
        yield rec


def xd_violence_type(split, cap, rng):
    rows = _pick(split, cap, rng, single=True)
    _convert(rows)
    for r in rows:
        media = cached_item("xd_violence", r["key"])
        if not media:
            continue
        rec = choice_record(rid("xd_violence_type", split, r["path"]), {"clip": "<video:1>", "question": "What kind of event does this video show?"},
                            "Watch and listen to the video. Which kind of event does it show?", dict(TYPES), CODES[r["codes"][0]],
                            area="safety", codes=r["codes"][0])
        rec.media = [media]
        yield rec


register(DatasetSpec("xd_violence", xd_violence, ("train", "val", "test"), _REPO, "mit", "safety", multimodal=True,
                     description="XD-Violence: does the video (with sound) contain violence? (small-file subset, class-balanced)"))
register(DatasetSpec("xd_violence_type", xd_violence_type, ("train", "val", "test"), _REPO, "mit", "safety", multimodal=True,
                     description="XD-Violence: normal / fighting / shooting / riot / abuse / car accident / explosion"))
