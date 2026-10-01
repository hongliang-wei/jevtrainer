"""Screen recordings of GUI use (desktop software, websites, Android, iOS, XR, multi-app workflows).

* ``gui_world``       - GUI-World (Chen et al. 2024): the dataset's own multiple-choice question about a recording
                        (train = Annotation/train, test = Annotation/benchmark).
* ``gui_world_goal``  - which task goal describes the recording; distractors are the goals of other recordings of the
                        same environment (derived from the ``goal`` annotation).
* ``gui_world_env``   - what kind of GUI environment is recorded; the label is the environment folder of the video.

The three share one set of converted clips (same videos per split). Clips are capped by size (screen recordings of up to
a few minutes) and by a per-split count, class-balanced over the six environments, because the hub mirror is slow.
Frames are kept at 640 px so that on-screen text stays legible; screen recordings carry no sound.
"""

from __future__ import annotations

import json
import os
import random
import re

from jevtrainer.data import bizkit, vidkit
from jevtrainer.data.avkit_video import balanced_take
from jevtrainer.data.base import DatasetSpec, choice_record, mcq_record, register, rid

_GW = "ONE-Lab/GUI-World"
_CATS = ["android", "IOS", "XR", "multi", "software", "website"]
_MAX_BYTES = int(float(os.environ.get("JEVTRAINER_GW_MAX_MB", "8")) * 1e6)
_N = {"train": int(os.environ.get("JEVTRAINER_GW_TRAIN", "2000")), "test": int(os.environ.get("JEVTRAINER_GW_TEST", "500"))}
_PART = {"train": "train", "test": "benchmark"}
_VID = dict(frames=8, max_side=640, keep_audio=False)
_DEADLINE: dict[str, float] = {}  # one download budget per split and process (the three datasets share the clips)

_ENV = {
    "android": "An Android phone app (touch interface, system navigation bar, mobile layouts)",
    "IOS": "An iPhone or iPad app (iOS interface, gestures, mobile layouts)",
    "XR": "An extended-reality (VR/AR headset) interface with floating windows and spatial controls",
    "multi": "A workflow that switches between several applications or windows to reach one goal",
    "software": "Desktop software on Windows, macOS or Linux (menus, toolbars, dialogs, side panels)",
    "website": "A web page used inside a browser (page content, links, forms, tabs)",
}
_LETTER = re.compile(r"\[\[\s*([A-H])\s*\]\]")
_PREFIX = re.compile(r"^\s*[A-H][\.\):]\s*")
_Q_MCQ = "Which option correctly answers the question about this screen recording?"
_Q_GOAL = "Which task is being carried out in this screen recording?"
_Q_ENV = "What kind of interface is shown in this screen recording?"


def _sizes() -> dict[str, int]:
    from huggingface_hub import HfApi

    info = vidkit.retry(lambda: HfApi().dataset_info(_GW, files_metadata=True))
    return {s.rfilename: s.size or 0 for s in info.siblings if s.rfilename.lower().endswith((".mp4", ".mov"))}


def _select(split: str, cap: int):
    """(rows with a converted clip, {key: media item}) - identical for the three datasets of one split."""
    sizes = _sizes()
    rows = []
    for cat in _CATS:
        with open(vidkit.get_file(_GW, f"Annotation/{_PART[split]}/{cat}.jsonl"), encoding="utf8") as f:
            for line in f:
                r = json.loads(line)
                if sizes.get(r.get("video_path"), 1e12) <= _MAX_BYTES:
                    r["cat"] = cat
                    rows.append(r)
    rows = balanced_take(rows, lambda r: r["cat"], min(cap, _N[split]), random.Random(f"gui_world:{split}"))
    keys = {r["video_path"]: r["video_path"].replace("/", "_").rsplit(".", 1)[0] for r in rows}
    files = {keys[r["video_path"]]: r["video_path"] for r in rows}
    items = bizkit.convert_hub_timed(_GW, "gui_world", files, _DEADLINE.setdefault(split, bizkit.deadline()), **_VID)
    out = [(r, keys[r["video_path"]]) for r in rows if keys[r["video_path"]] in items]
    return out, items


def _state():
    return {"recording": "<video:1>"}


def _mcq(r) -> tuple[list[str], int] | None:
    m = r.get("MCQA") or {}
    opts = [_PREFIX.sub("", o).strip() for o in m.get("Options") or []]
    g = _LETTER.search(str(m.get("Correct Answer", "")))
    if not g or not 2 <= len(opts) <= 8:
        return None
    gold = ord(g.group(1)) - 65
    return (opts, gold) if gold < len(opts) else None


def gui_world(split, cap, rng):
    rows, items = _select(split, cap)
    for r, key in rows:
        parsed = _mcq(r)
        if not parsed:
            continue
        st = {**_state(), "question": r["MCQA"]["Question"]}
        rec = mcq_record(rid("gui_world", split, key), st, _Q_MCQ, parsed[0], parsed[1], area="video",
                         environment=r["cat"], system=r.get("system"), app=",".join(r.get("app") or []))
        if rec:
            rec.media = [items[key]]
            yield rec


def gui_world_goal(split, cap, rng):
    rows, items = _select(split, cap)
    goals: dict[str, list[str]] = {}
    rows = [(r, k) for r, k in rows if r.get("goal")]
    for r, _ in rows:
        goals.setdefault(r["cat"], []).append(str(r["goal"]).strip().rstrip("."))
    for r, key in rows:
        gold = str(r["goal"]).strip().rstrip(".")
        pool = sorted(set(goals[r["cat"]]) - {gold})
        if len(pool) < 3:
            continue
        options = [gold, *rng.sample(pool, 3)]
        order = list(range(4))
        rng.shuffle(order)
        rec = mcq_record(rid("gui_world_goal", split, key), _state(), _Q_GOAL, [options[i] for i in order], order.index(0),
                         area="video", environment=r["cat"], app=",".join(r.get("app") or []))
        if rec:
            rec.media = [items[key]]
            yield rec


def gui_world_env(split, cap, rng):
    rows, items = _select(split, cap)
    for r, key in rows:
        rec = choice_record(rid("gui_world_env", split, key), _state(), _Q_ENV, dict(_ENV), r["cat"], name="environment",
                            area="video", app=",".join(r.get("app") or []))
        rec.media = [items[key]]
        yield rec


_SRC = "ONE-Lab/GUI-World"
register(DatasetSpec("gui_world", gui_world, ("train", "test"), _SRC, "other (research)", "video", multimodal=True,
                     description="GUI-World: multiple-choice questions about screen recordings of software, web, mobile, XR; test = official benchmark"))
register(DatasetSpec("gui_world_goal", gui_world_goal, ("train", "test"), _SRC, "other (research)", "video", multimodal=True,
                     description="GUI-World: which task goal is being carried out (distractors: goals of other recordings in the same environment)"))
register(DatasetSpec("gui_world_env", gui_world_env, ("train", "test"), _SRC, "other (research)", "video", multimodal=True,
                     description="GUI-World: kind of GUI environment (Android, iOS, XR, desktop, web, multi-app)"))
