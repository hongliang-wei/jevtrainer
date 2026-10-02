"""ActivityNet-QA (Yu et al. 2019): questions about long (~2 min) real-world videos, with their sound.

Questions: `lmms-eval/ActivityNetQA` (the official test questions, 8,000 over 800 videos; nine kinds: motion, spatial /
temporal relation, yes-no, colour, object, location, number, other). Videos: the zip chunks of the same repo (`all_test/v_*.mp4`,
~25 MB each); every chunk's directory is read with range requests and only the wanted videos are fetched, at most
JT_ANQA_VIDEOS (default 160). Only the official test questions are public in this copy, so they are split by video
(hash of the video name): ~10 % test, ~10 % val, the rest train.

Questions whose answer is yes / no become yes-no decisions; the other ones become 4-option choices whose distractors are
answers given to other questions of the same kind (colours for colour questions, numbers for number questions, ...).
Long videos are cut to their first 120 s for 16 frames.
"""

from __future__ import annotations

import os
import re
import threading

from jevtrainer.data import avkit, vidkit
from jevtrainer.data.avkit_video import cached_item, convert_remote_zip_videos, hash_bucket, remote_zip_index
from jevtrainer.data.base import DatasetSpec, mcq_record, noul_record, register, rid

_REPO = "lmms-eval/ActivityNetQA"
_LOCK = threading.Lock()
_KIND = {0: "motion", 1: "spatial", 2: "temporal", 3: "yes_no", 4: "colour", 5: "object", 6: "location", 7: "number", 8: "other"}


def _split(video: str) -> str:
    b = hash_bucket("anqa", video)
    return "test" if b < 10 else "val" if b < 20 else "train"


def _locate() -> dict[str, tuple[str, str]]:
    """{video name: (zip file, member)} over all chunks (directories cached on disk)."""
    import json

    p = avkit.raw_dir("activitynet_qa") / "index.json"
    if p.exists():
        return {k: tuple(v) for k, v in json.loads(p.read_text()).items()}
    out = {}
    for z in sorted(f for f in vidkit.listing(_REPO) if f.endswith(".zip")):
        _, idx = vidkit.retry(lambda: remote_zip_index(_REPO, z), tries=6, wait=10)
        for m in idx:
            name = re.sub(r"\.[a-z0-9]+$", "", m.split("/")[-1])
            out[name] = (z, m)
    p.write_text(json.dumps(out))
    return out


def activitynet_qa(split, cap, rng):
    df = vidkit.read_parquet(_REPO, "data/test-00000-of-00001.parquet")
    where = _locate()
    rows = [r for r in df.to_dict("records") if r["video_name"] in where or f"v_{r['video_name']}" in where]
    for r in rows:
        r["vid"] = r["video_name"] if r["video_name"] in where else f"v_{r['video_name']}"
    vids = sorted({r["vid"] for r in rows})
    # the video budget is fixed per dataset (not per split) so that every split sees the same videos
    vrng = __import__("random").Random("anqa-videos")
    vrng.shuffle(vids)
    keep = set(vids[:int(os.environ.get("JT_ANQA_VIDEOS", "160"))])
    pools: dict[int, list[str]] = {}
    for r in df.to_dict("records"):
        if str(r["answer"]).strip().lower() not in ("yes", "no"):
            pools.setdefault(int(r["type"]), []).append(str(r["answer"]).strip().lower())
    pools = {k: sorted(set(v)) for k, v in pools.items()}
    rows = [r for r in rows if r["vid"] in keep and _split(r["vid"]) == split]
    rows = avkit.balanced(rows, lambda r: int(r["type"]), cap, rng)
    todo = {where[v][1]: v for v in {r["vid"] for r in rows} if not cached_item("activitynet_qa", v)}
    if todo:
        by_zip: dict[str, dict] = {}
        for member, v in todo.items():
            by_zip.setdefault(where[v][0], {})[member] = (v, {"end": 120.0})
        with _LOCK:
            for z, wanted in by_zip.items():
                convert_remote_zip_videos(_REPO, z, "activitynet_qa", wanted, workers=4, frames=16, max_side=448, audio_s=120)
    for r in rows:
        m = cached_item("activitynet_qa", r["vid"])
        if not m:
            continue
        ans = str(r["answer"]).strip().lower()
        q = str(r["question"]).strip()
        q = q[:1].upper() + q[1:] + ("" if q.endswith("?") else "?")
        kind = _KIND.get(int(r["type"]), "other")
        state = {"clip": "<video:1>", "question": q}
        id_ = rid("activitynet_qa", split, r["question_id"])
        if ans in ("yes", "no"):
            rec = noul_record(id_, state, "Watch and listen to the video. Is the answer to the question yes?", ans == "yes",
                              "yes: the statement in the question is true", "no: the statement in the question is false",
                              area="video", dataset="activitynet_qa", kind=kind)
        else:
            others = [a for a in pools.get(int(r["type"]), []) if a != ans]
            if len(others) < 3:
                continue
            opts = [ans, *rng.sample(others, 3)]
            order = list(range(4))
            rng.shuffle(order)
            rec = mcq_record(id_, state, "Watch and listen to the video. Which option correctly answers the question?", [opts[i] for i in order],
                             order.index(0), area="video", dataset="activitynet_qa", kind=kind)
        if rec:
            rec.media = [m]
            yield rec


register(DatasetSpec("activitynet_qa", activitynet_qa, ("train", "val", "test"), _REPO, "other (research)", "video", multimodal=True,
                     description="ActivityNet-QA on long videos: yes/no and 4-option questions (subset of videos, split by video)"))
