"""TGIF-QA (Jang et al. 2017): questions about short animated GIFs (here as mp4, no sound).

Questions: `allenai/Molmo2-TGIF` (the official TGIF-QA csv files); videos: `Xiaodong/TGIF_Zero_Shot_QA_` (the 9,575 test
GIFs as mp4, read member by member with range requests). Only the official *test* questions have their videos on the hub
in a usable size (the 130 GB train archive is not used), so all of them are re-split by video: a fixed 10 % of the GIFs
(hash of the gif name) -> test, 10 % -> val, the rest -> train. Question types:

    action       what does X do N times?          5 options (official)
    transition   what does X do before / after?   5 options (official)
    frameqa      object / number / colour / place 4 options: the answer + 3 answers given to other questions of the same kind
    count        how many times ...?              0 .. 10 (answers above 10 are dropped)
"""

from __future__ import annotations

import threading

from jevtrainer.data import avkit, vidkit
from jevtrainer.data.avkit_video import cached_item, convert_remote_zip_videos, hash_bucket
from jevtrainer.data.base import DatasetSpec, mcq_record, register, rid

_Q = "allenai/Molmo2-TGIF"
_V = "Xiaodong/TGIF_Zero_Shot_QA_"
_LOCK = threading.Lock()
_FRAME_KIND = {0: "object", 1: "number", 2: "colour", 3: "place"}


def _split(gif: str) -> str:
    b = hash_bucket("tgifqa", gif)
    return "test" if b < 10 else "val" if b < 20 else "train"


def _rows(rng):
    rows = []
    for kind, fn in (("action", "Test_action_question.csv"), ("transition", "Test_transition_question.csv")):
        for r in vidkit.read_csv(_Q, fn, sep=None, engine="python").to_dict("records"):
            opts = [str(r[f"a{i}"]) for i in range(1, 6)]
            rows.append({"kind": kind, "gif": r["gif_name"], "q": str(r["question"]).strip(), "options": opts, "gold": int(r["answer"]),
                         "vid": str(r["vid_id"])})
    fq = vidkit.read_csv(_Q, "Test_frameqa_question.csv", sep=None, engine="python").to_dict("records")
    pools: dict[int, list[str]] = {}
    for r in fq:
        pools.setdefault(int(r["type"]), []).append(str(r["answer"]).strip())
    pools = {k: sorted(set(v)) for k, v in pools.items()}
    for r in fq:
        t, gold = int(r["type"]), str(r["answer"]).strip()
        rows.append({"kind": "frameqa", "gif": r["gif_name"], "q": str(r["question"]).strip(), "answer": gold, "pool": pools[t], "ftype": _FRAME_KIND.get(t, "?"),
                     "vid": str(r["vid_id"])})
    for r in vidkit.read_csv(_Q, "Test_count_question.csv", sep=None, engine="python").to_dict("records"):
        if 0 <= int(r["answer"]) <= 10:
            rows.append({"kind": "count", "gif": r["gif_name"], "q": str(r["question"]).strip(), "options": [str(i) for i in range(11)],
                         "gold": int(r["answer"]), "vid": str(r["vid_id"])})
    return rows


def tgif_qa(split, cap, rng):
    rows = [r for r in _rows(rng) if _split(r["gif"]) == split]
    rows = avkit.balanced(rows, lambda r: r["kind"], cap, rng)
    gifs = sorted({r["gif"] for r in rows})
    todo = {f"TGIF_Zero_Shot_QA/mp4/{g}.mp4": g for g in gifs if not cached_item("tgif_qa", g)}
    if todo:
        with _LOCK:
            convert_remote_zip_videos(_V, "TGIF_Zero_Shot_QA.zip", "tgif_qa", todo, workers=6, frames=8, max_side=448, keep_audio=False)
    for r in rows:
        m = cached_item("tgif_qa", r["gif"])
        if not m:
            continue
        if r["kind"] == "frameqa":
            others = [a for a in r["pool"] if a != r["answer"]]
            if len(others) < 3:
                continue
            opts = [r["answer"], *rng.sample(others, 3)]
            order = list(range(4))
            rng.shuffle(order)
            options, gold = [opts[i] for i in order], order.index(0)
        else:
            options, gold = r["options"], r["gold"]
        q = r["q"][:1].upper() + r["q"][1:]
        rec = mcq_record(rid("tgif_qa", split, r["vid"], r["gif"], r["q"]), {"clip": "<video:1>", "question": q},
                         "Look at the animated clip. Which option correctly answers the question?", options, gold,
                         area="video", dataset="tgif_qa", kind=r["kind"])
        if rec:
            rec.media = [m]
            yield rec


register(DatasetSpec("tgif_qa", tgif_qa, ("train", "val", "test"), _V, "other (research)", "video", multimodal=True,
                     description="TGIF-QA on animated GIFs: action, transition, frame QA and count (split by GIF)"))
