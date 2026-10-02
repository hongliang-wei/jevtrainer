"""More video question answering sets (multiple choice, one question per video).

* ``star``       - STAR situated reasoning over Charades clips: native 4-option questions (Interaction / Sequence /
  Prediction / Feasibility); the clip is the question's own [start, end] window of the Charades video.
* ``clevrer_mc`` - CLEVRER multiple choice (video-R1 release of the official questions): explanatory / predictive /
  counterfactual questions about colliding synthetic objects.
* ``msvd_qa`` / ``msrvtt_qa`` - open-ended one-word answers; the 4 options are the true answer + 3 answers that the
  dataset gives to *other*, similarly worded questions (see `OpenPool`; `task` = msvd_what, msvd_who, ...).
* ``activitynet_qa`` - open-ended answers over long YouTube videos, options built the same way (by question type).

The hub copies of MSVD-QA / MSRVTT-QA / ActivityNet-QA only carry the validation / test questions, so the split is cut
by a fixed hash of the video id (about 8 % test). Videos are read straight out of the hub zips (`zipkit`).
"""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict

from jevtrainer.data import vidkit
from jevtrainer.data.avkit_video import fetch_ranged, hash_bucket
from jevtrainer.data.base import DatasetSpec, mcq_record, register, rid
from jevtrainer.data.zipkit import convert_zip_clips, open_zip

_Q = "Which option correctly answers the question about the video?"
_FIRST = re.compile(r"^\s*(what|who|how|when|where|why|which|is|are|do|does|did|can|was|were)\b", re.I)


def _in_split(split: str, key: str, name: str) -> bool:
    return (hash_bucket(name, key) < 8) == (split == "test")


def _state(question: str) -> dict:
    return {"clip": "<video:1>", "question": question}


def _one_per_video(rows: list[dict], vid, rng) -> list[dict]:
    rng.shuffle(rows)
    seen, out = set(), []
    for r in rows:
        if vid(r) not in seen:
            seen.add(vid(r))
            out.append(r)
    return out


class OpenPool:
    """Distractors for open-ended answers: answers the dataset gives to *other* questions that start with the same words.

    Looks for questions with the same first four words ("what is the gender"), then the first three, then the same
    question type (what / who / ...); the first level that offers at least 3 other distinct answers is used (so the answer
    "male" is contrasted with "female", not with "to dive"). A question whose closest pool has only 1-2 other answers
    gets a 2-3 option question; `None` when nothing else exists.
    """

    def __init__(self, questions, answers, types):
        self.p4, self.p3, self.pt = defaultdict(set), defaultdict(set), defaultdict(set)
        for q, a, t in zip(questions, answers, types):
            w = str(q).lower().split()
            self.p4[" ".join(w[:4])].add(a)
            self.p3[" ".join(w[:3])].add(a)
            self.pt[t].add(a)

    def options(self, question: str, qtype: str, gold: str, rng, n: int = 4) -> list[str] | None:
        w = str(question).lower().split()
        levels = [self.p4[" ".join(w[:4])], self.p3[" ".join(w[:3])], self.pt[qtype]]
        best = None
        for pool in levels:
            cands = sorted(pool - {gold})
            if len(cands) >= n - 1:
                best = cands
                break
            if best is None and cands:
                best = cands
        if not best:
            return None
        options = [gold, *rng.sample(best, min(n - 1, len(best)))]
        rng.shuffle(options)
        return options


def _open_qa(name: str, split: str, cap: int, rng, df, vid_col: str, zip_repo: str, zip_file: str, member_of, label: str,
             answer_col: str = "answer", type_col: str | None = None):
    df = df.copy()
    df["_a"] = df[answer_col].astype(str).str.strip().str.lower()
    df["_t"] = df["question"].map(lambda q: (_FIRST.match(str(q)).group(1).lower() if _FIRST.match(str(q)) else "other"))
    pool = OpenPool(df["question"], df["_a"], df["_t"])
    z = open_zip(zip_repo, zip_file)
    names = set(z.names())
    rows = [r for r in df.to_dict("records") if _in_split(split, str(r[vid_col]), name) and member_of(r[vid_col]) in names]
    rows = _one_per_video(rows, lambda r: r[vid_col], rng)[:cap]
    jobs = {str(r[vid_col]): (z, member_of(r[vid_col]), {}) for r in rows}
    media = convert_zip_clips(name, jobs, frames=8, max_side=448, audio_s=30)
    for r in rows:
        m = media.get(str(r[vid_col]))
        if not m:
            continue
        options = pool.options(r["question"], r["_t"], r["_a"], rng)
        if not options:
            continue
        q = str(r["question"]).strip()
        rec = mcq_record(rid(name, split, r[vid_col], r.get("question_id", "")), _state(q if q.endswith("?") else q + "?"), _Q,
                         options, options.index(r["_a"]), area="video", dataset=name, task=f"{label}_{r['_t']}")
        if rec:
            rec.media = [m]
            yield rec


# ---- MSVD-QA / MSRVTT-QA (Xu et al. 2017): short open answers over short clips -------------------------------
def msvd_qa(split, cap, rng):
    """MSVD-QA validation questions (245 videos of the hub copy): what / who / how / when / where, 1-word answers."""
    df = vidkit.read_parquet("Rohollah903/msvd_qa", "val.parquet")
    yield from _open_qa("msvd_qa", split, cap, rng, df, "video_name", "Rohollah903/msvd_qa", "videos.zip",
                        lambda v: f"videos/{v}.avi", "msvd")


def msrvtt_qa(split, cap, rng):
    """MSRVTT-QA validation questions (497 videos of the hub copy): what / who / how / when / where, 1-word answers."""
    df = vidkit.read_parquet("Rohollah903/msrvtt_qa", "val.parquet")
    yield from _open_qa("msrvtt_qa", split, cap, rng, df, "video_name", "Rohollah903/msrvtt_qa", "videos.zip",
                        lambda v: f"videos/{v}.mp4", "msrvtt")


# ---- STAR (Wu et al. 2021): situated reasoning in real-world videos ---------------------------------------------
_STAR = "tdat1465/star-charades"
_STAR_TYPES = {"Interaction": "interaction", "Sequence": "sequence", "Prediction": "prediction", "Feasibility": "feasibility"}


def star(split, cap, rng):
    """STAR multiple choice (4 options as annotated). The clip is the [start, end] window of the Charades video the
    question refers to; one random question per video. Splits are the official train / val files; `task` is the
    question family (Interaction, Sequence, Prediction, Feasibility)."""
    path = fetch_ranged(_STAR, {"train": "STAR_train.json", "val": "STAR_val.json"}[split], "star")
    rows = json.load(open(path, encoding="utf8"))
    rows = [{k: r[k] for k in ("question_id", "question", "video_id", "start", "end", "answer", "choices")} for r in rows]
    z = open_zip(_STAR, "Charades_v1_480.zip")
    names = set(z.names())
    rows = [r for r in rows if f"Charades_v1_480/{r['video_id']}.mp4" in names and r["end"] - r["start"] > 1]
    # balance the four question families, then one question per video
    fam = defaultdict(list)
    for r in rows:
        fam[r["question_id"].split("_")[0]].append(r)
    picked, used = [], set()
    for g in fam.values():
        rng.shuffle(g)
    order = list(fam)
    while len(picked) < cap and any(fam.values()):
        for f in order:
            while fam[f]:
                r = fam[f].pop()
                if r["video_id"] not in used:
                    used.add(r["video_id"])
                    picked.append(r)
                    break
    jobs = {f"{r['video_id']}_{int(r['start'] * 10)}": (z, f"Charades_v1_480/{r['video_id']}.mp4", {"start": r["start"], "end": r["end"]})
            for r in picked}
    media = convert_zip_clips("star", jobs, frames=8, max_side=448, audio_s=15)
    for r in picked:
        m = media.get(f"{r['video_id']}_{int(r['start'] * 10)}")
        if not m:
            continue
        opts = [c["choice"] for c in sorted(r["choices"], key=lambda c: c["choice_id"])]
        gold = next((i for i, o in enumerate(opts) if o == r["answer"]), None)
        if gold is None:
            continue
        rec = mcq_record(rid("star", split, r["question_id"]), _state(r["question"]), _Q, opts, gold, area="video", dataset="star",
                         task=_STAR_TYPES.get(r["question_id"].split("_")[0], "other"))
        if rec:
            rec.media = [m]
            yield rec


# ---- CLEVRER multiple choice (Yi et al. 2020; video-R1 release) --------------------------------------------------
def _clevrer_kind(q: str) -> str:
    """explanatory = "responsible"; counterfactual = "without ..." / "if ... removed"; predictive = "will happen next / collide"."""
    ql = q.lower()
    if "responsible" in ql:
        return "explanatory"
    if "without" in ql or "removed" in ql or "if the" in ql:
        return "counterfactual"
    if "next" in ql or "will" in ql:
        return "predictive"
    return "other"


def clevrer_mc(split, cap, rng):
    """CLEVRER multiple-choice questions (8,220 in the video-R1 release; the answer letter is the dataset's own).
    One question per training video of the synthetic collision videos (no sound track); `task` is the question kind
    (explanatory / predictive / counterfactual) derived from the question wording."""
    df = vidkit.read_parquet("ngqtrung/video-r1-clevrer-mc-v1", "data/train-00000-of-00001.parquet")
    z = open_zip("pengxiang/clevrer", "video_train.zip")
    by_base = {n.rsplit("/", 1)[-1]: n for n in z.names() if n.endswith(".mp4")}
    rows = []
    for r in df.to_dict("records"):
        txt = r["prompt"][1]["content"]
        m = re.search(r"Question:\s*(.*?)\nOptions:\s*(\[.*\])", txt, re.S)
        vid = str(r["videos"][0]["video"]).rsplit("/", 1)[-1]
        if not m or vid not in by_base:
            continue
        try:
            opts = [re.sub(r"^[A-Z]\.\s*", "", o).strip() for o in ast.literal_eval(m.group(2))]
        except (ValueError, SyntaxError):
            continue
        gold = ord(str(r["reward_model"]["ground_truth"]).strip()[:1].upper()) - 65
        if 0 <= gold < len(opts):
            rows.append({"video": vid, "question": m.group(1).strip(), "options": opts, "gold": gold})
    rows = [r for r in rows if _in_split(split, r["video"], "clevrer") ]
    rows = _one_per_video(rows, lambda r: r["video"], rng)
    # even over the question kinds
    kinds = defaultdict(list)
    for r in rows:
        kinds[_clevrer_kind(r["question"])].append(r)
    picked = []
    while len(picked) < cap and any(kinds.values()):
        for k in list(kinds):
            if kinds[k] and len(picked) < cap:
                picked.append(kinds[k].pop())
    jobs = {r["video"][:-4]: (z, by_base[r["video"]], {}) for r in picked}
    media = convert_zip_clips("clevrer", jobs, frames=8, max_side=448, keep_audio=False)
    for r in picked:
        m = media.get(r["video"][:-4])
        if not m:
            continue
        rec = mcq_record(rid("clevrer_mc", split, r["video"], r["question"]), _state(r["question"]), _Q, r["options"], r["gold"],
                         area="video", dataset="clevrer", task=_clevrer_kind(r["question"]))
        if rec:
            rec.media = [m]
            yield rec


# ---- ActivityNet-QA (Yu et al. 2019): open questions over 1-3 min YouTube videos ---------------------------------
_ANQA = "lmms-eval/ActivityNetQA"
_ANQA_TYPES = {"0": "motion", "1": "spatial", "2": "temporal", "3": "yes_no", "4": "color", "5": "object", "6": "location",
               "7": "number", "8": "other"}


def activitynet_qa(split, cap, rng):
    """ActivityNet-QA test questions (8,000 over 800 videos). The videos live in 28 hub zips of ~210 videos each; only the
    first `JEVTRAINER_ANQA_ZIPS` (default 8) are indexed, so the questions about the ~230 videos inside are used. Yes/no questions
    keep the two options yes / no; others get up to 3 answers of similarly worded questions (`OpenPool`); `task` is the question type."""
    import os
    from concurrent.futures import ThreadPoolExecutor

    df = vidkit.read_parquet(_ANQA, "data/test-00000-of-00001.parquet")
    df["_a"] = df["answer"].astype(str).str.strip().str.lower()
    pool = OpenPool(df["question"], df["_a"], df["type"].astype(str))
    n = int(os.environ.get("JEVTRAINER_ANQA_ZIPS", "8"))
    with ThreadPoolExecutor(3) as ex:
        zips = list(ex.map(lambda i: open_zip(_ANQA, f"videos_chunked_{i:02d}.zip"), range(1, n + 1)))
    where = {m.rsplit("/", 1)[-1][2:-4]: (z, m) for z in zips for m in z.names() if m.endswith(".mp4")}
    rows = [r for r in df.to_dict("records") if r["video_name"] in where and _in_split(split, r["video_name"], "activitynet_qa")]
    rows = _one_per_video(rows, lambda r: r["video_name"], rng)[:cap]
    media = convert_zip_clips("activitynet_qa", {r["video_name"]: (*where[r["video_name"]], {}) for r in rows}, frames=8, max_side=448,
                              audio_s=30)
    for r in rows:
        m = media.get(r["video_name"])
        if not m:
            continue
        t = str(r["type"])
        options = ["yes", "no"] if r["_a"] in ("yes", "no") else pool.options(r["question"], t, r["_a"], rng)
        if not options or r["_a"] not in options:
            continue
        q = str(r["question"]).strip()
        rec = mcq_record(rid("activitynet_qa", split, r["question_id"]), _state(q if q.endswith("?") else q + "?"), _Q, options,
                         options.index(r["_a"]), area="video", dataset="activitynet_qa", task=_ANQA_TYPES.get(t, "other"))
        if rec:
            rec.media = [m]
            yield rec


register(DatasetSpec("activitynet_qa", activitynet_qa, ("train", "test"), _ANQA, "other (research)", "video", multimodal=True,
                     description="ActivityNet-QA: questions (motion, spatial, temporal, yes/no, colour, object, location, number) about long videos", version="2"))
register(DatasetSpec("msvd_qa", msvd_qa, ("train", "test"), "Rohollah903/msvd_qa", "other (research)", "video", multimodal=True,
                     description="MSVD-QA: one-word answers about a short clip; options are answers to similarly worded questions", version="2"))
register(DatasetSpec("msrvtt_qa", msrvtt_qa, ("train", "test"), "Rohollah903/msrvtt_qa", "other (research)", "video", multimodal=True,
                     description="MSRVTT-QA: one-word answers about a short clip; options are answers to similarly worded questions", version="2"))
register(DatasetSpec("star", star, ("train", "val"), _STAR, "other (research, Charades terms)", "video", multimodal=True,
                     description="STAR: situated reasoning (interaction, sequence, prediction, feasibility) on Charades clip windows"))
register(DatasetSpec("clevrer_mc", clevrer_mc, ("train", "test"), "ngqtrung/video-r1-clevrer-mc-v1 + pengxiang/clevrer", "cc0-1.0", "video",
                     multimodal=True, description="CLEVRER multiple choice: explanatory / predictive / counterfactual questions about collisions"))
