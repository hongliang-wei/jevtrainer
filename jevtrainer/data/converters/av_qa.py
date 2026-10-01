"""Audio-visual question answering over short video clips (video + its sound track in one record)."""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict

from jevtrainer.data import avkit
from jevtrainer.data.base import DatasetSpec, hf_file, mcq_record, register, rid

_Q = "Which option correctly answers the question about the video and its sound?"
_WORKERS = int(os.environ.get("JEVTRAINER_AV_WORKERS", "8"))


def _state(question: str) -> dict:
    return {"clip": "<video:1>", "question": question}


# ---- AVQA (Yang et al. 2022): 4-way multiple choice over VGGSound clips -------------------------
def avqa(split, cap, rng):
    repo = "juyil/AVQA-videos"
    rows = json.load(open(hf_file(repo, {"train": "train_qa.json", "val": "val_qa.json"}[split]), encoding="utf8"))
    rng.shuffle(rows)
    rows = rows[:cap]
    items = avkit.convert_videos(repo, "avqa", {r["video_name"]: f"videos/{r['video_name']}.mp4" for r in rows}, _WORKERS)
    for r in rows:
        media = items.get(r["video_name"])
        if not media:
            continue
        rec = mcq_record(rid("avqa", split, r["id"]), _state(r["question_text"]), _Q, r["multi_choice"], int(r["answer"]),
                         area="av", question_type=r.get("question_type"), modality=r.get("question_relation"))
        if rec:
            rec.media = [media]
            yield rec


# ---- MUSIC-AVQA (Li et al. 2022): open answers, options drawn from the answers of the same question ----
_SLOT = re.compile(r"<[^<>]+>")


def _question(r) -> str:
    vals = json.loads(r["templ_values"]) if r.get("templ_values") else []
    it = iter(vals)
    return _SLOT.sub(lambda m: str(next(it, "")), r["question_content"]).strip()


def music_avqa(split, cap, rng):
    repo = "UnFaZeD07/Music-AVQA"
    rows = [r for r in json.load(open(hf_file(repo, f"avqa-{split}.json"), encoding="utf8")) if not r.get("question_deleted")]
    train = rows if split == "train" else json.load(open(hf_file(repo, "avqa-train.json"), encoding="utf8"))
    pool_q, pool_t = defaultdict(set), defaultdict(set)  # candidate answers per question text / per question type
    for r in train:
        a = str(r["anser"]).strip().lower()
        pool_q[r["question_content"]].add(a)
        pool_t[r["type"]].add(a)
    rng.shuffle(rows)
    rows = rows[:cap]
    items = avkit.convert_videos(repo, "music_avqa", {r["video_id"]: f"videos/{r['video_id']}.mp4" for r in rows}, _WORKERS)
    for r in rows:
        media = items.get(r["video_id"])
        if not media:
            continue
        gold = str(r["anser"]).strip().lower()
        pool = sorted(pool_q[r["question_content"]] - {gold})
        if len(pool) < 3:
            pool = sorted((pool_t[r["type"]] | set(pool)) - {gold})
        options = [gold, *rng.sample(pool, min(3, len(pool)))]
        if len(options) < 2:
            continue
        order = list(range(len(options)))
        rng.shuffle(order)
        rec = mcq_record(rid("music_avqa", split, r["question_id"], r["video_id"]), _state(_question(r)), _Q,
                         [options[i] for i in order], order.index(0), area="av", question_type=r["type"])
        if rec:
            rec.media = [media]
            yield rec


register(DatasetSpec("avqa", avqa, ("train", "val"), "juyil/AVQA-videos", "other (research)", "av", multimodal=True,
                     description="AVQA: 4-way multiple choice about a VGGSound clip (video and sound)"))
register(DatasetSpec("music_avqa", music_avqa, ("train", "val", "test"), "UnFaZeD07/Music-AVQA", "gpl-3.0", "av", multimodal=True,
                     description="MUSIC-AVQA: audio-visual questions about instrument performances; distractors come from other answers"))
