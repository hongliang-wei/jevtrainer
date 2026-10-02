"""Affect, intent and speaking style in short talking-head / sitcom clips (video + voice + transcript).

Clips come as tar archives of the MMLA collection (THUIAR/MMLA-Datasets, cc-by-4.0): one pass over the archive
converts every clip of every split, the archive is then deleted. Official train / dev / test lists give the splits
(dev is registered as `val`). The transcript of the utterance is part of the state next to the clip.

    meld       MELD (Friends)        7-way emotion of the speaker (choice). Rows labelled only "positive"/"negative"
                                     (sentiment, not emotion) are dropped.
    mustard    MUStARD (sitcoms)     is the speaker being sarcastic? (noul; yes = sarcastic, no = sincere)
    urfunny    UR-FUNNY v2 (TED)     is this punchline moment humorous? (noul; yes = humorous, no = serious)
    mintrec    MIntRec (sitcoms)     20 communicative intents (choice)
    chsims2    CH-SIMS v2 (Chinese)  sentiment score of the speaker, 5 levels from the annotated multimodal score in [-1, 1]
                                     (<= -0.8 / -0.6..-0.2 / 0 / 0.2..0.6 / >= 0.8), labels from tamb2203579/CH-SIMSv2 meta.csv
    cmu_mosei  CMU-MOSEI             sentiment score -3..3 (score, 7 levels; mean annotator score rounded). Clips are read
                                     from the raw zip member by member; train is class-balanced and capped (3000), val 300,
                                     test 800 (random, natural distribution).
"""

from __future__ import annotations

import os

from jevtrainer.data import avkit
from jevtrainer.data.avkit_video import (balanced_take, cached_item, convert_remote_zip_videos, convert_tar_videos, fetch_ranged)
from jevtrainer.data.base import DatasetSpec, choice_record, hf_file, noul_record, register, rid
from jevtrainer.schema import Question, Record, Target

_WORKERS = int(os.environ.get("JEVTRAINER_AV_WORKERS", "8"))  # parallel ffmpeg jobs
_CONNS = int(os.environ.get("JEVTRAINER_AV_CONNS", "4"))  # parallel range requests per big file
_MMLA = "THUIAR/MMLA-Datasets"
_SPLIT_FILE = {"train": "train", "val": "dev", "test": "test"}
# The download from the mirror is slow; the clips in these tars are in random order, so a prefix of the archive is a random subset.
_MAX_GB = {"meld": float(os.environ.get("JEVTRAINER_MELD_GB", "5")), "mintrec": 1.5, "chsims2": float(os.environ.get("JEVTRAINER_CHSIMS_GB", "4"))}


def _tsv(folder: str, split: str) -> list[dict]:
    import pandas as pd

    d = pd.read_csv(hf_file(_MMLA, f"{folder}/{_SPLIT_FILE[split]}.tsv"), sep="\t", dtype=str, keep_default_na=False)
    return [{"id": r.id.strip(), "text": r.text.strip(), "label": r.label.strip()} for r in d.itertuples()]


def _mmla_clips(name: str, folder: str, tar: str, prefix: str, wanted_ids: list[str], max_gb: float | None = None) -> None:
    """Convert every wanted clip of an MMLA video tar once (marker file), then delete the archive."""
    mark = avkit.media_dir(name, "_done") / "ok"
    if mark.exists():
        return
    path = fetch_ranged(_MMLA, f"{folder}/{tar}", name, conns=_CONNS, max_bytes=int(max_gb * 1e9) if max_gb else None)
    if path is None:
        raise RuntimeError(f"cannot download {folder}/{tar}")
    convert_tar_videos(path, name, {f"{prefix}{i}.mp4": i for i in wanted_ids}, _WORKERS, frames=8)
    avkit.drop_raw(name)
    mark.write_text("1")


def _all_ids(folder: str) -> list[str]:
    return sorted({r["id"] for s in _SPLIT_FILE for r in _tsv(folder, s)})


def _clips(name: str, folder: str, tar: str, prefix: str, split: str, cap: int, rng, keep=None, max_gb: float | None = None) -> list[tuple[dict, dict]]:
    """(row, media item) pairs of a split, shuffled, at most `cap` of them (class-balanced when the cap bites)."""
    _mmla_clips(name, folder, tar, prefix, _all_ids(folder), max_gb)
    rows = [r for r in _tsv(folder, split) if keep is None or keep(r)]
    rng.shuffle(rows)
    pairs = [(r, m) for r, m in ((r, cached_item(name, r["id"])) for r in rows) if m]
    return balanced_take(pairs, lambda p: p[0]["label"], cap, rng) if cap < len(pairs) else pairs

def _state(r: dict) -> dict:
    return {"clip": "<video:1>", "utterance": r["text"]}


# ---- MELD ------------------------------------------------------------------------------------------------------
EMOTIONS = {
    "anger": "anger: annoyed, irritated, angry or shouting at someone",
    "disgust": "disgust: repelled, grossed out or revolted by something",
    "fear": "fear: scared, anxious, nervous or alarmed",
    "joy": "joy: happy, amused, laughing, excited or delighted",
    "neutral": "neutral: calm, matter-of-fact, no particular emotion",
    "sadness": "sadness: sad, hurt, disappointed, upset or crying",
    "surprise": "surprise: shocked, astonished or taken aback",
}


def meld(split, cap, rng):
    for r, media in _clips("meld", "MELD", "MELD_video.tar.gz", "MELD_", split, cap, rng, keep=lambda r: r["label"] in EMOTIONS,
                           max_gb=_MAX_GB["meld"]):
        rec = choice_record(rid("meld", split, r["id"]), _state(r),
                            "Watch and listen to the clip (face, voice and words). Which emotion does the speaker express?",
                            dict(EMOTIONS), r["label"], area="av")
        rec.media = [media]
        yield rec


# ---- MUStARD ---------------------------------------------------------------------------------------------------
def mustard(split, cap, rng):
    for r, media in _clips("mustard", "MUStARD", "MUStARD_video.tar.gz", "MUStARD_", split, cap, rng,
                           keep=lambda r: r["label"] in ("sarcastic", "sincere")):
        rec = noul_record(rid("mustard", split, r["id"]), _state(r),
                          "Watch and listen to the clip. Is the speaker being sarcastic?", r["label"] == "sarcastic",
                          true="sarcastic: the speaker means the opposite of what the words say, mocking or ironic",
                          false="sincere: the speaker means what is said", area="av")
        rec.media = [media]
        yield rec


# ---- UR-FUNNY --------------------------------------------------------------------------------------------------
def urfunny(split, cap, rng):
    for r, media in _clips("urfunny", "UR-FUNNY-v2", "UR-FUNNYv2_video.tar.gz", "UR-FUNNY_", split, cap, rng,
                           keep=lambda r: r["label"] in ("humorous", "serious")):
        rec = noul_record(rid("urfunny", split, r["id"]), _state(r),
                          "Watch and listen to this moment of a talk. Is the speaker delivering a punchline that is meant to be funny?",
                          r["label"] == "humorous", true="humorous: a joke or punchline the audience would laugh at",
                          false="serious: informative or earnest speech, not a joke", area="av")
        rec.media = [media]
        yield rec


# ---- MIntRec ---------------------------------------------------------------------------------------------------
INTENTS = {
    "complain": "complain: express dissatisfaction or annoyance about something",
    "inform": "inform: give information or state a fact to the listener",
    "praise": "praise: compliment or admire someone or something",
    "apologise": "apologise: say sorry or express regret",
    "thank": "thank: express gratitude",
    "advise": "advise: suggest what the listener should do",
    "criticize": "criticize: point out faults of someone or something",
    "arrange": "arrange: propose or settle plans, times or tasks",
    "introduce": "introduce: present oneself or another person",
    "care": "care: show concern for the listener's well-being",
    "comfort": "comfort: console someone who is upset",
    "leave": "leave: say goodbye or announce going away",
    "prevent": "prevent: try to stop someone from doing something",
    "taunt": "taunt: mock or tease someone in a provoking way",
    "greet": "greet: say hello to someone",
    "agree": "agree: accept or approve what was said or proposed",
    "flaunt": "flaunt: show off achievements or possessions",
    "oppose": "oppose: disagree or object to what was said",
    "ask for help": "ask for help: request assistance or a favour",
    "joke": "joke: say something to make others laugh",
}


def mintrec(split, cap, rng):
    keys = {k: k.replace(" ", "_") for k in INTENTS}
    for r, media in _clips("mintrec", "MIntRec", "MIntRec_video.tar.gz", "MIntRec_", split, cap, rng, keep=lambda r: r["label"] in INTENTS,
                           max_gb=_MAX_GB["mintrec"]):
        rec = choice_record(rid("mintrec", split, r["id"]), _state(r),
                            "Watch and listen to the clip. What is the speaker's intent in this utterance?",
                            {keys[k]: v for k, v in INTENTS.items()}, keys[r["label"]], area="av")
        rec.media = [media]
        yield rec


# ---- CH-SIMS v2 (Chinese) --------------------------------------------------------------------------------------
CHSIMS_LEVELS = ["强烈负面：明显生气、厌恶或悲伤", "轻微负面：略带不满、失落或担忧", "中性：没有明显的情感倾向",
                 "轻微正面：略带愉快、认同或期待", "强烈正面：明显开心、兴奋或赞赏"]


def _chsims_level(x: float) -> int:
    return 0 if x <= -0.7 else 1 if x < -0.1 else 2 if x <= 0.1 else 3 if x < 0.7 else 4


def chsims2(split, cap, rng):
    import pandas as pd

    meta = pd.read_csv(hf_file("tamb2203579/CH-SIMSv2", "CH-SIMS v2(s)/meta.csv"), dtype={"video_id": str, "clip_id": str})
    meta["id"] = meta.video_id + "_" + meta.clip_id
    mode = {"train": "train", "val": "valid", "test": "test"}
    mark = avkit.media_dir("chsims2", "_done") / "ok"
    if not mark.exists():
        path = fetch_ranged(_MMLA, "CH-SIMSv2.0/Ch-simsv2_video.tar.gz", "chsims2", conns=_CONNS, max_bytes=int(_MAX_GB["chsims2"] * 1e9))
        if path is None:
            raise RuntimeError("cannot download CH-SIMS v2 videos")
        convert_tar_videos(path, "chsims2", {f"Ch-sims_{i}.mp4": i for i in meta.id}, _WORKERS, frames=8)
        avkit.drop_raw("chsims2")
        mark.write_text("1")
    rows = [r for r in meta[meta["mode"] == mode[split]].to_dict("records") if cached_item("chsims2", r["id"])]
    rng.shuffle(rows)
    rows = balanced_take(rows, lambda r: _chsims_level(float(r["label"])), cap, rng) if cap < len(rows) else rows
    for r in rows:
        rec = Record(rid("chsims2", split, r["id"]), {"clip": "<video:1>", "文本": str(r["text"]).strip()},
                     {"sentiment": Question("score", "观看并聆听这段视频（表情、语气和内容），说话人表达的情感倾向是？", list(CHSIMS_LEVELS))},
                     {"sentiment": Target(str(_chsims_level(float(r["label"]))))}, meta={"area": "av", "raw_score": float(r["label"])})
        rec.media = [cached_item("chsims2", r["id"])]
        yield rec


# ---- CMU-MOSEI ---------------------------------------------------------------------------------------------------
MOSEI_LEVELS_EN = ["-3: highly negative", "-2: negative", "-1: slightly negative", "0: neutral", "+1: slightly positive",
                   "+2: positive", "+3: highly positive"]
_MOSEI_N = {"train": 3000, "val": 300, "test": 800}


def cmu_mosei(split, cap, rng):
    import pandas as pd

    repo = "tamb2203579/CMU-MOSEI"
    lab = pd.read_csv(hf_file(repo, "label.csv"), dtype={"video_id": str, "clip_id": str})
    lab["id"] = lab.video_id + "/" + lab.clip_id
    mode = {"train": "train", "val": "valid", "test": "test"}[split]
    rows = lab[lab["mode"] == mode].to_dict("records")
    rng.shuffle(rows)
    n = min(cap, _MOSEI_N[split])
    level = lambda r: int(round(float(r["label"]))) + 3
    rows = balanced_take(rows, level, n, rng) if split == "train" else rows[:n]
    convert_remote_zip_videos(repo, "Raw.zip", "cmu_mosei", {f"Raw/{r['video_id']}/{r['clip_id']}.mp4": r["id"].replace("/", "_") for r in rows},
                              _WORKERS, frames=8)
    for r in rows:
        media = cached_item("cmu_mosei", r["id"].replace("/", "_"))
        if not media:
            continue
        rec = Record(rid("cmu_mosei", split, r["id"]), {"clip": "<video:1>", "transcript": str(r["text"]).strip()},
                     {"sentiment": Question("score", "Watch and listen to the clip (face, voice and words). How positive or negative is the speaker's sentiment?",
                                            list(MOSEI_LEVELS_EN))},
                     {"sentiment": Target(str(level(r)))}, meta={"area": "av", "raw_score": float(r["label"])})
        rec.media = [media]
        yield rec


register(DatasetSpec("meld", meld, ("train", "val", "test"), f"{_MMLA}:MELD", "cc-by-4.0", "av", multimodal=True,
                     description="MELD: 7-way emotion of a Friends utterance (video, voice, transcript)"))
register(DatasetSpec("mustard", mustard, ("train", "val", "test"), f"{_MMLA}:MUStARD", "cc-by-4.0", "av", multimodal=True,
                     description="MUStARD: sarcasm in sitcom utterances (video, voice, transcript)"))
register(DatasetSpec("urfunny", urfunny, ("train", "val", "test"), f"{_MMLA}:UR-FUNNY-v2", "cc-by-4.0", "av", multimodal=True,
                     description="UR-FUNNY v2: humorous punchline vs serious talk (video, voice, transcript)"))
register(DatasetSpec("mintrec", mintrec, ("train", "val", "test"), f"{_MMLA}:MIntRec", "cc-by-4.0", "av", multimodal=True,
                     description="MIntRec: 20 speaker intents in sitcom utterances (video, voice, transcript)"))
register(DatasetSpec("chsims2", chsims2, ("train", "val", "test"), "tamb2203579/CH-SIMSv2 + THUIAR/MMLA-Datasets", "apache-2.0", "av",
                     multimodal=True, description="CH-SIMS v2: Chinese sentiment score (5 levels) of a speaker clip"))
register(DatasetSpec("cmu_mosei", cmu_mosei, ("train", "val", "test"), "tamb2203579/CMU-MOSEI", "apache-2.0", "av", multimodal=True,
                     description="CMU-MOSEI: sentiment score -3..3 of a YouTube monologue segment (7 levels)"))
