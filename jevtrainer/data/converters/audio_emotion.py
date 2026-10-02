"""Speech emotion recognition on acted emotional-speech corpora (16 kHz mono FLAC).

One clip + one question per record; the label set is fixed per corpus ("which emotion does the speaker express?"), the
answer is the corpus' own emotion tag. All corpora are split by *speaker/actor* so that no voice is in both train and test.
Classes are sampled evenly.
"""

from __future__ import annotations

import os

from jevtrainer.data import avkit
from jevtrainer.data.base import DatasetSpec, register

Q_EMO = "Which emotion does the speaker express?"
_NORM = {"anger": "angry", "angry": "angry", "disgust": "disgusted", "disgusted": "disgusted", "fear": "fearful", "fearful": "fearful",
         "happy": "happy", "happiness": "happy", "joy": "happy", "neutral": "neutral", "sad": "sad", "sadness": "sad",
         "surprise": "surprised", "surprised": "surprised", "calm": "calm", "boredom": "bored", "bored": "bored"}
_DESC = {
    "angry": "angry: the voice is tense, loud or sharp, expressing anger or irritation",
    "calm": "calm: a relaxed, peaceful voice with low arousal",
    "disgusted": "disgusted: the voice expresses disgust or revulsion",
    "fearful": "fearful: a frightened, anxious or trembling voice",
    "happy": "happy: a cheerful, joyful voice",
    "neutral": "neutral: a plain voice without a particular emotion",
    "sad": "sad: a low, slow, sorrowful voice",
    "surprised": "surprised: a startled voice expressing surprise",
    "bored": "bored: a flat, uninterested voice",
}


def _criteria(labels) -> dict[str, str]:
    return {k: _DESC[k] for k in labels}


def _emotion(r, col="emotion"):
    return _NORM.get(str(r[col]).strip().lower())


# ---- CREMA-D: 7442 clips, 91 actors, 6 emotions; official-style speaker-disjoint train / validation / test ----------
CREMAD = ["angry", "disgusted", "fearful", "happy", "neutral", "sad"]


def crema_d(split, cap, rng):
    """6 acted emotions in 12 scripted sentences; train / validation (val) / test come from the `confit/cremad-parquet` split."""
    repo = "confit/cremad-parquet"
    part = {"train": "train", "val": "validation", "test": "test"}[split]
    files = avkit.parquet_files(repo, f"data/{part}-")
    return avkit.clip_label(
        "crema_d", split, cap, rng, repo, files, ["file", "emotion"], label_of=_emotion, question=Q_EMO, criteria=_criteria(CREMAD),
        key_of=lambda r: os.path.basename(r["file"]), meta_of=lambda r: {"source_label": r["emotion"]},
    )


# ---- RAVDESS (speech part): 1440 clips, 24 actors, 8 emotions; actors 1-20 train, 21-24 test ------------------------
RAVDESS = ["angry", "calm", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]


def _ravdess_actor(r) -> int:
    return int(os.path.basename(r["file"]).rsplit(".", 1)[0].split("-")[-1])


def ravdess(split, cap, rng):
    """8 acted emotions (two statements, two intensities). The corpus ships no split: actors 1-20 are train and actors
    21-24 (two women, two men) are test."""
    repo = "confit/ravdess-parquet"
    files = avkit.parquet_files(repo, "fold1/")
    rows = avkit.parquet_rows(repo, files, ["file", "emotion"], "ravdess")
    seen, uniq = set(), []
    for r in rows:  # fold1 train + test together are all 1440 clips
        b = os.path.basename(r["file"])
        if b not in seen and (_ravdess_actor(r) >= 21) == (split == "test"):
            seen.add(b)
            uniq.append(r)
    return avkit.clip_label(
        "ravdess", split, cap, rng, repo, files, [], label_of=_emotion, question=Q_EMO, criteria=_criteria(RAVDESS),
        key_of=lambda r: os.path.basename(r["file"]), rows=uniq,
        meta_of=lambda r: {"source_label": r["emotion"], "actor": _ravdess_actor(r)},
    )


# ---- SAVEE: 480 clips, 4 male British speakers, 7 emotions; speaker KL is the test speaker ------------------------
SAVEE = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]


def savee(split, cap, rng):
    """7 acted emotions read by four male speakers (DC, JE, JK, KL); speaker KL is held out as test."""
    repo = "AbstractTTS/SAVEE"
    files = avkit.parquet_files(repo, "data/")
    is_test = lambda r: r["file"].split("_")[0].upper() == "KL"  # noqa: E731
    return avkit.clip_label(
        "savee", split, cap, rng, repo, files, ["file", "emotion"], label_of=_emotion, question=Q_EMO, criteria=_criteria(SAVEE),
        key_of=lambda r: r["file"], keep=is_test if split == "test" else (lambda r: not is_test(r)),
        meta_of=lambda r: {"source_label": r["emotion"], "speaker": r["file"].split("_")[0]},
    )


register(DatasetSpec("crema_d", crema_d, ("train", "val", "test"), "confit/cremad-parquet", "odbl", "audio", multimodal=True,
                     description="CREMA-D: acted emotion (6 classes) of a short spoken sentence"))
register(DatasetSpec("ravdess", ravdess, ("train", "test"), "confit/ravdess-parquet", "cc-by-nc-sa-4.0", "audio", multimodal=True,
                     description="RAVDESS speech: acted emotion (8 classes); actors 1-20 train, 21-24 test"))
register(DatasetSpec("savee", savee, ("train", "test"), "AbstractTTS/SAVEE", "other (research)", "audio", multimodal=True,
                     description="SAVEE: acted emotion (7 classes) by four male speakers; speaker KL is test"))
