"""More acted-emotion speech corpora: TESS, ESD (English part), EmoDB (16 kHz mono FLAC).

Same task as `audio_emotion.py` ("which emotion does the speaker express?", fixed label set, classes sampled evenly).
Splits never share a speaker (ESD, EmoDB: official) or a target word (TESS has only two speakers).
"""

from __future__ import annotations

import os

from jevtrainer.data import aukit, avkit
from jevtrainer.data.base import DatasetSpec, register
from jevtrainer.data.converters.audio_emotion import _NORM, Q_EMO, _criteria

_NORM = {**_NORM, "ps": "surprised", "anxiety": "fearful", "wut": "angry"}


def _emo(r):
    e = str(r["emotion"]).strip().lower()
    return "surprised" if "surpris" in e else _NORM.get(e)  # TESS: "pleasant surprise"


# ---- TESS: 2800 clips, two actresses (26 and 64 years), 7 emotions, 200 target words -----------------------------
TESS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]


def _tess_word(r) -> str:
    return os.path.basename(r["file"]).split("_")[1].lower()


def tess(split, cap, rng):
    """The Toronto Emotional Speech Set: 'Say the word X' in 7 emotions by two actresses. The corpus ships one split, so
    12 % of the 200 target words (fixed hash) are held out as test: no word is in both splits."""
    repo = "AbstractTTS/TESS"
    files = aukit.pq_files(repo, "data/", ["data/train-00000-of-00001.parquet"])
    test = lambda r: avkit.hash_pct(_tess_word(r), 12)  # noqa: E731
    return avkit.clip_label(
        "tess", split, cap, rng, repo, files, ["file", "emotion"], label_of=_emo, question=Q_EMO, criteria=_criteria(TESS),
        key_of=lambda r: os.path.basename(r["file"]), keep=test if split == "test" else (lambda r: not test(r)),
        meta_of=lambda r: {"source_label": r["emotion"], "speaker": os.path.basename(r["file"]).split("_")[0]},
    )


# ---- ESD (English part): 10 speakers (5 f, 5 m), 5 emotions, 350 utterances each --------------------------------------
ESD = ["angry", "happy", "neutral", "sad", "surprised"]


def esd(split, cap, rng):
    """Emotional Speech Dataset, English speakers 0011-0020 (5 emotions: neutral, happy, angry, sad, surprise). The corpus
    ships no split here: the last-numbered speaker of each gender (a woman and a man) is held out as test."""
    repo = "AbstractTTS/ESD_english"
    files = aukit.pq_files(repo, "data/", [f"data/train-0000{i}-of-00003.parquet" for i in range(3)])
    rows = avkit.parquet_rows(repo, files, ["file", "emotion", "gender", "transcription"], "esd")
    spk = lambda r: os.path.basename(r["file"]).split("_")[0]  # noqa: E731
    last: dict[str, str] = {}
    for r in rows:
        g = str(r["gender"]).lower()
        last[g] = max(last.get(g, ""), spk(r))
    held = set(last.values())
    rows = [r for r in rows if (spk(r) in held) == (split == "test")]
    return avkit.clip_label(
        "esd", split, cap, rng, repo, files, [], label_of=_emo, question=Q_EMO, criteria=_criteria(ESD),
        key_of=lambda r: os.path.basename(r["file"]), rows=rows,
        meta_of=lambda r: {"source_label": r["emotion"], "speaker": spk(r), "gender": r["gender"]},
    )


# ---- EmoDB: 535 German acted utterances, 10 actors, 7 emotions ------------------------------------------------------------
EMODB = ["angry", "bored", "disgusted", "fearful", "happy", "neutral", "sad"]


def emodb(split, cap, rng):
    """Berlin Database of Emotional Speech (German sentences, 7 emotions). train / test of the `confit/emodb-parquet`
    copy; the speech is German, the question is asked in English."""
    repo = "confit/emodb-parquet"
    files = aukit.pq_files(repo, f"data/{split}-", ["data/test-00000-of-00001.parquet", "data/train-00000-of-00001.parquet"])
    return avkit.clip_label(
        "emodb", split, cap, rng, repo, files, ["emotion"], label_of=_emo, question=Q_EMO, criteria=_criteria(EMODB),
        key_of=lambda r: f"{r['_file']}:{r['_i']}", meta_of=lambda r: {"source_label": r["emotion"]},
    )


register(DatasetSpec("tess", tess, ("train", "test"), "AbstractTTS/TESS", "cc-by-nc-nd-4.0", "audio", multimodal=True,
                     description="TESS: acted emotion (7 classes) of 'Say the word X'; 12% of the words are test"))
register(DatasetSpec("esd", esd, ("train", "test"), "AbstractTTS/ESD_english", "other (research)", "audio", multimodal=True,
                     description="ESD English: acted emotion (5 classes) of 10 speakers; one woman and one man held out as test"))
register(DatasetSpec("emodb", emodb, ("train", "test"), "confit/emodb-parquet", "other (research)", "audio", multimodal=True,
                     description="EmoDB: acted emotion (7 classes) of German sentences"))
