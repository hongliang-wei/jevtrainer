"""Sound-event recognition over short audio clips (16 kHz mono FLAC, at most 30 s).

Every record is one audio clip plus one 4-way multiple-choice question ("which sound is in this clip?"): the answer is the
dataset's own label, the three distractors are random other class names of the same dataset (never a label the clip
also carries). Classes are sampled evenly. Splits follow the official protocol; where a corpus ships only cross-validation
folds the last fold is held out as `test` (folds group clips of the same source recording, so nothing leaks).
"""

from __future__ import annotations

import csv
from collections import Counter

from jevtrainer.data import avkit
from jevtrainer.data.base import DatasetSpec, hf_file, register

Q_SOUND = "Which sound can be heard in this audio clip?"


# ---- ESC-50: 2000 five-second environmental recordings, 50 classes, folds 1-4 train / fold 5 test ------------
def esc50(split, cap, rng):
    repo = "ashraq/esc50"
    files = avkit.parquet_files(repo, "data/")
    want = (lambda r: r["fold"] <= 4) if split == "train" else (lambda r: r["fold"] == 5)
    return avkit.clip_choice(
        "esc50", split, cap, rng, repo, files, ["filename", "fold", "category"],
        label_of=lambda r: avkit.humanize(r["category"]), question=Q_SOUND, key_of=lambda r: r["filename"], keep=want,
        meta_of=lambda r: {"source_label": r["category"], "fold": r["fold"]},
    )


# ---- UrbanSound8K: 8732 clips (<= 4 s) of 10 urban sound classes, folds 1-9 train / fold 10 test --------------
def urbansound8k(split, cap, rng):
    repo = "danavery/urbansound8K"
    files = avkit.parquet_files(repo, "data/")
    want = (lambda r: r["fold"] <= 9) if split == "train" else (lambda r: r["fold"] == 10)
    return avkit.clip_choice(
        "urbansound8k", split, cap, rng, repo, files, ["slice_file_name", "fold", "class"],
        label_of=lambda r: avkit.humanize(r["class"]), question="Which urban sound can be heard in this audio clip?",
        key_of=lambda r: r["slice_file_name"], keep=want, meta_of=lambda r: {"source_label": r["class"], "fold": r["fold"]},
    )


# ---- FSD50K: 51k Freesound clips, 200 AudioSet-ontology classes, multi-label ----------------------------------
# train / val = the official dev split, test = the official eval set. The answer is the most specific label of the clip
# (labels are listed leaf first); distractors are leaf classes none of whose labels the clip carries.
def fsd50k(split, cap, rng):
    repo = "philgzl/fsd50k"
    sub = "eval" if split == "test" else "dev"
    if sub == "eval":
        labels = {r["fname"]: r["labels"].split(",") for r in csv.DictReader(open(hf_file("Fhrozen/FSD50k", "labels/eval.csv"), encoding="utf8"))}
        part = {k: "test" for k in labels}
    else:
        raw = list(csv.DictReader(open(hf_file("Fhrozen/FSD50k", "labels/dev.csv"), encoding="utf8")))
        labels = {r["fname"]: r["labels"].split(",") for r in raw}
        part = {r["fname"]: ("val" if r["split"] == "val" else "train") for r in raw}
    leaves = sorted({v[0] for v in labels.values()})
    files = avkit.parquet_files(repo, f"data/{sub}-")
    rows = avkit.parquet_rows(repo, files, ["name"], "fsd50k")
    for r in rows:
        r["fname"] = r["name"].rsplit("/", 1)[-1].rsplit(".", 1)[0].rsplit("_", 1)[0]
    rows = [r for r in rows if r["name"].endswith("_0.ogg") and part.get(r["fname"]) == split]
    for r in rows:
        r["labels"] = labels[r["fname"]]
    rng.shuffle(rows)
    picked = avkit.balanced(rows, lambda r: r["labels"][0], cap, rng)
    media = avkit.parquet_audio(repo, picked, "fsd50k", lambda r: avkit.rid_("fsd50k", split, r["fname"]))
    for r, m in zip(picked, media):
        if not m:
            continue
        gold = avkit.humanize(r["labels"][0])
        pool = [avkit.humanize(x) for x in leaves if x not in r["labels"]]
        rec = avkit.audio_mcq(avkit.rid_("fsd50k", split, r["fname"]), m, Q_SOUND, gold, pool, rng, source_label=r["labels"][0],
                              labels=r["labels"])
        if rec:
            yield rec
    avkit.release("fsd50k")


# ---- AudioSet: 10 s YouTube clips, 527 classes, multi-label ---------------------------------------------------
# train = the class-balanced training subset (bal_train, ~22k clips), test = the evaluation set. The answer is the clip's
# rarest label (so that frequent classes such as Speech / Music do not dominate); distractors are labels seen in the
# dataset that the clip does not carry. Very generic ontology nodes are never used as distractors because they would
# also be true (Animal, Music, Vehicle, ...).
_GENERIC = {"Animal", "Domestic animals, pets", "Livestock, farm animals, working animals", "Wild animals", "Bird", "Music",
            "Musical instrument", "Vehicle", "Sound effect", "Noise", "Environmental noise", "Inside, small room",
            "Inside, large room or hall", "Inside, public space", "Outside, urban or manmade", "Outside, rural or natural",
            "Mechanisms", "Engine", "Tools", "Domestic sounds, home sounds", "Human sounds", "Human voice", "Water",
            "Natural sounds", "Sounds of things", "Source-ambiguous sounds", "Generic impact sounds", "Miscellaneous sources",
            "Hubbub, speech noise, speech babble", "Silence", "Music genre", "Music role", "Music mood", "Channel, environment and background"}


def audioset(split, cap, rng):
    repo = "agkphysics/AudioSet"
    sub = "bal_train" if split == "train" else "eval"
    files = avkit.parquet_files(repo, f"data/{sub}/")
    rows = avkit.parquet_rows(repo, files, ["video_id", "human_labels"], "audioset")
    freq = Counter(x for r in rows for x in r["human_labels"])
    pool_all = sorted(x for x, n in freq.items() if x not in _GENERIC and n >= 5)
    rows = [r for r in rows if r["human_labels"]]
    for r in rows:
        r["gold"] = min(r["human_labels"], key=lambda x: (freq[x], x))
    rng.shuffle(rows)
    picked = avkit.balanced(rows, lambda r: r["gold"], cap, rng)
    media = avkit.parquet_audio(repo, picked, "audioset", lambda r: avkit.rid_("audioset", split, r["video_id"]), max_s=10.0)
    for r, m in zip(picked, media):
        if not m:
            continue
        pool = [x for x in pool_all if x not in r["human_labels"]]
        rec = avkit.audio_mcq(avkit.rid_("audioset", split, r["video_id"]), m, Q_SOUND, r["gold"], pool, rng, source_label=r["gold"],
                              labels=r["human_labels"])
        if rec:
            yield rec
    avkit.release("audioset")


register(DatasetSpec("esc50", esc50, ("train", "test"), "ashraq/esc50", "cc-by-nc-3.0", "audio", multimodal=True,
                     description="ESC-50: which of 4 environmental sounds is in the 5 s clip (folds 1-4 train, fold 5 test)"))
register(DatasetSpec("urbansound8k", urbansound8k, ("train", "test"), "danavery/urbansound8K", "cc-by-nc-3.0", "audio",
                     multimodal=True, description="UrbanSound8K: which of 4 urban sounds is in the clip (folds 1-9 train, fold 10 test)"))
register(DatasetSpec("fsd50k", fsd50k, ("train", "val", "test"), "philgzl/fsd50k", "cc-by-4.0", "audio", multimodal=True,
                     description="FSD50K: which of 4 sound classes (AudioSet ontology) is the main event of the Freesound clip"))
register(DatasetSpec("audioset", audioset, ("train", "test"), "agkphysics/AudioSet", "cc-by-4.0", "audio", multimodal=True,
                     description="AudioSet: which of 4 sound classes is heard in the 10 s YouTube clip (balanced train subset / eval set)"))
