"""Audio-visual event recognition: VGGSound (sound source of a clip) and AVE (event class, audio-visual match).

VGGSound  (11hu83/vggsound, cc-by-4.0)
    The repo holds the 15,446 clips of the official VGGSound *test* split (309 classes, 10 s, video + sound).
    There is no other released split with media on the mirror, so we cut our own: a clip goes to `test` when
    the hash of its id falls in 10 % of the buckets, else to `train` (class-balanced, capped).
    Task: 4-way choice "what is making the sound?". Gold = the clip's class label; 3 distractors = other classes,
    two of them drawn from the 6 most similar class names (shared content words, e.g. "playing acoustic guitar" vs
    "playing electric guitar"), one uniformly at random.

AVE  (UnFaZeD07/AVE-Dataset, mit)  official train / val / test lists
    ave:       28-way choice over the annotated audio-visual event of the clip (video cropped to the annotated event
               segment, at least 2 s, plus its own sound).
    ave_match: yes/no "does the sound belong to what is shown?". Every clip yields exactly one record, positive or
               negative by a fixed coin (hash of the video id). Positive: the clip's own sound. Negative: the sound
               of another clip *of the same split* whose annotated category differs and is not acoustically
               confusable with it (male/female speech; acoustic guitar/banjo/mandolin/ukulele; truck/bus/race car/
               motorcycle are never swapped with each other). The video is stored without sound and the (own or
               foreign) sound as a separate audio item: `<video:1>` + `<audio:1>`.
"""

from __future__ import annotations

import csv
import os
import re

from jevtrainer.data import avkit
from jevtrainer.data.avkit_video import (balanced_take, cached_item, convert_parquet_videos, convert_zip_videos, fetch_ranged,
                                         hash_bucket)
from jevtrainer.data.base import DatasetSpec, choice_record, hf_file, hf_listing, mcq_record, noul_record, register, rid

_WORKERS = int(os.environ.get("JEVTRAINER_AV_WORKERS", "8"))  # parallel ffmpeg jobs
_CONNS = int(os.environ.get("JEVTRAINER_AV_CONNS", "4"))  # parallel range requests per big file


# ---- VGGSound ------------------------------------------------------------------------------------------
_STOP = {"playing", "a", "an", "the", "of", "and", "in", "on", "with", "by", "to", "from"}


def _toks(label: str) -> set[str]:
    out = set()
    for w in re.findall(r"[a-z]+", label.lower()):
        if w in _STOP:
            continue
        w = re.sub(r"(ing|s)$", "", w) if len(w) > 4 else w
        out.add(w)
    return out


def _similar_classes(classes: list[str]) -> dict[str, list[str]]:
    toks = {c: _toks(c) for c in classes}
    sim = {}
    for c in classes:
        scored = []
        for d in classes:
            if d == c or not toks[c] or not toks[d]:
                continue
            j = len(toks[c] & toks[d]) / len(toks[c] | toks[d])
            if j > 0:
                scored.append((-j, d))
        sim[c] = [d for _, d in sorted(scored)[:6]]
    return sim


def _vgg_convert(ids: set[str]) -> None:
    """Shard by shard: ranged download of a ~1 GB parquet (video + sound bytes), convert every clip, delete the shard."""
    repo = "11hu83/vggsound"
    limit = int(os.environ.get("JEVTRAINER_VGG_SHARDS", "31"))
    marks = avkit.media_dir("vggsound", "_shards")
    for i in range(min(limit, 31)):
        fn = f"data/vggsound_test_{i:04d}.parquet"
        if (marks / f"{i:04d}.done").exists():
            continue
        path = fetch_ranged(repo, fn, "vggsound", conns=_CONNS)
        if path is None:
            continue
        n = convert_parquet_videos(path, "vggsound", "video_id", "video", {v: v for v in ids}, _WORKERS, frames=8)
        path.unlink(missing_ok=True)
        (marks / f"{i:04d}.done").write_text(str(n))
        print(f"[vggsound] shard {i}: {n} new clips", flush=True)


def vggsound(split, cap, rng):
    repo = "11hu83/vggsound"
    meta = []
    with open(hf_file(repo, "metadata.csv"), encoding="utf8") as f:  # one row per clip file of the repo
        for r in csv.DictReader(f):
            meta.append((r["file_name"].split("/")[1], r["text"].strip()))
    _vgg_convert({v for v, _ in meta})  # both splits at once, so the second split needs no download
    rows = [(vid, c) for vid, c in meta if (hash_bucket("vggsound", vid) < 10) == (split == "test") and cached_item("vggsound", vid)]
    classes = sorted({c for _, c in meta})
    sim = _similar_classes(classes)
    rows = balanced_take(rows, lambda r: r[1], cap, rng)
    for vid, label in rows:
        media = cached_item("vggsound", vid)
        hard = rng.sample(sim[label], min(2, len(sim[label])))
        rest = [c for c in classes if c != label and c not in hard]
        options = [label, *hard, *rng.sample(rest, 3 - len(hard))]
        order = list(range(4))
        rng.shuffle(order)
        rec = mcq_record(rid("vggsound", split, vid), {"clip": "<video:1>", "question": "What is making the sound in this clip?"},
                         "Looking at and listening to the clip, which option best describes the main sound-producing event?",
                         [options[i] for i in order], order.index(0), area="av", category=label)
        if rec:
            rec.media = [media]
            yield rec


# ---- AVE -----------------------------------------------------------------------------------------------
AVE_CLASSES = {
    "church_bell": "a church bell ringing",
    "male_speech": "a man speaking",
    "bark": "a dog barking",
    "airplane": "a fixed-wing aircraft (airplane) flying or taxiing",
    "race_car": "a race car driving, auto racing",
    "female_speech": "a woman speaking",
    "helicopter": "a helicopter flying",
    "violin": "a violin (fiddle) being played",
    "flute": "a flute being played",
    "ukulele": "a ukulele being played",
    "frying": "food frying in a pan",
    "truck": "a truck driving",
    "shofar": "a shofar (ram's-horn trumpet) being blown",
    "motorcycle": "a motorcycle riding",
    "acoustic_guitar": "an acoustic guitar being played",
    "train_horn": "a train sounding its horn",
    "clock": "a clock ticking or chiming",
    "banjo": "a banjo being played",
    "goat": "a goat bleating",
    "baby_cry": "a baby crying",
    "bus": "a bus driving",
    "chainsaw": "a chainsaw cutting",
    "cat": "a cat meowing",
    "horse": "a horse neighing or galloping",
    "toilet_flush": "a toilet flushing",
    "rodent": "a rodent (rat, mouse) squeaking or moving",
    "accordion": "an accordion being played",
    "mandolin": "a mandolin being played",
}
_AVE_NAME = {
    "Church bell": "church_bell", "Male speech, man speaking": "male_speech", "Bark": "bark",
    "Fixed-wing aircraft, airplane": "airplane", "Race car, auto racing": "race_car",
    "Female speech, woman speaking": "female_speech", "Helicopter": "helicopter", "Violin, fiddle": "violin",
    "Flute": "flute", "Ukulele": "ukulele", "Frying (food)": "frying", "Truck": "truck", "Shofar": "shofar",
    "Motorcycle": "motorcycle", "Acoustic guitar": "acoustic_guitar", "Train horn": "train_horn", "Clock": "clock",
    "Banjo": "banjo", "Goat": "goat", "Baby cry, infant cry": "baby_cry", "Bus": "bus", "Chainsaw": "chainsaw",
    "Cat": "cat", "Horse": "horse", "Toilet flush": "toilet_flush", "Rodents, rats, mice": "rodent",
    "Accordion": "accordion", "Mandolin": "mandolin",
}
_CONFUSABLE = [{"male_speech", "female_speech"}, {"acoustic_guitar", "banjo", "mandolin", "ukulele"},
               {"truck", "bus", "race_car", "motorcycle"}]


def _confusable(a: str, b: str) -> bool:
    return a == b or any(a in g and b in g for g in _CONFUSABLE)


def _ave_rows(split: str) -> list[dict]:
    repo = "UnFaZeD07/AVE-Dataset"
    f = {"train": "trainSet.txt", "val": "valSet.txt", "test": "testSet.txt"}[split]
    rows = []
    for line in open(hf_file(repo, f), encoding="utf8").read().splitlines():
        p = line.strip().split("&")
        if len(p) < 5 or p[0] not in _AVE_NAME:
            continue
        t0, t1 = float(p[3]), float(p[4])
        if t1 - t0 < 2.0:  # keep at least 2 s of picture and sound around the event
            t1 = min(10.0, t0 + 2.0)
            t0 = max(0.0, t1 - 2.0)
        rows.append({"vid": p[1], "cls": _AVE_NAME[p[0]], "t0": t0, "t1": t1})
    return rows


def _ave_clips(split: str, cap: int, rng) -> tuple[list[dict], dict[str, dict]]:
    allrows = [r for s in ("train", "val", "test") for r in _ave_rows(s)]  # one pass over the archive serves every split
    mark = avkit.media_dir("ave", "_done") / "ok"
    if not mark.exists():
        archive = fetch_ranged("UnFaZeD07/AVE-Dataset", "videos.zip", "ave", conns=_CONNS)
        if archive is None:
            raise RuntimeError("cannot download AVE videos.zip")
        wanted = {f"videos/{r['vid']}.mp4": (r["vid"], {"start": r["t0"], "end": r["t1"]}) for r in allrows}
        convert_zip_videos(archive, "ave", wanted, _WORKERS, frames=8)
        avkit.drop_raw("ave")
        mark.write_text("1")
    rows = _ave_rows(split)
    rng.shuffle(rows)
    rows = balanced_take(rows, lambda r: r["cls"], cap, rng)
    items = {r["vid"]: cached_item("ave", r["vid"]) for r in rows}
    return rows, {k: v for k, v in items.items() if v}


def ave(split, cap, rng):
    rows, items = _ave_clips(split, cap, rng)
    for r in rows:
        media = items.get(r["vid"])
        if not media:
            continue
        rec = choice_record(rid("ave", split, r["vid"]), {"clip": "<video:1>", "question": "Which audio-visual event happens in this clip?"},
                            "Watch and listen to the clip. Which event is both seen and heard?", dict(AVE_CLASSES), r["cls"],
                            area="av")
        rec.media = [media]
        yield rec


def ave_match(split, cap, rng):
    rows, items = _ave_clips(split, cap, rng)
    rows = [r for r in rows if r["vid"] in items]
    by_cls = {}
    for r in rows:
        by_cls.setdefault(r["cls"], []).append(r)
    for r in rows:
        own = items[r["vid"]]
        silent = {k: v for k, v in own.items() if k != "audio"}
        if "audio" not in own:
            continue
        positive = hash_bucket("ave_match", r["vid"], 2) == 0
        sound = own["audio"]
        if not positive:
            pool = [o for c, os_ in by_cls.items() if not _confusable(c, r["cls"]) for o in os_ if "audio" in items[o["vid"]]]
            if not pool:
                continue
            other = rng.choice(pool)
            sound = items[other["vid"]]["audio"]
        rec = noul_record(rid("ave_match", split, r["vid"]),
                          {"clip": "<video:1>", "sound": "<audio:1>", "question": "Does this sound come from what is shown in the video?"},
                          "Look at the video and listen to the separate sound. Is the sound the one produced by the scene on screen?",
                          positive, true="the sound belongs to the scene shown (same event)",
                          false="the sound comes from a different event than the scene shown", area="av")
        rec.media = [silent, {"type": "audio", "path": sound}]
        yield rec


register(DatasetSpec("vggsound", vggsound, ("train", "test"), "11hu83/vggsound", "cc-by-4.0", "av", multimodal=True,
                     description="VGGSound: what makes the sound in a 10 s clip (309 classes, 4-way, similar-class distractors)"))
register(DatasetSpec("ave", ave, ("train", "val", "test"), "UnFaZeD07/AVE-Dataset", "mit", "av", multimodal=True,
                     description="AVE: 28-way audio-visual event classification"))
register(DatasetSpec("ave_match", ave_match, ("train", "val", "test"), "UnFaZeD07/AVE-Dataset", "mit", "av", multimodal=True,
                     description="AVE-derived: does the (separate) sound match the video? negatives swap in the sound of another class"))
