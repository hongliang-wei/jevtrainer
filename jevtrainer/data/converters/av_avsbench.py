"""AVSBench (Zhou et al. 2022): which sound source is audible (and visible) in a 5 s clip? 23 sounding-object classes.

Source: `UnFaZeD07/AVSBench` (the single-source S4 subset, one tar.gz of 4,932 five-second mp4 clips with their sound).
Official train / val / test lists come from `s4_meta_data.csv` (3452 / 740 / 740). One streaming pass over the first
3.6 GB of the archive (it holds all the mp4 clips) converts every wanted clip (frames + sound); the partial archive is
deleted at the end.
"""

from __future__ import annotations

import threading

from jevtrainer.data import avkit, vidkit
from jevtrainer.data.avkit_video import cached_item, fetch_ranged, convert_tar_videos
from jevtrainer.data.base import DatasetSpec, choice_record, register, rid

_REPO = "UnFaZeD07/AVSBench"
_LOCK = threading.Lock()

LABELS = {
    "helicopter": "helicopter: rotor blades chopping the air",
    "female_singing": "female singing: a woman singing, with or without music",
    "playing_acoustic_guitar": "playing acoustic guitar",
    "playing_piano": "playing piano",
    "playing_violin": "playing violin",
    "typing_on_computer_keyboard": "typing on a computer keyboard",
    "playing_tabla": "playing tabla (Indian hand drums)",
    "dog_barking": "dog barking",
    "lions_roaring": "lion roaring",
    "chainsawing_trees": "chainsawing trees: a chainsaw cutting wood",
    "male_speech": "male speech: a man talking",
    "playing_glockenspiel": "playing glockenspiel (metal bell percussion)",
    "race_car": "race car: a racing car engine passing",
    "mynah_bird_singing": "mynah bird singing or calling",
    "playing_ukulele": "playing ukulele",
    "ambulance_siren": "ambulance siren",
    "horse_clip-clop": "horse hooves clip-clopping",
    "cat_meowing": "cat meowing",
    "driving_buses": "driving buses: a bus engine and road noise",
    "cap_gun_shooting": "cap gun shooting: toy gun pops",
    "baby_laughter": "baby laughing",
    "lawn_mowing": "lawn mowing: a mower engine",
    "coyote_howling": "coyote howling",
}


def _rows():
    df = vidkit.read_csv(_REPO, "s4_meta_data.csv")
    return [{"name": r["name"], "cat": r["category"], "split": r["split"],
             "member": f"s4_data/raw_videos/{r['split']}/{r['category']}/{r['name']}.mp4"} for r in df.to_dict("records")
            if r["category"] in LABELS]


def _prepare(rows):
    """Convert all clips that are not converted yet (one pass over the archive)."""
    todo = {r["member"]: (f"{r['split']}_{r['name']}", {}) for r in rows if not cached_item("avsbench", f"{r['split']}_{r['name']}")}
    if not todo:
        return
    # the 4,932 raw_videos come first in the tar (3.4 GB of 7.4 GB); the rest (frames, masks, spectrograms) is not needed
    arc = fetch_ranged(_REPO, "s4_data.tar.gz", "avsbench", conns=6, max_bytes=3_600_000_000)
    if arc is None:
        raise RuntimeError("could not download s4_data.tar.gz")
    convert_tar_videos(arc, "avsbench", todo, workers=6, by_basename=False, frames=8, max_side=448, audio_s=6)
    arc.unlink(missing_ok=True)


def avsbench(split, cap, rng):
    every = _rows()
    with _LOCK:
        _prepare(every)  # all splits in one pass: the archive is deleted afterwards
    rows = [r for r in every if r["split"] == split]
    rows = avkit.balanced(rows, lambda r: r["cat"], cap, rng)
    for r in rows:
        m = cached_item("avsbench", f"{r['split']}_{r['name']}")
        if not m:
            continue
        rec = choice_record(rid("avsbench", split, r["name"]),
                            {"clip": "<video:1>", "question": "Which object or being is making the sound in this clip?"},
                            "Watch and listen to the 5-second clip. Which sound source is both audible and visible?", dict(LABELS), r["cat"],
                            area="av", dataset="avsbench")
        rec.media = [m]
        yield rec


register(DatasetSpec("avsbench", avsbench, ("train", "val", "test"), _REPO, "other (research only)", "av", multimodal=True,
                     description="AVSBench single-source clips: which of 23 sound sources is audible and visible (5 s mp4 with sound)"))
