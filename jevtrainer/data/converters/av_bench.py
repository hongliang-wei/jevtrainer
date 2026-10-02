"""Audio-visual evaluation sets (eval_only): the sound track is needed to answer. Suite "av-omni".

    worldsense      WorldSense (lmms-lab/WorldSense): 1662 videos / 3172 questions, 8 frames + the video's sound
    daily_omni      Daily-Omni (liarliar/Daily-Omni + xxayt/Daily-Omni metadata): 1197 questions over 30 s / 60 s clips
    omnibench       OmniBench (m-a-p/OmniBench): one picture + one sound per question, 1142 questions
    av_odyssey      AV-Odyssey Bench: sound / picture / video combinations, 4555 questions
    av_speakerbench AV-SpeakerBench: speaker-aware questions over short clips, 3212 questions

Not convertible here: AVI-Bench (FudanCVL/AVIBench is gated and the token has no access), AVHBench (the hub copy
MYMY-young/AVHBench has the questions but not the videos).
Every record carries `source_id` (YouTube id when the set uses YouTube clips) for the train / eval overlap check.
"""

from __future__ import annotations

import ast
import json
import re
import tarfile
from pathlib import Path

from jevtrainer.data import avkit, benchkit, vidkit
from jevtrainer.data.base import DatasetSpec, register, rid
from jevtrainer.data.benchkit import Q_AV, mcq, strip_letter, letter_index, text_index, youtube_source

_WORKERS = benchkit.WORKERS


def _video_state(question: str) -> dict:
    return {"clip": "<video:1>", "question": question}


def _options(v) -> list[str]:
    return [strip_letter(o) for o in list(v)]


# ---- WorldSense -----------------------------------------------------------------------------------------
def worldsense(split, cap, rng):
    repo = "lmms-lab/WorldSense"
    rows = vidkit.read_parquet(repo, "data/test-00000-of-00001.parquet").to_dict("records")
    rng.shuffle(rows)
    rows = rows[:cap]
    index = {}  # video id -> (zip, member): the sound is part of these mp4 files
    for z in sorted(f for f in vidkit.listing(repo) if f.startswith("videos_chunk_")):
        for m in benchkit.zip_names(repo, z):
            if m.endswith(".mp4"):
                index[Path(m).stem] = (z, m)
    jobs = {r["video"]: {"repo": repo, "zip": index[r["video"]][0], "member": index[r["video"]][1]}
            for r in rows if r["video"] in index}
    items = benchkit.convert_zip_members("worldsense", jobs, _WORKERS, frames=8)
    for r in rows:
        opts = _options(r["candidates"])
        rec = mcq(rid("worldsense", r["index"]), _video_state(r["question"]), Q_AV, opts, letter_index(r["answer"], len(opts)),
                  [items.get(r["video"])], "av", category=r["task_domain"], task_type=r["task_type"], domain=r["domain"],
                  duration=r["duration"], sub_category=r["sub_category"], audio_class=list(r["audio_class"]),
                  source_id=f"worldsense:{r['video']}")
        if rec:
            yield rec


# ---- Daily-Omni -------------------------------------------------------------------------------------------
def daily_omni(split, cap, rng):
    repo = "liarliar/Daily-Omni"
    rows = vidkit.read_json(repo, "qa.json")
    rng.shuffle(rows)
    rows = rows[:cap]
    tar_path = avkit.fetch_file(repo, "Videos.tar", "daily_omni")
    wanted = {r["video_id"] for r in rows}
    jobs = {}
    raw = avkit.raw_dir("daily_omni") / "videos"
    raw.mkdir(exist_ok=True)
    with tarfile.open(tar_path) as t:
        pick = {}
        for m in t.getmembers():
            if m.isfile() and m.name.lower().endswith(".mp4"):
                for vid in wanted:
                    if vid in m.name:
                        pick.setdefault(vid, m)
        for vid, m in pick.items():
            if vidkit.cached_item("daily_omni", vid):
                continue
            dest = raw / f"{vid}.mp4"
            with t.extractfile(m) as src, open(dest, "wb") as out:
                out.write(src.read())
            jobs[vid] = {"src": dest}
    items = vidkit.convert_local("daily_omni", jobs, _WORKERS, frames=8)
    Path(tar_path).unlink(missing_ok=True)
    for vid in wanted:
        if vid not in items:
            cached = vidkit.cached_item("daily_omni", vid)
            if cached:
                items[vid] = cached
    for i, r in enumerate(rows):
        opts = _options(r["Choice"])
        rec = mcq(rid("daily_omni", r["video_id"], i, r["Question"][:40]), _video_state(r["Question"]), Q_AV, opts,
                  letter_index(r["Answer"], len(opts)), [items.get(r["video_id"])], "av", category=r["Type"], task_type=r["Type"],
                  domain=r.get("content_parent_category"), duration=r.get("video_duration"), sub_category=r.get("content_fine_category"),
                  video_category=r.get("video_category"), source_id=youtube_source(r["video_id"]) or f"daily_omni:{r['video_id']}")
        if rec:
            yield rec


# ---- OmniBench ----------------------------------------------------------------------------------------------
def omnibench(split, cap, rng):
    repo = "m-a-p/OmniBench"
    files = avkit.parquet_files(repo, "data/")

    def handle(r):
        key = f"{r['index']}"
        img = _image(r["image"], "omnibench", key)
        aud = avkit.audio_from_bytes(benchkit.blob(r["audio"]), "omnibench", key, 30.0)
        opts = [str(o).strip() for o in r["options"]]
        return mcq(rid("omnibench", r["index"]), {"image": "<image:1>", "audio": "<audio:1>", "question": r["question"]},
                   "Which option correctly answers the question about the picture and the sound?", opts,
                   text_index(r["answer"], opts), [img, aud], "av", category=r["task type"], task_type=r["task type"],
                   modality=r["audio type"], source_id=f"omnibench:{r['image_path']}")

    recs = benchkit.parquet_media(repo, files, "omnibench", handle)
    rng.shuffle(recs)
    yield from recs[:cap]


def _image(v, name: str, key: str) -> dict | None:
    import io

    from PIL import Image

    data = benchkit.blob(v)
    return avkit.image_item(Image.open(io.BytesIO(data)), name, key) if data else None


# ---- AV-Odyssey -------------------------------------------------------------------------------------------
_TAGS = [(re.compile(r"\[\s*audio\s*(\d)\s*\]", re.I), r"<audio:\1>"), (re.compile(r"\[\s*img\s*(\d)\s*\]", re.I), r"<image:\1>"),
         (re.compile(r"\[\s*video\s*(\d)\s*\]", re.I), r"<video:\1>")]


def av_odyssey(split, cap, rng):
    repo = "AV-Odyssey/AV_Odyssey_Bench"
    files = sorted(f for f in vidkit.listing(repo) if re.match(r"^av_odyssey_part\d+\.parquet$", f))

    def handle(r):
        key = str(r["question_id"])
        media, kinds = [], {"image": [], "video": [], "audio": []}
        for i in range(1, 5):
            if r.get(f"image_{i}"):
                kinds["image"].append(_image(r[f"image_{i}"], "av_odyssey", f"{key}_i{i}"))
        for i in range(1, 5):
            if r.get(f"video_{i}"):
                kinds["video"].append(_video_bytes(benchkit.blob(r[f"video_{i}"]), "av_odyssey", f"{key}_v{i}"))
        for i in range(1, 5):
            if r.get(f"audio_{i}"):
                kinds["audio"].append(avkit.audio_from_bytes(benchkit.blob(r[f"audio_{i}"]), "av_odyssey", f"{key}_a{i}", 30.0))
        media = kinds["image"] + kinds["video"] + kinds["audio"]
        q = r["question"]
        for pat, rep in _TAGS:
            q = pat.sub(rep, q)
        # media the question never mentions are placed in front of it by jevtrainer.media
        opts = [str(o).strip() for o in r["options"]]
        return mcq(rid("av_odyssey", key), {"question": q}, "Which option correctly answers the question about the given pictures, "
                   "videos and sounds?", opts, letter_index(r["answer"], len(opts)), media, "av",
                   category=r["subfield"], task_type=str(r["question_type_id"]), modality=r["data_type"], source_id=f"av_odyssey:{key}")

    recs = benchkit.parquet_media(repo, files, "av_odyssey", handle, shards=2, inner=3, batch=16)
    rng.shuffle(recs)
    yield from recs[:cap]


def _video_bytes(data: bytes | None, name: str, key: str) -> dict | None:
    if not data:
        return None
    src = avkit.raw_dir(name) / f"{key}.mp4"
    src.write_bytes(data)
    try:
        return vidkit.finish(name, key, src, frames=8)
    finally:
        src.unlink(missing_ok=True)


# ---- AV-SpeakerBench ---------------------------------------------------------------------------------------
def av_speakerbench(split, cap, rng):
    repo = "plnguyen2908/AV-SpeakerBench"
    df = vidkit.read_csv(repo, "test.csv")
    rows = [r for r in df.to_dict("records") if r.get("mp4_ok", True)]
    rng.shuffle(rows)
    rows = rows[:cap]
    index = {}  # clip name -> (zip, member)
    for z in sorted(f for f in vidkit.listing(repo, "audiovisual_chunk_") if f.endswith(".zip")):
        for m in benchkit.zip_names(repo, z):
            if m.endswith(".mp4"):
                index[Path(m).stem] = (z, m)
    key = lambda r: Path(str(r["audio_visual_path"])).stem  # noqa: E731
    jobs = {key(r): {"repo": repo, "zip": index[key(r)][0], "member": index[key(r)][1]} for r in rows if key(r) in index}
    items = benchkit.convert_zip_members("av_speakerbench", jobs, _WORKERS, frames=8)
    for r in rows:
        opts = _options(ast.literal_eval(r["choices"]) if isinstance(r["choices"], str) else r["choices"])
        rec = mcq(rid("av_speakerbench", r["question_id"]), _video_state(r["question"]), Q_AV, opts, letter_index(r["answer"], len(opts)),
                  [items.get(key(r))], "av", category=r["category"], task_type=r["task_id"], sub_category=r["sub_category"],
                  source_id=youtube_source(r["video_id"]) or f"av_speakerbench:{r['video_id']}")
        if rec:
            yield rec


register(DatasetSpec("worldsense", worldsense, ("test",), "lmms-lab/WorldSense", "cc-by-nc-sa-4.0", "av", eval_only=True, multimodal=True,
                     description="WorldSense: omni-modal video questions that need the audio track (eval only)"))
register(DatasetSpec("daily_omni", daily_omni, ("test",), "liarliar/Daily-Omni", "apache-2.0", "av", eval_only=True, multimodal=True,
                     description="Daily-Omni: audio-visual event alignment, comparison, reasoning over 30 s / 60 s clips (eval only)"))
register(DatasetSpec("omnibench", omnibench, ("test",), "m-a-p/OmniBench", "apache-2.0", "av", eval_only=True, multimodal=True,
                     description="OmniBench: one image and one audio per question (eval only)"))
register(DatasetSpec("av_odyssey", av_odyssey, ("test",), "AV-Odyssey/AV_Odyssey_Bench", "cc-by-sa-4.0", "av", eval_only=True,
                     multimodal=True, description="AV-Odyssey Bench: audio-visual deaf-test style multiple choice (eval only)"))
register(DatasetSpec("av_speakerbench", av_speakerbench, ("test",), "plnguyen2908/AV-SpeakerBench", "cc-by-nc-4.0", "av", eval_only=True,
                     multimodal=True, description="AV-SpeakerBench: speaker-centric audio-visual questions (eval only)"))
