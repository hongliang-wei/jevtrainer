"""General video understanding evaluation sets (eval_only). Suite "video-bench".

    mvbench          MVBench (OpenGVLab/MVBench): 20 temporal tasks x 200 questions, 8 frames
    tempcompass      TempCompass multi-choice (lmms-eval/TempCompass): 1580 questions on action / speed / direction / order / attribute change
    egoschema        EgoSchema Subset (lmms-eval/egoschema): the 500 publicly scored questions, 3 minute egocentric clips, 16 frames
    longvideobench   LongVideoBench validation (Jialuo21/LongVideoBench, ungated copy): 1337 questions, 16 frames, no interleaved subtitles
    video_mme        Video-MME (lmms-lab/Video-MME): 2700 questions over 900 videos up to one hour, 16 frames + the first 30 s of sound
    video_mme_sub    Video-MME with subtitles: same media, the state also carries a subtitle excerpt
    perceptiontest_val  Perception Test multiple-choice validation (lmms-eval/PerceptionTest_Val): videos sampled to 5000 questions, 8 frames

The long sets (egoschema, longvideobench, video_mme*) store 16 frames per clip; use `readout_options: {video_frames: 16}`
at evaluation time, otherwise jevtrainer.media keeps 8 of them (configs/eval/video_bench.yaml).
`source_id` carries the YouTube id when the clip comes from YouTube.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

from jevtrainer.data import avkit, benchkit, vidkit
from jevtrainer.data.base import DatasetSpec, register, rid
from jevtrainer.data.benchkit import LONG_FRAMES, Q_VIDEO, letter_index, mcq, strip_letter, text_index, youtube_source

_WORKERS = benchkit.WORKERS


def _state(question: str, **extra) -> dict:
    return {"clip": "<video:1>", "question": question, **extra}


# ---- MVBench --------------------------------------------------------------------------------------------------
# task json -> zip(s) of the video repo that hold its clips (OpenGVLab/MVBench README)
_MV_ZIPS = {
    "action_sequence": ["star"], "action_prediction": ["star"], "object_interaction": ["star"], "action_antonym": ["ssv2_video"],
    "fine_grained_action": ["Moments_in_Time_Raw"], "unexpected_action": ["FunQA_test"], "object_existence": ["clevrer"],
    "moving_direction": ["clevrer"], "moving_count": ["clevrer"], "moving_attribute": ["clevrer"], "counterfactual_inference": ["clevrer"],
    "object_shuffle": ["perception"], "action_count": ["perception"], "state_change": ["perception"], "character_order": ["perception"],
    "action_localization": ["sta"], "scene_transition": ["scene_qa"], "fine_grained_pose": ["data0613", "star"],
    "egocentric_navigation": ["vlnqa"], "episodic_reasoning": ["tvqa"],
}


def mvbench(split, cap, rng):
    repo = "OpenGVLab/MVBench"
    names = sorted(f for f in vidkit.listing(repo, "json/") if f.endswith(".json"))
    index: dict[str, dict[str, str]] = {}  # zip stem -> {video file name or frame folder -> member}
    frames_dir: dict[str, list[str]] = {}  # TVQA: folder -> frame members
    for z in sorted({z for zs in _MV_ZIPS.values() for z in zs}):
        mem = benchkit.zip_names(repo, f"video/{z}.zip")
        d = index.setdefault(z, {})
        for m in mem:
            if m.endswith("/"):
                continue
            if m.lower().endswith((".jpg", ".png")):
                frames_dir.setdefault(Path(m).parent.name, []).append(m)
            else:
                d.setdefault(m.split("/", 1)[1] if "/" in m else m, m)
                d.setdefault(Path(m).name, m)
    rows = []
    for fn in names:
        task = Path(fn).stem
        if task not in _MV_ZIPS:
            continue
        for i, r in enumerate(vidkit.read_json(repo, fn)):
            r["_task"], r["_i"] = task, i
            for z in _MV_ZIPS[task]:
                member = index[z].get(r["video"]) or index[z].get(Path(r["video"]).name)
                if member or r["video"] in frames_dir:
                    r["_zip"], r["_member"] = z, member
                    break
            if "_zip" in r:
                rows.append(r)
    rng.shuffle(rows)
    rows = rows[:cap]
    jobs, frame_jobs = {}, {}
    for r in rows:
        key = f"{r['_zip']}__{Path(r['video']).stem}" + (f"_{r['start']:.1f}_{r['end']:.1f}" if r.get("start") is not None and r["_task"] == "action_localization" else "")
        r["_key"] = key
        if r["video"] in frames_dir and not r["_member"]:
            frame_jobs[key] = r
        else:
            extra = {"start": r["start"], "end": r["end"]} if r["_task"] == "action_localization" and r.get("end") else {}
            jobs[key] = {"repo": repo, "zip": f"video/{r['_zip']}.zip", "member": r["_member"], **extra}
    items = benchkit.convert_zip_members("mvbench", jobs, _WORKERS, frames=8)
    for key, r in frame_jobs.items():  # TVQA: a folder of frames
        item = vidkit.cached_item("mvbench", key)
        if not item:
            members = sorted(frames_dir[r["video"]])
            tmp = avkit.raw_dir("mvbench") / key
            paths = [benchkit.extract_member(repo, f"video/{r['_zip']}.zip", m, tmp / Path(m).name) for m in _spread(members, 8)]
            item = benchkit.image_frames_item([p for p in paths if p], "mvbench", key, n=8, fps=3.0)
            if item:
                item["duration"] = round(len(members) / 3.0, 3)
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)
        if item:
            items[key] = item
    for r in rows:
        opts = [str(c).strip() for c in r["candidates"]]
        extra = {"subtitle": r["subtitle"][:1200]} if r.get("subtitle") else {}
        rec = mcq(rid("mvbench", r["_task"], r["_i"]), _state(r["question"], **extra), Q_VIDEO, opts, text_index(r["answer"], opts),
                  [items.get(r["_key"])], "video", category=r["_task"], task_type=r["_task"], source_id=f"mvbench:{r['_key']}")
        if rec:
            yield rec


def _spread(xs: list, n: int) -> list:
    return [xs[int((i + 0.5) * len(xs) / n)] for i in range(n)] if len(xs) > n else list(xs)


# ---- TempCompass (multi-choice) -----------------------------------------------------------------------------------
def tempcompass(split, cap, rng):
    repo = "lmms-eval/TempCompass"
    rows = vidkit.read_parquet(repo, "multi-choice/test-00000-of-00001.parquet").to_dict("records")
    rng.shuffle(rows)
    rows = rows[:cap]
    zpath = avkit.fetch_file(repo, "tempcompass_videos.zip", "tempcompass")
    need = {r["video_id"] for r in rows}
    jobs = {}
    raw = avkit.raw_dir("tempcompass") / "videos"
    raw.mkdir(exist_ok=True)
    with zipfile.ZipFile(zpath) as z:
        for vid in need:
            if vidkit.cached_item("tempcompass", vid):
                continue
            m = f"videos/{vid}.mp4"
            if m in z.namelist():
                dest = raw / f"{vid}.mp4"
                dest.write_bytes(z.read(m))
                jobs[vid] = {"src": dest}
    items = vidkit.convert_local("tempcompass", jobs, _WORKERS, frames=8)
    Path(zpath).unlink(missing_ok=True)
    for vid in need:
        if vid not in items and vidkit.cached_item("tempcompass", vid):
            items[vid] = vidkit.cached_item("tempcompass", vid)
    for i, r in enumerate(rows):
        lines = r["question"].split("\n")
        stem = [ln for ln in lines if not re.match(r"^[A-Z][\.\)]\s", ln)]
        opts = [strip_letter(ln) for ln in lines if re.match(r"^[A-Z][\.\)]\s", ln)]
        rec = mcq(rid("tempcompass", r["video_id"], i, r["question"][:60]), _state(" ".join(stem).strip()), Q_VIDEO, opts,
                  text_index(r["answer"], opts), [items.get(r["video_id"])], "video", category=r["dim"], task_type=r["dim"],
                  source_id=f"tempcompass:{r['video_id'].replace('_reverse', '')}")
        if rec:
            yield rec


# ---- EgoSchema (Subset, 500 scored questions) ------------------------------------------------------------------------
def egoschema(split, cap, rng):
    repo = "lmms-eval/egoschema"
    rows = vidkit.read_parquet(repo, "Subset/test-00000-of-00001.parquet").to_dict("records")
    rng.shuffle(rows)
    rows = rows[:cap]
    index = {}
    for z in sorted(f for f in vidkit.listing(repo) if f.startswith("videos_chunked_")):
        for m in benchkit.zip_names(repo, z):
            if m.endswith(".mp4"):
                index[Path(m).stem] = (z, m)
    jobs = {r["video_idx"]: {"repo": repo, "zip": index[r["video_idx"]][0], "member": index[r["video_idx"]][1]}
            for r in rows if r["video_idx"] in index}
    items = benchkit.convert_zip_members("egoschema", jobs, _WORKERS, frames=LONG_FRAMES, long=True, keep_audio=False)
    for r in rows:
        opts = [strip_letter(o) for o in list(r["option"])]
        rec = mcq(rid("egoschema", r["question_idx"]), _state(r["question"]), Q_VIDEO, opts, int(r["answer"]),
                  [items.get(r["video_idx"])], "video", category="egoschema", source_id=f"egoschema:{r['video_idx']}")
        if rec:
            yield rec


# ---- LongVideoBench (validation) ------------------------------------------------------------------------------------
def longvideobench(split, cap, rng):
    import json

    repo = "Jialuo21/LongVideoBench"
    rows = json.load(open(vidkit.get_file(repo, "lvb_val.json"), encoding="utf8"))
    rng.shuffle(rows)
    rows = rows[:cap]
    have = vidkit.listing(repo, "videos/")
    files = {r["video_id"]: f"videos/{Path(r['video_path']).name}" for r in rows if f"videos/{Path(r['video_path']).name}" in have}
    items = benchkit.convert_hub_long(repo, "longvideobench", files, LONG_FRAMES, _WORKERS, audio_s=30.0)
    for r in rows:
        opts = [str(c).strip() for c in r["candidates"]]
        rec = mcq(rid("longvideobench", r["id"]), _state(r["question"]), Q_VIDEO, opts, int(r["correct_choice"]), [items.get(r["video_id"])],
                  "video", category=r["question_category"], task_type=r["level"], domain=r["topic_category"],
                  duration=f"{r['duration_group']}s", source_id=youtube_source(r["video_id"]) or f"longvideobench:{r['video_id']}")
        if rec:
            yield rec


# ---- Video-MME (with and without subtitles) -------------------------------------------------------------------------
_SRT_TIME = re.compile(r"^\d+:\d\d:\d\d")


def _subtitle_excerpt(srt: str, max_chars: int = 1500) -> str:
    lines = [ln.strip() for ln in srt.splitlines() if ln.strip() and not ln.strip().isdigit() and "-->" not in ln and not _SRT_TIME.match(ln)]
    text = " ".join(lines)
    if len(text) <= max_chars:
        return text
    n = 8  # evenly spaced excerpts so that the whole video is represented
    step = len(text) // n
    return " … ".join(text[i * step: i * step + max_chars // n] for i in range(n))


def _video_mme(name: str, with_subtitles: bool, split, cap, rng):
    repo = "lmms-lab/Video-MME"
    rows = vidkit.read_parquet(repo, "videomme/test-00000-of-00001.parquet").to_dict("records")
    rng.shuffle(rows)
    rows = rows[:cap]
    index = {}
    for z in sorted(f for f in vidkit.listing(repo) if f.startswith("videos_chunked_")):
        for m in benchkit.zip_names(repo, z):
            if m.endswith(".mp4"):
                index[Path(m).stem] = (z, m)
    vids = {r["videoID"] for r in rows}
    jobs = {v: {"repo": repo, "zip": index[v][0], "member": index[v][1]} for v in vids if v in index}
    items = benchkit.convert_zip_members("video_mme", jobs, _WORKERS, frames=LONG_FRAMES, long=True, audio_s=30.0)
    subs = {}
    if with_subtitles:
        for v in vids:
            try:
                dest = benchkit.extract_member(repo, "subtitle.zip", f"subtitle/{v}.srt", avkit.raw_dir(name) / f"{v}.srt")
            except KeyError:
                dest = None
            if dest:
                subs[v] = _subtitle_excerpt(dest.read_text(encoding="utf8", errors="ignore"))
                dest.unlink()
    for r in rows:
        opts = [strip_letter(o) for o in list(r["options"])]
        extra = {"subtitles": subs[r["videoID"]]} if with_subtitles and subs.get(r["videoID"]) else {}
        rec = mcq(rid(name, r["question_id"]), _state(r["question"], **extra), Q_VIDEO, opts, letter_index(r["answer"], len(opts)),
                  [items.get(r["videoID"])], "video", category=r["task_type"], task_type=r["task_type"], domain=r["domain"],
                  duration=r["duration"], sub_category=r["sub_category"], source_id=youtube_source(r["videoID"]) or f"video_mme:{r['videoID']}")
        if rec:
            yield rec


def video_mme(split, cap, rng):
    yield from _video_mme("video_mme", False, split, cap, rng)


def video_mme_sub(split, cap, rng):
    yield from _video_mme("video_mme_sub", True, split, cap, rng)


# ---- Perception Test (multiple-choice validation) --------------------------------------------------------------------
PT_MAX_QUESTIONS = 5000


def perceptiontest_val(split, cap, rng):
    repo = "lmms-eval/PerceptionTest_Val"
    rows = vidkit.read_parquet(repo, "mc_question_val/validation-00000-of-00001.parquet").to_dict("records")
    by_video: dict[str, list] = {}
    for r in rows:
        by_video.setdefault(r["video_name"], []).append(r)
    names = sorted(by_video)
    rng.shuffle(names)
    picked, n = [], 0
    for v in names:  # whole videos until the cap is reached
        if n >= min(cap, PT_MAX_QUESTIONS):
            break
        picked.append(v)
        n += len(by_video[v])
    index = {}
    for z in sorted(f for f in vidkit.listing(repo) if f.startswith("videos_chunked_")):
        for m in benchkit.zip_names(repo, z):
            if m.endswith(".mp4"):
                index[Path(m).stem] = (z, m)
    jobs = {v: {"repo": repo, "zip": index[v][0], "member": index[v][1]} for v in picked if v in index}
    items = benchkit.convert_zip_members("perceptiontest_val", jobs, _WORKERS, frames=8)
    for v in picked:
        for r in by_video[v]:
            opts = [str(o).strip() for o in list(r["options"])]
            rec = mcq(rid("perceptiontest_val", v, r["question_id"]), _state(r["question"]), Q_VIDEO, opts, int(r["answer_id"]),
                      [items.get(v)], "video", category=r["area"], task_type=r["reasoning"], sub_category=(list(r["tag"]) or [None])[0],
                      source_id=f"perceptiontest:{v}")
            if rec:
                yield rec


register(DatasetSpec("mvbench", mvbench, ("test",), "OpenGVLab/MVBench", "mit", "video", eval_only=True, multimodal=True,
                     description="MVBench: 20 temporal video understanding tasks (eval only)"))
register(DatasetSpec("tempcompass", tempcompass, ("test",), "lmms-eval/TempCompass", "other (research)", "video", eval_only=True,
                     multimodal=True, description="TempCompass multi-choice: temporal aspects of short videos (eval only)"))
register(DatasetSpec("egoschema", egoschema, ("test",), "lmms-eval/egoschema", "other (research)", "video", eval_only=True, multimodal=True,
                     description="EgoSchema Subset: 500 long-form egocentric questions, 16 frames (eval only)"))
register(DatasetSpec("longvideobench", longvideobench, ("val",), "Jialuo21/LongVideoBench", "cc-by-nc-sa-4.0", "video", eval_only=True,
                     multimodal=True, description="LongVideoBench validation, 16 frames, no subtitles (eval only)"))
register(DatasetSpec("video_mme", video_mme, ("test",), "lmms-lab/Video-MME", "other (research)", "video", eval_only=True, multimodal=True,
                     description="Video-MME: 16 frames and the first 30 s of the video's sound (eval only)"))
register(DatasetSpec("video_mme_sub", video_mme_sub, ("test",), "lmms-lab/Video-MME", "other (research)", "video", eval_only=True,
                     multimodal=True, description="Video-MME with a subtitle excerpt in the state (eval only)"))
register(DatasetSpec("perceptiontest_val", perceptiontest_val, ("val",), "lmms-eval/PerceptionTest_Val", "cc-by-4.0", "video",
                     eval_only=True, multimodal=True, description="Perception Test multiple-choice validation, <= 5000 questions (eval only)"))
