"""VideoGUI (Sun et al. 2024) high-level planning: from the screen recordings of a task's start and end state.

`VideoGUI/VideoGUI-High-Plan` holds, for each of ~80 software tasks (Photoshop, After Effects, PowerPoint, Premiere, Runway,
Stable Diffusion, ...), a `start.mp4` and `end.mp4` clip plus the full-task query and the ordered sub-task sequence. Both clips
are given to the model (2 video items; screen recordings have no sound).

    videogui_goal   which task query describes the change from the first clip to the second (4 options, distractors = other
                    tasks of the same application)
    videogui_plan   given the task query, which sub-task sequence turns the first clip into the second (distractors = sequences
                    of other tasks of the same application)

Split by task (hash of task id): ~12 % test, ~10 % val, rest train (the corpus is tiny: ~80 tasks).
"""

from __future__ import annotations

import re
import threading

from jevtrainer.data import avkit, vidkit
from jevtrainer.data.avkit_video import cached_item, hash_bucket
from jevtrainer.data.base import DatasetSpec, mcq_record, register, rid

_REPO = "VideoGUI/VideoGUI-High-Plan"
_LOCK = threading.Lock()
_VID = dict(frames=8, max_side=640, keep_audio=False)


def _split(task: str) -> str:
    b = hash_bucket("videogui", task)
    return "test" if b < 12 else "val" if b < 22 else "train"


def _steps(seq) -> list[str]:
    if isinstance(seq, str):
        seq = re.split(r"\s*(?:\n|;\s*(?=[A-Z\"'‘“]))\s*", seq.strip("[] \n;"))
    out = [re.sub(r"\s+", " ", str(s)).strip(" \"';") for s in list(seq)]
    return [s for s in out if s]


def _tasks() -> list[dict]:
    files = vidkit.listing(_REPO, "high_plan/")
    rows = []
    for task in sorted({f.split("/")[1] for f in files}):
        if f"high_plan/{task}/metadata.parquet" not in files:
            continue
        df = vidkit.read_parquet(_REPO, f"high_plan/{task}/metadata.parquet")
        r = df.to_dict("records")[0]
        sp, ep = f"high_plan/{task}/start.mp4", f"high_plan/{task}/end.mp4"
        steps = _steps(r["subtask_sequence"])
        if sp in files and ep in files and steps and str(r["fulltask_query"]).strip():
            rows.append({"task": task, "app": str(r["app"]), "query": re.sub(r"\s+", " ", str(r["fulltask_query"])).strip(), "steps": steps,
                         "start": sp, "end": ep})
    return rows


def _media(rows: list[dict]) -> None:
    todo = {}
    for r in rows:
        for k in ("start", "end"):
            if not cached_item("videogui", f"{r['task']}_{k}"):
                todo[f"{r['task']}_{k}"] = r[k]
    if todo:
        with _LOCK:
            vidkit.convert_hub(_REPO, "videogui", todo, workers=4, **_VID)


def _plan_text(steps: list[str]) -> str:
    t = " ".join(f"{i + 1}) {s}" for i, s in enumerate(steps))
    return t if len(t) <= 600 else t[:597] + "..."


def _records(kind: str, split: str, cap: int, rng):
    all_rows = _tasks()
    rows = [r for r in all_rows if _split(r["task"]) == split]
    rng.shuffle(rows)
    rows = rows[:cap]
    _media(rows)
    for r in rows:
        a, b = cached_item("videogui", f"{r['task']}_start"), cached_item("videogui", f"{r['task']}_end")
        if not a or not b:
            continue
        same = [o for o in all_rows if o["app"] == r["app"] and o["task"] != r["task"]]
        pool = same if len(same) >= 3 else [o for o in all_rows if o["task"] != r["task"]]
        if len(pool) < 3:
            continue
        others = rng.sample(pool, 3)
        if kind == "goal":
            opts, gold_txt = [o["query"] for o in others], r["query"]
            state = {"start": "<video:1>", "end": "<video:2>", "question": "The first clip shows a software screen before a task, the second after it. Which task was performed?"}
            instr = "Which task query describes what was done between the first and the second screen recording?"
        else:
            opts, gold_txt = [_plan_text(o["steps"]) for o in others], _plan_text(r["steps"])
            state = {"start": "<video:1>", "end": "<video:2>", "task": r["query"],
                     "question": "The first clip shows the screen before the task, the second after it. Which sub-task sequence carries out the task?"}
            instr = "Which ordered sequence of sub-tasks accomplishes the task and leads from the first to the second recording?"
        options = [gold_txt, *opts]
        order = list(range(4))
        rng.shuffle(order)
        rec = mcq_record(rid("videogui", kind, split, r["task"]), state, instr, [options[i] for i in order], order.index(0),
                         area="video", dataset=f"videogui_{kind}", app=r["app"], task=r["task"])
        if rec:
            rec.media = [a, b]
            yield rec


def videogui_goal(split, cap, rng):
    yield from _records("goal", split, cap, rng)


def videogui_plan(split, cap, rng):
    yield from _records("plan", split, cap, rng)


register(DatasetSpec("videogui_goal", videogui_goal, ("train", "val", "test"), _REPO, "mit", "video", multimodal=True, version="2",
                     description="VideoGUI: which task query explains the change between a start and an end screen recording"))
register(DatasetSpec("videogui_plan", videogui_plan, ("train", "val", "test"), _REPO, "mit", "video", multimodal=True, version="2",
                     description="VideoGUI: which sub-task sequence turns the start recording into the end recording"))
