"""GUI, browser and computer-use agents with screenshots.

Two decision shapes:

* element choice: candidate boxes are drawn on the screenshot with numbers (Set-of-Mark) and
  the question asks which mark the next action targets. Used when the source has element boxes.
* next-step choice: for coordinate-only trajectories (AGUVIS), options are natural-language step
  descriptions; distractors are later steps of the same task, then steps of other tasks.
  A second question on the same state asks for the action type.

Screenshots are resized to MAX_SIDE on the long side before caching, which bounds vision tokens.
"""

from __future__ import annotations

import base64
import io
import json
import re
import zipfile
from collections import Counter, defaultdict

from jevtrainer.data.base import DatasetSpec, cache_dir, register, rid
from jevtrainer.data.converters.agents import _elem
from jevtrainer.schema import Question, Record, Target

MAX_SIDE = 1280
MARK_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#008080", "#9a6324", "#800000", "#000075", "#f032e6"]
ELEMENT_Q = "The screenshot shows candidate elements outlined and numbered. Which numbered element should the next action target?"
STEP_Q = "Which step should the agent take next to make progress on the task?"
TYPE_Q = "What kind of action should the agent take next?"


def _dl(repo, path):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, path, repo_type="dataset")


def save_image(img, name: str, key: str) -> str:
    d = cache_dir() / "images" / name
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{key}.jpg"
    if not p.exists():
        img = img.convert("RGB")
        s = MAX_SIDE / max(img.size)
        if s < 1:
            img = img.resize((round(img.width * s), round(img.height * s)))
        img.save(p, quality=90)
    return str(p)


def draw_marks(img, boxes):
    """Outline each (x, y, w, h) box and put its 1-based number at the top-left corner."""
    from PIL import ImageDraw, ImageFont

    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    size = max(14, img.width // 60)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:
        font = ImageFont.load_default()
    for i, (x, y, w, h) in enumerate(boxes):
        c = MARK_COLORS[i % len(MARK_COLORS)]
        draw.rectangle([x, y, x + w, y + h], outline=c, width=max(2, size // 6))
        tag = str(i + 1)
        tw, th = draw.textbbox((0, 0), tag, font=font)[2:]
        ty = y - th - 4 if y - th - 4 >= 0 else y
        draw.rectangle([x, ty, x + tw + 6, ty + th + 4], fill=c)
        draw.text((x + 3, ty + 1), tag, fill="white", font=font)
    return img


def _overlap(a, b) -> float:
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy / max(1.0, min(a[2] * a[3], b[2] * b[3]))


def pick_marks(gold, others, rng, k: int, area: float):
    """Gold box plus up to k distractors that are reasonably sized and do not overlap each other."""
    chosen = [gold]
    pool = [o for o in others if o[0][2] >= 6 and o[0][3] >= 6 and o[0][2] * o[0][3] < 0.3 * area]
    rng.shuffle(pool)
    for o in pool:
        if len(chosen) > k:
            break
        if o[1] and o[1] not in {c[1] for c in chosen} and all(_overlap(o[0], c[0]) < 0.3 for c in chosen):
            chosen.append(o)
    rng.shuffle(chosen)
    return chosen, chosen.index(gold)


def element_record(key, state, img, gold, others, rng, name: str, k=(3, 8), extra: dict | None = None, **meta) -> Record | None:
    """gold / others are (box, description); boxes in image pixels."""
    chosen, g = pick_marks(gold, others, rng, rng.randint(*k), img.width * img.height)
    if len(chosen) < 3:
        return None
    crit = {f"element_{i + 1}": f"[{i + 1}] {d}" for i, (_, d) in enumerate(chosen)}
    qs = {"target": Question("choice", ELEMENT_Q, crit)}
    ts = {"target": Target(f"element_{g + 1}")}
    for qn, (q, t) in (extra or {}).items():
        qs[qn], ts[qn] = q, t
    rec = Record(key, state, qs, ts, meta={"area": "agents", **meta})
    rec.images = [save_image(draw_marks(img, [b for b, _ in chosen]), name, key)]
    return rec


def type_question(actions: dict[str, str], gold: str):
    """Action-type question over a platform's action vocabulary (label -> description)."""
    return Question("choice", TYPE_Q, dict(actions)), Target(gold)


# ---- Multimodal-Mind2Web ---------------------------------------------------------
M2W_SPLITS = {"train": "train", "test_task": "test_task", "test_website": "test_website", "test_domain": "test_domain"}


def mm_mind2web(split, cap, rng):
    from huggingface_hub import HfApi
    from PIL import Image
    import pyarrow.parquet as pq

    repo = "osunlp/Multimodal-Mind2Web"
    shards = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.startswith(f"data/{split}-"))
    rng.shuffle(shards)
    cols = ["operation", "pos_candidates", "neg_candidates", "website", "domain", "annotation_id", "confirmed_task",
            "screenshot", "action_reprs", "target_action_index"]
    n = 0
    for shard in shards:
        for batch in pq.ParquetFile(_dl(repo, shard)).iter_batches(batch_size=8, columns=cols):
            for r in batch.to_pylist():
                if n >= cap:
                    return
                rec = _m2w_record(r, rng, Image)
                if rec:
                    n += 1
                    yield rec


def _m2w_box(c):
    try:
        x, y, w, h = map(float, json.loads(c["attributes"])["bounding_box_rect"].split(","))
    except (KeyError, ValueError):
        return None
    return (x, y, w, h) if w > 0 and h > 0 else None


def _m2w_record(r, rng, Image):
    pos = [json.loads(c) for c in r["pos_candidates"]]
    if not pos or not r["screenshot"]:
        return None
    gold_box = _m2w_box(pos[0])
    if gold_box is None:
        return None
    img = Image.open(io.BytesIO(r["screenshot"]["bytes"]))
    view_h = min(img.height, int(img.width * 0.75))
    top = int(min(max(0, gold_box[1] + gold_box[3] / 2 - view_h / 2), img.height - view_h))
    if gold_box[1] < top or gold_box[1] + gold_box[3] > top + view_h:
        return None
    others = []
    for c in (json.loads(x) for x in r["neg_candidates"]):
        b = _m2w_box(c)
        if b and b[1] >= top and b[1] + b[3] <= top + view_h and b[0] + b[2] <= img.width:
            others.append(((b[0], b[1] - top, b[2], b[3]), _elem(c)))
    img = img.crop((0, top, img.width, top + view_h))
    op = json.loads(r["operation"])
    idx = int(r["target_action_index"])
    state = {"website": f"{r['website']} ({r['domain']})", "task": r["confirmed_task"],
             "previous_actions": r["action_reprs"][:idx][-5:] or "none",
             "next_operation": op["op"] + (f" {op['value']!r}" if op.get("value") else "")}
    gold = ((gold_box[0], gold_box[1] - top, gold_box[2], gold_box[3]), _elem(pos[0]))
    return element_record(rid("mm_m2w", r["annotation_id"], idx), state, img, gold, others, rng, "mm_mind2web")


# ---- GUIAct (web-single, web-multi, smartphone) ----------------------------------
GUIACT_ACTIONS = {
    "click": "click an element", "tap": "tap an element or point", "input": "type text into the focused field",
    "scroll": "scroll the page", "swipe": "swipe the screen", "hover": "hover over an element",
    "select": "choose an option from a dropdown", "select_text": "select a span of text", "copy": "copy the selected text",
    "enter": "press Enter", "answer": "stop and give the answer / report the task complete",
}


def _point(a):
    p = a.get("point") or {}
    m = re.findall(r"[\d.]+", p.get("absolute", "")) if isinstance(p, dict) else []
    return (float(m[0]), float(m[1])) if len(m) >= 2 else None


def _guiact(platform):
    def build(split, cap, rng):
        from PIL import Image
        import pyarrow.parquet as pq

        repo = "yiye2023/GUIAct"
        data = json.load(open(_dl(repo, f"{platform}_{split}_data.json"), encoding="utf-8"))
        by_image = defaultdict(list)
        for d in data:
            by_image[d["image_id"]].append(d)
        names = Counter(a["name"] for d in data for a in _acts(d))
        vocab = {k: GUIACT_ACTIONS.get(k, k) for k, c in names.items() if c >= 20}
        pf = pq.ParquetFile(_dl(repo, f"{platform}_{split}_images.parquet"))
        n = 0
        for batch in pf.iter_batches(batch_size=16):
            for row in batch.to_pylist():
                for d in by_image.get(row["__index_level_0__"], []):
                    if n >= cap:
                        return
                    rec = _guiact_record(d, row, vocab, rng, platform, Image)
                    if rec:
                        n += 1
                        yield rec

    return build


def _acts(d):
    a = d["actions_label"]
    return a if isinstance(a, list) else [a]


def _guiact_record(d, row, vocab, rng, platform, Image):
    a = _acts(d)[0]
    if a["name"] not in vocab:
        return None
    img = Image.open(io.BytesIO(base64.b64decode(row["base64"])))
    sx, sy = img.width / d["image_size"]["width"], img.height / d["image_size"]["height"]
    elems = []
    for e in row["elements"] or []:
        p = e.get("position") or e.get("rect")  # smartphone: position/id, web: rect/uid
        desc = " ".join(x for x in (e.get("ui_type"), (e.get("text") or "").strip()[:60]) if x) or "element"
        elems.append(((p["x"] * sx, p["y"] * sy, p["width"] * sx, p["height"] * sy), desc, e.get("id", e.get("uid"))))
    state = {"task": d["question"], "previous_actions": d["actions_history"] or "none"}
    key = rid("guiact", platform, d["uid"])
    tq = type_question(vocab, a["name"])
    gold = None
    if a.get("element") and elems:
        gold = next((e for e in elems if e[2] == a["element"].get("id")), None)
    elif _point(a) and elems:
        px, py = _point(a)[0] * sx, _point(a)[1] * sy
        inside = [e for e in elems if e[0][0] <= px <= e[0][0] + e[0][2] and e[0][1] <= py <= e[0][1] + e[0][3]]
        gold = min(inside, key=lambda e: e[0][2] * e[0][3]) if inside else None
    if gold is not None:
        others = [(b, t) for b, t, i in elems if i != gold[2]]
        return element_record(key, state, img, (gold[0], gold[1]), others, rng, f"guiact_{platform}", extra={"action_type": tq})
    rec = Record(key, state, {"action_type": tq[0]}, {"action_type": tq[1]}, meta={"area": "agents"})
    rec.images = [save_image(img, f"guiact_{platform}", key)]
    return rec


# ---- AGUVIS stage-2 trajectories ---------------------------------------------------
AGUVIS_ACTIONS = {
    "click": "click or tap a point on the screen", "long_press": "long-press a point", "type": "type text",
    "press": "press a keyboard key", "scroll": "scroll or swipe", "drag": "drag from one point to another",
    "home": "go to the home screen", "back": "press the back button", "open_app": "open an app",
    "wait": "wait for the screen to change", "finish": "stop: the task is complete",
}
_CODE_TYPE = [("terminate", "finish"), ("long_press", "long_press"), ("swipe", "scroll"), ("scroll", "scroll"), ("drag", "drag"),
              ("write", "type"), ("press", "press"), ("hotkey", "press"), ("home", "home"), ("back", "back"),
              ("open_app", "open_app"), ("wait", "wait"), ("click", "click"), ("moveTo", "click")]


def _code_type(code: str) -> str | None:
    head = code.split("(")[0]
    return next((t for k, t in _CODE_TYPE if k in head), None)


def _aguvis_parse(row):
    human = next(c["value"] for c in row["conversations"] if c["from"] == "human")
    m = re.search(r"Instruction:\s*(.+?)\n\s*\nPrevious actions:\s*(.*)$", human, re.S)
    if not m:
        return None
    prev = [re.sub(r"^Step \d+:\s*", "", s).strip() for s in m.group(2).strip().splitlines() if s.strip() and s.strip() != "None"]
    gpt = " ".join(c["value"] for c in row["conversations"] if c["from"] == "gpt" and c.get("recipient") != "os")
    step = re.search(r"Action:\s*(.+?)\s*$", gpt, re.S)
    code = " ".join(c["value"] for c in row["conversations"] if c.get("recipient") == "os")
    return {"task": m.group(1).strip(), "prev": prev, "step": step.group(1).strip() if step else None, "type": _code_type(code)}


_EPISODE = re.compile(r"(_step|step|/screenshot_|_)\d+(\.(jpg|png))+$")


def _similar(a: str, b: str) -> bool:
    """Paraphrases of the gold step are not valid distractors."""
    x, y = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    return len(x & y) / max(1, len(x | y)) > 0.5


def _aguvis(json_name: str, images: str, name: str, max_rows: int | None = None):
    """images: 'zip:<file>' inside xlangai/aguvis-stage2, or 'odyssey' for per-file OpenGVLab/GUI-Odyssey screenshots."""

    def build(split, cap, rng):
        rows = json.load(open(_dl("xlangai/aguvis-stage2", json_name), encoding="utf-8"))
        parsed = [(r["image"][-1] if isinstance(r["image"], list) else r["image"], p)  # list = history screenshots, last is current
                  for r in rows if (p := _aguvis_parse(r)) and p["step"] and p["type"]]
        by_episode = defaultdict(list)
        for img, p in parsed:
            by_episode[_EPISODE.sub("", img)].append(p)
        steps = [p["step"] for _, p in parsed]
        types = Counter(p["type"] for _, p in parsed)
        vocab = {t: AGUVIS_ACTIONS[t] for t, c in types.items() if c >= 20}
        opener = _image_opener(images)
        rng.shuffle(parsed)
        n = 0
        for img_name, p in parsed:
            if n >= min(cap, max_rows or cap):
                return
            if p["type"] not in vocab:
                continue
            later = [q["step"] for q in by_episode[_EPISODE.sub("", img_name)] if len(q["prev"]) > len(p["prev"])]
            later = [s for s in dict.fromkeys(later) if not _similar(s, p["step"])]
            opts = [p["step"]] + rng.sample(later, min(len(later), 2))
            want = rng.randint(4, 6)
            while len(opts) < want:
                s = rng.choice(steps)
                if not any(_similar(s, o) for o in opts):
                    opts.append(s)
            rng.shuffle(opts)
            img = opener(img_name)
            if img is None:
                continue
            key = rid(name, img_name)
            tq = type_question(vocab, p["type"])
            rec = Record(key, {"task": p["task"], "previous_steps": p["prev"][-6:] or "none"},
                         {"next_step": Question("choice", STEP_Q, {f"opt_{i + 1}": o for i, o in enumerate(opts)}), "action_type": tq[0]},
                         {"next_step": Target(f"opt_{opts.index(p['step']) + 1}"), "action_type": tq[1]}, meta={"area": "agents"})
            rec.images = [save_image(img, name, key)]
            n += 1
            yield rec

    return build


def _image_opener(spec: str):
    from PIL import Image

    if spec.startswith("zip:"):
        z = zipfile.ZipFile(_dl("xlangai/aguvis-stage2", spec[4:]))
        index = {}
        for f in z.namelist():
            index.setdefault(f.rsplit("/", 1)[-1], f)
            index.setdefault(f.split("/", 1)[-1], f)

        def open_zip(name):
            f = index.get(name) or index.get(name.rsplit("/", 1)[-1])
            return Image.open(io.BytesIO(z.read(f))) if f else None

        return open_zip
    if spec == "odyssey":
        from huggingface_hub import HfApi

        repo = "OpenGVLab/GUI-Odyssey"
        where = {f.rsplit("/", 1)[-1]: f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.startswith("screenshots/")}

        def open_hub(name):
            f = where.get(name.rsplit("/", 1)[-1])
            if not f:
                return None
            try:
                return Image.open(_dl(repo, f))
            except OSError:
                return None

        return open_hub
    raise ValueError(spec)


# ---- OmniACT (desktop + web, labelled element boxes) --------------------------------
_XY = re.compile(r"pyautogui\.(?:click|doubleClick|rightClick|moveTo)\(\s*([\d.]+)\s*,\s*([\d.]+)")


def omniact(split, cap, rng):
    from PIL import Image

    repo = "Writer/omniact"
    z = zipfile.ZipFile(_dl(repo, "data.zip"))
    by_screen = {}  # index paths say screen_1.png / screen_1.json; web files are screen1.png / screen1_boxes.json
    for f in z.namelist():
        m = re.match(r"(.+)/screen_?(\d+)(?:_boxes)?\.(png|json)$", f)
        if m:
            by_screen[(m.group(1), m.group(2), m.group(3))] = f

    def resolve(p):
        m = re.match(r"(.+)/screen_?(\d+)(?:_boxes)?\.(png|json)$", p)
        return by_screen.get(m.groups()) if m else None

    file = "val.json" if split == "validation" else f"{split}.json"
    index = json.load(open(_dl(repo, file), encoding="utf-8"))
    names = set(z.namelist())
    n = 0
    for k, e in index.items():
        if n >= cap:
            return
        paths = {"task": e["task"], "image": resolve(e["image"]), "box": resolve(e["box"])}
        if not all(p in names for p in paths.values()):
            continue
        text = z.read(paths["task"]).decode("utf-8", "ignore")
        task = re.search(r"Task:\s*(.+)", text)
        xy = _XY.search(text)
        if not task or not xy:
            continue
        boxes = [(tuple(b["top_left"]) + (b["bottom_right"][0] - b["top_left"][0], b["bottom_right"][1] - b["top_left"][1]),
                  b["label"].replace("_", " ")) for b in json.loads(z.read(paths["box"])).values() if b.get("valid") and isinstance(b.get("label"), str) and b["label"] != "NA"]
        px, py = map(float, xy.groups())
        inside = [b for b in boxes if b[0][0] <= px <= b[0][0] + b[0][2] and b[0][1] <= py <= b[0][1] + b[0][3]]
        if not inside:
            continue
        gold = min(inside, key=lambda b: b[0][2] * b[0][3])
        img = Image.open(io.BytesIO(z.read(paths["image"])))
        rec = element_record(rid("omniact", split, k), {"platform": e["image"].split("/")[2], "task": task.group(1).strip()},
                             img, gold, [b for b in boxes if b is not gold], rng, "omniact")
        if rec:
            n += 1
            yield rec


# ---- WebLINX (conversational web navigation) -----------------------------------------
_WL_CAND = re.compile(r"\(uid = ([\w-]+)\) \[\[tag\]\] (\S+).*?\[\[bbox\]\] x=([\d.-]+) y=([\d.-]+) width=([\d.]+) height=([\d.]+)(.*)")
_WL_TEXT = re.compile(r"\[\[text\]\] (.*?) \[\[")


def weblinx(split, cap, rng):
    import gzip

    from PIL import Image

    rows = [json.loads(l) for l in gzip.open(_dl("McGill-NLP/WebLINX", f"data/chat/{split}.json.gz"), "rt", encoding="utf-8")]
    rows = [r for r in rows if re.search(r'uid="([\w-]+)"', r["action"])]
    rng.shuffle(rows)
    replays, n = {}, 0
    for r in rows:
        if n >= cap:
            return
        uid = re.search(r'uid="([\w-]+)"', r["action"]).group(1)
        cands = []
        for line in r["candidates"].split("\n"):
            m = _WL_CAND.match(line.strip())
            if m:
                t = _WL_TEXT.search(line)
                desc = f"{m.group(2)} {t.group(1).strip()[:60]}" if t else m.group(2)
                cands.append((m.group(1), tuple(map(float, m.group(3, 4, 5, 6))), desc))
        gold = next((c for c in cands if c[0] == uid), None)
        if gold is None:
            continue
        try:
            if r["demo"] not in replays:
                replays[r["demo"]] = json.load(open(_dl("McGill-NLP/WebLINX-full", f"demonstrations/{r['demo']}/replay.json"), encoding="utf-8"))["data"]
            turn = replays[r["demo"]][r["turn"]]
            shot = turn["state"]["screenshot"]
            img = Image.open(_dl("McGill-NLP/WebLINX-full", f"demonstrations/{r['demo']}/screenshots/{shot}"))
            vw = turn["action"]["arguments"]["metadata"]["viewportWidth"]
        except (OSError, KeyError, IndexError, TypeError, ValueError):
            continue
        s = img.width / vw
        scaled = [((b[0] * s, b[1] * s, b[2] * s, b[3] * s), d) for _, b, d in cands]
        keep = [c for c in scaled if c[0][0] >= 0 and c[0][1] >= 0 and c[0][0] + c[0][2] <= img.width and c[0][1] + c[0][3] <= img.height]
        g = scaled[cands.index(gold)]
        if g not in keep:
            continue
        history = [h.strip() for h in re.split(r"</s><s>\[INST\]", r["action_history"]) if h.strip()][-4:]
        state = {"conversation_and_actions": history or "none", "action": r["action"].split("(")[0]}
        rec = element_record(rid("weblinx", split, r["demo"], r["turn"]), state, img, g, [c for c in keep if c is not g], rng, "weblinx")
        if rec:
            n += 1
            yield rec


# ---- registration ------------------------------------------------------------------
register(DatasetSpec("omniact", omniact, ("train", "validation", "test"), "Writer/omniact", "mit", "agents", multimodal=True,
                     description="desktop/web task -> marked target element"))
register(DatasetSpec("weblinx", weblinx, ("train", "valid"), "McGill-NLP/WebLINX(+full screenshots)", "cc-by-nc-sa-4.0", "agents",
                     multimodal=True, tags=["non-commercial"], description="conversational web navigation -> marked target element"))
register(DatasetSpec("mm_mind2web", mm_mind2web, tuple(M2W_SPLITS), "osunlp/Multimodal-Mind2Web", "openrail", "agents", multimodal=True,
                     description="web next-element choice on marked screenshots (Set-of-Mark)"))
for _p in ("web-single", "web-multi", "smartphone"):
    register(DatasetSpec(f"guiact_{_p.replace('-', '_')}", _guiact(_p), ("train", "test"), f"yiye2023/GUIAct:{_p}", "apache-2.0", "agents",
                         multimodal=True, description="GUIAct: action type + marked target element"))
_AGUVIS = {  # name: (json, images, license, max rows)
    "aitw": ("aitw-l2.json", "zip:aitw.zip", "apache-2.0", None),
    "coat": ("coat.json", "zip:coat.zip", "apache-2.0", None),
    "miniwob": ("miniwob-l2.json", "zip:miniwob.zip", "apache-2.0", None),
    "guide": ("guide.json", "zip:guide.zip", "apache-2.0", None),
    "android_control": ("android_control.json", "zip:android_control.zip", "apache-2.0", None),
    "amex": ("amex-l2.json", "zip:amex.zip", "cc-by-4.0", None),
    "gui_odyssey": ("gui-odyssey-l2.json", "odyssey", "cc-by-4.0", 20000),
}
for _n, (_j, _img, _lic, _max) in _AGUVIS.items():
    register(DatasetSpec(f"aguvis_{_n}", _aguvis(_j, _img, f"aguvis_{_n}", _max), ("train",), f"xlangai/aguvis-stage2:{_j}", _lic, "agents",
                         multimodal=True, description="next-step choice + action type on GUI trajectories"))
