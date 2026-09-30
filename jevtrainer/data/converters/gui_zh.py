"""CAGUI (AgentCPM-GUI): 600 Chinese-app episodes in AITW format, eval-only. Each step asks the action type
and, for taps, which numbered element (Set-of-Mark over the step's UI boxes) is tapped."""

from __future__ import annotations

import json

from jevtrainer.data.base import DatasetSpec, register, rid
from jevtrainer.data.converters.gui import _dl, _threaded, element_record, save_image
from jevtrainer.schema import Question, Record, Target

CAGUI_ACTIONS = {"click": "点击某个元素", "long_press": "长按某个元素", "swipe": "滑动屏幕", "type": "输入文字", "back": "按返回键",
                 "home": "按主页键", "enter": "按回车键", "finish": "任务已完成", "impossible": "任务无法完成"}
ELEMENT_ZH = "截图中候选元素已用编号框标出。下一步操作应当点击哪个编号的元素？"
TYPE_ZH = "为推进任务，智能体下一步应当执行哪类操作？"


def _kind(s) -> str | None:
    t = s["result_action_type"]
    if t == 4:
        if json.loads(s["result_touch_yx"]) != json.loads(s["result_lift_yx"]):
            return "swipe"
        return "long_press" if (s.get("duration") or 0) >= 1000 else "click"
    return {3: "type", 5: "back", 6: "home", 7: "enter", 10: "finish", 11: "impossible"}.get(t)


def _where(x: float, y: float) -> str:
    col = "左" if x < 1 / 3 else "中" if x < 2 / 3 else "右"
    row = "上" if y < 1 / 3 else "中" if y < 2 / 3 else "下"
    return f"屏幕{row}{col}部（{x:.0%}, {y:.0%}）"


def cagui(split, cap, rng):
    from huggingface_hub import list_repo_files
    from PIL import Image

    eps = sorted(f for f in list_repo_files("openbmb/CAGUI", repo_type="dataset") if f.startswith("CAGUI_agent/") and f.endswith(".json"))
    n = 0
    for f, path in _threaded(lambda f: _dl("openbmb/CAGUI", f), eps, chunk=32, workers=8):
        steps = json.load(open(path, encoding="utf-8"))
        history = []
        for s in steps:
            kind = _kind(s)
            if kind is None:
                continue
            key = rid("cagui", s["episode_id"], s["step_id"])
            state = {"任务": s["instruction"], "已执行的操作": "；".join(history[-8:]) or "（无）", "步骤": f"第 {s['step_id'] + 1} 步"}
            img = Image.open(_dl("openbmb/CAGUI", "CAGUI_agent/" + s["image_path"]))
            tq, tt = Question("choice", TYPE_ZH, dict(CAGUI_ACTIONS)), Target(kind)
            rec = None
            if kind in ("click", "long_press"):
                ty, tx = json.loads(s["result_touch_yx"])
                boxes = [(b[1] * img.width, b[0] * img.height, b[3] * img.width, b[2] * img.height) for b in json.loads(s["ui_positions"])]
                inside = [b for b in boxes if b[0] <= tx * img.width <= b[0] + b[2] and b[1] <= ty * img.height <= b[1] + b[3]]
                if inside:
                    gold = min(inside, key=lambda b: b[2] * b[3])
                    desc = lambda b: _where((b[0] + b[2] / 2) / img.width, (b[1] + b[3] / 2) / img.height)
                    others = [(b, desc(b)) for b in boxes if b != gold]
                    rec = element_record(key, state, img, (gold, desc(gold)), others, rng, "cagui", extra={"action_type": (tq, tt)}, lang="zh")
                    if rec:
                        rec.questions["target"].instructions = ELEMENT_ZH
            if rec is None:
                rec = Record(key, state, {"action_type": tq}, {"action_type": tt}, meta={"area": "agents", "lang": "zh"})
                rec.images = [save_image(img, "cagui", key)]
            history.append(CAGUI_ACTIONS[kind] + (f"「{s['result_action_text']}」" if kind == "type" and s["result_action_text"] else ""))
            yield rec
            n += 1
            if n >= cap:
                return


register(DatasetSpec("cagui", cagui, ("test",), "openbmb/CAGUI:CAGUI_agent (600 Chinese-app episodes)", "cc-by-nc-4.0", "agents",
                     eval_only=True, multimodal=True, tags=["zh", "nc", "gui"], description="action type + Set-of-Mark tap target"))
