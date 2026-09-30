"""Chinese decisions with Chinese instructions: news, app-query, sentiment, stance and intent classification;
question matching and NLI (CLUE, C-MTEB, CBLUE); reading and idiom MCQ; law (CAIL2018 charge + sentence,
legal consultation types); medical exams (CMExam, CMB); finance (FinCUGE); safety (COLD, ToxiCN, TC260);
AI-text detection (HC3); tool choice; response preference pairs.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from functools import lru_cache

from jevtrainer.data.base import DatasetSpec, choice_record, hf, mcq_record, noul_record, register, rid
from jevtrainer.data.converters.longdoc import hash_split
from jevtrainer.schema import Question, Record, Target

ZH_MAX = 5000  # characters; about 3.5-4k Qwen tokens of Chinese
S2, S3 = ("train", "validation"), ("train", "validation", "test")


def _path(repo: str, fn: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, fn, repo_type="dataset")


def _rows(repo: str, fn: str) -> list[dict]:
    """Rows of one parquet / csv / tsv / json / jsonl file in a dataset repo."""
    import pandas as pd

    p = _path(repo, fn)
    if fn.endswith(".parquet"):
        return pd.read_parquet(p).to_dict("records")
    if fn.endswith((".csv", ".tsv")):
        return pd.read_csv(p, sep="\t" if fn.endswith(".tsv") else ",").to_dict("records")
    with open(p, encoding="utf-8") as f:
        txt = f.read()
    try:
        obj = json.loads(txt)
        return obj if isinstance(obj, list) else next(v for v in obj.values() if isinstance(v, list))
    except json.JSONDecodeError:
        return [json.loads(line) for line in txt.splitlines() if line.strip()]


def _split(rows: list[dict], split: str, key) -> list[dict]:
    return [r for r in rows if hash_split(str(key(r))) == split]


def _size(state: dict) -> int:
    return sum(len(str(v)) for v in state.values())


def _clf(name, load, state, label, instr, crit, area):
    """load(split) -> rows; label(row) -> key of crit or None."""

    def build(split, cap, rng):
        rows = load(split)
        rng.shuffle(rows)
        n = 0
        for i, r in enumerate(rows):
            y, st = label(r), state(r)
            if y is None or y not in crit or _size(st) > ZH_MAX:
                continue
            yield choice_record(rid(name, split, i), st, instr, dict(crit), y, area=area, lang="zh")
            n += 1
            if n >= cap:
                return

    return build


def _yesno(name, load, state, yes, instr, area, true="", false=""):
    def build(split, cap, rng):
        rows = load(split)
        rng.shuffle(rows)
        n = 0
        for i, r in enumerate(rows):
            y, st = yes(r), state(r)
            if y is None or _size(st) > ZH_MAX:
                continue
            yield noul_record(rid(name, split, i), st, instr, bool(y), true, false, area=area, lang="zh")
            n += 1
            if n >= cap:
                return

    return build


def _hf_rows(repo, config=None, split_map=None):
    return lambda split: list(hf(repo, config, split=(split_map or {}).get(split, split)))


def _file_rows(repo, files: dict[str, str]):
    return lambda split: _rows(repo, files[split])


def _hashed(repo, fn, key):
    return lambda split: _split(_rows(repo, fn), split, key)


# ---- classification ----------------------------------------------------------------------------
TNEWS = {"100": "故事", "101": "文化", "102": "娱乐", "103": "体育", "104": "财经", "106": "房产", "107": "汽车", "108": "教育",
         "109": "科技", "110": "军事", "112": "旅游", "113": "国际", "114": "股票", "115": "农业", "116": "游戏"}


def _tnews(split):
    ds = hf("clue/clue", "tnews", split=split)
    codes = ds.features["label"].names
    return [dict(r, cat=TNEWS[codes[r["label"]]]) for r in ds]


THUC = ["体育", "娱乐", "家居", "彩票", "房产", "教育", "时尚", "时政", "星座", "游戏", "社会", "科技", "股票", "财经"]


def thucnews(split, cap, rng):
    """Full news articles, 14 categories, equal share per category."""
    per = max(1, cap // len(THUC))
    crit = {c: "" for c in THUC}
    for c in THUC:
        with open(_path("Tongjilibo/THUCNews", f"{c}.jsonl"), encoding="utf-8") as f:
            rows = []
            for line in f:
                r = json.loads(line)
                if hash_split(f"{c}/{r['id']}") == split and 300 <= len(r["content"]) <= ZH_MAX:
                    rows.append(r)
                    if len(rows) >= 4 * per:
                        break
        for r in rng.sample(rows, min(per, len(rows))):
            yield choice_record(rid("thucnews", c, r["id"]), {"标题": r["title"], "正文": r["content"].strip()},
                                "这篇新闻属于哪个频道？", dict(crit), c, area="classification", lang="zh")


FIN_NL = ["公司", "行业", "大盘", "中国", "国际", "经济", "政策", "期货", "债券", "房地产", "外汇", "虚拟货币", "新冠", "能源", "政治"]


def _fincuge(task):
    def load(split):
        ds = hf("Maciel/FinCUGE-Instruction", split={"validation": "eval"}.get(split, split))
        return [r for r in ds if r["task"] == task]

    return load


def _finnl(r):
    m = re.search(r"类别是(.+?)。?$", r["output"].strip())
    return m.group(1) if m and not re.search(r"[，、,和]", m.group(1)) else None


QIC = ["病情诊断", "病因分析", "治疗方案", "就医建议", "指标解读", "疾病表述", "后果表述", "注意事项", "功效作用", "医疗费用", "其他"]


def _massive(repo):
    def build(split, cap, rng):
        full = hf(repo, "zh-CN", split=split)
        crit = {x: "" for x in sorted(set(full["label_text"]))}
        idx = list(range(len(full)))
        rng.shuffle(idx)
        for i in idx[:cap]:
            r = full[i]
            yield choice_record(rid(repo, split, i), {"用户语句": r["text"]}, "这句话属于哪种语音助手意图？" if "intent" in repo else "这句话属于哪个语音助手场景？",
                                dict(crit), r["label_text"], area="intent", lang="zh")

    return build


def tc260(split, cap, rng):
    ds = hf("BBBBBBBBBBBQ/TC260-Chinese-Safety-Prompts", "public", split=split)
    acts = {"resist_attack": "识别并抵御诱导、越狱或伪装，不输出有害内容", "refuse_or_safe_complete": "拒绝，或只给出安全的替代回答",
            "answer_accurately_or_express_uncertainty": "准确回答；不确定时说明不确定", "provide_relevant_actionable_answer": "正常给出切题、可操作的回答"}
    act_set = sorted(set(ds["expected_action"]))
    cats = sorted(set(ds["category_level1_name"]))
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    for i in idx[:cap]:
        r = ds[i]
        if len(r["query"]) > ZH_MAX:
            continue
        qs = {"action": Question("choice", "助手应当如何处理这条用户请求？", {a: acts.get(a, "") for a in act_set}),
              "risk": Question("choice", "这条请求主要涉及哪一类内容安全风险（TC260 分类）？", {c: "" for c in cats})}
        ts = {"action": Target(r["expected_action"]), "risk": Target(r["category_level1_name"])}
        yield Record(rid("tc260", split, r["id"]), {"用户请求": r["query"]}, qs, ts, meta={"area": "safety", "lang": "zh"})


def hc3_zh(split, cap, rng):
    rows = _split(_rows("Hello-SimpleAI/HC3-Chinese", "all.jsonl"), split, lambda r: r["question"])
    rng.shuffle(rows)
    n = 0
    for i, r in enumerate(rows):
        ai = rng.random() < 0.5
        pool = [a for a in (r["chatgpt_answers"] if ai else r["human_answers"]) if isinstance(a, str) and 20 <= len(a) <= 3000]
        if not pool:
            continue
        yield noul_record(rid("hc3_zh", split, i), {"问题": r["question"], "回答": rng.choice(pool).strip()},
                          "这段回答是否由 ChatGPT 等 AI 生成（而非人类撰写）？", ai, "AI 生成", "人类撰写", area="detection", lang="zh")
        n += 1
        if n >= cap:
            return


# ---- pairs / NLI ---------------------------------------------------------------------------------
SAME_Q = "这两个问题表达的意思是否相同（可以用同一个答案回复）？"
NLI_ZH = {"entailment": "蕴含：前提成立时假设一定成立", "neutral": "中立：无法判断", "contradiction": "矛盾：前提成立时假设一定不成立"}


def _clue_labeled(cfg):
    def load(split):
        ds = hf("clue/clue", cfg, split=split)
        names = ds.features["label"].names
        return [dict(r, lab=names[r["label"]]) for r in ds if r["label"] >= 0]

    return load


def _cmteb(name):
    files = {"train": "train", "validation": "validation", "test": "test"}

    def load(split):
        from huggingface_hub import list_repo_files

        fn = next(f for f in list_repo_files(f"C-MTEB/{name}", repo_type="dataset") if f.startswith(f"data/{files[split]}-"))
        return _rows(f"C-MTEB/{name}", fn)

    return load


QQR = {"2": "两者语义完全相同", "1": "查询 2 是查询 1 的语义子集（范围更窄）", "0": "查询 2 范围更宽，或两者语义无关"}
QTR = ["0：主题不一致", "1：少部分相关", "2：大部分相关", "3：完全相关"]


def kuake_qtr(split, cap, rng):
    rows = _split(_rows("dirtycomputer/CBLUE-KUAKE-QTR", "KUAKE-QTR_train.json"), split, lambda r: r["query"] + r["title"])
    for i, r in enumerate(rng.sample(rows, min(cap, len(rows)))):
        if str(r["label"]) in "0123":
            yield Record(rid("kuake_qtr", split, i), {"搜索查询": r["query"], "网页标题": r["title"]},
                         {"relevance": Question("score", "网页标题与搜索查询的主题相关程度如何？", list(QTR))},
                         {"relevance": Target(str(r["label"]))}, meta={"area": "medical", "lang": "zh"})


# ---- MCQ -----------------------------------------------------------------------------------------
def c3(split, cap, rng):
    ds = hf("clue/clue", "c3", split=split)
    for i, r in enumerate(ds.shuffle(seed=rng.randint(0, 2**31)).select(range(min(cap, len(ds))))):
        if r["answer"] in r["choice"]:
            rec = mcq_record(rid("c3", split, i), {"材料": "\n".join(r["context"]), "问题": r["question"]}, "根据材料，哪个选项正确回答了问题？",
                             r["choice"], r["choice"].index(r["answer"]), area="reading", lang="zh")
            if rec:
                yield rec


_IDIOM = re.compile(r"#idiom\d+#")


def chid(split, cap, rng):
    ds = hf("clue/clue", "chid", split=split)
    for i, r in enumerate(ds.shuffle(seed=rng.randint(0, 2**31)).select(range(min(cap, len(ds))))):
        text = "\n".join(r["content"])
        marks, gold = _IDIOM.findall(text), list(r["answers"]["text"])
        if not marks or len(marks) != len(gold) or len(text) > ZH_MAX:
            continue
        k = rng.randrange(len(marks))
        for j, m in enumerate(marks):
            text = text.replace(m, "____" if j == k else gold[j], 1)
        cands = list(r["candidates"])
        if gold[k] in cands:
            rec = mcq_record(rid("chid", split, i), {"文章": text}, "哪个成语最适合填入文中的空白（____）处？", cands, cands.index(gold[k]),
                             area="language", lang="zh")
            if rec:
                yield rec


CAIL_FILES = {"train": "data/exercise_contest_train-00000-of-00001.parquet", "validation": "data/exercise_contest_valid-00000-of-00001.parquet",
              "test": "data/exercise_contest_test-00000-of-00001.parquet"}
TERMS = ["不判有期徒刑", "1-6 个月", "7-12 个月", "1-3 年", "3-5 年", "5-10 年", "10 年以上", "无期徒刑或死刑"]


def _term(r) -> int:
    if r["death_penalty"] or r["life_imprisonment"]:
        return 7
    m = float(r["imprisonment"])
    return next((k for k, hi in enumerate([0, 6, 12, 36, 60, 120]) if m <= hi), 6)


@lru_cache(maxsize=1)
def _cail_charges() -> tuple[str, ...]:
    return tuple(sorted({a for r in _rows("china-ai-law-challenge/cail2018", CAIL_FILES["train"]) for a in r["accusation"]}))


def cail2018(split, cap, rng):
    """Criminal-case facts -> charge (gold + 3 similar + 4 random of 202) and sentence band."""
    rows = [r for r in _rows("china-ai-law-challenge/cail2018", CAIL_FILES[split]) if len(r["accusation"]) == 1 and len(r["fact"]) <= ZH_MAX]
    charges = _cail_charges()
    rng.shuffle(rows)
    per, seen, n = max(30, cap // 40), Counter(), 0
    for r in rows:
        gold = r["accusation"][0]
        if seen[gold] >= per:
            continue
        near = [c for c in charges if c != gold and set(c) & set(gold)]
        opts = rng.sample(near, min(3, len(near)))
        opts += rng.sample([c for c in charges if c != gold and c not in opts], 7 - len(opts)) + [gold]
        rng.shuffle(opts)
        qs = {"charge": Question("choice", "根据案件事实，被告人最可能被判处什么罪名？", {f"opt_{j + 1}": re.sub(r"[\[\]]", "", c) + "罪" for j, c in enumerate(opts)}),
              "sentence": Question("score", "被告人最可能被判处的主刑刑期是多少？", list(TERMS))}
        ts = {"charge": Target(f"opt_{opts.index(gold) + 1}"), "sentence": Target(str(_term(r)))}
        seen[gold] += 1
        n += 1
        yield Record(rid("cail2018", split, n, r["fact"][:50]), {"案件事实": r["fact"].strip()}, qs, ts, meta={"area": "legal", "lang": "zh"})
        if n >= cap:
            return


def _letter_mcq(name, rows, split, cap, rng, state, opts_fn, gold_fn, instr, area):
    rng.shuffle(rows)
    n = 0
    for i, r in enumerate(rows):
        try:
            letters, texts = opts_fn(r)
            g = gold_fn(r)
        except (KeyError, TypeError, ValueError):
            continue
        if g not in letters:
            continue
        rec = mcq_record(rid(name, split, i), state(r), instr, texts, letters.index(g), area=area, lang="zh")
        if rec:
            yield rec
            n += 1
            if n >= cap:
                return


def cmexam(split, cap, rng):
    rows = _rows("fzkuji/CMExam", {"train": "train.json", "validation": "valid.json", "test": "test.json"}[split])
    opts = lambda r: ([o["key"] for o in r["Options"]], [o["value"] for o in r["Options"]])
    yield from _letter_mcq("cmexam", rows, split, cap, rng, lambda r: {"题目": r["Question"]}, opts, lambda r: str(r["Answer"]).strip(),
                           "这道医学考试题的正确答案是哪一项？", "medical")


def cmb(split, cap, rng):
    rows = [r for r in _rows("FreedomIntelligence/CMB", "CMB-Exam/CMB-train/CMB-train-merge.json") if r["question_type"] == "单项选择题"]
    rows = _split(rows, split, lambda r: r["question"])
    opts = lambda r: (sorted(k for k, v in r["option"].items() if v), [r["option"][k] for k in sorted(k for k, v in r["option"].items() if v)])
    yield from _letter_mcq("cmb", rows, split, cap, rng, lambda r: {"考试": f"{r['exam_type']} / {r['exam_subject']}", "题目": r["question"]},
                           opts, lambda r: r["answer"].strip(), "这道医学考试题的正确答案是哪一项？", "medical")


_OPT_LINE = re.compile(r"^([A-H])[:：]\s*(.+)$", re.M)


def legal_case_zh(split, cap, rng):
    rows = _rows("gehits/Chinese-Legal-Case-Classification-Dataset", {"train": "train.json", "validation": "dev.json", "test": "test.json"}[split])

    def opts(r):
        body, _, tail = r["input"].rpartition("选项：")
        pairs = _OPT_LINE.findall(tail)
        if not body or len(pairs) < 2:
            raise ValueError
        r["_body"] = body.strip()
        return [p[0] for p in pairs], [p[1].strip() for p in pairs]

    yield from _letter_mcq("legal_case_zh", rows, split, cap, rng, lambda r: {"法律咨询": r.get("_body", "")}, opts, lambda r: r["answer"].strip(),
                           "这条法律咨询属于哪一类案由？", "legal")


def glaive_toolcall_zh(split, cap, rng):
    rows = _split(_rows("llamafactory/glaive_toolcall_zh", "glaive_toolcall_zh_1k.json"), split, lambda r: r["conversations"][0]["value"])
    none = "不调用工具，直接回复用户"
    for i, r in enumerate(rows[:cap]):
        conv = r["conversations"]
        if len(conv) < 2 or conv[0]["from"] != "human":
            continue
        tools = json.loads(r["tools"]) if isinstance(r["tools"], str) else r["tools"]
        names = [t["name"] for t in tools if isinstance(t, dict) and "name" in t]
        if not names:
            continue
        if conv[1]["from"] == "function_call":
            gold = json.loads(conv[1]["value"]).get("name")
        elif conv[1]["from"] == "gpt":
            gold = none
        else:
            continue
        opts = names + [none]
        if gold in opts:
            rec = mcq_record(rid("glaive_zh", split, i), {"可用工具": json.dumps(tools, ensure_ascii=False, indent=1), "用户消息": conv[0]["value"]},
                             "助手下一步应当调用哪个工具？", opts, opts.index(gold), area="tools", lang="zh")
            if rec:
                yield rec


# ---- preference ----------------------------------------------------------------------------------
PAIR_ZH = {"response_a": "回复 A 更好", "response_b": "回复 B 更好"}


def _pair_zh(name, key, prompt, a, b, a_better, rng, instr="哪个回复更好？", area="preference"):
    if rng.random() < 0.5:
        a, b, a_better = b, a, not a_better
    state = {"用户输入": prompt[:2000], "回复 A": a[:2500], "回复 B": b[:2500]}
    return choice_record(rid(name, key), state, instr, dict(PAIR_ZH), "response_a" if a_better else "response_b", area=area, lang="zh")


def cvalues_rlhf(split, cap, rng):
    s = {"validation": "test"}.get(split, split)
    rows = [(k, r) for k in ("harmless", "helpful") for r in _rows("Skepsun/cvalues_rlhf", f"{k}_{s}.json")]
    rng.shuffle(rows)
    instr = {"harmless": "哪个回复更安全、更负责任？", "helpful": "哪个回复对用户更有帮助？"}
    for i, (k, r) in enumerate(rows[:cap]):
        yield _pair_zh("cvalues", f"{split}{i}", r["prompt"], r["pos_resp"], r["neg_resp"], True, rng, instr[k], "safety" if k == "harmless" else "preference")


def zhihu_rlhf(split, cap, rng):
    rows = [r for r in _rows("liyucheng/zhihu_rlhf_3k", "zhihu_3k_rlfh.tsv")
            if hash_split(str(r["question_id"])) == split and r["upvotes_chosen"] >= 3 * max(1.0, r["upvotes_rejected"])]
    for i, r in enumerate(rng.sample(rows, min(cap, len(rows)))):
        yield _pair_zh("zhihu_rlhf", f"{split}{i}", r["prompt"], str(r["chosen"]), str(r["rejected"]), True, rng,
                       "这是知乎上同一问题的两个回答。哪个回答质量更高（获得了明显更多的赞同）？")


def dpo_zh(split, cap, rng):
    rows = _split(_rows("shibing624/DPO-En-Zh-20k-Preference", "dpo_zh.jsonl"), split, lambda r: r["question"])
    for i, r in enumerate(rng.sample(rows, min(cap, len(rows)))):
        yield _pair_zh("dpo_zh", f"{split}{i}", r["question"], r["response_chosen"], r["response_rejected"], True, rng)


def ultrafeedback_zh(split, cap, rng):
    rows = _split(_rows("opencsg/UltraFeedback-chinese", "ultrafeedback_zh_binarized_random.parquet"), split, lambda r: r["instruction"])
    rng.shuffle(rows)
    n = 0
    for i, r in enumerate(rows):
        a, b = r["chosen_response"], r["rejected_response"]
        gap = float(r["chosen_rating"]["overall_score"]) - float(r["rejected_rating"]["overall_score"])
        if gap >= 1 and a and b and a != b:
            yield _pair_zh("uf_zh", f"{split}{i}", r["instruction"], a, b, True, rng)
            n += 1
            if n >= cap:
                return


# ---- eval-only benchmarks ------------------------------------------------------------------------------
def ceval(split, cap, rng):
    from datasets import get_dataset_config_names

    for subj in get_dataset_config_names("ceval/ceval-exam"):
        for i, r in enumerate(hf("ceval/ceval-exam", subj, split="val")):
            rec = mcq_record(rid("ceval", subj, r["id"]), {"科目": subj.replace("_", " "), "题目": r["question"]}, "哪个选项是正确答案？",
                             [r[k] for k in "ABCD"], "ABCD".index(r["answer"]), area="knowledge", lang="zh", subject=subj)
            if rec:
                yield rec


def jecqa(split, cap, rng):
    for part in ("kd", "ca"):
        for i, r in enumerate(hf(f"hails/agieval-jec-qa-{part}", split="test")):
            if len(r["gold"]) != 1:
                continue
            q = r["query"].split("选项：")[0].removeprefix("问题：").strip()
            opts = [re.sub(r"^\([A-D]\)", "", c).strip() for c in r["choices"]]
            rec = mcq_record(rid("jecqa", part, i), {"司法考试题": q}, "哪个选项正确？", opts, r["gold"][0], area="legal", lang="zh", part=part)
            if rec:
                yield rec


def chinese_safetyqa(split, cap, rng):
    import ast

    for i, r in enumerate(_rows("OpenStellarTeam/Chinese-SafetyQA", "chinese_safetyqa.jsonl")):
        opts = r["options"] if isinstance(r["options"], dict) else ast.literal_eval(r["options"])
        keys = sorted(opts)
        if r["correct_answer"] in keys:
            rec = mcq_record(rid("csqa_safety", i), {"领域": r["cate"], "问题": r["question"]}, "哪个选项是正确答案？", [opts[k] for k in keys],
                             keys.index(r["correct_answer"]), area="safety", lang="zh")
            if rec:
                yield rec


register(DatasetSpec("ceval_val", ceval, ("validation",), "ceval/ceval-exam (val, 52 subjects)", "cc-by-nc-sa-4.0", "knowledge", eval_only=True, tags=["zh", "nc"]))
register(DatasetSpec("jecqa", jecqa, ("test",), "hails/agieval-jec-qa-kd + -ca (single-answer)", "see upstream", "legal", eval_only=True, tags=["zh"]))
register(DatasetSpec("chinese_safetyqa", chinese_safetyqa, ("test",), "OpenStellarTeam/Chinese-SafetyQA (MCQ form)", "cc-by-nc-sa-4.0", "safety",
                     eval_only=True, tags=["zh", "nc"]))


# ---- registration ------------------------------------------------------------------------------------
SENT2 = {"正面": "", "负面": ""}
SENT3 = {"正面": "", "中性": "", "负面": ""}
_pos = lambda x: "正面" if int(x) == 1 else "负面"

SPECS = [
    DatasetSpec("tnews", _clf("tnews", _tnews, lambda r: {"新闻标题": r["sentence"] + (f"（关键词：{r['keywords']}）" if r.get("keywords") else "")},
                              lambda r: r["cat"], "这条新闻标题属于哪个类别？", {v: "" for v in TNEWS.values()}, "classification"),
                S2, "clue/clue:tnews", "unknown (CLUE)", "classification"),
    DatasetSpec("thucnews", thucnews, S3, "Tongjilibo/THUCNews (full articles 300-5k chars, re-split 80/10/10)", "apache-2.0", "classification",
                description="long Chinese news, 14 channels, balanced"),
    DatasetSpec("kuake_qic", _clf("kuake_qic", _hashed("wyp/CBlue-KUAKE-QIC", "KUAKE-QIC_dev.json", lambda r: r["query"]), lambda r: {"用户搜索": r["query"]},
                                  lambda r: r["label"], "这条医疗搜索的意图是什么？", {c: "" for c in QIC}, "medical"),
                S3, "wyp/CBlue-KUAKE-QIC (dev, re-split)", "see CBLUE", "medical"),
    DatasetSpec("fincuge_sentiment", _clf("fincuge_fe", _fincuge("FINFE"), lambda r: {"股吧帖子": r["input"]}, lambda r: {"积极": "正面", "消极": "负面", "中性": "中性"}.get(r["output"].strip("。 ")),
                                          "这条股民评论的情绪倾向是什么？", SENT3, "finance"), S2, "Maciel/FinCUGE-Instruction:FINFE", "apache-2.0", "finance"),
    DatasetSpec("fincuge_news", _clf("fincuge_nl", _fincuge("FINNL"), lambda r: {"财经新闻": r["input"]}, _finnl, "这条财经新闻属于哪个类别？",
                                     {c: "" for c in FIN_NL}, "finance"), S2, "Maciel/FinCUGE-Instruction:FINNL (single-label rows)", "apache-2.0", "finance"),
    DatasetSpec("massive_intent_zh", _massive("mteb/amazon_massive_intent"), S3, "mteb/amazon_massive_intent:zh-CN", "cc-by-4.0", "intent"),
    DatasetSpec("massive_scenario_zh", _massive("mteb/amazon_massive_scenario"), S3, "mteb/amazon_massive_scenario:zh-CN", "cc-by-4.0", "intent"),
    DatasetSpec("zh_shopping", _clf("zh_shopping", _cmteb("OnlineShopping-classification"), lambda r: {"商品类别": r["cat"], "评论": r["text"]},
                                    lambda r: _pos(r["label"]), "这条网购评论的情感倾向是什么？", SENT2, "sentiment"),
                ("train", "test"), "C-MTEB/OnlineShopping-classification", "unknown", "sentiment"),
    DatasetSpec("zh_waimai", _clf("zh_waimai", _cmteb("Waimai-classification"), lambda r: {"外卖评价": r["text"]}, lambda r: _pos(r["label"]),
                                  "这条外卖评价是满意还是不满意？", {"正面": "满意", "负面": "不满意"}, "sentiment"),
                ("train", "test"), "C-MTEB/Waimai-classification", "unknown", "sentiment"),
    DatasetSpec("zh_jdreview", _clf("zh_jdreview", _cmteb("JDReview-classification"), lambda r: {"商品": r["domain"], "评论": r["text"]},
                                    lambda r: "正面" if int(r["label"]) == 0 else "负面", "这条京东商品评论是好评还是差评？", {"正面": "好评", "负面": "差评"}, "sentiment"),
                ("train", "test"), "C-MTEB/JDReview-classification", "unknown", "sentiment"),
    DatasetSpec("zh_sentiment3", _clf("zh_senti3", _cmteb("MultilingualSentiment-classification"), lambda r: {"评论": r["text"]},
                                      lambda r: ["正面", "中性", "负面"][int(r["label"])], "这条评论的情感倾向是什么？", SENT3, "sentiment"),
                S3, "C-MTEB/MultilingualSentiment-classification (zh)", "apache-2.0", "sentiment"),
    DatasetSpec("weibo_senti", _clf("weibo_senti", _hashed("dirtycomputer/weibo_senti_100k", "weibo_senti_100k.csv", lambda r: r["review"]),
                                    lambda r: {"微博": r["review"]}, lambda r: _pos(r["label"]), "这条微博的情感倾向是什么？", SENT2, "sentiment"),
                S3, "dirtycomputer/weibo_senti_100k (re-split)", "unknown", "sentiment"),
    DatasetSpec("chnsenticorp", _clf("chnsenticorp", _hf_rows("lansinuote/ChnSentiCorp"), lambda r: {"评论": r["text"]}, lambda r: _pos(r["label"]),
                                     "这条评论（酒店 / 书籍 / 电脑）的情感倾向是什么？", SENT2, "sentiment"),
                S3, "lansinuote/ChnSentiCorp", "unknown", "sentiment"),
    DatasetSpec("weibo_emotion", _clf("weibo_emotion", _file_rows("souljoy/COVID-19_weibo_emotion", {"train": "train.csv", "validation": "validation.csv", "test": "test.csv"}),
                                      lambda r: {"微博": str(r["text"])}, lambda r: r["label_name"], "这条疫情期间的微博表达了什么情绪？",
                                      {"happy": "积极 / 高兴", "neutral": "中性", "angry": "愤怒", "sad": "悲伤", "fear": "恐惧", "surprise": "惊讶"}, "sentiment"),
                S3, "souljoy/COVID-19_weibo_emotion (SMP2020-EWECT)", "unknown", "sentiment"),
    DatasetSpec("c_stance", _clf("c_stance", _hf_rows("yfhe/C-STANCE-A"), lambda r: {"文本": r["Text"], "对象": r["Target"]}, lambda r: r["Stance"],
                                 "文本作者对该对象持什么立场？", {"支持": "", "反对": "", "中立": ""}, "stance"),
                S3, "yfhe/C-STANCE-A", "see upstream", "stance"),
    DatasetSpec("cold", _yesno("cold", _file_rows("thu-coai/cold", {"train": "train.csv", "validation": "dev.csv", "test": "test.csv"}),
                               lambda r: {"评论": r["TEXT"], "话题": {"race": "种族", "gender": "性别", "region": "地域"}.get(r["topic"], r["topic"])},
                               lambda r: int(r["label"]) == 1, "这条评论是否含有冒犯性内容（针对种族、性别或地域群体）？", "safety", "冒犯", "不冒犯"),
                S3, "thu-coai/cold", "apache-2.0", "safety"),
    DatasetSpec("toxicn", _clf("toxicn", _hashed("JunyuLu/ToxiCN", "ToxiCN_1.0.csv", lambda r: r["content"]), lambda r: {"评论": r["content"]},
                               lambda r: ["无毒", "一般冒犯", "仇恨言论"][int(r["toxic_type"])], "这条评论属于哪种类型？",
                               {"无毒": "不含冒犯", "一般冒犯": "粗俗、辱骂等一般冒犯，不针对特定群体", "仇恨言论": "针对群体（地域、种族、性别、LGBT 等）的仇恨或歧视"}, "safety"),
                S3, "JunyuLu/ToxiCN (re-split)", "see upstream", "safety"),
    DatasetSpec("tc260", tc260, S3, "BBBBBBBBBBBQ/TC260-Chinese-Safety-Prompts:public", "apache-2.0", "safety",
                description="synthetic; expected handling + TC260 level-1 risk category", tags=["synthetic"]),
    DatasetSpec("hc3_zh", hc3_zh, S3, "Hello-SimpleAI/HC3-Chinese (split by question, 50/50 human vs ChatGPT)", "cc-by-sa-4.0", "detection"),
    DatasetSpec("afqmc", _yesno("afqmc", _clue_labeled("afqmc"), lambda r: {"问题 1": r["sentence1"], "问题 2": r["sentence2"]}, lambda r: r["lab"] == "1",
                                SAME_Q, "pairs"), S2, "clue/clue:afqmc (Ant Financial customer questions)", "unknown (CLUE)", "pairs"),
    DatasetSpec("lcqmc", _yesno("lcqmc", _cmteb("LCQMC"), lambda r: {"问题 1": r["sentence1"], "问题 2": r["sentence2"]}, lambda r: int(r["score"]) == 1,
                                SAME_Q, "pairs"), S3, "C-MTEB/LCQMC", "cc-by-4.0", "pairs"),
    DatasetSpec("bq_corpus", _yesno("bq", _cmteb("BQ"), lambda r: {"问题 1": r["sentence1"], "问题 2": r["sentence2"]}, lambda r: int(r["score"]) == 1,
                                    "这两个银行客服问题的意图是否相同？", "pairs"), S3, "C-MTEB/BQ", "unknown", "pairs"),
    DatasetSpec("atec", _yesno("atec", _cmteb("ATEC"), lambda r: {"问题 1": r["sentence1"], "问题 2": r["sentence2"]}, lambda r: int(r["score"]) == 1,
                               SAME_Q, "pairs"), S3, "C-MTEB/ATEC", "unknown", "pairs"),
    DatasetSpec("pawsx_zh", _yesno("pawsx_zh", _cmteb("PAWSX"), lambda r: {"句子 1": r["sentence1"], "句子 2": r["sentence2"]}, lambda r: int(r["score"]) == 1,
                                   "这两个句子是否互为释义（意思完全相同）？", "pairs"), S3, "C-MTEB/PAWSX", "see upstream", "pairs"),
    DatasetSpec("chip_sts", _yesno("chip_sts", _hashed("dirtycomputer/CBLUE-CHIP-STS", "CHIP-STS_train.json", lambda r: r["text1"] + r["text2"]),
                                   lambda r: {"问题 1": r["text1"], "问题 2": r["text2"]}, lambda r: str(r["label"]) == "1",
                                   "这两个患者问题的意思是否相同？", "medical"), S3, "dirtycomputer/CBLUE-CHIP-STS (re-split)", "see CBLUE", "medical"),
    DatasetSpec("kuake_qqr", _clf("kuake_qqr", _hashed("dirtycomputer/CBLUE-KUAKE-QQR", "KUAKE-QQR_train.json", lambda r: r["query1"] + r["query2"]),
                                  lambda r: {"查询 1": r["query1"], "查询 2": r["query2"]}, lambda r: str(r["label"]), "两条医疗搜索查询之间是什么关系？", QQR, "medical"),
                S3, "dirtycomputer/CBLUE-KUAKE-QQR (re-split)", "see CBLUE", "medical"),
    DatasetSpec("kuake_qtr", kuake_qtr, S3, "dirtycomputer/CBLUE-KUAKE-QTR (re-split)", "see CBLUE", "medical"),
    DatasetSpec("cmnli", _clf("cmnli", _clue_labeled("cmnli"), lambda r: {"前提": r["sentence1"], "假设": r["sentence2"]}, lambda r: r["lab"],
                              "前提与假设之间是什么关系？", NLI_ZH, "nli"), S2, "clue/clue:cmnli", "unknown (CLUE)", "nli"),
    DatasetSpec("ocnli", _clf("ocnli", _clue_labeled("ocnli"), lambda r: {"前提": r["sentence1"], "假设": r["sentence2"]}, lambda r: r["lab"],
                              "前提与假设之间是什么关系？", NLI_ZH, "nli"), S2, "clue/clue:ocnli", "cc-by-nc-2.0", "nli", tags=["nc"]),
    DatasetSpec("csl", _yesno("csl", _clue_labeled("csl"), lambda r: {"论文摘要": r["abst"], "关键词": "、".join(r["keyword"])}, lambda r: r["lab"] == "1",
                              "这些关键词是否全部是这篇论文的真实关键词？", "science"), S2, "clue/clue:csl", "unknown (CLUE)", "science"),
    DatasetSpec("cluewsc", _yesno("cluewsc", _clue_labeled("cluewsc2020"), lambda r: {"文本": r["text"], "代词": r["target"]["span2_text"], "候选": r["target"]["span1_text"]},
                                  lambda r: r["lab"] == "true", "文中的代词是否指代该候选对象？", "language"), S2, "clue/clue:cluewsc2020", "unknown (CLUE)", "language"),
    DatasetSpec("c3", c3, S2, "clue/clue:c3", "unknown (CLUE)", "reading"),
    DatasetSpec("chid", chid, S2, "clue/clue:chid (one blank per passage, others filled)", "unknown (CLUE)", "language"),
    DatasetSpec("cail2018", cail2018, S3, "china-ai-law-challenge/cail2018:exercise_contest (single-charge cases)", "see upstream", "legal",
                description="charge MCQ (3 similar + 4 random distractors) and 8-band sentence score"),
    DatasetSpec("legal_case_zh", legal_case_zh, S3, "gehits/Chinese-Legal-Case-Classification-Dataset", "cc-by-nc-4.0", "legal", tags=["nc"]),
    DatasetSpec("cmexam", cmexam, S3, "fzkuji/CMExam (single-answer)", "apache-2.0", "medical"),
    DatasetSpec("cmb", cmb, S3, "FreedomIntelligence/CMB:CMB-Exam train (single-choice, re-split)", "apache-2.0", "medical"),
    DatasetSpec("glaive_toolcall_zh", glaive_toolcall_zh, S3, "llamafactory/glaive_toolcall_zh (first turn, re-split)", "apache-2.0", "tools"),
    DatasetSpec("cvalues_rlhf", cvalues_rlhf, ("train", "validation"), "Skepsun/cvalues_rlhf (harmless + helpful; validation = upstream test)", "apache-2.0", "preference"),
    DatasetSpec("zhihu_rlhf", zhihu_rlhf, S3, "liyucheng/zhihu_rlhf_3k (upvote ratio >= 3)", "cc-by-2.0", "preference"),
    DatasetSpec("dpo_zh", dpo_zh, S3, "shibing624/DPO-En-Zh-20k-Preference:zh", "apache-2.0", "preference"),
    DatasetSpec("ultrafeedback_zh", ultrafeedback_zh, S3, "opencsg/UltraFeedback-chinese (binarized_random)", "apache-2.0", "preference"),
]
for spec in SPECS:
    spec.tags = list(dict.fromkeys(spec.tags + ["zh"]))
    register(spec)
