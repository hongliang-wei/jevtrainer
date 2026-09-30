"""More Chinese decisions: app-description categories (IFLYTEK), sentence similarity and query-title
relevance scores, passage reranking (T2Ranking, cMedQA), paper disciplines (CSL); eval-only exams
(AGIEval Chinese, CMMLU, FinanceIQ) and Chinese vision MCQ (MMBench-CN, CMMMU).
"""

from __future__ import annotations

import io
import re
import zipfile

from jevtrainer.data.base import DatasetSpec, hf, mcq_record, noul_record, register, rid
from jevtrainer.data.converters.gui import save_image
from jevtrainer.data.converters.longdoc import hash_split
from jevtrainer.data.converters.zh import S3, ZH_MAX, _clf, _cmteb, _path, _rows, _size, _split
from jevtrainer.schema import Question, Record, Target

IFLYTEK = [
    "打车", "地图导航", "免费WIFI", "租车", "同城服务", "快递物流", "婚庆", "家政", "公共交通", "政务", "社区服务", "薅羊毛", "魔幻", "仙侠",
    "卡牌", "飞行空战", "射击游戏", "休闲益智", "动作类", "体育竞技", "棋牌中心", "经营养成", "策略", "MOBA", "辅助工具", "约会社交", "即时通讯",
    "工作社交", "论坛圈子", "婚恋社交", "情侣社交", "社交工具", "生活社交", "微博博客", "新闻", "漫画", "小说", "技术", "教辅", "问答交流", "搞笑",
    "杂志", "百科", "影视娱乐", "求职", "兼职", "视频", "短视频", "音乐", "直播", "电台", "K歌", "成人", "中小学", "职考", "公务员", "英语",
    "视频教育", "高等教育", "成人教育", "艺术", "语言(非英语)", "旅游资讯", "综合预定", "民航", "铁路", "酒店", "行程管理", "民宿短租", "出国",
    "工具", "亲子儿童", "母婴", "驾校", "违章", "汽车咨询", "汽车交易", "日常养车", "行车辅助", "租房", "买房", "装修家居", "电子产品", "问诊挂号",
    "养生保健", "医疗服务", "减肥瘦身", "美妆美业", "菜谱", "餐饮店", "体育咨讯", "运动健身", "支付", "保险", "股票", "借贷", "理财", "彩票",
    "记账", "银行", "美颜", "影像剪辑", "摄影修图", "相机", "绘画", "二手", "电商", "团购", "外卖", "电影票务", "社区超市", "购物咨询", "笔记",
    "办公", "日程管理", "女性", "经营", "收款", "其他",
]


def _iflytek(split):
    ds = hf("clue/clue", "iflytek", split=split)
    names = ds.features["label"].names
    return [dict(r, cat=IFLYTEK[int(names[r["label"]])]) for r in ds if r["label"] >= 0]


# ---- scores ----------------------------------------------------------------------------------------
STS_ZH = ["0：意思完全不同", "1：意思不同，但话题相关", "2：意思不同，但有部分细节相同", "3：大致相同，但缺少或多出重要信息",
          "4：基本相同，只有次要细节不同", "5：意思完全相同"]
QBQTC = ["0：不相关", "1：部分相关", "2：高度相关"]


def _score(name, load, state, score, instr, levels, area):
    def build(split, cap, rng):
        rows = load(split)
        rng.shuffle(rows)
        n = 0
        for i, r in enumerate(rows):
            y, st = score(r), state(r)
            if y is None or not 0 <= y < len(levels) or _size(st) > ZH_MAX:
                continue
            yield Record(rid(name, split, i), st, {"score": Question("score", instr, list(levels))}, {"score": Target(str(y))},
                         meta={"area": area, "lang": "zh"})
            n += 1
            if n >= cap:
                return

    return build


def _qbqtc(split):
    return _split(_cmteb("QBQTC")("test"), split, lambda r: r["sentence1"])


# ---- reranking --------------------------------------------------------------------------------------
_TAG = re.compile(r"<[^>]{0,20}>")


def _clean(t: str) -> str:
    return re.sub(r"\s*\n\s*", "\n", _TAG.sub("\n", str(t))).strip()


def _rerank(name, repo, fn, qkey, pkey, instr, true, false, area, per=2, max_len=2000):
    """Per query up to `per` positive and `per` negative passages, each a yes/no relevance record."""

    def build(split, cap, rng):
        rows = _split(_rows(repo, fn), split, lambda r: r["query"])
        rng.shuffle(rows)
        n = 0
        for i, r in enumerate(rows):
            for lab, pool in ((True, r["positive"]), (False, r["negative"])):
                pool = [p for p in (_clean(x) for x in pool) if 10 <= len(p) <= max_len]
                for j, p in enumerate(rng.sample(pool, min(per, len(pool)))):
                    yield noul_record(rid(name, split, i, lab, j), {qkey: r["query"], pkey: p}, instr, lab, true, false, area=area, lang="zh")
                    n += 1
                    if n >= cap:
                        return

    return build


def _cmteb_file(name):
    from huggingface_hub import list_repo_files

    return next(f for f in list_repo_files(f"C-MTEB/{name}", repo_type="dataset") if f.startswith("data/") and f.endswith(".parquet"))


def t2_rerank(split, cap, rng):
    yield from _rerank("t2_rerank", "C-MTEB/T2Reranking", _cmteb_file("T2Reranking"), "搜索查询", "段落",
                       "这个段落是否与搜索查询相关、能满足用户的信息需求？", "相关", "不相关", "retrieval")(split, cap, rng)


def cmedqa_rerank(split, cap, rng):
    yield from _rerank("cmedqa_rerank", "C-MTEB/CMedQAv2-reranking", _cmteb_file("CMedQAv2-reranking"), "患者提问", "医生回答",
                       "这个医生回答是否针对这位患者的提问？", "针对该提问", "答非所问", "medical")(split, cap, rng)


# ---- paper discipline ---------------------------------------------------------------------------------
def _csl_rows(split):
    seen, rows = set(), []
    for name in ("CLSClusteringP2P", "CLSClusteringS2S"):
        try:
            data = _rows(f"C-MTEB/{name}", _cmteb_file(name))
        except Exception:
            continue
        for r in data:
            for text, lab in zip(r["sentences"], r["labels"]):
                if text not in seen:
                    seen.add(text)
                    if hash_split(text) == split:
                        rows.append({"text": text, "lab": lab, "p2p": name.endswith("P2P")})
        if rows:
            break
    return rows


CSL_DISC = ["工学", "理学", "医学", "农学", "管理学", "经济学", "法学", "教育学", "文学", "历史学", "哲学", "艺术学", "军事学"]


# ---- eval-only exams ------------------------------------------------------------------------------------
AGIEVAL_ZH = ["logiqa-zh", "gaokao-chinese", "gaokao-english", "gaokao-geography", "gaokao-history", "gaokao-biology", "gaokao-chemistry",
              "gaokao-physics", "gaokao-mathqa"]


def agieval_zh(split, cap, rng):
    for part in AGIEVAL_ZH:
        for i, r in enumerate(hf(f"hails/agieval-{part}", split="test")):
            if len(r["gold"]) != 1:
                continue
            q = r["query"].rsplit("选项：", 1)[0].strip()
            q = re.sub(r"^问题：", "", q).replace("问题：", "\n问题：").strip()
            opts = [re.sub(r"^\([A-Z]\)", "", c).strip() for c in r["choices"]]
            rec = mcq_record(rid("agieval_zh", part, i), {"科目": part, "题目": q}, "哪个选项正确？", opts, r["gold"][0],
                             area="reasoning" if part == "logiqa-zh" else "knowledge", lang="zh", subject=part)
            if rec:
                yield rec


def cmmlu(split, cap, rng):
    import pandas as pd

    with zipfile.ZipFile(_path("haonan-li/cmmlu", "cmmlu_v1_0_1.zip")) as z:
        for fn in sorted(n for n in z.namelist() if re.search(r"(^|/)test/[^/]+\.csv$", n)):
            subj = fn.rsplit("/", 1)[-1][:-4]
            df = pd.read_csv(io.BytesIO(z.read(fn)))
            for i, r in enumerate(df.to_dict("records")):
                if r.get("Answer") not in ("A", "B", "C", "D"):
                    continue
                rec = mcq_record(rid("cmmlu", subj, i), {"科目": subj.replace("_", " "), "题目": str(r["Question"])}, "哪个选项是正确答案？",
                                 [r[k] for k in "ABCD"], "ABCD".index(r["Answer"]), area="knowledge", lang="zh", subject=subj)
                if rec:
                    yield rec


def financeiq(split, cap, rng):
    import pandas as pd
    from huggingface_hub import list_repo_files

    sub = {"validation": "dev", "test": "test"}[split]
    for fn in sorted(f for f in list_repo_files("Duxiaoman-DI/FinanceIQ", repo_type="dataset") if f.startswith(f"data/{sub}/") and f.endswith(".csv")):
        exam = fn.rsplit("/", 1)[-1][:-4]
        for i, r in enumerate(pd.read_csv(_path("Duxiaoman-DI/FinanceIQ", fn)).to_dict("records")):
            if r.get("Answer") not in ("A", "B", "C", "D"):
                continue
            rec = mcq_record(rid("financeiq", sub, exam, i), {"考试": exam, "题目": str(r["Question"])}, "哪个选项是正确答案？",
                             [r[k] for k in "ABCD"], "ABCD".index(r["Answer"]), area="finance", lang="zh", subject=exam)
            if rec:
                yield rec


# ---- eval-only vision -----------------------------------------------------------------------------------
def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and x != x) or str(x).strip() in ("", "nan", "None")


def mmbench_cn(split, cap, rng):
    for cfg, sp in (("default", "dev"), ("chinese_culture", "test")):
        for i, r in enumerate(hf("lmms-lab/MMBench_CN", cfg, split=sp)):
            keys = [k for k in "ABCD" if not _nan(r.get(k))]
            if r.get("answer") not in keys or r.get("image") is None:
                continue
            st = {"问题": r["question"]}
            if not _nan(r.get("hint")):
                st = {"提示": r["hint"], **st}
            rec = mcq_record(rid("mmbench_cn", cfg, r["index"]), st, "根据图片，哪个选项正确？", [r[k] for k in keys], keys.index(r["answer"]),
                             area="vision", lang="zh", category=r.get("category"), subset=cfg)
            if rec:
                rec.images = [save_image(r["image"], "mmbench_cn", f"{cfg}_{r['index']}")]
                yield rec


_IMG = re.compile(r'<img="?([^">]+)"?>')


def cmmmu(split, cap, rng):
    for i, r in enumerate(hf("lmms-lab/CMMMU", split="val")):
        if r["type"] != "选择" or r["answer"] not in ("A", "B", "C", "D"):
            continue
        imgs = [(k, r[f"image_{k}"]) for k in range(1, 6) if r.get(f"image_{k}") is not None]
        if not imgs:
            continue
        names = {r.get(f"image_{k}_filename"): f"[图{j + 1}]" for j, (k, _) in enumerate(imgs)}
        q = _IMG.sub(lambda m: names.get(m.group(1), "[图]"), r["question"])
        rec = mcq_record(rid("cmmmu", r["id"]), {"学科": f"{r['category']} / {r['subcategory']}", "题目": q}, "哪个选项是正确答案？",
                         [r[f"option{k}"] for k in range(1, 5)], "ABCD".index(r["answer"]), area="vision", lang="zh", subject=r["subcategory"])
        if rec:
            rec.images = [save_image(img, "cmmmu", f"{r['id']}_{k}") for k, img in imgs]
            yield rec


# ---- registration ------------------------------------------------------------------------------------
SPECS = [
    DatasetSpec("iflytek", _clf("iflytek", _iflytek, lambda r: {"应用描述": r["sentence"]}, lambda r: r["cat"], "这款手机应用属于哪个类别？",
                                {c: "" for c in IFLYTEK}, "classification"),
                ("train", "validation"), "clue/clue:iflytek (119 app categories)", "unknown (CLUE)", "classification"),
    DatasetSpec("zh_stsb", _score("zh_stsb", _cmteb("STSB"), lambda r: {"句子 1": r["sentence1"], "句子 2": r["sentence2"]}, lambda r: int(r["score"]),
                                  "这两个句子的语义相似程度如何？", STS_ZH, "pairs"), S3, "C-MTEB/STSB (Chinese STS-B)", "see upstream", "pairs"),
    DatasetSpec("qbqtc", _score("qbqtc", _qbqtc, lambda r: {"搜索查询": r["sentence1"], "网页标题": r["sentence2"]}, lambda r: int(r["score"]),
                                "网页标题与搜索查询的相关程度如何？", QBQTC, "retrieval"), S3, "C-MTEB/QBQTC (QQ Browser, re-split)", "see upstream", "retrieval"),
    DatasetSpec("t2_rerank", t2_rerank, S3, "C-MTEB/T2Reranking (T2Ranking, split by query, 2 pos + 2 neg per query)", "apache-2.0", "retrieval"),
    DatasetSpec("cmedqa_rerank", cmedqa_rerank, S3, "C-MTEB/CMedQAv2-reranking (split by query)", "see upstream", "medical"),
    DatasetSpec("csl_discipline", _clf("csl_disc", _csl_rows, lambda r: {"论文": r["text"]}, lambda r: r["lab"], "这篇论文属于哪个学科门类？",
                                       {c: "" for c in CSL_DISC}, "science"),
                S3, "C-MTEB/CLSClusteringP2P (CSL title + abstract, re-split)", "apache-2.0", "science"),
    DatasetSpec("agieval_zh", agieval_zh, ("test",), "hails/agieval-{logiqa-zh, gaokao-*} (single-answer)", "mit", "knowledge", eval_only=True),
    DatasetSpec("cmmlu", cmmlu, ("test",), "haonan-li/cmmlu (67 subjects, test)", "cc-by-nc-4.0", "knowledge", eval_only=True, tags=["nc"]),
    DatasetSpec("financeiq", financeiq, ("validation", "test"), "Duxiaoman-DI/FinanceIQ (10 finance certification exams)", "cc-by-nc-sa-4.0", "finance",
                eval_only=True, tags=["nc"]),
    DatasetSpec("mmbench_cn", mmbench_cn, ("test",), "lmms-lab/MMBench_CN (dev + chinese_culture)", "see upstream", "vision",
                eval_only=True, multimodal=True),
    DatasetSpec("cmmmu", cmmmu, ("test",), "lmms-lab/CMMMU (val, multiple-choice)", "see upstream", "vision", eval_only=True, multimodal=True),
]
for spec in SPECS:
    spec.tags = list(dict.fromkeys(spec.tags + ["zh"]))
    register(spec)
