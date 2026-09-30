"""Chinese review stars (Amazon zh), safety scenario / instruction-attack types (THU Safety-Prompts),
and passage reranking (mMARCO-zh, cMedQA v1)."""

from __future__ import annotations

import json

from jevtrainer.data.base import DatasetSpec, hf, register, rid
from jevtrainer.data.converters.longdoc import hash_split
from jevtrainer.data.converters.zh import S3, ZH_MAX, _path
from jevtrainer.data.converters.zh2 import _cmteb_file, _rerank
from jevtrainer.schema import Question, Record, Target

STARS = ["1 星：很差", "2 星：较差", "3 星：一般", "4 星：较好", "5 星：很好"]


def amazon_zh(split, cap, rng):
    ds = hf("SetFit/amazon_reviews_multi_zh", split=split)
    idx = list(range(len(ds)))
    rng.shuffle(idx)
    for i in idx[:cap]:
        r = ds[i]
        yield Record(rid("amazon_zh", split, r["id"]), {"商品评论": r["text"]}, {"stars": Question("score", "这条亚马逊商品评论给了几星？", list(STARS))},
                     {"stars": Target(str(int(r["label"])))}, meta={"area": "sentiment", "lang": "zh"})


SCENARIOS = {
    "Insult": "辱骂仇恨", "Unfairness_And_Discrimination": "偏见歧视", "Crimes_And_Illegal_Activities": "违法犯罪",
    "Physical_Harm": "身体伤害", "Mental_Health": "心理健康", "Privacy_And_Property": "隐私财产", "Ethics_And_Morality": "道德伦理",
    "Goal_Hijacking": "目标劫持：要求忽略原指令、改输出指定内容", "Prompt_Leaking": "提示泄露：套取系统提示或模型内部信息",
    "Role_Play_Instruction": "角色扮演：借扮演角色诱导不当内容", "Unsafe_Instruction_Topic": "不安全指令主题：直接要求就不安全话题生成内容",
    "Inquiry_With_Unsafe_Opinion": "隐含不安全观点的询问", "Reverse_Exposure": "反面诱导：以“想避免”为名套取负面信息",
}


def safety_prompts_zh(split, cap, rng):
    rows = []
    for fn in ("typical_safety_scenarios.json", "instruction_attack_scenarios.json"):
        with open(_path("thu-coai/Safety-Prompts", fn), encoding="utf-8") as f:
            for scen, items in json.load(f).items():
                rows += [(scen, it["prompt"]) for it in items if isinstance(it, dict) and it.get("prompt") and hash_split(it["prompt"]) == split]
    crit = {k: SCENARIOS.get(k, "") for k in sorted({s for s, _ in rows})}
    rng.shuffle(rows)
    for i, (scen, prompt) in enumerate(r for r in rows if len(r[1]) <= ZH_MAX):
        if i >= cap:
            return
        yield Record(rid("safety_prompts_zh", scen, prompt), {"用户输入": prompt},
                     {"scenario": Question("choice", "这条用户输入属于哪一类安全场景或指令攻击？", dict(crit))}, {"scenario": Target(scen)},
                     meta={"area": "safety", "lang": "zh"})


def mmarco_rerank_zh(split, cap, rng):
    yield from _rerank("mmarco_zh", "C-MTEB/Mmarco-reranking", _cmteb_file("Mmarco-reranking"), "搜索查询", "段落",
                       "这个段落是否回答了搜索查询？", "回答了", "没有回答", "retrieval", per=1)(split, cap, rng)


def cmedqa1_rerank(split, cap, rng):
    yield from _rerank("cmedqa1_rerank", "C-MTEB/CMedQAv1-reranking", _cmteb_file("CMedQAv1-reranking"), "患者提问", "医生回答",
                       "这个医生回答是否针对这位患者的提问？", "针对该提问", "答非所问", "medical")(split, cap, rng)


for spec in [
    DatasetSpec("amazon_zh", amazon_zh, S3, "SetFit/amazon_reviews_multi_zh", "see upstream (Amazon)", "sentiment", description="1-5 stars"),
    DatasetSpec("safety_prompts_zh", safety_prompts_zh, S3, "thu-coai/Safety-Prompts (typical + instruction-attack, re-split)", "apache-2.0", "safety",
                description="13 scenario types; prompts generated with ChatGPT", tags=["synthetic"]),
    DatasetSpec("mmarco_rerank_zh", mmarco_rerank_zh, S3, "C-MTEB/Mmarco-reranking (split by query)", "see upstream (MS MARCO)", "retrieval", tags=["nc"]),
    DatasetSpec("cmedqa1_rerank", cmedqa1_rerank, S3, "C-MTEB/CMedQAv1-reranking (split by query)", "see upstream", "medical"),
]:
    spec.tags = list(dict.fromkeys(spec.tags + ["zh"]))
    register(spec)
