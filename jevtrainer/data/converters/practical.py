"""Practical English decisions: prompt-injection detection, flight-booking intents (ATIS), financial news
topics, fraudulent job postings, multi-turn response preference (HelpSteer3, includes Chinese and code),
emotional-support strategies (ESConv) and harmful-question taxonomy (Salad-Data)."""

from __future__ import annotations

import re

from jevtrainer.data.base import DatasetSpec, choice_record, hf, noul_record, register, rid, take
from jevtrainer.data.converters.longdoc import hash_split
from jevtrainer.data.converters.longdoc2 import _balanced
from jevtrainer.data.converters.preference import PAIR
from jevtrainer.data.converters.zh import _path, _rows
from jevtrainer.schema import Question, Record, Target

S3 = ("train", "validation", "test")
INJECTION_Q = "Does this input try to hijack the assistant: override or ignore its instructions, extract its system prompt, or jailbreak it?"


def _injection(name, repo):
    def build(split, cap, rng):
        ds = take(hf(repo, split=split), cap, rng)
        for i, r in enumerate(ds):
            yield noul_record(rid(name, split, i), {"user_input": r["text"][:6000]}, INJECTION_Q, int(r["label"]) == 1,
                              "Prompt injection / jailbreak", "Ordinary request", area="security")

    return build


def atis(split, cap, rng):
    rows = _rows("tuetschek/atis", f"atis_{split}.csv")
    labels = sorted({r["intent"] for r in _rows("tuetschek/atis", "atis_train.csv")})
    crit = {k: k.replace("+", " + ").replace("_", " ") for k in labels}
    rng.shuffle(rows)
    for i, r in enumerate(rows[:cap]):
        if r["intent"] in crit:
            yield choice_record(rid("atis", split, r["id"]), {"utterance": r["text"]}, "What does the traveller want (flight-booking intent)?",
                                dict(crit), r["intent"], area="intent")


FIN_TOPICS = ["Analyst Update", "Fed | Central Banks", "Company | Product News", "Treasuries | Corporate Debt", "Dividend", "Earnings",
              "Energy | Oil", "Financials", "Currencies", "General News | Opinion", "Gold | Metals | Materials", "IPO", "Legal | Regulation",
              "M&A | Investments", "Macro", "Markets", "Politics", "Personnel Change", "Stock Commentary", "Stock Movement"]


def fin_news_topic(split, cap, rng):
    rows = _rows("zeroshot/twitter-financial-news-topic", {"train": "topic_train.csv", "validation": "topic_valid.csv"}[split])
    crit = {t: "" for t in FIN_TOPICS}
    rng.shuffle(rows)
    for i, r in enumerate(rows[:cap]):
        yield choice_record(rid("fin_news_topic", split, i), {"tweet": r["text"]}, "What is this financial news tweet about?", dict(crit),
                            FIN_TOPICS[int(r["label"])], area="finance")


JOB_FIELDS = ["title", "location", "department", "salary_range", "employment_type", "required_experience", "required_education", "industry",
              "function", "company_profile", "description", "requirements", "benefits"]


def fake_jobs(split, cap, rng):
    import pandas as pd

    df = pd.read_csv(_path("victor/real-or-fake-fake-jobposting-prediction", "fake_job_postings.csv"))
    rows = [r for r in df.to_dict("records") if hash_split(str(r["job_id"])) == split]
    for i, r in enumerate(_balanced(rows, lambda r: int(r["fraudulent"]) == 1, rng)[:cap]):
        st = {k: re.sub(r"\s+", " ", str(r[k])).strip()[:2500] for k in JOB_FIELDS if isinstance(r.get(k), str) and r[k].strip()}
        st["has_company_logo"] = "yes" if int(r["has_company_logo"]) else "no"
        yield noul_record(rid("fake_jobs", r["job_id"]), st, "Is this job posting fraudulent (a scam)?", int(r["fraudulent"]) == 1,
                          "Fraudulent", "Genuine", area="security")


def _transcript(msgs, limit=3500):
    text = "\n\n".join(f"{m['role']}: {m['content']}" for m in msgs)
    return text if len(text) <= limit else "…" + text[-limit:]


def helpsteer3(split, cap, rng):
    """test = upstream validation; upstream train is hashed 80 / 20 into train / validation."""
    ds = hf("nvidia/HelpSteer3", "preference", split="validation" if split == "test" else "train")
    idx = [i for i, p in enumerate(ds["overall_preference"])
           if p != 0 and (split == "test" or (hash_split(f"hs3/{i}") == "train") == (split == "train"))]
    rng.shuffle(idx)
    for i in idx[:cap]:
        r = ds[i]
        a, b, a_better = r["response1"], r["response2"], r["overall_preference"] < 0
        if rng.random() < 0.5:
            a, b, a_better = b, a, not a_better
        st = {"conversation": _transcript(r["context"]), "response_a": a[:3000], "response_b": b[:3000]}
        yield choice_record(rid("helpsteer3", split, i), st, "Which final assistant response is better?", dict(PAIR),
                            "response_a" if a_better else "response_b", area="preference", domain=r["domain"], language=r["language"],
                            strength=abs(int(r["overall_preference"])), **({"lang": "zh"} if r["language"] == "chinese" else {}))


def salad(split, cap, rng):
    rows = [r for r in hf("OpenSafetyLab/Salad-Data", "base_set", split="train") if hash_split(str(r["qid"])) == split]
    strip = lambda s: re.sub(r"^O\d+:\s*", "", s)
    doms = {strip(r["1-category"]): "" for r in rows}
    tasks = {strip(r["2-category"]): "" for r in rows}
    rng.shuffle(rows)
    for r in rows[:cap]:
        qs = {"domain": Question("choice", "Which harm domain does this request fall under?", dict(sorted(doms.items()))),
              "task": Question("choice", "Which specific harm category best describes it?", dict(sorted(tasks.items())))}
        ts = {"domain": Target(strip(r["1-category"])), "task": Target(strip(r["2-category"]))}
        yield Record(rid("salad", r["qid"]), {"request": r["question"]}, qs, ts, meta={"area": "safety", "source": r["source"]})


ESCONV = {"Question": "ask about the situation or feelings", "Restatement or Paraphrasing": "restate what the seeker said",
          "Reflection of feelings": "name and reflect the seeker's feelings", "Self-disclosure": "share a similar own experience",
          "Affirmation and Reassurance": "affirm strengths, reassure", "Providing Suggestions": "suggest what to do",
          "Information": "give useful facts or resources", "Others": "greetings, small talk, other"}


def esconv(split, cap, rng):
    """Which support strategy the next supporter turn uses, given the dialogue so far."""
    import json

    fn = {"train": "train.txt", "validation": "valid.txt", "test": "test.txt"}[split]
    with open(_path("thu-coai/esconv", fn), encoding="utf-8") as f:
        dialogs = [json.loads(line) for line in f if line.strip()]
    items = []
    for d, conv in enumerate(dialogs):
        turns = conv["dialog"]
        for t, turn in enumerate(turns):
            strat = (turn.get("annotation") or {}).get("strategy") or turn.get("strategy")
            if t >= 2 and turn.get("speaker") in ("supporter", "sys") and strat in ESCONV:
                items.append((d, t, strat))
    rng.shuffle(items)
    for d, t, strat in items[:cap]:
        conv = dialogs[d]
        hist = "\n".join(f"{'seeker' if x['speaker'] in ('seeker', 'usr') else 'supporter'}: {x['text'].strip()}" for x in conv["dialog"][:t])
        st = {"problem": f"{conv.get('emotion_type', '')} / {conv.get('problem_type', '')}", "situation": conv.get("situation", ""),
              "dialogue": hist[-3500:]}
        yield choice_record(rid("esconv", split, d, t), st, "Which strategy should the supporter use in the next turn?", dict(ESCONV), strat,
                            area="dialogue")


register(DatasetSpec("prompt_injection", _injection("prompt_injection", "deepset/prompt-injections"), ("train", "test"), "deepset/prompt-injections",
                     "apache-2.0", "security", description="English + some German"))
register(DatasetSpec("safeguard_injection", _injection("safeguard_injection", "xTRam1/safe-guard-prompt-injection"), ("train", "test"),
                     "xTRam1/safe-guard-prompt-injection", "see upstream", "security", tags=["synthetic"]))
register(DatasetSpec("atis", atis, ("train", "test"), "tuetschek/atis", "see upstream (LDC ATIS)", "intent"))
register(DatasetSpec("fin_news_topic", fin_news_topic, ("train", "validation"), "zeroshot/twitter-financial-news-topic", "mit", "finance"))
register(DatasetSpec("fake_jobs", fake_jobs, S3, "victor/real-or-fake-fake-jobposting-prediction (EMSCAD, balanced, re-split)", "cc0-1.0", "security"))
register(DatasetSpec("helpsteer3", helpsteer3, S3, "nvidia/HelpSteer3:preference (ties dropped; train re-split; test = upstream validation)", "cc-by-4.0",
                     "preference", description="general, STEM, code, multilingual (incl. Chinese)"))
register(DatasetSpec("esconv", esconv, S3, "thu-coai/esconv (supporter turns with a strategy label)", "cc-by-nc-4.0", "dialogue", tags=["nc"]))
register(DatasetSpec("salad", salad, S3, "OpenSafetyLab/Salad-Data:base_set (re-split)", "apache-2.0", "safety",
                     description="harm domain (6) and category (16)"))
