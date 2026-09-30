"""More long-document decisions: legal (ECtHR, LEDGAR, CUAD, MAUD), numeric (DROP, TabFact), verification and
faithfulness (DocNLI, WiCE, HaluEval), multi-hop (HotpotQA, 2Wiki, WikiHop), papers and long-document
classification (QASPER, patents, hyperpartisan news), LLM-output judging (RewardBench 1/2, MT-Bench human),
agent outcomes (SWE-agent patches) and phishing triage.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from jevtrainer.data.base import DatasetSpec, choice_record, hf, mcq_record, noul_record, register, rid, take
from jevtrainer.data.converters.longdoc import MAX_CHARS, S3, _list, hash_split, numeric_options
from jevtrainer.schema import Question, Record, Target

PARQUET = "refs/convert/parquet"


def _balanced(rows: list, is_yes, rng) -> list:
    """Downsample the majority class to the minority count."""
    yes, no = [r for r in rows if is_yes(r)], [r for r in rows if not is_yes(r)]
    k = min(len(yes), len(no))
    out = rng.sample(yes, k) + rng.sample(no, k)
    rng.shuffle(out)
    return out


# ---- legal --------------------------------------------------------------------------------
ECHR = {"2": "Article 2 (right to life)", "3": "Article 3 (prohibition of torture and inhuman treatment)", "5": "Article 5 (liberty and security)",
        "6": "Article 6 (fair trial)", "8": "Article 8 (private and family life)", "9": "Article 9 (thought, conscience and religion)",
        "10": "Article 10 (freedom of expression)", "11": "Article 11 (assembly and association)", "14": "Article 14 (prohibition of discrimination)",
        "P1-1": "Article 1 of Protocol 1 (protection of property)"}


def ecthr(split, cap, rng):
    a, b = hf("coastalcph/lex_glue", "ecthr_a", split=split), hf("coastalcph/lex_glue", "ecthr_b", split=split)
    names = a.features["labels"].feature.names
    violated = {"\n".join(r["text"]): {names[i] for i in r["labels"]} for r in a}
    n = 0
    for i, r in enumerate(b):
        facts = "\n".join(r["text"])
        alleged = [names[j] for j in r["labels"]]
        if not alleged or facts not in violated or len(facts) > MAX_CHARS or n >= cap:
            continue
        qs = {f"art_{x}": Question("noul", f"The applicant alleges a breach of {ECHR[x]}. Did the European Court of Human Rights find a violation of it?")
              for x in alleged}
        ts = {f"art_{x}": Target("yes" if x in violated[facts] else "no") for x in alleged}
        n += 1
        yield Record(rid("ecthr", split, i), {"case_facts": facts}, qs, ts, meta={"area": "legal"})


def ledgar(split, cap, rng):
    ds = take(hf("coastalcph/lex_glue", "ledgar", split=split), cap, rng)
    names = ds.features["label"].names
    for i, r in enumerate(ds):
        gold = names[r["label"]]
        opts = rng.sample([x for x in names if x != gold], 7) + [gold]
        rng.shuffle(opts)
        yield mcq_record(rid("ledgar", split, i), {"contract_provision": r["text"]}, "Which heading best fits this contract provision?", opts,
                         opts.index(gold), area="legal")


def legalbench_cuad(split, cap, rng):
    from datasets import get_dataset_config_names

    rows = []
    for task in [t for t in get_dataset_config_names("nguha/legalbench") if t.startswith("cuad_")]:
        topic = task[5:].replace("_", " ").replace("-", " ")
        for r in hf("nguha/legalbench", task, split="test"):
            y = {"yes": True, "no": False}.get(str(r["answer"]).strip().lower())
            if y is not None and hash_split(task + r["text"]) == split:
                rows.append(noul_record(rid("lb_cuad", task, r["index"]), {"contract_clause": r["text"]},
                                        f"Is this a '{topic}' clause (CUAD category)?", y, area="legal", task=task))
    rng.shuffle(rows)
    yield from rows[:cap]


MAUD_FILES = {"train": "MAUD_v1/MAUD_train.csv", "validation": "MAUD_v1/MAUD_dev.csv", "test": "MAUD_v1/MAUD_test.csv"}


def maud(split, cap, rng):
    import pandas as pd
    from huggingface_hub import hf_hub_download

    frames = {s: pd.read_csv(hf_hub_download("theatticusproject/maud", f, repo_type="dataset")) for s, f in MAUD_FILES.items()}
    answers = defaultdict(set)
    for df in frames.values():
        for q, sq, a in zip(df["question"], df["subquestion"], df["answer"]):
            answers[(q, sq)].add(str(a).strip())
    df = frames[split]
    df = df[df["data_type"] == "main"].sample(frac=1.0, random_state=rng.randint(0, 2**31))
    n = 0
    for _, r in df.iterrows():
        opts = sorted(answers[(r["question"], r["subquestion"])])
        gold, text = str(r["answer"]).strip(), str(r["text"])
        if not 2 <= len(opts) <= 10 or len(text) > MAX_CHARS or n >= cap:
            continue
        ask = r["question"].replace("-Answer", "") + ("" if r["subquestion"] == "<NONE>" else f" ({r['subquestion']})")
        rng.shuffle(opts)
        n += 1
        yield mcq_record(rid("maud", split, r["id"], r["contract_name"]), {"merger_agreement_excerpt": text, "deal_point": r["text_type"]},
                         f"Which answer describes this merger agreement's deal point: {ask}?", opts, opts.index(gold), area="legal")


# ---- numeric ------------------------------------------------------------------------------
_NUMS = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")


def drop(split, cap, rng):
    ds = take(hf("ucinlp/drop", split=split), cap, rng)
    for i, r in enumerate(ds):
        sp = r["answers_spans"]
        if not sp["types"] or sp["types"][0] != "number":
            continue
        related = [float(x) for x in dict.fromkeys(_NUMS.findall(r["passage"]))]
        rng.shuffle(related)
        got = numeric_options(sp["spans"][0], related[:3], rng, rng.randint(3, 5))
        if got:
            yield mcq_record(rid("drop", split, i), {"passage": r["passage"].strip(), "question": r["question"]},
                             "Which number answers the question (count or compute from the passage)?", *got, area="numeric")


TF_YES, TF_NO = {"1", "true", "yes", "entailed", "entailment"}, {"0", "false", "no", "refuted", "contradiction"}


def tabfact(split, cap, rng):
    for i, r in enumerate(take(hf("table-benchmark/tabfact", split=split), cap, rng)):
        a = str(r["answer"]).strip().lower()
        if a in TF_YES | TF_NO:
            yield noul_record(rid("tabfact", split, i), {"table_title": r["table_title"], "table": r["table"], "statement": r["question"]},
                              "Is the statement true according to the table?", a in TF_YES, area="numeric")


# ---- verification / faithfulness -----------------------------------------------------------------
def docnli(split, cap, rng):
    ds = hf("tasksource/doc-nli", split=split).filter(lambda r: 1500 <= len(r["premise"]) <= MAX_CHARS)
    ds = take(ds, 4 * cap, rng)
    rows = _balanced(list(ds), lambda r: r["label"] == "entailment", rng)[:cap]
    for i, r in enumerate(rows):
        yield noul_record(rid("docnli", split, i), {"document": r["premise"], "hypothesis": r["hypothesis"]},
                          "Is everything the hypothesis says supported by the document?", r["label"] == "entailment", area="verification")


WICE_FILES = {"train": "train.jsonl", "validation": "dev.jsonl", "test": "test.jsonl"}
WICE = {"supported": "Supported: the evidence supports every part of the claim", "partially_supported": "Partially supported: only some parts are supported",
        "not_supported": "Not supported by the evidence"}


def wice(split, cap, rng):
    for i, r in enumerate(take(hf("tasksource/wice", split="train", data_files={"train": WICE_FILES[split]}), cap, rng)):
        ev = "\n".join(r["evidence"])
        if r["label"] in WICE and len(ev) <= MAX_CHARS:
            yield choice_record(rid("wice", split, i), {"claim": r["claim"], "claim_context": r["meta"]["claim_context"], "evidence_article": ev},
                                "Does the cited article support the claim?", dict(WICE), r["label"], area="verification")


def halueval_summ(split, cap, rng):
    for i, r in enumerate(hf("pminervini/HaluEval", "summarization", split="data")):
        if hash_split(r["document"][:500]) != split:
            continue
        ok = rng.random() < 0.5
        yield noul_record(rid("halueval_summ", i), {"document": r["document"], "summary": r["right_summary" if ok else "hallucinated_summary"]},
                          "Is every statement in the summary supported by the document?", ok, area="verification")


# ---- multi-hop over paragraph sets --------------------------------------------------------------
def _multihop(name, repo, config, split, cap, rng):
    ds = take(hf(repo, config, split=split), cap, rng)
    for i, r in enumerate(ds):
        titles, sents = r["context"]["title"], r["context"]["sentences"]
        state = {"question": r["question"], "paragraphs": [f"{t}: {''.join(s).strip()}" for t, s in zip(titles, sents)]}
        ans = r["answer"].strip()
        if ans.lower() in ("yes", "no"):
            yield noul_record(rid(name, split, i), state, "Based on the paragraphs, is the answer to the question yes?", ans.lower() == "yes", area="multi_hop")
            continue
        support = [t for t in dict.fromkeys(r["supporting_facts"]["title"]) if t.lower() != ans.lower()]
        others = [t for t in titles if t not in support and t.lower() != ans.lower()]
        rng.shuffle(others)
        opts = [ans] + (support + others)[: rng.randint(3, 4)]
        opts = list(dict.fromkeys(opts))
        rng.shuffle(opts)
        rec = mcq_record(rid(name, split, i), state, "Which option answers the question?", opts, opts.index(ans), area="multi_hop")
        if rec:
            yield rec


def hotpotqa(split, cap, rng):
    yield from _multihop("hotpotqa", "hotpotqa/hotpot_qa", "distractor", split, cap, rng)


def wiki2mh(split, cap, rng):
    yield from _multihop("2wikimh", "framolfese/2WikiMultihopQA", None, split, cap, rng)


def wikihop(split, cap, rng):
    for i, r in enumerate(take(hf("QAngaroo/wiki_hop", split=split, revision=PARQUET), cap, rng)):
        docs = "\n\n".join(r["supports"])
        if r["answer"].startswith("___MASK") or len(docs) > MAX_CHARS or r["answer"] not in r["candidates"]:
            continue
        rel, _, subj = r["question"].partition(" ")
        cands = [c for c in r["candidates"] if c != r["answer"]]
        opts = rng.sample(cands, min(len(cands), rng.randint(3, 7))) + [r["answer"]]
        rng.shuffle(opts)
        rec = mcq_record(rid("wikihop", split, i), {"documents": docs}, f"Which candidate is the '{rel.replace('_', ' ')}' of '{subj}'?", opts,
                         opts.index(r["answer"]), area="multi_hop")
        if rec:
            yield rec


# ---- papers and long-document classification ---------------------------------------------------
def qasper(split, cap, rng):
    for i, r in enumerate(take(hf("allenai/qasper-yesno", split=split), cap, rng)):
        a = str(r["answer"]).strip().lower()
        if a not in ("yes", "no"):
            continue
        ev = set(r["evidence"])
        budget = MAX_CHARS - len(r["abstract"]) - sum(map(len, ev))
        body = []
        for sec, paras in zip(r["full_text"]["section_name"], r["full_text"]["paragraphs"]):
            keep = []
            for p in paras:
                if p in ev or len(p) <= budget:
                    keep.append(p)
                    budget -= 0 if p in ev else len(p)
            if keep:
                body.append(f"## {sec}\n" + "\n".join(keep))
        extra = [e for e in ev if not any(e in b for b in body)]
        state = {"title": r["title"], "abstract": r["abstract"], "paper": "\n\n".join(body + extra), "question": r["question"]}
        yield noul_record(rid("qasper", split, i), state, "According to the paper, is the answer to the question yes?", a == "yes", area="science")


PATENT = ["Human Necessities", "Performing Operations; Transporting", "Chemistry; Metallurgy", "Textiles; Paper", "Fixed Constructions",
          "Mechanical Engineering; Lighting; Heating; Weapons; Blasting", "Physics", "Electricity", "General tagging of new or cross-sectional technology"]


def patents(split, cap, rng):
    crit = {f"sec_{k}": v for k, v in zip("ABCDEFGHY", PATENT)}
    for i, r in enumerate(take(hf("ccdv/patent-classification", "patent", split=split), cap, rng)):
        text = r["text"] if len(r["text"]) <= MAX_CHARS else r["text"][:12_000] + " [...]"
        yield choice_record(rid("patent", split, i), {"patent_description": text}, "Which CPC section does this patent belong to?", dict(crit),
                            list(crit)[r["label"]], area="classification")


HP_FILES = ("data/articles-training-byarticle-20181122.zip", "data/ground-truth-training-byarticle-20181122.zip")


def hyperpartisan(split, cap, rng):
    import html
    import xml.etree.ElementTree as ET
    import zipfile

    from huggingface_hub import hf_hub_download

    arts, truth = (zipfile.ZipFile(hf_hub_download("SemEvalWorkshop/hyperpartisan_news_detection", f, repo_type="dataset")) for f in HP_FILES)
    labels = {a.get("id"): a.get("hyperpartisan") == "true" for a in ET.fromstring(truth.read(truth.namelist()[0])).iter("article")}
    for a in ET.fromstring(arts.read(arts.namelist()[0])).iter("article"):
        text = html.unescape(re.sub(r"\s+", " ", " ".join(a.itertext()))).strip()
        aid = a.get("id")
        if aid in labels and hash_split("hp" + aid) == split and 300 <= len(text) <= MAX_CHARS:
            yield noul_record(rid("hyperpartisan", aid), {"title": a.get("title"), "article": text},
                              "Is this news article hyperpartisan (extremely one-sided, blindly partisan)?", labels[aid], area="classification")


# ---- judging LLM outputs -----------------------------------------------------------------------
def _pick(prefix, split, key, prompt, responses: list[str], gold: int, rng, ask, **meta):
    order = list(range(len(responses)))
    rng.shuffle(order)
    state = {"request": prompt, **{f"response_{j + 1}": responses[o] for j, o in enumerate(order)}}
    crit = {f"r{j + 1}": f"Response {j + 1}" for j in range(len(responses))}
    return choice_record(rid(prefix, split, key), state, ask, crit, f"r{order.index(gold) + 1}", area="judge", **meta)


def reward_bench(split, cap, rng):
    for r in hf("allenai/reward-bench", split="filtered"):
        if hash_split(f"rb{r['id']}") == split and len(r["chosen"]) + len(r["rejected"]) <= MAX_CHARS:
            yield _pick("rewardbench", split, r["id"], r["prompt"], [r["chosen"], r["rejected"]], 0, rng,
                        "Which response is better (correct, safe, and follows the request)?", subset=r["subset"])


def reward_bench2(split, cap, rng):
    for r in hf("allenai/reward-bench-2", split="test"):
        resp = list(r["chosen"])[:1] + list(r["rejected"])
        if r["subset"] == "Ties" or hash_split(f"rb2{r['id']}") != split or sum(map(len, resp)) > MAX_CHARS or len(resp) < 2:
            continue
        yield _pick("rewardbench2", split, r["id"], r["prompt"], resp, 0, rng,
                    "Which response is best (correct, precise, safe, and follows every instruction)?", subset=r["subset"])


def mt_bench_human(split, cap, rng):
    votes = defaultdict(Counter)
    convs = {}
    for r in hf("lmsys/mt_bench_human_judgments", split="human"):
        k = (r["question_id"], r["model_a"], r["model_b"], r["turn"])
        votes[k][r["winner"]] += 1
        convs[k] = (r["conversation_a"], r["conversation_b"])
    crit = {"a": "Assistant A is better", "b": "Assistant B is better", "tie": "About equally good"}
    for k, v in votes.items():
        if hash_split(f"mtb{k[0]}") != split:
            continue
        n = 2 * k[3]
        ca, cb = (c[:n] for c in convs[k])
        if rng.random() < 0.5:
            ca, cb, v = cb, ca, Counter({"model_a": v["model_b"], "model_b": v["model_a"], "tie": v["tie"]})
        fmt = lambda c: "\n\n".join(f"[{m['role']}]: {m['content']}" for m in c)
        total = sum(v.values())
        probs = {"a": v["model_a"] / total, "b": v["model_b"] / total, "tie": v["tie"] / total}
        state = {"conversation_with_assistant_a": fmt(ca), "conversation_with_assistant_b": fmt(cb),
                 "judge": "the final assistant turn" if k[3] == 2 else "the assistant's answer"}
        if len(state["conversation_with_assistant_a"]) + len(state["conversation_with_assistant_b"]) > MAX_CHARS:
            continue
        yield Record(rid("mtbench_h", *k), state, {"decision": Question("choice", "Which assistant answered the user better?", dict(crit))},
                     {"decision": Target(max(probs, key=probs.get), probs)}, meta={"area": "judge", "votes": total})


# ---- agent outcomes / security -------------------------------------------------------------------
def _issue(traj) -> str | None:
    for m in traj:
        t = m.get("text") or ""
        if m["role"] == "user" and "ISSUE:" in t:
            return t.split("ISSUE:", 1)[1].split("INSTRUCTIONS:", 1)[0].strip()
    return None


def swe_agent(split, cap, rng):
    ds = hf("nebius/SWE-agent-trajectories", split="train", verification_mode="no_checks")
    ds = ds.filter(lambda r: hash_split(r["instance_id"]) == split and bool((r["generated_patch"] or "").strip()), num_proc=4)
    rows = []
    for j, r in enumerate(ds):
        issue = _issue(r["trajectory"])
        if issue and len(issue) + len(r["generated_patch"]) <= MAX_CHARS:
            rows.append((j, r["instance_id"], issue, r["generated_patch"].strip(), bool(r["target"])))
    for j, inst, issue, patch, ok in _balanced(rows, lambda x: x[4], rng)[:cap]:
        yield noul_record(rid("swe_agent", split, j), {"repository_issue": issue, "instance": inst, "candidate_patch": patch},
                          "Will this patch resolve the issue (make the repository's hidden tests pass)?", ok, area="agents")


def phishing(split, cap, rng):
    rows = [r for r in hf("zefang-liu/phishing-email-dataset", split="train")
            if isinstance(r["Email Text"], str) and 800 <= len(r["Email Text"]) <= MAX_CHARS and hash_split(r["Email Text"][:300]) == split]
    for i, r in enumerate(_balanced(rows, lambda r: r["Email Type"] == "Phishing Email", rng)[:cap]):
        yield noul_record(rid("phishing", split, i), {"email": r["Email Text"].strip()}, "Is this email phishing or a scam?", r["Email Type"] == "Phishing Email",
                          "Phishing / scam", "Legitimate", area="security")


S2 = ("train", "validation")
register(DatasetSpec("ecthr", ecthr, S3, "coastalcph/lex_glue:ecthr_a+ecthr_b (facts <= 14k chars)", "cc-by-4.0", "legal",
                     description="per alleged article: did the court find a violation"))
register(DatasetSpec("ledgar", ledgar, S3, "coastalcph/lex_glue:ledgar", "cc-by-4.0", "legal", description="provision heading, gold + 7 of 100 labels"))
register(DatasetSpec("legalbench_cuad", legalbench_cuad, S3, "nguha/legalbench:cuad_* (re-split 80/10/10)", "cc-by-4.0", "legal"))
register(DatasetSpec("maud", maud, S3, "theatticusproject/maud (main rows)", "cc-by-4.0", "legal", description="merger-agreement deal points, options = the question's answer set"))
register(DatasetSpec("drop", drop, S2, "ucinlp/drop (number answers)", "cc-by-sa-4.0", "numeric", description="numeric MCQ; distractors from the passage's numbers"))
register(DatasetSpec("tabfact", tabfact, S3, "table-benchmark/tabfact", "cc-by-4.0", "numeric"))
register(DatasetSpec("docnli", docnli, S3, "tasksource/doc-nli (premise 1.5k-14k chars, balanced)", "bsd", "verification"))
register(DatasetSpec("wice", wice, S3, "tasksource/wice", "cc-by-sa-4.0", "verification"))
register(DatasetSpec("halueval_summ", halueval_summ, S3, "pminervini/HaluEval:summarization (re-split 80/10/10)", "apache-2.0", "verification"))
register(DatasetSpec("hotpotqa", hotpotqa, S2, "hotpotqa/hotpot_qa:distractor", "cc-by-sa-4.0", "multi_hop"))
register(DatasetSpec("wiki2mh", wiki2mh, S2, "framolfese/2WikiMultihopQA", "apache-2.0", "multi_hop"))
register(DatasetSpec("wikihop", wikihop, S2, "QAngaroo/wiki_hop (parquet)", "cc-by-sa-3.0", "multi_hop"))
register(DatasetSpec("qasper", qasper, S3, "allenai/qasper-yesno (evidence kept, other paragraphs to 14k chars)", "cc-by-4.0", "science"))
register(DatasetSpec("patents", patents, S3, "ccdv/patent-classification:patent (first 12k chars)", "cc-by-4.0", "classification"))
register(DatasetSpec("hyperpartisan", hyperpartisan, S3, "SemEvalWorkshop/hyperpartisan_news_detection (by-article, re-split)", "cc-by-4.0", "classification"))
register(DatasetSpec("reward_bench", reward_bench, S3, "allenai/reward-bench:filtered (re-split 80/10/10)", "odc-by", "judge"))
register(DatasetSpec("reward_bench2", reward_bench2, S3, "allenai/reward-bench-2 (no Ties; re-split 80/10/10)", "odc-by", "judge"))
register(DatasetSpec("mt_bench_human", mt_bench_human, S3, "lmsys/mt_bench_human_judgments:human (split by question)", "cc-by-4.0", "judge",
                     description="soft targets from expert votes"))
register(DatasetSpec("swe_agent", swe_agent, S3, "nebius/SWE-agent-trajectories (issue + final patch, split by instance, balanced)", "cc-by-4.0", "agents"))
register(DatasetSpec("phishing", phishing, S3, "zefang-liu/phishing-email-dataset (800-14k chars, balanced)", "lgpl-3.0", "security"))
