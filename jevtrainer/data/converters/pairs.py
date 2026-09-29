"""Sentence-pair and relevance sources: GLUE/SuperGLUE, PAWS, WikiQA, SciTail, STS-B, ESCI."""

from __future__ import annotations

from jevtrainer.data.base import DatasetSpec, choice_record, hf, mcq_record, noul_record, register, rid, take
from jevtrainer.schema import Question, Record, Target


def _pair(name, repo, config, fields, label_fn, instructions, criteria, split_map=None, area="nli"):
    def build(split, cap, rng):
        ds = take(hf(repo, config, split=(split_map or {}).get(split, split)), cap, rng)
        for i, r in enumerate(ds):
            y = label_fn(r)
            if y is None:
                continue
            yield choice_record(rid(name, split, i), {k: r[v] for k, v in fields.items()}, instructions, dict(criteria), y, area=area)

    return build


def _noul_pair(name, repo, config, fields, yes_fn, instructions, true, false, split_map=None, area="nli"):
    def build(split, cap, rng):
        ds = take(hf(repo, config, split=(split_map or {}).get(split, split)), cap, rng)
        for i, r in enumerate(ds):
            y = yes_fn(r)
            if y is None:
                continue
            yield noul_record(rid(name, split, i), {k: r[v] for k, v in fields.items()}, instructions, y, true, false, area=area)

    return build


ENT2 = {"entailment": "The second text follows from the first", "not_entailment": "It does not follow"}
VAL = {"test": "validation"}


def copa(split, cap, rng):
    ds = take(hf("aps/super_glue", "copa", split=VAL.get(split, split)), cap, rng)
    for i, r in enumerate(ds):
        ask = "What was the cause?" if r["question"] == "cause" else "What happened as a result?"
        rec = mcq_record(rid("copa", split, i), {"premise": r["premise"]}, ask, [r["choice1"], r["choice2"]], r["label"], area="commonsense")
        if rec:
            yield rec


def stsb(split, cap, rng):
    ds = take(hf("sentence-transformers/stsb", split=split), cap, rng)
    levels = ["0: unrelated", "1: same topic only", "2: some shared details", "3: roughly equivalent, details differ", "4: mostly equivalent", "5: same meaning"]
    for i, r in enumerate(ds):
        q = {"similarity": Question("score", "How similar in meaning are the two sentences?", levels)}
        yield Record(rid("stsb", split, i), {"sentence1": r["sentence1"], "sentence2": r["sentence2"]}, q,
                     {"similarity": Target(str(int(round(5 * float(r["score"])))))}, meta={"area": "similarity"})


ESCI = {"Exact": "The product matches the query exactly", "Substitute": "Not what was asked for but could replace it",
        "Complement": "Does not match but goes with the requested item", "Irrelevant": "Unrelated to the query"}


def esci(split, cap, rng):
    ds = take(hf("tasksource/esci", split=split).filter(lambda r: r["product_locale"] == "us"), cap, rng)
    for i, r in enumerate(ds):
        state = {"query": r["query"].strip(), "product": {"title": r["product_title"], "brand": r["product_brand"],
                                                          "bullets": (r["product_bullet_point"] or "")[:800]}}
        yield choice_record(rid("esci", split, i), state, "How does this product relate to the shopping query?", dict(ESCI), r["esci_label"], area="retrieval")


def contract_nli(split, cap, rng):
    ds = take(hf("kiddothe2b/contract-nli", "contractnli_a", split=split, revision="refs/convert/parquet"), cap, rng)
    names = ["contradiction", "entailment", "neutral"]
    for i, r in enumerate(ds):
        label = names[r["label"]] if isinstance(r["label"], int) else str(r["label"]).lower()
        yield choice_record(rid("contractnli", split, i), {"contract": r["premise"][:12000], "statement": r["hypothesis"]},
                            "Does the NDA support, contradict, or not mention the statement?",
                            {"entailment": "The contract supports it", "contradiction": "The contract contradicts it", "neutral": "The contract does not say"},
                            label, area="legal")


register(DatasetSpec("contract_nli", contract_nli, ("train", "validation", "test"), "kiddothe2b/contract-nli:contractnli_a (parquet)", "cc-by-4.0", "legal"))
register(DatasetSpec("qnli", _pair("qnli", "nyu-mll/glue", "qnli", {"question": "question", "sentence": "sentence"},
                                   lambda r: ["entailment", "not_entailment"][r["label"]], "Does the sentence contain the answer to the question?", ENT2, VAL),
                     ("train", "test"), "nyu-mll/glue:qnli", "unknown", "nli"))
register(DatasetSpec("rte", _pair("rte", "nyu-mll/glue", "rte", {"text": "sentence1", "hypothesis": "sentence2"},
                                  lambda r: ["entailment", "not_entailment"][r["label"]], "Does the text entail the hypothesis?", ENT2, VAL),
                     ("train", "test"), "nyu-mll/glue:rte", "unknown", "nli"))
register(DatasetSpec("mrpc", _noul_pair("mrpc", "nyu-mll/glue", "mrpc", {"sentence1": "sentence1", "sentence2": "sentence2"},
                                        lambda r: r["label"] == 1, "Are the two sentences paraphrases of each other?", "Same meaning", "Different meaning", VAL, "similarity"),
                     ("train", "test"), "nyu-mll/glue:mrpc", "unknown", "similarity"))
register(DatasetSpec("qqp", _noul_pair("qqp", "nyu-mll/glue", "qqp", {"question1": "question1", "question2": "question2"},
                                       lambda r: r["label"] == 1, "Do the two questions ask the same thing?", "Duplicates", "Different questions", VAL, "similarity"),
                     ("train", "test"), "nyu-mll/glue:qqp", "unknown", "similarity"))
register(DatasetSpec("paws", _noul_pair("paws", "google-research-datasets/paws", "labeled_final", {"sentence1": "sentence1", "sentence2": "sentence2"},
                                        lambda r: r["label"] == 1, "Do the two sentences mean the same thing?", "Paraphrase", "Word order changes the meaning", area="similarity"),
                     ("train", "validation", "test"), "google-research-datasets/paws:labeled_final", "custom (free)", "similarity"))
register(DatasetSpec("wic", _noul_pair("wic", "aps/super_glue", "wic", {"word": "word", "sentence1": "sentence1", "sentence2": "sentence2"},
                                       lambda r: r["label"] == 1, "Is the word used with the same meaning in both sentences?", "Same sense", "Different sense", VAL, "language"),
                     ("train", "test"), "aps/super_glue:wic", "cc-by-nc-4.0", "language"))
register(DatasetSpec("cb", _pair("cb", "aps/super_glue", "cb", {"premise": "premise", "hypothesis": "hypothesis"},
                                 lambda r: ["entailment", "contradiction", "neutral"][r["label"]], "How does the premise relate to the hypothesis?",
                                 {"entailment": "Follows", "contradiction": "Contradicts", "neutral": "Neither"}, VAL),
                     ("train", "test"), "aps/super_glue:cb", "unknown", "nli"))
register(DatasetSpec("multirc", _noul_pair("multirc", "aps/super_glue", "multirc", {"paragraph": "paragraph", "question": "question", "answer": "answer"},
                                           lambda r: r["label"] == 1, "Is the proposed answer correct according to the paragraph?", "Correct", "Incorrect", VAL, "reading"),
                     ("train", "test"), "aps/super_glue:multirc", "unknown", "reading"))
register(DatasetSpec("wiki_qa", _noul_pair("wiki_qa", "microsoft/wiki_qa", None, {"question": "question", "sentence": "answer"},
                                           lambda r: r["label"] == 1, "Does this sentence answer the question?", "Answers it", "Does not answer it", area="retrieval"),
                     ("train", "validation", "test"), "microsoft/wiki_qa", "other", "retrieval"))
register(DatasetSpec("scitail", _pair("scitail", "allenai/scitail", "tsv_format", {"premise": "premise", "hypothesis": "hypothesis"},
                                      lambda r: {"entails": "entailment", "neutral": "not_entailment"}.get(r["label"]), "Does the premise support the science claim?", ENT2),
                     ("train", "validation", "test"), "allenai/scitail", "apache-2.0", "nli"))
register(DatasetSpec("copa", copa, ("train", "test"), "aps/super_glue:copa", "bsd-2-clause", "commonsense"))
register(DatasetSpec("stsb", stsb, ("train", "validation", "test"), "sentence-transformers/stsb", "unknown", "similarity", description="similarity as a 0-5 score"))
register(DatasetSpec("esci", esci, ("train", "test"), "tasksource/esci (us)", "apache-2.0", "retrieval", description="Amazon ESCI exact/substitute/complement/irrelevant"))
