"""Intent, topic and sentiment classification sources."""

from __future__ import annotations

from jevtrainer.data.base import DatasetSpec, choice_record, hf, noul_record, register, rid, take
from jevtrainer.schema import Question, Record, Target


def _names(ds, col="label"):
    return ds.features[col].names


def _humanize(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").strip()


# ---- intent -------------------------------------------------------------------------
def banking77(split, cap, rng):
    full = hf("mteb/banking77", split=split)
    names = sorted(set(full["label_text"]))
    crit = {n: _humanize(n) for n in names}
    for i, r in enumerate(take(full, cap, rng)):
        yield choice_record(rid("banking77", split, i), {"message": r["text"]}, "Which banking intent best describes this customer message?", crit, r["label_text"], area="intent")


def clinc150(split, cap, rng):
    ds = take(hf("clinc/clinc_oos", "plus", split=split), cap, rng)
    names = _names(ds, "intent")
    crit = {n: ("The request is outside every supported intent." if n == "oos" else _humanize(n)) for n in names}
    for i, r in enumerate(ds):
        yield choice_record(rid("clinc150", split, i), {"request": r["text"]}, "Which assistant intent does this request express? Pick oos when none fits.", crit, names[r["intent"]], area="intent")


# ---- topic --------------------------------------------------------------------------
AGNEWS_CRIT = {
    "world": "World news, international politics and current affairs.",
    "sports": "Sports and athletics.",
    "business": "Business, finance, markets and the economy.",
    "sci_tech": "Science, technology, computers and the internet.",
}


def agnews(split, cap, rng):
    ds = take(hf("fancyzhx/ag_news", split=split), cap, rng)
    keys = list(AGNEWS_CRIT)
    for i, r in enumerate(ds):
        yield choice_record(rid("agnews", split, i), {"text": r["text"]}, "Which news topic does this article belong to?", dict(AGNEWS_CRIT), keys[r["label"]], area="topic")


# ---- sentiment -------------------------------------------------------------------------
SST5 = ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]


def sst5(split, cap, rng):
    ds = take(hf("SetFit/sst5", split=split), cap, rng)
    for i, r in enumerate(ds):
        q = {"sentiment": Question("score", "How positive is the sentiment of this sentence?", list(SST5))}
        yield Record(rid("sst5", split, i), {"text": r["text"]}, q, {"sentiment": Target(str(r["label"]))}, meta={"area": "sentiment"})


YELP = ["1 star: terrible", "2 stars: poor", "3 stars: average", "4 stars: good", "5 stars: excellent"]


def yelp(split, cap, rng):
    ds = take(hf("Yelp/yelp_review_full", split=split), cap, rng)
    for i, r in enumerate(ds):
        text = " ".join(r["text"].split()[:300])
        qs = {
            "stars": Question("score", "How many stars did this reviewer give?", list(YELP)),
            "recommend": Question("noul", "Would this reviewer recommend the business?", {"true": "Clearly positive overall", "false": "Negative or mixed"}),
        }
        ts = {"stars": Target(str(r["label"])), "recommend": Target("yes" if r["label"] >= 3 else "no")}
        yield Record(rid("yelp", split, i), {"review": text}, qs, ts, meta={"area": "sentiment"})


def sms_spam(split, cap, rng):
    ds = take(hf("ucirvine/sms_spam", split="train"), cap, rng)
    for i, r in enumerate(ds):
        yield noul_record(rid("sms_spam", i), {"message": r["sms"]}, "Is this SMS message spam?", r["label"] == 1,
                          "Unsolicited advertising, scam or bulk message", "A normal personal or service message", area="spam")


for spec in [
    DatasetSpec("banking77", banking77, ("train", "test"), "mteb/banking77", "cc-by-4.0", "intent", description="77 banking intents"),
    DatasetSpec("clinc150", clinc150, ("train", "validation", "test"), "clinc/clinc_oos:plus", "cc-by-3.0", "intent", description="150 intents + out-of-scope"),
    DatasetSpec("agnews", agnews, ("train", "test"), "fancyzhx/ag_news", "unknown", "topic", description="4 news topics (Intern-Decision wording)"),
    DatasetSpec("sst5", sst5, ("train", "validation", "test"), "SetFit/sst5", "unknown", "sentiment", description="5-level sentiment as a score question"),
    DatasetSpec("yelp", yelp, ("train", "test"), "Yelp/yelp_review_full", "yelp-dataset-terms", "sentiment", description="star score + recommend noul, two questions per state"),
    DatasetSpec("sms_spam", sms_spam, ("train",), "ucirvine/sms_spam", "cc-by-4.0", "spam", description="SMS spam noul"),
]:
    register(spec)
