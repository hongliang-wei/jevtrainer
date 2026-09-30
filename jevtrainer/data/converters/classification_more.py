"""More single-text classification sources (the Kev / Decider / lev registries, all public)."""

from __future__ import annotations

from jevtrainer.data.base import DatasetSpec, choice_record, classification, hf, register, rid, take
from jevtrainer.schema import Question, Record, Target

PARQUET = "refs/convert/parquet"  # Hub auto-conversion of script-based datasets


def _h(s: str) -> str:
    return s.replace("_", " ").replace("-", " ").strip()


def _crit(names, desc=None):
    return {n: (desc or {}).get(n, _h(n)) for n in names}


DBPEDIA = ["Company", "EducationalInstitution", "Artist", "Athlete", "OfficeHolder", "MeanOfTransportation", "Building",
           "NaturalPlace", "Village", "Animal", "Plant", "Album", "Film", "WrittenWork"]
YAHOO = ["Society & Culture", "Science & Mathematics", "Health", "Education & Reference", "Computers & Internet", "Sports",
         "Business & Finance", "Entertainment & Music", "Family & Relationships", "Politics & Government"]
EMOTION = ["sadness", "joy", "love", "anger", "fear", "surprise"]
GO_EMOTIONS = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire",
               "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy",
               "love", "nervousness", "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]
BIOS = ["accountant", "architect", "attorney", "chiropractor", "comedian", "composer", "dentist", "dietitian", "dj", "filmmaker",
        "interior_designer", "journalist", "model", "nurse", "painter", "paralegal", "pastor", "personal_trainer", "photographer",
        "physician", "poet", "professor", "psychologist", "rapper", "software_engineer", "surgeon", "teacher", "yoga_teacher"]
TREC = {"ABBR": "Asks what an abbreviation stands for", "ENTY": "Asks about a thing, object, animal, product or creative work",
        "DESC": "Asks for a definition, description, reason or manner", "HUM": "Asks about a person, group or organisation",
        "LOC": "Asks about a place", "NUM": "Asks for a number, date, amount or quantity"}
LIAR = ["pants_on_fire", "false", "barely_true", "half_true", "mostly_true", "true"]


def _index(names):
    return lambda r, col="label": names[int(r[col])] if 0 <= int(r[col]) < len(names) else None


SPECS = [
    DatasetSpec("dbpedia14", classification("dbpedia14", "fancyzhx/dbpedia_14", None, lambda r: {"title": r["title"], "text": r["content"]},
                lambda r: DBPEDIA[r["label"]], "Which kind of entity does this Wikipedia article describe?", _crit(DBPEDIA), area="topic"),
                ("train", "test"), "fancyzhx/dbpedia_14", "cc-by-sa-3.0", "topic"),
    DatasetSpec("yahoo_topics", classification("yahoo_topics", "community-datasets/yahoo_answers_topics", None,
                lambda r: {"question": r["question_title"], "details": r["question_content"], "best_answer": r["best_answer"][:1500]},
                lambda r: YAHOO[r["topic"]], "Which Yahoo! Answers category does this question belong to?", _crit(YAHOO), area="topic"),
                ("train", "test"), "community-datasets/yahoo_answers_topics", "unknown", "topic"),
    DatasetSpec("newsgroups20", classification("newsgroups20", "SetFit/20_newsgroups", None, lambda r: {"post": r["text"][:3000]},
                lambda r: r["label_text"], "Which newsgroup was this post written for?",
                {n: n for n in ["alt.atheism", "comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware",
                                "comp.windows.x", "misc.forsale", "rec.autos", "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey",
                                "sci.crypt", "sci.electronics", "sci.med", "sci.space", "soc.religion.christian", "talk.politics.guns",
                                "talk.politics.mideast", "talk.politics.misc", "talk.religion.misc"]}, keep=lambda r: bool(r["text"].strip()), area="topic"),
                ("train", "test"), "SetFit/20_newsgroups", "unknown", "topic"),
    DatasetSpec("trec", classification("trec", "CogComp/trec", None, lambda r: {"question": r["text"]}, lambda r: list(TREC)[r["coarse_label"]],
                "What kind of answer does this question ask for?", TREC, revision=PARQUET, area="intent"),
                ("train", "test"), "CogComp/trec (parquet)", "unknown", "intent"),
    DatasetSpec("emotion", classification("emotion", "dair-ai/emotion", "split", lambda r: {"text": r["text"]}, lambda r: EMOTION[r["label"]],
                "Which emotion does the writer express?", _crit(EMOTION), area="sentiment"),
                ("train", "validation", "test"), "dair-ai/emotion", "unknown", "sentiment"),
    DatasetSpec("go_emotions", classification("go_emotions", "google-research-datasets/go_emotions", "simplified", lambda r: {"comment": r["text"]},
                lambda r: GO_EMOTIONS[r["labels"][0]] if len(r["labels"]) == 1 else None, "Which emotion does this Reddit comment express?",
                _crit(GO_EMOTIONS), area="sentiment"),
                ("train", "validation", "test"), "google-research-datasets/go_emotions:simplified", "apache-2.0", "sentiment",
                description="single-label comments only; 28 classes"),
    DatasetSpec("imdb", classification("imdb", "stanfordnlp/imdb", None, lambda r: {"review": " ".join(r["text"].split()[:400])},
                lambda r: ["negative", "positive"][r["label"]], "Is this movie review positive or negative?",
                {"negative": "The reviewer dislikes the film", "positive": "The reviewer likes the film"}, area="sentiment"),
                ("train", "test"), "stanfordnlp/imdb", "unknown", "sentiment"),
    DatasetSpec("amazon_polarity", classification("amazon_polarity", "fancyzhx/amazon_polarity", None, lambda r: {"title": r["title"], "review": r["content"]},
                lambda r: ["negative", "positive"][r["label"]], "Is this product review positive or negative?",
                {"negative": "1-2 stars", "positive": "4-5 stars"}, area="sentiment"),
                ("train", "test"), "fancyzhx/amazon_polarity", "apache-2.0", "sentiment"),
    DatasetSpec("sst2", classification("sst2", "stanfordnlp/sst2", None, lambda r: {"sentence": r["sentence"]}, lambda r: ["negative", "positive"][r["label"]],
                "Is the sentiment of this movie-review sentence positive or negative?", {"negative": "", "positive": ""},
                split_map={"test": "validation"}, area="sentiment"),
                ("train", "test"), "stanfordnlp/sst2", "unknown", "sentiment"),
    DatasetSpec("rotten_tomatoes", classification("rotten_tomatoes", "cornell-movie-review-data/rotten_tomatoes", None, lambda r: {"review": r["text"]},
                lambda r: ["negative", "positive"][r["label"]], "Is this movie review positive or negative?", {"negative": "", "positive": ""}, area="sentiment"),
                ("train", "validation", "test"), "cornell-movie-review-data/rotten_tomatoes", "unknown", "sentiment"),
    DatasetSpec("tweet_sentiment", classification("tweet_sentiment", "cardiffnlp/tweet_eval", "sentiment", lambda r: {"tweet": r["text"]},
                lambda r: ["negative", "neutral", "positive"][r["label"]], "What is the sentiment of this tweet?",
                {"negative": "", "neutral": "", "positive": ""}, area="sentiment"),
                ("train", "validation", "test"), "cardiffnlp/tweet_eval:sentiment", "unknown", "sentiment"),
    DatasetSpec("tweet_emotion", classification("tweet_emotion", "cardiffnlp/tweet_eval", "emotion", lambda r: {"tweet": r["text"]},
                lambda r: ["anger", "joy", "optimism", "sadness"][r["label"]], "Which emotion does this tweet express?",
                _crit(["anger", "joy", "optimism", "sadness"]), area="sentiment"),
                ("train", "validation", "test"), "cardiffnlp/tweet_eval:emotion", "unknown", "sentiment"),
    DatasetSpec("tweet_irony", classification("tweet_irony", "cardiffnlp/tweet_eval", "irony", lambda r: {"tweet": r["text"]},
                lambda r: ["literal", "ironic"][r["label"]], "Is this tweet ironic?", {"literal": "Meant literally", "ironic": "Says the opposite of what it means"},
                area="sentiment"),
                ("train", "validation", "test"), "cardiffnlp/tweet_eval:irony", "unknown", "sentiment"),
    DatasetSpec("tweet_offensive", classification("tweet_offensive", "cardiffnlp/tweet_eval", "offensive", lambda r: {"tweet": r["text"]},
                lambda r: ["not_offensive", "offensive"][r["label"]], "Is this tweet offensive?",
                {"not_offensive": "No insults, threats or profanity aimed at someone", "offensive": "Contains insults, threats or targeted profanity"}, area="safety"),
                ("train", "validation", "test"), "cardiffnlp/tweet_eval:offensive", "unknown", "safety"),
    DatasetSpec("tweet_hate", classification("tweet_hate", "cardiffnlp/tweet_eval", "hate", lambda r: {"tweet": r["text"]},
                lambda r: ["not_hate", "hate"][r["label"]], "Is this tweet hate speech against immigrants or women?",
                {"not_hate": "", "hate": "Attacks or dehumanises people for who they are"}, area="safety"),
                ("train", "validation", "test"), "cardiffnlp/tweet_eval:hate", "unknown", "safety"),
    DatasetSpec("hate_offensive", classification("hate_offensive", "tdavidson/hate_speech_offensive", None, lambda r: {"tweet": r["tweet"]},
                lambda r: ["hate_speech", "offensive", "neither"][r["class"]], "How would you classify this tweet?",
                {"hate_speech": "Attacks a group for protected traits", "offensive": "Offensive or profane but not hate speech", "neither": "Neither"}, area="safety"),
                ("train",), "tdavidson/hate_speech_offensive", "mit", "safety"),
    DatasetSpec("civil_comments", classification("civil_comments", "google/civil_comments", None, lambda r: {"comment": r["text"]},
                lambda r: "toxic" if r["toxicity"] >= 0.5 else ("civil" if r["toxicity"] < 0.1 else None), "Is this comment toxic?",
                {"toxic": "Rude, disrespectful or unreasonable enough to make someone leave a discussion", "civil": "Civil"}, area="safety"),
                ("train", "validation", "test"), "google/civil_comments", "cc0-1.0", "safety", description="toxicity >= 0.5 vs < 0.1; ambiguous rows dropped"),
    DatasetSpec("enron_spam", classification("enron_spam", "SetFit/enron_spam", None, lambda r: {"subject": r["subject"], "body": r["message"][:2500]},
                lambda r: r["label_text"], "Is this email spam?", {"spam": "Unsolicited bulk or scam email", "ham": "A normal email"}, area="spam"),
                ("train", "test"), "SetFit/enron_spam", "unknown", "spam"),
    DatasetSpec("cola", classification("cola", "nyu-mll/glue", "cola", lambda r: {"sentence": r["sentence"]}, lambda r: ["unacceptable", "acceptable"][r["label"]],
                "Is this sentence grammatically acceptable English?", {"unacceptable": "", "acceptable": ""}, split_map={"test": "validation"}, area="language"),
                ("train", "test"), "nyu-mll/glue:cola", "unknown", "language"),
    DatasetSpec("subjectivity", classification("subjectivity", "SetFit/subj", None, lambda r: {"sentence": r["text"]}, lambda r: r["label_text"],
                "Is this sentence an objective statement or a subjective opinion?", {"objective": "", "subjective": ""}, area="language"),
                ("train", "test"), "SetFit/subj", "unknown", "language"),
    DatasetSpec("massive_intent", classification("massive_intent", "mteb/amazon_massive_intent", "en", lambda r: {"utterance": r["text"]},
                lambda r: r["label_text"], "Which assistant intent does this utterance express?", {}, area="intent"),
                ("train", "validation", "test"), "mteb/amazon_massive_intent:en", "cc-by-4.0", "intent"),
    DatasetSpec("massive_scenario", classification("massive_scenario", "mteb/amazon_massive_scenario", "en", lambda r: {"utterance": r["text"]},
                lambda r: r["label_text"], "Which assistant scenario does this utterance belong to?", {}, area="intent"),
                ("train", "validation", "test"), "mteb/amazon_massive_scenario:en", "cc-by-4.0", "intent"),
    DatasetSpec("bitext_support", classification("bitext_support", "bitext/Bitext-customer-support-llm-chatbot-training-dataset", None,
                lambda r: {"message": r["instruction"]}, lambda r: r["intent"], "Which customer-support intent does this message express?", {}, area="intent"),
                ("train",), "bitext/Bitext-customer-support-llm-chatbot-training-dataset", "cdla-sharing-1.0", "intent"),
    DatasetSpec("fin_tweets", classification("fin_tweets", "zeroshot/twitter-financial-news-sentiment", None, lambda r: {"tweet": r["text"]},
                lambda r: ["bearish", "bullish", "neutral"][r["label"]], "What is the market sentiment of this financial tweet?",
                {"bearish": "Expects prices to fall", "bullish": "Expects prices to rise", "neutral": ""}, split_map={"test": "validation"}, area="finance"),
                ("train", "test"), "zeroshot/twitter-financial-news-sentiment", "mit", "finance"),
    DatasetSpec("fin_phrasebank", classification("fin_phrasebank", "atrost/financial_phrasebank", None, lambda r: {"sentence": r["sentence"]},
                lambda r: ["negative", "neutral", "positive"][r["label"]], "What is the sentiment of this financial news sentence for the company?",
                {"negative": "", "neutral": "", "positive": ""}, area="finance"),
                ("train", "validation", "test"), "atrost/financial_phrasebank", "cc-by-nc-sa-3.0", "finance"),
    DatasetSpec("bias_in_bios", classification("bias_in_bios", "LabHC/bias_in_bios", None, lambda r: {"biography": r["hard_text"]},
                lambda r: BIOS[r["profession"]], "What is this person's profession?", _crit(BIOS), split_map={"validation": "dev"}, area="classification"),
                ("train", "validation", "test"), "LabHC/bias_in_bios", "mit", "classification"),
]


def _fill_open_label_sets():
    """Datasets whose label set is read from the data (MASSIVE, Bitext)."""
    for spec in SPECS:
        if spec.name in ("massive_intent", "massive_scenario", "bitext_support"):
            repo, cfg, col = {"massive_intent": ("mteb/amazon_massive_intent", "en", "label_text"),
                              "massive_scenario": ("mteb/amazon_massive_scenario", "en", "label_text"),
                              "bitext_support": ("bitext/Bitext-customer-support-llm-chatbot-training-dataset", None, "intent")}[spec.name]
            spec.build = _open_labels(spec.name, repo, cfg, col, spec)


def _open_labels(name, repo, cfg, col, spec):
    instr = {"massive_intent": "Which assistant intent does this utterance express?",
             "massive_scenario": "Which assistant scenario does this utterance belong to?",
             "bitext_support": "Which customer-support intent does this message express?"}[name]
    field = "instruction" if name == "bitext_support" else "text"
    key = "message" if name == "bitext_support" else "utterance"

    def build(split, cap, rng):
        full = hf(repo, cfg, split=split)
        names = sorted(set(full[col]))
        crit = _crit(names)
        for i, r in enumerate(take(full, cap, rng)):
            yield choice_record(rid(name, split, i), {key: r[field]}, instr, dict(crit), r[col], area="intent")

    return build


def liar2(split, cap, rng):
    ds = take(hf("chengxuphd/liar2", split=split), cap, rng)
    for i, r in enumerate(ds):
        state = {"statement": r["statement"], "speaker": r["speaker"], "context": r["context"], "date": r["date"]}
        q = {"truth": Question("score", "How true is this statement, as rated by PolitiFact?", ["Pants on fire", "False", "Barely true", "Half true", "Mostly true", "True"])}
        yield Record(rid("liar2", split, i), state, q, {"truth": Target(str(r["label"]))}, meta={"area": "factcheck"})


def support_tickets(split, cap, rng):
    ds = hf("Tobi-Bueck/customer-support-tickets", split="train").filter(lambda r: r["language"] == "en" and r["body"] and r["queue"] and r["priority"] and r["type"])
    queues = sorted(set(ds["queue"]))
    types = sorted(set(ds["type"]))
    for i, r in enumerate(take(ds, cap, rng)):
        qs = {
            "queue": Question("choice", "Which support queue should handle this ticket?", {q: q for q in queues}),
            "type": Question("choice", "What type of ticket is this?", {t: t for t in types}),
            "priority": Question("score", "What priority should this ticket get?", ["low", "medium", "high"]),
        }
        pr = {"low": "0", "medium": "1", "high": "2"}.get(r["priority"])
        ts = {"queue": Target(r["queue"]), "type": Target(r["type"])}
        if pr is not None:
            ts["priority"] = Target(pr)
        yield Record(rid("tickets", i), {"subject": r["subject"], "body": r["body"]}, qs, ts, meta={"area": "support"})


def amazon_stars(split, cap, rng):
    ds = take(hf("SetFit/amazon_reviews_multi_en", split=split), cap, rng)
    levels = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
    for i, r in enumerate(ds):
        yield Record(rid("amazon_stars", split, i), {"review": r["text"]}, {"stars": Question("score", "How many stars did the reviewer give?", levels)},
                     {"stars": Target(str(r["label"]))}, meta={"area": "sentiment"})


_fill_open_label_sets()
for spec in SPECS:
    register(spec)
register(DatasetSpec("liar2", liar2, ("train", "validation", "test"), "chengxuphd/liar2", "unknown", "factcheck", description="6-level truthfulness score"))
register(DatasetSpec("support_tickets", support_tickets, ("train",), "Tobi-Bueck/customer-support-tickets (en)", "cc-by-nc-4.0", "support",
                     description="queue + type choices and a priority score on one ticket"))
register(DatasetSpec("amazon_stars", amazon_stars, ("train", "validation", "test"), "SetFit/amazon_reviews_multi_en", "see upstream", "sentiment"))
