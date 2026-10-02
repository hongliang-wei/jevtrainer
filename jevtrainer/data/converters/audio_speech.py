"""Speech understanding from the sound alone: intent, keyword, spoken language (16 kHz mono FLAC, at most 30 s).

One audio clip + one question per record. Answers come straight from the corpus labels; distractors are other labels of
the same corpus. Classes are sampled evenly. Splits follow the corpus' official splits; corpora that ship one split get a
fixed hash-based 10 % test split.
"""

from __future__ import annotations

import random

from jevtrainer.data import avkit
from jevtrainer.data.base import DatasetSpec, register

# ---- SLURP: 70 spoken-assistant intents (real recordings only, no synthetic TTS part) --------------------------
_SCENARIO = {
    "alarm": "alarm", "audio": "audio and volume", "calendar": "calendar", "cooking": "cooking", "datetime": "date and time",
    "email": "email", "general": "general chat", "iot": "smart home", "lists": "lists and to-dos", "music": "music",
    "news": "news", "play": "media playback", "qa": "question answering", "recommendation": "recommendations",
    "social": "social media", "takeaway": "food takeaway", "transport": "transport and taxis", "weather": "weather",
}


def _intent_text(intent: str) -> str:
    scenario, _, action = intent.partition("_")
    return f"{_SCENARIO.get(scenario, scenario)}: {avkit.humanize(action)}"


def slurp(split, cap, rng):
    """Intent of a spoken command to a home assistant. Answer = SLURP's `scenario_action` intent, written as
    '<scenario>: <action>'; distractors are three other intents. train / devel (val) / test are the official splits."""
    repo = "marcel-gohsen/slurp"
    part = {"train": "train", "val": "devel", "test": "test"}[split]
    files = avkit.parquet_files(repo, f"data/{part}-")
    return avkit.clip_choice(
        "slurp", split, cap, rng, repo, files, ["intent"], label_of=lambda r: _intent_text(r["intent"]),
        question="What does the speaker ask the assistant to do?", key_of=lambda r: f"{r['_file']}:{r['_i']}",
        meta_of=lambda r: {"source_label": r["intent"]},
    )


# ---- MINDS-14: banking intents in 14 languages ---------------------------------------------------------------
_MINDS = {
    "abroad": "using the card abroad", "address": "changing the address on the account", "app_error": "reporting an error in the banking app",
    "atm_limit": "the ATM cash withdrawal limit", "balance": "checking the account balance", "business_loan": "applying for a business loan",
    "card_issues": "a problem with the card", "cash_deposit": "depositing cash", "direct_debit": "setting up or checking a direct debit",
    "freeze": "freezing the card or account", "high_value_payment": "making a high-value payment", "joint_account": "opening a joint account",
    "latest_transactions": "checking the latest transactions", "pay_bill": "paying a bill",
}


def minds14(split, cap, rng):
    """What a bank customer calls about. The corpus has one split: 10 % of the recordings (fixed hash of the file name)
    are the test split. Answer = the intent folder of the recording, described in plain English; the speech itself is in
    14 different languages."""
    repo = "PolyAI/minds14"
    files = avkit.parquet_files(repo, "all/")
    test = lambda r: avkit.hash_pct(r["path"], 10)  # noqa: E731
    intent = lambda r: r["path"].split("~", 1)[1].split("/", 1)[0].lower()  # noqa: E731
    return avkit.clip_choice(
        "minds14", split, cap, rng, repo, files, ["path"], label_of=lambda r: _MINDS.get(intent(r)),
        question="What is the caller phoning the bank about?", key_of=lambda r: r["path"],
        keep=(lambda r: test(r)) if split == "test" else (lambda r: not test(r)),
        meta_of=lambda r: {"source_label": intent(r), "language": r["path"].split("~", 1)[0]},
    )


# ---- Speech Commands v0.02: 35 spoken keywords, official train / validation / test lists -------------------------
def speech_commands(split, cap, rng):
    """Which keyword was spoken (one second clips, 35 words). Answer = the word; distractors = three other words."""
    repo = "pollen-robotics/speech-commands-v0.02"
    part = {"train": "train", "val": "validation", "test": "test"}[split]
    files = avkit.parquet_files(repo, f"data/{part}/")
    return avkit.clip_choice(
        "speech_commands", split, cap, rng, repo, files, ["label", "file"],
        label_of=lambda r: r["label"] if not r["label"].startswith("_") else None,
        question="Which word does the speaker say?", key_of=lambda r: r["file"],
        meta_of=lambda r: {"source_label": r["label"]},
    )


# ---- FLEURS: spoken language identification, 102 languages -----------------------------------------------------
def fleurs_langid(split, cap, rng):
    """Which language is spoken. Answer = the language name of the recording, distractors = three other FLEURS languages.
    Every language contributes the same number of clips (cap / 102). The clips are the first 100-row groups of one shard
    per language, fetched with a single range request, so each language costs ~100-150 MB instead of its whole split."""
    from concurrent.futures import ThreadPoolExecutor

    repo = "mteb/fleurs"
    part = {"train": "train", "val": "validation", "test": "test"}[split]
    names = [f for f in avkit.parquet_files(repo) if f.rsplit("/", 1)[-1].startswith(part)]
    first: dict[str, str] = {}
    for f in sorted(names):
        first.setdefault(f.split("/")[0], f)
    per_lang = max(1, cap // len(first))

    def one(item):
        lang, fn = item
        for attempt in range(3):
            try:
                rows = avkit.parquet_head(repo, fn, "fleurs_langid", per_lang, ["id", "language", "audio"])
                break
            except Exception:
                rows = []
                import time

                time.sleep(15 * (attempt + 1))
        local = random.Random(f"fleurs:{split}:{lang}")
        local.shuffle(rows)
        out = []
        for r in rows[:per_lang]:
            key = f"{lang}:{r['id']}"
            m = avkit.audio_from_bytes(r["audio"]["bytes"], "fleurs_langid", avkit.rid_("fleurs_langid", split, key))
            if m:
                out.append((key, r["language"], m))
        return lang, out

    with ThreadPoolExecutor(6) as ex:
        results = list(ex.map(one, sorted(first.items())))
    pool = sorted({lang_name for _, out in results for _, lang_name, _ in out})
    for lang, out in results:
        for key, lang_name, m in out:
            rec = avkit.audio_mcq(avkit.rid_("fleurs_langid", split, key), m, "Which language is spoken in this recording?",
                                  lang_name, pool, rng, source_label=lang)
            if rec:
                yield rec


register(DatasetSpec("slurp", slurp, ("train", "val", "test"), "marcel-gohsen/slurp", "cc-by-4.0", "audio", multimodal=True,
                     description="SLURP: intent of a spoken home-assistant command (69 scenario/action intents, real recordings)"))
register(DatasetSpec("minds14", minds14, ("train", "test"), "PolyAI/minds14", "cc-by-4.0", "audio", multimodal=True,
                     description="MINDS-14: banking intent of a spoken call in 14 languages"))
register(DatasetSpec("speech_commands", speech_commands, ("train", "val", "test"), "pollen-robotics/speech-commands-v0.02", "cc-by-4.0",
                     "audio", multimodal=True, description="Speech Commands v0.02: which of 35 keywords is spoken"))
register(DatasetSpec("fleurs_langid", fleurs_langid, ("train", "val", "test"), "mteb/fleurs", "cc-by-4.0", "audio", multimodal=True,
                     description="FLEURS: spoken language identification over 102 languages (language-balanced)"))
