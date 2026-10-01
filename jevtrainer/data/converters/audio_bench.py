"""Audio understanding evaluation sets (eval_only): one audio clip (<= 30 s kept) + a multiple-choice question. Suite "audio-bench".

    mmau_mini              MMAU test-mini (gamma-lab-umd/MMAU-test-mini, MMAU-v05.15.25): 1000 questions on sound / music / speech.
                           The 9000-question MMAU test split ships without answers and cannot be scored.
    mmar                   MMAR (BoJack/MMAR): 1000 deep-reasoning questions over mixed sound / music / speech
    mmsu                   MMSU (ddwang2000/MMSU): ~5000 spoken language understanding questions
    voicebench_mmsu        VoiceBench mmsu: spoken MMLU questions (the question and its options are in the audio)
    voicebench_openbookqa  VoiceBench openbookqa: spoken science questions, 455 items

VoiceBench's other subsets (alpacaeval, commoneval, sd-qa, ifeval, advbench, ...) have no gold answer to score a choice.
"""

from __future__ import annotations

import json
import re
import tarfile
from pathlib import Path

from jevtrainer.data import avkit, benchkit, vidkit
from jevtrainer.data.base import DatasetSpec, register, rid
from jevtrainer.data.benchkit import Q_AUDIO, blob, letter_index, mcq, strip_letter, text_index

_WORKERS = benchkit.WORKERS


def _state(question: str) -> dict:
    return {"clip": "<audio:1>", "question": question}


# ---- MMAU test-mini -----------------------------------------------------------------------------------------
def mmau_mini(split, cap, rng):
    repo = "gamma-lab-umd/MMAU-test-mini"

    def handle(r):
        a = json.loads(r["other_attributes"])
        opts = [strip_letter(c) for c in r["choices"]]
        gold = text_index(r["answer"], opts)
        if gold is None:
            gold = letter_index(r["answer"], len(opts))
        media = avkit.audio_from_bytes(blob(r["context"]), "mmau_mini", a["id"], 30.0)
        return mcq(rid("mmau_mini", a["id"]), _state(r["instruction"]), Q_AUDIO, opts, gold, [media], "audio",
                   category=a.get("task"), task_type=a.get("task"), sub_category=a.get("sub-category"),
                   difficulty=a.get("difficulty"), mmau_category=a.get("category"), origin=a.get("dataset"),
                   source_id=f"mmau:{a['id']}")

    recs = benchkit.parquet_media(repo, ["test_mini.parquet"], "mmau_mini", handle, shards=1, inner=6)
    rng.shuffle(recs)
    yield from recs[:cap]


# ---- MMAR -----------------------------------------------------------------------------------------------------
def mmar(split, cap, rng):
    repo = "BoJack/MMAR"
    rows = vidkit.read_json(repo, "MMAR-meta.json")
    rng.shuffle(rows)
    rows = rows[:cap]
    todo = [r for r in rows if not _cached_audio("mmar", r["id"])]
    base = avkit.raw_dir("mmar") / "audio"
    if todo:
        tar_path = avkit.fetch_file(repo, "mmar-audio.tar.gz", "mmar")
        with tarfile.open(tar_path) as t:
            t.extractall(base)
        Path(tar_path).unlink(missing_ok=True)
    from concurrent.futures import ThreadPoolExecutor

    found = {p.name: p for p in base.rglob("*.wav")} if base.exists() else {}

    def one(r):
        src = found.get(Path(r["audio_path"]).name)
        return r["id"], (avkit.audio_item(src, "mmar", r["id"], 30.0) if src else _cached_audio("mmar", r["id"]))

    with ThreadPoolExecutor(_WORKERS) as ex:
        items = dict(ex.map(one, rows))
    avkit.drop_raw("mmar")
    for r in rows:
        opts = [str(c).strip() for c in r["choices"]]
        yt = r.get("source") == "youtube" and re.match(r"^[A-Za-z0-9_-]{11}", r["id"])
        rec = mcq(rid("mmar", r["id"]), _state(r["question"]), Q_AUDIO, opts, text_index(r["answer"], opts), [items.get(r["id"])], "audio",
                  category=r["category"], task_type=r["modality"], modality=r["modality"], sub_category=r["sub-category"],
                  language=r.get("language"), source_id=f"yt:{yt.group(0)}" if yt else f"mmar:{r['id']}")
        if rec:
            yield rec


def _cached_audio(name: str, key: str) -> dict | None:
    p = avkit.media_dir(name, key) / "a.flac"
    return {"type": "audio", "path": str(p)} if p.exists() and p.stat().st_size > 0 else None


# ---- MMSU (ddwang2000) -----------------------------------------------------------------------------------------
def mmsu(split, cap, rng):
    repo = "ddwang2000/MMSU"
    files = avkit.parquet_files(repo, "data/")

    def handle(r):
        opts = [str(r[f"choice_{c}"]).strip() for c in "abcd" if r.get(f"choice_{c}")]
        gold = text_index(r["answer_gt"], opts)
        if gold is None:
            gold = letter_index(r["answer_gt"], len(opts))
        media = avkit.audio_from_bytes(blob(r["audio"]), "mmsu", r["id"], 30.0)
        return mcq(rid("mmsu", r["id"]), _state(r["question"]), Q_AUDIO, opts, gold, [media], "audio", category=r["category"],
                   task_type=r["task_name"], sub_category=r["sub-category"], domain=r.get("sub-sub-category"),
                   source_id=f"mmsu:{r['id']}")

    recs = benchkit.parquet_media(repo, files, "mmsu", handle, shards=2, inner=4)
    rng.shuffle(recs)
    yield from recs[:cap]


# ---- VoiceBench (spoken multiple choice subsets) ------------------------------------------------------------------
_OPTION_LINE = re.compile(r"^\s*([A-J])[\.\)]\s+(.*\S)\s*$")


def _parse_prompt(prompt: str) -> list[str]:
    """Option texts of a VoiceBench prompt ('question\\nA. x\\nB. y\\n...\\nWhat is the answer ...')."""
    opts, expect = [], "A"
    for line in prompt.splitlines():
        m = _OPTION_LINE.match(line)
        if m and m.group(1) == expect:
            opts.append(m.group(2))
            expect = chr(ord(expect) + 1)
    return opts


def _voicebench(name: str, prefix: str, split, cap, rng):
    repo = "hlt-lab/voicebench"
    files = avkit.parquet_files(repo, prefix)

    def handle(r):
        opts = _parse_prompt(r["prompt"])
        gold = letter_index(r["reference"], len(opts))
        key = rid(name, r["prompt"][:200], r.get("question_id", ""))
        media = avkit.audio_from_bytes(blob(r["audio"]), name, key, 30.0)
        # the question and all options are spoken: the text side only lists the option strings
        return mcq(key, _state("Listen to the spoken multiple-choice question and pick the answer."),
                   "Which option is the correct answer to the spoken question?", opts, gold, [media], "audio",
                   category=r.get("category") or name.split("_", 1)[1], source_id=f"{name}:{key}", task_type=r.get("src"))

    recs = benchkit.parquet_media(repo, files, name, handle, shards=2, inner=4)
    rng.shuffle(recs)
    yield from recs[:cap]


def voicebench_mmsu(split, cap, rng):
    yield from _voicebench("voicebench_mmsu", "mmsu/", split, cap, rng)


def voicebench_openbookqa(split, cap, rng):
    yield from _voicebench("voicebench_openbookqa", "openbookqa/", split, cap, rng)


register(DatasetSpec("mmau_mini", mmau_mini, ("test",), "gamma-lab-umd/MMAU-test-mini", "cc-by-4.0", "audio", eval_only=True, multimodal=True,
                     description="MMAU (v05.15.25) test-mini: 1000 audio questions with answers (eval only)"))
register(DatasetSpec("mmar", mmar, ("test",), "BoJack/MMAR", "mit", "audio", eval_only=True, multimodal=True,
                     description="MMAR: deep reasoning over mixed sound, music and speech (eval only)"))
register(DatasetSpec("mmsu", mmsu, ("test",), "ddwang2000/MMSU", "cc-by-4.0", "audio", eval_only=True, multimodal=True,
                     description="MMSU: massive multi-task spoken language understanding (eval only)"))
register(DatasetSpec("voicebench_mmsu", voicebench_mmsu, ("test",), "hlt-lab/voicebench", "apache-2.0", "audio", eval_only=True,
                     multimodal=True, description="VoiceBench mmsu: spoken MMLU multiple choice (eval only)"))
register(DatasetSpec("voicebench_openbookqa", voicebench_openbookqa, ("test",), "hlt-lab/voicebench", "apache-2.0", "audio",
                     eval_only=True, multimodal=True, description="VoiceBench openbookqa: spoken science multiple choice (eval only)"))
