"""Graded speech judgements: how well does a non-native speaker pronounce (speechocean762), how good does a recording sound (NISQA).

One clip + one `score` question per record. The grade is the corpus' own expert / listener rating, cut into five levels (the
cut points are in each docstring). Clips are sampled evenly over the levels. Official train / test splits are kept.
"""

from __future__ import annotations

from jevtrainer.data import audiokit, avkit
from jevtrainer.data.base import DatasetSpec, register

_SO = "mispeech/speechocean762"

# speechocean762 scores are integers 0-10 given by five experts (averaged and rounded in the corpus release)
SO_EDGES = [5, 7, 8, 9]  # <5 | 5-6 | 7 | 8 | 9-10
SO_LEVELS = {
    "accuracy": [
        "very poor: many words are mispronounced, the sentence is hard to understand (score 0-4 of 10)",
        "poor: a good number of mispronounced phones, a strong foreign accent (score 5-6)",
        "fair: some mispronunciations, but most words are clear (score 7)",
        "good: only a few mispronounced phones (score 8)",
        "excellent: pronunciation close to a native speaker (score 9-10)",
    ],
    "fluency": [
        "very disfluent: long pauses, repetitions and broken sentences (score 0-4 of 10)",
        "disfluent: frequent hesitations and stops, the sentence is split into pieces (score 5-6)",
        "fair: some hesitation or repetition, the sentence is mostly connected (score 7)",
        "fluent: smooth, with only minor hesitation (score 8)",
        "very fluent: natural speed and rhythm without hesitation (score 9-10)",
    ],
    "total": [
        "very poor: overall pronunciation is hard to understand (total score 0-4 of 10)",
        "poor: overall pronunciation has many problems (total score 5-6)",
        "fair: overall pronunciation is understandable with some problems (total score 7)",
        "good: overall pronunciation is good with minor problems (total score 8)",
        "excellent: overall pronunciation is near native (total score 9-10)",
    ],
    "prosodic": [
        "very poor: monotonous or erratic intonation, stress and rhythm (score 0-4 of 10)",
        "poor: intonation and stress are often wrong (score 5-6)",
        "fair: intonation and rhythm are acceptable with some mistakes (score 7)",
        "good: natural intonation and stress with small slips (score 8)",
        "excellent: intonation, stress and rhythm like a native speaker (score 9-10)",
    ],
}
SO_Q = {
    "accuracy": "How accurate is the pronunciation of the speaker (a non-native English learner reading a sentence)?",
    "fluency": "How fluent is the speaker (a non-native English learner reading a sentence)?",
    "total": "What is the overall pronunciation quality of the speaker (a non-native English learner reading a sentence)?",
    "prosodic": "How good are the intonation, stress and rhythm of the speaker (a non-native English learner reading a sentence)?",
}


def _speechocean(dim: str):
    def build(split, cap, rng):
        files = avkit.parquet_files(_SO, f"data/{split}-")
        return audiokit.clip_score(
            f"speechocean762_{dim}", split, cap, rng, _SO, files, ["accuracy", "fluency", "total", "prosodic", "text", "speaker"],
            level_of=lambda r: audiokit.bucket(r[dim], SO_EDGES), question=SO_Q[dim], levels=SO_LEVELS[dim],
            key_of=lambda r: f"{r['_file']}:{r['_i']}", media_name="speechocean762",
            meta_of=lambda r: {"task": dim, "score10": r[dim], "speaker": r["speaker"]},
        )

    build.__doc__ = (f"Pronunciation {dim} of a non-native English learner reading a sentence (speechocean762, 5000 clips by 250 "
                     f"speakers, official train / test). Answer = the expert `{dim}` score (0-10) cut into five levels: <5, 5-6, 7, 8, 9-10.")
    return build


for _dim in ("accuracy", "fluency", "total", "prosodic"):
    register(DatasetSpec(f"speechocean762_{_dim}", _speechocean(_dim), ("train", "test"), _SO, "apache-2.0", "audio", multimodal=True,
                         description=f"speechocean762: expert {_dim} score of a learner's English pronunciation (5 levels)"))
