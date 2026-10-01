"""General video question answering with answer options (multiple choice straight from the dataset)."""

from __future__ import annotations

import io
from pathlib import Path

from jevtrainer.data import avkit, vidkit
from jevtrainer.data.base import DatasetSpec, mcq_record, register, rid

_NQ = "VLM2Vec/nextqa-rawvideo"
_NQ_ANN = "https://raw.githubusercontent.com/doc-doc/NExT-QA/main/dataset/nextqa/"
Q_NQ = "Which option correctly answers the question about the video?"


def nextqa(split, cap, rng):
    """NExT-QA multiple choice (5 options; causal, temporal and descriptive questions about everyday videos). The hub only
    hosts the raw videos of the official val and test sets (no train videos), so the splits are `val` (val.csv from the
    authors' repository) and `test` (official test questions); one random question per video."""
    import pandas as pd

    if split == "val":
        df = pd.read_csv(io.StringIO(vidkit.http_text(_NQ_ANN + "val.csv", "nextqa", "val.csv")))
    else:
        df = vidkit.read_parquet("lmms-eval/NExTQA", "MC/test-00000-of-00001.parquet")
    files = {Path(f).stem: f for f in vidkit.listing(_NQ) if f.endswith(".mp4")}
    rows = [r for r in df.to_dict("records") if str(r["video"]) in files]
    rng.shuffle(rows)
    seen, uniq = set(), []
    for r in rows:
        if r["video"] not in seen:
            seen.add(r["video"])
            uniq.append(r)
    uniq = uniq[:cap]
    media = vidkit.convert_hub(_NQ, "nextqa", {str(r["video"]): files[str(r["video"])] for r in uniq}, frames=8, max_side=448, audio_s=30)
    for r in uniq:
        m = media.get(str(r["video"]))
        if not m:
            continue
        rec = mcq_record(rid("nextqa", split, r["video"], r["qid"]), {"clip": "<video:1>", "question": str(r["question"]).strip() + "?"}, Q_NQ,
                         [r[f"a{i}"] for i in range(5)], int(r["answer"]), area="video", dataset="nextqa", question_type=str(r.get("type")))
        if rec:
            rec.media = [m]
            yield rec


register(DatasetSpec("nextqa", nextqa, ("val", "test"), "lmms-eval/NExTQA", "mit", "video", multimodal=True,
                     description="NExT-QA multiple choice video QA (val and test videos only)"))
