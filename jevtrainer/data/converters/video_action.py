"""Human action recognition on real videos (sound track kept when the source has one).

One video, one question "which option describes the action?": the true action class plus 3 distractor classes.
Distractors are other classes of the same dataset; for Something-Something v2 two of them share the first word of the
true template (hard negatives such as "Putting ..." vs "Putting ... on a surface"), the rest are random.
"""

from __future__ import annotations

import io
import os
import re
import zipfile
from pathlib import Path

from jevtrainer.data import avkit, vidkit
from jevtrainer.data.base import DatasetSpec, register, rid

Q_ACT = "Which option best describes the action shown in the video?"
Q_SSV2 = "Which option best describes the action in the video? 'something' stands for the object(s) involved."


def _item(name, split, cap, rng, rows, key_of, label_of, labels, media, hard=None, **extra):
    n = 0
    for r in rows:
        m = media.get(key_of(r))
        if not m:
            continue
        gold = label_of(r)
        others = [x for x in labels if x != gold]
        pool = rng.sample(others, min(3, len(others)))
        if hard:
            near = [x for x in hard(gold) if x != gold]
            if near:
                pool = list(dict.fromkeys(rng.sample(near, min(2, len(near))) + pool))[:3]
        rec = vidkit.option_record(rid(name, split, key_of(r)), m, "What action is shown in the video?", gold, pool, rng,
                                   Q_ACT, area="video", dataset=name)
        if rec:
            rec.media = [m]
            n += 1
            yield rec


# ---- Something-Something v2 ---------------------------------------------------------------------------------
_SSV2 = "morpheushoc/something-something-v2"


def _ssv2_rows(split):
    if split == "test":  # labels of the official test set (released after the challenge): id;template
        p = vidkit.get_file(_SSV2, "test-answers.csv")
        rows = []
        for line in open(p, encoding="utf8"):
            i, _, t = line.strip().partition(";")
            if i and t:
                rows.append({"id": i, "label": t, "template": t})
        return rows
    return vidkit.read_json(_SSV2, {"train": "train.json", "val": "validation.json"}[split])


def ssv2(split, cap, rng):
    """20BN Something-Something v2 (community mirror of the archive). Class = caption template with 'something'
    placeholders (174 classes); at most ceil(cap/174) videos per class, taken in archive order."""
    rows = _ssv2_rows(split)
    tmpl = {str(r["id"]): re.sub(r"[\[\]]", "", r["template"]).strip() for r in rows}
    labels = sorted(set(tmpl.values()))
    quota = -(-cap // len(labels))
    raw = avkit.raw_dir("ssv2") / "videos"
    if not list(raw.glob("20bn-something-something-v2-[0-9][0-9]")):
        avkit.fetch(_SSV2, "ssv2", allow=["videos/*"], workers=4)
    parts = []  # the archive is one stream cut into pieces: only a gap-free prefix can be read
    for i in range(20):
        p = raw / f"20bn-something-something-v2-{i:02d}"
        if not p.exists():
            break
        parts.append(p)
    seen: dict[str, int] = {}
    total = [0]

    def keep(member: str) -> bool:
        vid = Path(member).stem
        c = tmpl.get(vid)
        if c is None or seen.get(c, 0) >= quota or total[0] >= cap:
            return False
        seen[c] = seen.get(c, 0) + 1
        total[0] += 1
        return True

    stream = ((Path(n).stem, n, b) for n, b in vidkit.tar_members(parts, keep, done=lambda: total[0] >= cap))
    media = vidkit.convert_stream("ssv2", stream, frames=8, max_side=448, keep_audio=False)
    first = {c.split()[0].lower(): [] for c in labels}
    for c in labels:
        first[c.split()[0].lower()].append(c)
    ids = [r["id"] for r in rows if str(r["id"]) in media]
    rng.shuffle(ids)
    yield from _item("ssv2", split, cap, rng, ids, str, lambda i: tmpl[str(i)], labels, media,
                     hard=lambda g: first.get(g.split()[0].lower(), []))


# ---- HMDB51 ---------------------------------------------------------------------------------------------------
_HMDB = "divm/hmdb51"


def hmdb51(split, cap, rng):
    """HMDB51 (51 classes, movie / web clips). Official-style split of the mirror: train 70% / validation 10% / test 20%."""
    import pandas as pd

    sp = {"train": "train", "val": "validation", "test": "test"}[split]
    df = vidkit.read_csv(_HMDB, f"{sp}/metadata.csv")
    rows = df.to_dict("records")
    rows = avkit.balanced(rows, lambda r: r["label"], cap, rng)
    have = vidkit.listing(_HMDB, sp + "/")
    rows = [r for r in rows if f"{sp}/{r['file_name']}" in have]
    media = vidkit.convert_hub(_HMDB, "hmdb51", {r["video_id"]: f"{sp}/{r['file_name']}" for r in rows}, frames=8, max_side=448)
    labels = sorted({str(x).replace("_", " ") for x in df["label"]})
    yield from _item("hmdb51", split, cap, rng, rows, lambda r: r["video_id"], lambda r: str(r["label"]).replace("_", " "), labels, media)


# ---- Kinetics-400 (partial: archive parts hold ~1000 random-class videos each) -------------------------------------
_K400 = "kiyoonkim/kinetics-400-targz"


def kinetics400(split, cap, rng):
    """Kinetics-400 from the community tar.gz mirror (the 700 version is only hosted as 40 GB zips). The archive parts
    hold ~1000 videos of random classes, so only the first JT_K400_PARTS (default 10 train / 2 val) parts are fetched
    and at most ceil(cap/400) videos per class are kept. 10-second clips with their sound track."""
    sp = {"train": "train", "val": "val"}[split]
    df = vidkit.read_csv(_K400, f"annotations/{sp}.csv")
    key = {f"{r.youtube_id}_{int(r.time_start):06d}_{int(r.time_end):06d}": r.label for r in df.itertuples()}
    labels = sorted(set(key.values()))
    nparts = int(os.environ.get("JT_K400_PARTS", "10" if split == "train" else "2"))
    nparts = min(nparts, -(-cap // 800) + 1)
    quota = -(-cap // len(labels))
    seen: dict[str, int] = {}
    media: dict[str, dict] = {}
    from concurrent.futures import ThreadPoolExecutor

    def fetch(i):
        return avkit.fetch_file(_K400, f"{sp}/part_{i}.tar.gz", "kinetics400")

    with ThreadPoolExecutor(1) as dl:
        nxt = dl.submit(fetch, 0)
        for i in range(nparts):
            part = nxt.result()
            if i + 1 < nparts:
                nxt = dl.submit(fetch, i + 1)
            if part is None:
                continue

            def keep(m):
                k = Path(m).stem
                c = key.get(k)
                if c is None or seen.get(c, 0) >= quota or sum(seen.values()) >= cap:
                    return False
                seen[c] = seen.get(c, 0) + 1
                return True

            stream = ((Path(n).stem, n, b) for n, b in vidkit.tar_members([part], keep))
            media.update(vidkit.convert_stream("kinetics400", stream, frames=8, max_side=448, audio_s=10))
            Path(part).unlink(missing_ok=True)
    ids = sorted(media)
    rng.shuffle(ids)
    yield from _item("kinetics400", split, cap, rng, ids[:cap], str, lambda k: key[k], labels, media)


# ---- UCF101 -----------------------------------------------------------------------------------------------------
def _camel(s: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s).lower().replace("y o y o", "yoyo")


def ucf101(split, cap, rng):
    """UCF101 (101 classes, YouTube clips with sound). Official split 1: trainlist01 -> train, testlist01 -> test."""
    repo = "quchenyuan/UCF101-ZIP"
    splits = avkit.fetch_file(repo, "UCF101TrainTestSplits-RecognitionTask.zip", "ucf101")
    with zipfile.ZipFile(splits) as z:
        name = {"train": "trainlist01.txt", "test": "testlist01.txt"}[split]
        member = next(n for n in z.namelist() if n.endswith(name))
        files = [ln.split()[0] for ln in z.read(member).decode().splitlines() if ln.strip()]
    rows = [{"file": f, "cls": f.split("/")[0]} for f in files]
    rows = avkit.balanced(rows, lambda r: r["cls"], cap, rng)
    archive = avkit.fetch_file(repo, "UCF-101.zip", "ucf101")
    tmp = avkit.raw_dir("ucf101") / "x"
    tmp.mkdir(exist_ok=True)
    jobs = {}
    with zipfile.ZipFile(archive) as z:
        names = {Path(n).name: n for n in z.namelist() if n.endswith(".avi")}
        for r in rows:
            n = names.get(Path(r["file"]).name)
            if n is None:
                continue
            key = Path(n).stem
            if vidkit.cached_item("ucf101", key) is None:
                z.extract(n, tmp)
            jobs[key] = {"src": tmp / n}
    media = vidkit.convert_local("ucf101", jobs, frames=8, max_side=448, audio_s=10)
    labels = sorted({_camel(r["cls"]) for r in rows})
    yield from _item("ucf101", split, cap, rng, [r for r in rows if Path(r["file"]).stem in media], lambda r: Path(r["file"]).stem,
                     lambda r: _camel(r["cls"]), labels, media)


register(DatasetSpec("ssv2", ssv2, ("train", "val", "test"), _SSV2, "other (20BN research licence)", "video", multimodal=True,
                     description="Something-Something v2: 174 object-agnostic hand/object action templates (no sound track)"))
register(DatasetSpec("hmdb51", hmdb51, ("train", "val", "test"), _HMDB, "other (CC BY 4.0 annotations)", "video", multimodal=True,
                     description="HMDB51: 51 human action classes, which one is shown"))
register(DatasetSpec("kinetics400", kinetics400, ("train", "val"), _K400, "CC BY 4.0 (annotations)", "video", multimodal=True,
                     description="Kinetics-400 subset (class balanced, with sound): choose the action among 4"))
register(DatasetSpec("ucf101", ucf101, ("train", "test"), "quchenyuan/UCF101-ZIP", "other (research)", "video", multimodal=True,
                     description="UCF101 split 1: 101 action classes in YouTube clips with sound"))
