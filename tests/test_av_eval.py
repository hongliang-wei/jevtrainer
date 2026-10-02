"""Evaluation ablations (mute / black / shuffle_audio) and source-id based train/eval overlap. No model is loaded."""

import pytest
from PIL import Image

from jevtrainer.config import EvalConfig, load_config
from jevtrainer.data.base import mcq_record
from jevtrainer.data.mixture import Source, build_mixture, source_ids, state_keys
from jevtrainer.eval.ablate import ABLATIONS, ablate_records
from jevtrainer.eval.runner import breakdown
from jevtrainer.media import MediaOptions, load_media
from jevtrainer.schema import Record


def _frames(tmp_path, key, size=(64, 48), n=4):
    d = tmp_path / key
    d.mkdir(exist_ok=True)
    out = []
    for i in range(n):
        p = d / f"f{i:02d}.jpg"
        Image.new("RGB", size, (200, 30, 30)).save(p)
        out.append(str(p))
    return out


def av_record(tmp_path, key, **meta):
    r = mcq_record(f"r-{key}", {"clip": "<video:1>", "question": "q?"}, "pick", ["a", "b", "c"], 1, **meta)
    r.media = [{"type": "video", "frames": _frames(tmp_path, key), "fps": 1.0, "duration": 4.0, "audio": str(tmp_path / f"{key}.flac")}]
    return r


def audio_record(key):
    r = mcq_record(f"a-{key}", {"clip": "<audio:1>", "question": "what is it?"}, "pick", ["a", "b"], 0)
    r.media = [{"type": "audio", "path": f"/data/{key}.flac"}]
    return r


def test_ablate_none_is_identity(tmp_path):
    recs = [av_record(tmp_path, "k1")]
    assert ablate_records(recs, "none") is recs
    with pytest.raises(ValueError):
        ablate_records(recs, "deaf")
    assert ABLATIONS == ("none", "mute", "black", "shuffle_audio")


def test_ablate_mute(tmp_path):
    v, a = av_record(tmp_path, "k1"), audio_record("s1")
    out = ablate_records([v, a], "mute")
    assert "audio" not in out[0].media[0] and out[0].media[0]["frames"] == v.media[0]["frames"]
    assert out[1].media == [] and "<audio:" not in out[1].state_text()
    assert out[0].meta["ablate"] == "mute" and out[1].meta["ablate"] == "mute"
    assert "audio" in v.media[0] and len(a.media) == 1  # inputs untouched
    out[0].validate()


def test_ablate_mute_keeps_other_media_tags():
    r = mcq_record("x", {"v": "<video:1>", "a": "<audio:1> and <audio:2>", "q": "?"}, "pick", ["a", "b"], 0)
    r.media = [{"type": "video", "path": "clip.mp4"}, {"type": "audio", "path": "x.flac"}, {"type": "audio", "path": "y.flac"}]
    m = ablate_records([r], "mute")[0]
    assert [i["type"] for i in m.media] == ["video"] and m.media[0]["mute"] is True
    assert m.state["v"] == "<video:1>" and "<audio" not in m.state["a"]


def test_ablate_black(tmp_path):
    v = av_record(tmp_path, "k1", task_type="x")
    out = ablate_records([v], "black", black_dir=tmp_path / "blk")[0]
    m = out.media[0]
    assert len(m["frames"]) == 4 and len(set(m["frames"])) == 1  # one reused picture
    assert m["audio"] == v.media[0]["audio"] and m["duration"] == 4.0  # sound and timing kept
    im = Image.open(m["frames"][0])
    assert im.size == (64, 48) and im.convert("L").getextrema() == (0, 0)
    assert out.meta["ablate"] == "black" and out.meta["task_type"] == "x"
    assert v.media[0]["frames"][0].endswith("f00.jpg")
    # a second record of the same frame size reuses the very same file
    w = ablate_records([av_record(tmp_path, "k2")], "black", black_dir=tmp_path / "blk")[0]
    assert w.media[0]["frames"][0] == m["frames"][0]


def test_ablate_black_loads_as_black(tmp_path):
    v = av_record(tmp_path, "k1")
    v.media[0].pop("audio")
    out = ablate_records([v], "black", black_dir=tmp_path / "blk")[0]
    clip = load_media(out, MediaOptions(video_frames=2)).videos[0]
    assert len(clip.frames) == 2 and all(f.convert("L").getextrema() == (0, 0) for f in clip.frames)


def test_ablate_shuffle_audio(tmp_path):
    recs = [av_record(tmp_path, f"k{i}") for i in range(5)] + [audio_record("s1"), audio_record("s2")]
    own = [r.media[0].get("audio") or r.media[0]["path"] for r in recs]
    out = ablate_records(recs, "shuffle_audio", seed=3)
    new = [r.media[0].get("audio") or r.media[0]["path"] for r in out]
    assert all(a != b for a, b in zip(own, new))  # every record hears something else
    assert sorted(new) == sorted(own)  # a permutation of the dataset's sounds
    assert [r.media[0]["frames"] for r in out[:5]] == [r.media[0]["frames"] for r in recs[:5]]
    assert ablate_records(recs, "shuffle_audio", seed=3)[0].media[0] == out[0].media[0]  # deterministic
    assert recs[0].media[0]["audio"] == own[0]
    assert all(r.meta["ablate"] == "shuffle_audio" for r in out)


def test_ablate_shuffle_audio_edge_cases(tmp_path):
    silent = mcq_record("t", "just text", "pick", ["a", "b"], 0)
    one = av_record(tmp_path, "only")
    out = ablate_records([silent, one], "shuffle_audio")  # nothing to swap with: unchanged sound
    assert out[1].media[0]["audio"] == one.media[0]["audio"] and out[0].media == []
    video_no_sound = av_record(tmp_path, "k9")
    video_no_sound.media[0].pop("audio")
    out = ablate_records([video_no_sound, audio_record("s1"), audio_record("s2")], "shuffle_audio")
    assert "audio" not in out[0].media[0]  # a silent clip stays silent


def test_ablate_in_eval_config(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model: m\nbenchmarks: av-omni\nablate: mute\nreadout_options: {video_frames: 16}\n")
    c = load_config(EvalConfig, p)
    assert c.ablate == "mute" and c.benchmarks == ["av-omni"] and c.readout_options == {"video_frames": 16}
    assert load_config(EvalConfig, p, ["ablate=black"]).ablate == "black"
    with pytest.raises(SystemExit):
        load_config(EvalConfig, p, ["ablate=deaf"])
    assert EvalConfig(model="m").ablate == "none"


def test_breakdown():
    import torch

    def o(logits, target, **meta):
        return {"logits": torch.tensor(logits), "target": target, "meta": meta}

    outs = [o([1.0, 0.0], 0, task_type="a", duration="short"), o([0.0, 1.0], 0, task_type="a", duration="short"),
            o([0.0, 1.0], 1, task_type="b", duration="short"), o([0.0, 1.0], None, task_type="b")]
    b = breakdown(outs)
    assert b["task_type"]["a"] == {"n": 2, "accuracy": 0.5} and b["task_type"]["b"] == {"n": 1, "accuracy": 1.0}
    assert "duration" not in b  # one value only: not a breakdown
    twice = breakdown([o([1.0, 0.0], 0, category="a", task_type="a"), o([0.0, 1.0], 0, category="b", task_type="b")])
    assert list(twice) == ["category"]  # task_type repeats category


# ---- overlap by source id -------------------------------------------------------------------------------------
def test_source_ids_explicit_and_derived(tmp_path):
    r = av_record(tmp_path, "k1", source_id="yt:abc")
    assert source_ids(r) == {"yt:abc"}
    r = av_record(tmp_path, "k1", source_id=["yt:a", "yt:b"])
    assert source_ids(r) == {"yt:a", "yt:b"}
    r = av_record(tmp_path, "-0Ou4eFM3is_000030")  # media key looks like <YouTube id>_<start>
    assert source_ids(r) == {"yt:-0Ou4eFM3is"}
    r = av_record(tmp_path, "00342")
    assert source_ids(r) == set()
    r = mcq_record("x", "t", "i", ["a", "b"], 0)
    r.media = [{"type": "audio", "path": "/c/media/ds/dQw4w9WgXcQ/a.flac"}]
    assert source_ids(r) == {"yt:dQw4w9WgXcQ"}


def test_overlap_by_source_id_not_by_path(tmp_path, monkeypatch):
    # same source video, different frame files and different question: only the source id links them
    ev = av_record(tmp_path, "evalcopy", source_id="yt:SAMEVIDEO01")
    ev.state = {"clip": "<video:1>", "question": "eval question"}
    tr_same = av_record(tmp_path, "traincopy", source_id="yt:SAMEVIDEO01")
    tr_same.state = {"clip": "<video:1>", "question": "a different train question"}
    tr_other = av_record(tmp_path, "other", source_id="yt:OTHERVIDEO2")
    tr_other.state = {"clip": "<video:1>", "question": "a different train question"}
    assert state_keys(ev) & state_keys(tr_same)
    assert not state_keys(ev) & state_keys(tr_other)
    # path-only overlap keeps working (no source ids)
    a, b = av_record(tmp_path, "p1"), av_record(tmp_path, "p1b")
    b.media = a.media
    assert state_keys(a) & state_keys(b)

    import jevtrainer.data.mixture as mx

    monkeypatch.setattr(mx, "check_trainable", lambda name: None)
    monkeypatch.setattr(mx, "load", lambda name, split, n, seed: [tr_same, tr_other])
    kept, report = build_mixture([Source("anything")], 0, exclude=[ev])
    assert [r.id for r in kept] == [tr_other.id]
    assert report["anything"]["overlap_removed"] == 1


def test_overlap_is_unchanged_for_text_records():
    a = mcq_record("a", "same text", "i", ["x", "y"], 0)
    b = mcq_record("b", "same text", "i", ["x", "y"], 1)
    c = mcq_record("c", "other text", "i", ["x", "y"], 1)
    assert state_keys(a) & state_keys(b) and not state_keys(a) & state_keys(c)
    assert isinstance(a, Record)
