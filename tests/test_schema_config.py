import pytest

from jevtrainer.config import TrainConfig, load_config
from jevtrainer.data.augment import AugmentConfig, augment, fit_options, pack
from jevtrainer.schema import Record


def test_record_roundtrip(record):
    d = record.to_dict()
    again = Record.from_dict(d).validate()
    assert again.targets["urgent"].label == "yes"
    assert again.targets["anger"].probs == {"0": 0.2, "1": 0.7, "2": 0.1}
    assert again.questions["anger"].labels() == ["0", "1", "2"]
    assert record.target_index("team") == 0


def test_bad_record():
    with pytest.raises(ValueError):
        Record.from_dict({"id": "x", "state": "", "questions": {"q": {"type": "choice", "instructions": "", "criteria": {"a": ""}}}}).validate()


def test_minimal_config(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model: m\nreadout: pointer\ndataset: banking77, boolq:validation\n")
    c = load_config(TrainConfig, p, ["lr=3e-5", "lora.r=8"])
    assert [s.name for s in c.dataset] == ["banking77", "boolq"] and c.dataset[1].split == "validation"
    assert c.lr == 3e-5 and c.lora.r == 8 and c.lora.alpha == 16


def test_unknown_field_is_explained(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model: m\nreadout: marker\ndataset: x\nlearning_rate: 1\n")
    with pytest.raises(SystemExit, match="unknown field"):
        load_config(TrainConfig, p)


def test_augment_keeps_gold(record):
    import random

    rng = random.Random(0)
    for _ in range(20):
        r = augment([record], AugmentConfig(p_none=0.5, p_distract=0.5), rng)[0].validate()
        assert r.targets["team"].label in r.questions["team"].criteria
    big = Record.from_dict({"id": "b", "state": "", "questions": {"q": {"type": "choice", "instructions": "", "criteria": {str(i): "" for i in range(100)}}},
                            "targets": {"q": {"label": "42"}}})
    small = fit_options(big, 10, rng).validate()
    assert len(small.questions["q"].criteria) == 10 and "42" in small.questions["q"].criteria
    packed = pack([record, record], rng).validate()
    assert len(packed.questions) == 6


def test_registries_load():
    from jevtrainer.registry import BENCHMARKS, DATASETS, FAMILIES, READOUTS

    assert {"marker", "slot", "pointer"} <= set(READOUTS.names())
    assert {"generic", "qwen", "encoder"} <= set(FAMILIES.names())
    assert "banking77" in DATASETS and "jevbench_hard" in BENCHMARKS
    with pytest.raises(KeyError, match="Did you mean"):
        DATASETS.get("banking78")
