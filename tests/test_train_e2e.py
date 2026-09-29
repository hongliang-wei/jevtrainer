import json

import pytest

from jevtrainer.config import TrainConfig
from jevtrainer.eval.runner import collect
from jevtrainer.model.load import load
from jevtrainer.schema import write_jsonl
from jevtrainer.train.trainer import Trainer


@pytest.mark.parametrize("readout,finetune", [("marker", "full"), ("pointer", "lora"), ("slot", "lora")])
def test_train_save_load(tiny_vl, record, tmp_path, readout, finetune):
    data = tmp_path / "train.jsonl"
    recs = []
    for i in range(24):
        r = type(record).from_dict(record.to_dict())
        r.id = f"r{i}"
        r.state = {"ticket": f"case {i}: " + ("double charge" if i % 2 else "app crash")}
        r.targets["team"].label = "billing" if i % 2 else "tech"
        recs.append(r)
    write_jsonl(recs, data)
    cfg = TrainConfig(model=tiny_vl, readout=readout, dataset=str(data), output_dir=str(tmp_path / "run"), finetune=finetune,
                      max_steps=4, batch_size=4, dtype="fp32", grad_ckpt=False, holdout=0.25, lora={"r": 4}, log_steps=1)
    result = Trainer(cfg).run()
    assert result["steps"] == 4 and "temperature" in result
    out = tmp_path / "run"
    assert (out / "readout.json").exists() and (out / "calibration.json").exists()
    losses = [json.loads(l)["loss"] for l in (out / "train_log.jsonl").read_text().splitlines()]
    assert len(losses) == 4
    b = load(out, "fp32")
    outs = collect(b, recs[:3])
    assert len(outs) == 9 and b.temperature["default"] > 0
