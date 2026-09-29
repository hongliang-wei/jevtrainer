import pytest
import torch
from conftest import DEVICE, on_device

from jevtrainer.model.load import build, prepare_finetune, trainable_parameters
from jevtrainer.predict import answer, probs
from jevtrainer.train.batching import Collator, DecisionModel, to_device
from jevtrainer.train.calibrate import fit_temperature

READOUTS = ["marker", "slot", "pointer"]


def run(b, records, grad=True):
    batch = Collator(b.readout, 4096)(records)
    with torch.set_grad_enabled(grad):
        return DecisionModel(b.model, b.readout)(to_device(batch, DEVICE)), batch


@pytest.mark.parametrize("readout", READOUTS)
def test_forward_backward_lora(tiny_vl, record, readout):
    b = build(tiny_vl, readout, dtype="fp32")
    prepare_finetune(b, "lora", {"r": 4}, grad_ckpt=False)
    on_device(b)
    logits, batch = run(b, [record, record])
    assert len(batch["reads"]) == 6
    assert [z.numel() for z in logits] == [3, 2, 3] * 2
    loss = sum(-torch.log_softmax(z, -1)[t["index"]] for z, t in zip(logits, batch["targets"]))
    loss.backward()
    lora_grads = [p.grad for n, p in b.model.named_parameters() if "lora_" in n and p.grad is not None]
    assert lora_grads and any(g.abs().sum() > 0 for g in lora_grads)
    assert all(p.grad is not None for p in b.readout.parameters() if p.requires_grad)
    assert not any("visual" in n for n, p in b.model.named_parameters() if p.requires_grad)


def test_marker_reads_before_placeholder(tiny_vl, record):
    b = build(tiny_vl, "marker", dtype="fp32")
    row = b.readout.encode(record)[0]
    dec = b.readout.special_ids["<decision>"]
    assert [row.input_ids[rd.info["pos"] + 1] for rd in row.reads] == [dec] * 3
    assert row.input_ids.count(dec) == 3  # the state's literal "<decision>" was neutralised


def test_slot_order_is_label_order(tiny_vl, record):
    b = build(tiny_vl, "slot", dtype="fp32")
    rows = b.readout.encode(record)
    slots = rows[0].reads[0].info["slots"]
    assert sorted(slots) == [0, 1, 2]
    text = b.tok.decode(rows[0].input_ids)
    for label, s in zip(["billing", "tech", "other"], slots):
        assert f'"t": "<|decision_{s:03d}|>", "n": "{label}"' in text


def test_pointer_rows_per_question(tiny_vl, record):
    b = build(tiny_vl, "pointer", dtype="fp32")
    rows = b.readout.encode(record)
    assert len(rows) == 3
    assert [len(r.reads[0].info["opts"]) for r in rows] == [3, 2, 3]


@pytest.mark.parametrize("readout", READOUTS)
def test_image_record(tiny_vl, image_record, readout):
    b = on_device(build(tiny_vl, readout, dtype="fp32"))
    b.model.eval()
    z, batch = run(b, [image_record], grad=False)
    assert batch["pixel_values"] is not None
    assert z[0].numel() == 3 and torch.isfinite(z[0]).all()


def test_text_only_model(tiny_text, record):
    for readout in READOUTS:
        b = build(tiny_text, readout, dtype="fp32")
        prepare_finetune(b, "full", {}, grad_ckpt=False)
        z, _ = run(on_device(b), [record])
        assert [x.numel() for x in z] == [3, 2, 3]
        assert len(trainable_parameters(b)) > 10


def test_predict_and_calibrate(record):
    p = probs(torch.tensor([2.0, 0.0, -1.0]), 3)
    a = answer(record.questions["team"], p)
    assert a["choice"] == "billing" and 0 < a["confidence"] < 1
    s = answer(record.questions["anger"], torch.tensor([0.0, 0.5, 0.5]))
    assert abs(s["score"] - 1.5) < 1e-6
    n = answer(record.questions["urgent"], torch.tensor([0.25, 0.75]))
    assert n["noul"] == 0.75
    over = [torch.tensor([6.0, 0.0])] * 50 + [torch.tensor([0.0, 6.0])] * 50
    t = fit_temperature(over, [2] * 100, [0] * 70 + [1] * 30)
    assert t > 1.5
