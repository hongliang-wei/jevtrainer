"""Chinese proofreading (spelling errors, domain text correction), dialogue contradiction (CDConv) and
translated preference pairs."""

from __future__ import annotations

from jevtrainer.data.base import DatasetSpec, choice_record, noul_record, register, rid
from jevtrainer.data.converters.longdoc import hash_split
from jevtrainer.data.converters.zh import S3, ZH_MAX, _pair_zh, _path, _rows

TYPO_Q = "这段文字中是否有错别字或用词错误？"


def csc(split, cap, rng):
    """Erroneous sentences as yes; their corrected text (or already-correct sentences) as no, 50 / 50."""
    rows = _rows("shibing624/CSC", {"train": "train.json", "validation": "dev.json", "test": "test.json"}[split])
    rng.shuffle(rows)
    for i, r in enumerate(rows[:cap]):
        bad = r["original_text"] != r["correct_text"]
        err = bad and rng.random() < 0.5
        text = r["original_text"] if err or not bad else r["correct_text"]
        yield noul_record(rid("csc", split, i), {"文本": text}, TYPO_Q, err, "有错误", "没有错误", area="language", lang="zh")


def _correction_files():
    from huggingface_hub import list_repo_files

    return sorted(f for f in list_repo_files("shibing624/chinese_text_correction", repo_type="dataset") if f.endswith(".tsv"))


def text_correction_zh(split, cap, rng):
    rows = []
    for fn in _correction_files():
        try:
            data = _rows("shibing624/chinese_text_correction", fn)
        except Exception:
            continue
        rows += [(fn[:-4], str(r["source"]), str(r["target"])) for r in data
                 if isinstance(r.get("source"), str) and isinstance(r.get("target"), str) and len(r["source"]) <= ZH_MAX
                 and hash_split(r["source"]) == split]
    rng.shuffle(rows)
    for i, (dom, src, tgt) in enumerate(rows[:cap]):
        yield noul_record(rid("text_corr_zh", split, i), {"文本": src}, "这段文字是否需要校对修改（错别字、语法或标点错误）？", src != tgt,
                          "需要修改", "无需修改", area="language", lang="zh", domain=dom)


CDCONV = {"0": "没有矛盾", "1": "句内矛盾：机器人最后一句话自身前后矛盾", "2": "角色混淆：机器人把用户说过的话当成自己的",
          "3": "历史矛盾：机器人最后一句话与它之前说过的话矛盾"}


def cdconv(split, cap, rng):
    import pandas as pd

    sub = "dev" if split == "validation" else split
    fn = _path("thu-coai/cdconv", f"split/4class_{sub}.tsv")
    df = pd.read_csv(fn, sep="\t", header=None, names=["u1", "b1", "u2", "b2", "label"], dtype=str, quoting=3)
    rows = [r for r in df.to_dict("records") if r["label"] in CDCONV]
    rng.shuffle(rows)
    for i, r in enumerate(rows[:cap]):
        dialog = f"用户：{r['u1']}\n机器人：{r['b1']}\n用户：{r['u2']}\n机器人：{r['b2']}"
        yield choice_record(rid("cdconv", split, i), {"对话": dialog}, "机器人的最后一句回复是否存在矛盾？属于哪一类？", dict(CDCONV), r["label"],
                            area="dialogue", lang="zh")


def dpo_pairs_zh(split, cap, rng):
    rows = [r for r in _rows("wenbopan/Chinese-dpo-pairs", "train/train-00000-of-00001.parquet")
            if hash_split(r["prompt"]) == split and len(str(r["rejected"]).strip()) >= 2 and r["chosen"] != r["rejected"]]
    for i, r in enumerate(rng.sample(rows, min(cap, len(rows)))):
        prompt = (f"{r['system']}\n\n" if r.get("system") else "") + r["prompt"]
        yield _pair_zh("dpo_pairs_zh", f"{split}{i}", prompt, str(r["chosen"]), str(r["rejected"]), True, rng, area="preference")


for spec in [
    DatasetSpec("csc", csc, S3, "shibing624/CSC (SIGHAN + Wang271K; corrected text as negatives)", "apache-2.0", "language"),
    DatasetSpec("text_correction_zh", text_correction_zh, S3, "shibing624/chinese_text_correction (law, medical, official docs, ...; re-split)",
                "apache-2.0", "language"),
    DatasetSpec("cdconv", cdconv, S3, "thu-coai/cdconv (4-class)", "cc-by-nc-4.0", "dialogue", tags=["nc"]),
    DatasetSpec("dpo_pairs_zh", dpo_pairs_zh, S3, "wenbopan/Chinese-dpo-pairs (re-split)", "mit", "preference", tags=["translated"]),
]:
    spec.tags = list(dict.fromkeys(spec.tags + ["zh"]))
    register(spec)
