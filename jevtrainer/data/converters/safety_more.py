"""Content-safety and preference judgments on real assistant traffic: Aegis 2.0 (prompt + response safety),
BeaverTails (response harmfulness), PKU-SafeRLHF (safer / better of two responses), Skywork preference pairs."""

from __future__ import annotations

from jevtrainer.data.base import DatasetSpec, hf, noul_record, register, rid, take
from jevtrainer.data.converters.longdoc import hash_split
from jevtrainer.schema import Question, Record, Target

PAIR = {"response_a": "Response A", "response_b": "Response B"}


def aegis2(split, cap, rng):
    ds = take(hf("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split=split), cap, rng)
    for i, r in enumerate(ds):
        if not r["prompt"] or r["prompt"] == "REDACTED" or r["prompt_label"] not in ("safe", "unsafe"):
            continue
        state = {"user_prompt": r["prompt"][:4000]}
        qs = {"prompt_unsafe": Question("noul", "Is the user's prompt unsafe (asks for or contains harmful content)?")}
        ts = {"prompt_unsafe": Target("yes" if r["prompt_label"] == "unsafe" else "no")}
        if r["response"] and r["response_label"] in ("safe", "unsafe"):
            state["assistant_response"] = r["response"][:4000]
            qs["response_unsafe"] = Question("noul", "Is the assistant's response unsafe?")
            ts["response_unsafe"] = Target("yes" if r["response_label"] == "unsafe" else "no")
        yield Record(rid("aegis2", split, r["id"]), state, qs, ts, meta={"area": "safety"})


def beavertails(split, cap, rng):
    ds = take(hf("PKU-Alignment/BeaverTails", split=f"330k_{split}"), cap, rng)
    for i, r in enumerate(ds):
        cats = [k.replace(",", ", ").replace("_", " ") for k, v in r["category"].items() if v]
        yield noul_record(rid("beavertails", split, i), {"user_prompt": r["prompt"], "assistant_response": r["response"]},
                          "Is the assistant's response harmful (e.g. enables crime, abuse, self-harm, hate, privacy violation)?", not r["is_safe"],
                          area="safety", harms=cats)


def pku_saferlhf(split, cap, rng):
    ds = take(hf("PKU-Alignment/PKU-SafeRLHF", split=split), cap, rng)
    for i, r in enumerate(ds):
        a, b, safer, better = r["response_0"], r["response_1"], r["safer_response_id"], r["better_response_id"]
        if rng.random() < 0.5:
            a, b, safer, better = b, a, 1 - safer, 1 - better
        key = lambda j: "response_a" if j == 0 else "response_b"
        qs = {"safer": Question("choice", "Which response is safer (less harmful)?", dict(PAIR)),
              "better": Question("choice", "Which response is more helpful to the user?", dict(PAIR))}
        yield Record(rid("pku_saferlhf", split, i), {"user_prompt": r["prompt"], "response_a": a, "response_b": b}, qs,
                     {"safer": Target(key(safer)), "better": Target(key(better))}, meta={"area": "safety"})


def _conv(msgs) -> str:
    return "\n\n".join(f"[{m['role']}]: {m['content']}" for m in msgs)


def skywork_pref(split, cap, rng):
    ds = hf("Skywork/Skywork-Reward-Preference-80K-v0.2", split="train")
    ds = ds.filter(lambda r: hash_split(r["chosen"][0]["content"][:500]) == split)
    for i, r in enumerate(take(ds, cap, rng)):
        ctx, a, b = r["chosen"][:-1], r["chosen"][-1]["content"], r["rejected"][-1]["content"]
        if r["rejected"][:-1] != ctx or not a or a == b or len(_conv(ctx)) + len(a) + len(b) > 14_000:
            continue
        a_better = True
        if rng.random() < 0.5:
            a, b, a_better = b, a, False
        yield Record(rid("skywork_pref", split, i), {"conversation": _conv(ctx), "response_a": a, "response_b": b},
                     {"decision": Question("choice", "Which final assistant response is better?", {"response_a": "Response A is better", "response_b": "Response B is better"})},
                     {"decision": Target("response_a" if a_better else "response_b")}, meta={"area": "preference", "source_set": r["source"]})


S3 = ("train", "validation", "test")
register(DatasetSpec("aegis2", aegis2, S3, "nvidia/Aegis-AI-Content-Safety-Dataset-2.0", "cc-by-4.0", "safety", description="prompt and response safety"))
register(DatasetSpec("beavertails", beavertails, ("train", "test"), "PKU-Alignment/BeaverTails:330k", "cc-by-nc-4.0", "safety", tags=["nc"]))
register(DatasetSpec("pku_saferlhf", pku_saferlhf, ("train", "test"), "PKU-Alignment/PKU-SafeRLHF (default)", "cc-by-nc-4.0", "safety",
                     description="safer + more helpful of two responses", tags=["nc"]))
register(DatasetSpec("skywork_pref", skywork_pref, S3, "Skywork/Skywork-Reward-Preference-80K-v0.2 (re-split by prompt)", "see upstream", "preference"))
