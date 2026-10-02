"""Who is speaking: speaker gender / accent of real recordings (16 kHz mono FLAC, at most 30 s).

One clip + one question per record, a fixed label set per corpus, classes sampled evenly. Splits never share a speaker.
"""

from __future__ import annotations

import os
import tarfile
from concurrent.futures import ThreadPoolExecutor

from jevtrainer.data import avkit
from jevtrainer.data.base import DatasetSpec, register

Q_GENDER = "Is the speaker male or female?"
GENDER = {
    "male": "male: the voice of a man",
    "female": "female: the voice of a woman",
}

# ---- AISHELL-1: 400 Mandarin speakers; the hub copy `AISHELL/AISHELL-1` ships one tar.gz per speaker S0002..S0101 -------------
_AISHELL = "AISHELL/AISHELL-1"


def aishell1_gender(split, cap, rng):
    """Speaker gender of Mandarin read speech. The copy on the hub has the first 100 speakers (S0002-S0101) only, so there is
    no official test split: 20 % of the speakers (fixed hash of the speaker id) are held out as test. Per speaker at most
    `per_spk` clips (the first ones of the tar) are used and only JT_AISHELL_TARS (default 70) archives are read, half of them women and half men (the low speaker ids are all men)."""
    name = "aishell1_gender"
    info = avkit.fetch_file(_AISHELL, "resource_aishell/speaker.info", name)
    gender = {}
    for line in info.read_text().splitlines():
        p = line.split()
        if len(p) == 2:
            gender[f"S{p[0]}"] = {"M": "male", "F": "female"}.get(p[1])
    n_tars = int(os.environ.get("JT_AISHELL_TARS", "70"))
    pool = [f"S{i:04d}" for i in range(2, 102) if gender.get(f"S{i:04d}")]  # the low ids are all men: take both genders evenly
    spks = [s for g in ("male", "female") for s in [x for x in pool if gender[x] == g][: n_tars // 2]]
    is_test = lambda s: avkit.hash_pct("aishell1:" + s, 20)  # noqa: E731
    spks = [s for s in spks if is_test(s) == (split == "test") and gender.get(s)]
    per_spk = max(8, min(120, cap // max(1, len(spks)) + 1))

    def one(spk):
        src = avkit.fetch_file(_AISHELL, f"data_aishell/wav/{spk}.tar.gz", name)
        if src is None:
            return []
        out = []
        try:
            for member, blob in avkit.tar_members(src):
                key = os.path.basename(member)
                m = avkit.audio_from_bytes(blob, name, avkit.rid_(name, split, key), 30.0)
                if m:
                    out.append((spk, key, m))
                if len(out) >= per_spk:
                    break
        except (tarfile.TarError, EOFError, OSError):
            pass
        finally:
            src.unlink(missing_ok=True)
        return out

    with ThreadPoolExecutor(4) as ex:
        items = [x for part in ex.map(one, spks) for x in part]
    picked = avkit.balanced(items, lambda x: gender[x[0]], cap, rng)
    for spk, key, m in picked:
        yield avkit.audio_choice(avkit.rid_(name, split, key), m, Q_GENDER, GENDER, gender[spk], speaker=spk, language="zh")
    avkit.release(name)


register(DatasetSpec("aishell1_gender", aishell1_gender, ("train", "test"), _AISHELL, "apache-2.0", "audio", multimodal=True,
                     description="AISHELL-1: gender of the Mandarin speaker; 20% of the speakers are test"))
