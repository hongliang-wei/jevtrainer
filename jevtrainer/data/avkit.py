"""Helpers shared by audio / video converters: fetch raw files, cut frames and sound with ffmpeg.

Layout (all under `$JEVTRAINER_CACHE`, default ~/.cache/jevtrainer):

    raw/<dataset>/...             downloaded originals (delete after conversion with `drop_raw`)
    media/<dataset>/<key>/        f00.jpg f01.jpg ... a.flac (16 kHz mono)  <- what records point to

A converter calls `video_item(...)` / `audio_item(...)` and puts the returned dict in `Record.media`.
ffmpeg comes from the system or from the `imageio-ffmpeg` wheel.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from jevtrainer.data.base import cache_dir


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def raw_dir(name: str) -> Path:
    p = Path(os.environ.get("JEVTRAINER_RAW", cache_dir() / "raw")) / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def media_dir(name: str, key: str) -> Path:
    p = cache_dir() / "media" / name / key
    p.mkdir(parents=True, exist_ok=True)
    return p


def drop_raw(name: str) -> None:
    shutil.rmtree(raw_dir(name), ignore_errors=True)


# ---- fetching -------------------------------------------------------------------------
def fetch(repo: str, name: str | None = None, allow: list[str] | None = None, ignore: list[str] | None = None,
          repo_type: str = "dataset", workers: int = 16) -> Path:
    """snapshot_download into raw/<name> (resumable; a second call only checks what is missing)."""
    from huggingface_hub import snapshot_download

    dest = raw_dir(name or repo.replace("/", "__"))
    snapshot_download(repo, repo_type=repo_type, local_dir=str(dest), allow_patterns=allow, ignore_patterns=ignore,
                      max_workers=workers)
    return dest


def fetch_file(repo: str, filename: str, name: str, repo_type: str = "dataset") -> Path | None:
    """Download one file into raw/<name>/<filename> (None when it is missing or fails)."""
    from huggingface_hub import hf_hub_download

    try:
        return Path(hf_hub_download(repo, filename, repo_type=repo_type, local_dir=str(raw_dir(name))))
    except Exception:
        return None


def convert_videos(repo: str, name: str, items: dict[str, str], workers: int = 8, delete_raw: bool = True, **kw) -> dict[str, dict]:
    """Fetch `{key: file in repo}` one by one, turn each into a media item and delete the raw file.

    Runs `workers` fetch+ffmpeg jobs in parallel; returns `{key: media item}` for the ones that worked.
    """
    from concurrent.futures import ThreadPoolExecutor

    def one(kv):
        key, fn = kv
        out = media_dir(name, key)
        if (out / "f01.jpg").exists():  # converted by an earlier run
            files = sorted(out.glob("f*.jpg"))
            dur = probe_cache(out)
            item = {"type": "video", "frames": [str(f) for f in files], "fps": len(files) / max(dur, 0.1), "duration": dur}
            if (out / "a.flac").exists():
                item["audio"] = str(out / "a.flac")
            return key, item
        src = fetch_file(repo, fn, name)
        if src is None:
            return key, None
        try:
            item = video_item(src, name, key, **kw)
            if item:
                (out / "meta.txt").write_text(str(item["duration"]))
        finally:
            if delete_raw:
                src.unlink(missing_ok=True)
        return key, item

    with ThreadPoolExecutor(workers) as ex:
        return {k: v for k, v in ex.map(one, items.items()) if v}


def probe_cache(out: Path) -> float:
    try:
        return float((out / "meta.txt").read_text())
    except Exception:
        return 10.0


def unzip(archive: Path, dest: Path, members: list[str] | None = None) -> Path:
    import zipfile

    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(dest, members=members)
    return dest


def untar(archive: Path, dest: Path) -> Path:
    import tarfile

    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as t:
        t.extractall(dest)
    return dest


# ---- ffmpeg ---------------------------------------------------------------------------------
def _run(args: list[str]) -> str:
    p = subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", *args],
                       capture_output=True, text=True)
    return p.stderr


def probe(path: Path) -> dict:
    """duration (s), has_audio, has_video from `ffmpeg -i`."""
    p = subprocess.run([ffmpeg_exe(), "-hide_banner", "-nostdin", "-i", str(path)], capture_output=True, text=True)
    err = p.stderr
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
    dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else 0.0
    return {"duration": dur, "has_audio": " Audio:" in err, "has_video": " Video:" in err}


def video_item(src: Path, name: str, key: str, frames: int = 8, max_side: int = 448, audio_s: float = 30.0,
               start: float | None = None, end: float | None = None, keep_audio: bool = True) -> dict | None:
    """Cut `frames` uniformly spaced JPEGs and the sound track (16 kHz mono FLAC) out of a video segment.

    Returns the `Record.media` item, or None when the file cannot be decoded.
    """
    info = probe(src)
    if not info["has_video"]:
        return None
    t0 = start or 0.0
    t1 = end if end is not None else info["duration"]
    dur = (t1 - t0) if t1 else info["duration"]
    if dur <= 0.2:
        return None
    out = media_dir(name, key)
    pat = out / "f%02d.jpg"
    seek = ["-ss", f"{t0:.3f}"] if start else []
    vf = f"fps={frames / dur:.6f},scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'"
    args = [*seek, "-t", f"{dur:.3f}", "-i", str(src), "-vf", vf, "-frames:v", str(frames), "-q:v", "3", "-y", str(pat)]
    use_audio = keep_audio and info["has_audio"]
    if use_audio:
        args += ["-vn", "-ac", "1", "-ar", "16000", "-t", f"{min(dur, audio_s):.3f}", "-y", str(out / "a.flac")]
    _run(args)
    files = sorted(out.glob("f*.jpg"))
    if not files:
        return None
    item = {"type": "video", "frames": [str(f) for f in files], "fps": len(files) / dur, "duration": round(dur, 3)}
    if use_audio and (out / "a.flac").exists() and (out / "a.flac").stat().st_size > 0:
        item["audio"] = str(out / "a.flac")
    return item


def audio_item(src: Path, name: str, key: str, max_s: float = 30.0, start: float | None = None,
               end: float | None = None) -> dict | None:
    """16 kHz mono FLAC of an audio (or video) file segment, at most `max_s` seconds."""
    out = media_dir(name, key) / "a.flac"
    if not out.exists():
        t = max_s if end is None else min(max_s, end - (start or 0.0))
        seek = ["-ss", f"{start:.3f}"] if start else []
        _run([*seek, "-t", f"{t:.3f}", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-y", str(out)])
    if not out.exists() or out.stat().st_size == 0:
        return None
    return {"type": "audio", "path": str(out)}


def audio_from_array(array, sr: int, name: str, key: str, max_s: float = 30.0) -> dict | None:
    """Write a decoded waveform (HF `Audio` feature) as 16 kHz mono FLAC."""
    import numpy as np
    import soundfile as sf

    x = np.asarray(array, dtype="float32")
    if x.ndim > 1:
        x = x.mean(-1 if x.shape[-1] <= 8 else 0)
    if sr != 16000:
        import librosa

        x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    x = x[: int(max_s * 16000)]
    if len(x) < 1600:
        return None
    out = media_dir(name, key) / "a.flac"
    sf.write(str(out), x, 16000, format="FLAC")
    return {"type": "audio", "path": str(out)}


def image_item(path_or_pil, name: str, key: str) -> dict:
    from PIL import Image

    out = media_dir(name, key) / "i.jpg"
    if not out.exists():
        im = path_or_pil if hasattr(path_or_pil, "save") else Image.open(path_or_pil)
        im.convert("RGB").save(out, quality=92)
    return {"type": "image", "path": str(out)}
