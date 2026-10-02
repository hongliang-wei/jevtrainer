"""Images, videos (with their own sound track) and audio inside a record.

A record's ``media`` is a list of items, referenced from the state as ``<image:N>``, ``<video:N>`` and
``<audio:N>`` (N counts items of that type, from 1)::

    {"type": "image", "path": "a.jpg"}
    {"type": "video", "frames": ["f0.jpg", ...], "fps": 1.0, "audio": "a.flac"}   # pre-extracted (jt media extract)
    {"type": "video", "path": "clip.mp4"}                                         # decoded on the fly
    {"type": "audio", "path": "a.flac"}

Relative paths are resolved against ``record.meta["media_root"]``. A video's ``fps`` is the *effective* rate of
its stored frames (frames / duration), which is what a model needs to place them on a time axis.

Anything not referenced by a tag in the state is placed in front of it, in order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from jevtrainer.schema import Record

TAG = re.compile(r"<(image|video|audio):(\d+)>")
SAMPLE_RATE = 16000


@dataclass
class VideoClip:
    frames: list  # PIL images, uniformly spaced in time
    fps: float  # effective frames per second of `frames`
    duration: float
    audio: np.ndarray | None = None  # mono float32 at SAMPLE_RATE, the clip's own sound


@dataclass
class MediaOptions:
    video_frames: int = 8  # frames kept per video (uniform subsample)
    video_fps: float | None = None  # if set: keep ~this many frames per second of the clip instead (needs densely extracted frames)
    video_max_frames: int = 32  # cap with video_fps
    frame_max_side: int = 448
    frame_tokens: int = 70  # soft tokens per video frame, for models with a configurable budget (Gemma 4: 70..1120)
    audio_max_s: float = 30.0
    video_audio_max_s: float | None = None  # defaults to audio_max_s
    use_audio_in_video: bool = True  # a video's own sound goes in with it
    image_pixel_budget: int | None = None

    @classmethod
    def from_options(cls, options: dict[str, Any]) -> "MediaOptions":
        names = cls.__dataclass_fields__
        return cls(**{k: v for k, v in options.items() if k in names})


@dataclass
class Media:
    images: list = field(default_factory=list)
    videos: list[VideoClip] = field(default_factory=list)
    audios: list[np.ndarray] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {"image": len(self.images), "video": len(self.videos), "audio": len(self.audios)}


# ---------------------------------------------------------------------------------------------------------
def _resolve(ref: str, root: str | None) -> Path:
    p = Path(ref)
    return p if p.is_absolute() or not root else Path(root) / p


def _resize(im, max_side: int):
    from PIL import Image

    w, h = im.size
    s = max_side / max(w, h)
    return im if s >= 1 else im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BICUBIC)


def _pick(n_have: int, n_want: int) -> list[int]:
    if n_have <= n_want:
        return list(range(n_have))
    return [int((i + 0.5) * n_have / n_want) for i in range(n_want)]


def load_audio(path: Path, max_s: float, start: float = 0.0) -> np.ndarray:
    import soundfile as sf

    with sf.SoundFile(str(path)) as f:
        sr = f.samplerate
        f.seek(int(start * sr))
        x = f.read(int(max_s * sr), dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SAMPLE_RATE:
        x = _resample(x, sr)
    return np.ascontiguousarray(x, dtype=np.float32)


def _resample(x: np.ndarray, sr: int) -> np.ndarray:
    try:
        import librosa

        return librosa.resample(x, orig_sr=sr, target_sr=SAMPLE_RATE)
    except ImportError:
        import torch

        t = torch.from_numpy(x)[None, None]
        n = round(len(x) * SAMPLE_RATE / sr)
        return torch.nn.functional.interpolate(t, size=n, mode="linear", align_corners=False)[0, 0].numpy()


def decode_video_file(path: Path, n_frames: int, max_audio_s: float, with_audio: bool) -> VideoClip:
    """Uniformly sample frames (and read the sound track) from a video file with PyAV."""
    import av
    from PIL import Image

    with av.open(str(path)) as c:
        vs = c.streams.video[0]
        duration = float(c.duration / av.time_base) if c.duration else 0.0
        frames = []
        for fr in c.decode(video=0):
            frames.append(fr.to_image())
        if not frames:
            raise ValueError(f"{path}: no video frames")
        if not duration:
            duration = len(frames) / float(vs.average_rate or 25)
    idx = _pick(len(frames), n_frames)
    clip_frames = [frames[i] for i in idx]
    audio = None
    if with_audio:
        audio = _decode_audio_av(path, max_audio_s)
    return VideoClip(clip_frames, len(clip_frames) / max(duration, 1e-3), duration, audio)


def _decode_audio_av(path: Path, max_s: float) -> np.ndarray | None:
    import av

    with av.open(str(path)) as c:
        if not c.streams.audio:
            return None
        rs = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
        chunks, total = [], 0
        for frame in c.decode(audio=0):
            for f in rs.resample(frame):
                a = f.to_ndarray().reshape(-1)
                chunks.append(a)
                total += len(a)
            if total >= max_s * SAMPLE_RATE:
                break
    if not chunks:
        return None
    return np.concatenate(chunks)[: int(max_s * SAMPLE_RATE)].astype(np.float32)


def load_media(record: Record, opts: MediaOptions) -> Media:
    from PIL import Image

    root = record.meta.get("media_root") or record.meta.get("image_root")
    m = Media()
    for it in record.media:
        kind = it["type"]
        if kind == "image":
            im = Image.open(_resolve(it["path"], root)).convert("RGB")
            m.images.append(_resize(im, opts.frame_max_side * 2))
        elif kind == "audio":
            m.audios.append(load_audio(_resolve(it["path"], root), opts.audio_max_s, float(it.get("start", 0.0))))
        elif kind == "video":
            va_max = opts.video_audio_max_s or opts.audio_max_s
            if "frames" in it:
                refs = it["frames"]
                dur = float(it.get("duration") or (len(refs) / float(it.get("fps") or 1.0)))
                want = opts.video_frames
                if opts.video_fps:  # ~video_fps frames per second; clips cached with fewer frames just keep what they have
                    want = max(2, min(opts.video_max_frames, round(dur * opts.video_fps)))
                keep = _pick(len(refs), want)
                frames = [_resize(Image.open(_resolve(refs[i], root)).convert("RGB"), opts.frame_max_side) for i in keep]
                audio = None
                if opts.use_audio_in_video and it.get("audio"):
                    audio = load_audio(_resolve(it["audio"], root), va_max)
                m.videos.append(VideoClip(frames, len(frames) / max(dur, 1e-3), dur, audio))
            else:
                own_sound = opts.use_audio_in_video and not it.get("mute") and not it.get("audio")
                clip = decode_video_file(_resolve(it["path"], root), opts.video_frames, va_max, own_sound)
                clip.frames = [_resize(f, opts.frame_max_side) for f in clip.frames]
                if it.get("black"):  # evaluation ablation: same timing, nothing to see
                    clip.frames = [Image.new("RGB", f.size) for f in clip.frames]
                if it.get("mute"):
                    clip.audio = None
                elif it.get("audio") and opts.use_audio_in_video:  # sound taken from another file (ablation)
                    clip.audio = load_audio(_resolve(it["audio"], root), va_max)
                m.videos.append(clip)
        else:
            raise ValueError(f"{record.id}: unknown media type {kind!r}")
    return m


def plan_order(text: str, counts: dict[str, int]) -> list[tuple[str, int]]:
    """Media items in the order they will appear in the final text: front-loaded unreferenced ones first."""
    seen: list[tuple[str, int]] = []
    for kind, n in TAG.findall(text):
        key = (kind, int(n) - 1)
        if key[1] < 0 or key[1] >= counts[kind]:
            raise ValueError(f"state refers to <{kind}:{n}> but the record has {counts[kind]} {kind} item(s)")
        if key not in seen:
            seen.append(key)
    front = [(k, i) for k in ("image", "video", "audio") for i in range(counts[k]) if (k, i) not in seen]
    return front + seen


def inline(text: str, media: Media, placeholder) -> tuple[str, list[tuple[str, int]]]:
    """Replace every tag by ``placeholder(kind, has_audio)``; a second mention of the same item becomes plain text.

    Unreferenced items get their placeholder in front of the text. Returns the new text and the items in the
    order their placeholders now appear (the order the processor will consume its inputs in).
    """
    order = plan_order(text, media.counts())

    def ph(key):
        kind, i = key
        return placeholder(kind, kind == "video" and media.videos[i].audio is not None)

    referenced = []
    for kind, n in TAG.findall(text):
        key = (kind, int(n) - 1)
        if key not in referenced:
            referenced.append(key)
    front = order[: len(order) - len(referenced)]
    used: set[tuple[str, int]] = set()

    def sub(mo):
        key = (mo.group(1), int(mo.group(2)) - 1)
        if key in used:
            return f"[{mo.group(1)} {mo.group(2)}]"
        used.add(key)
        return ph(key)

    body = TAG.sub(sub, text)
    return "".join(ph(k) for k in front) + body, order


def processor_inputs(media: Media, order: list[tuple[str, int]], silent_video_audio: bool = False) -> dict[str, list]:
    """images / videos / audios lists in placeholder order. A video's own sound joins the audio list at its place.

    `silent_video_audio`: a video without a sound track gets silence of its length (models that always read
    video and sound together).
    """
    out: dict[str, list] = {"images": [], "videos": [], "audios": []}
    for kind, i in order:
        if kind == "image":
            out["images"].append(media.images[i])
        elif kind == "video":
            v = media.videos[i]
            out["videos"].append(v)
            if v.audio is not None:
                out["audios"].append(v.audio)
            elif silent_video_audio:
                out["audios"].append(np.zeros(max(1600, int(v.duration * SAMPLE_RATE)), dtype=np.float32))
        else:
            out["audios"].append(media.audios[i])
    return out
