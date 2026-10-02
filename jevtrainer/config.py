"""YAML configs. Only `model`, `readout`, `dataset` are required; everything else has a default.

    model: Qwen/Qwen3.5-0.8B
    readout: pointer
    dataset: banking77,boolq
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from jevtrainer.data.augment import AugmentConfig


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class LoraCfg(Strict):
    r: int = 16
    alpha: int | None = None
    dropout: float = 0.05
    targets: str | list[str] = "auto"


class AugmentCfg(Strict):
    shuffle: bool = True
    p_none: float = 0.05
    p_distract: float = 0.1
    p_pack: float = 0.0
    pack_max: int = 4

    def to_config(self) -> AugmentConfig:
        return AugmentConfig(**self.model_dump())


class SourceCfg(Strict):
    name: str
    split: str = "train"
    max_samples: int | None = None
    weight: float = 1.0
    group_by: str | None = None  # meta key that splits the source into tasks, e.g. question_type
    per_group: int | None = None  # records per task (random subset), so tasks weigh the same
    repeat_to: int | None = None  # tasks / sources smaller than this are repeated (at most 3x)


def _sources(v: Any) -> list[dict]:
    if v is None:
        return []
    if isinstance(v, str):
        v = [s.strip() for s in v.split(",") if s.strip()]
    out = []
    for s in v:
        if isinstance(s, str) and (s.endswith(".jsonl") or Path(s).exists()):
            out.append({"name": s})
        elif isinstance(s, str):
            name, _, split = s.partition(":")
            out.append({"name": name, **({"split": split} if split else {})})
        else:
            out.append(s)
    return out


class TrainConfig(Strict):
    # what
    model: str
    readout: Literal["marker", "slot", "pointer"] | str
    dataset: list[SourceCfg]
    output_dir: str = "runs/latest"
    family: str = "auto"
    readout_options: dict[str, Any] = {}
    # how
    finetune: Literal["lora", "full", "frozen"] = "lora"
    lora: LoraCfg = LoraCfg()
    freeze_vision: bool = True
    loss: str = "ce"
    score_rps_weight: float = 0.0
    augment: AugmentCfg = AugmentCfg()
    max_samples: int | None = None  # per dataset, unless the source sets its own
    dataset_cap: int = 100_000  # rows converted per upstream split (cached)
    max_state_tokens: int = 2048
    max_length: int = 4096
    # optimisation
    epochs: float = 1.0
    max_steps: int | None = None
    lr: float | None = None  # default: 1e-4 lora, 1e-5 full
    head_lr: float = 1e-3
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    batch_size: int = 8  # records per step, before accumulation
    grad_accum: int = 1
    max_grad_norm: float = 1.0
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    quantize: Literal["none", "8bit", "4bit"] = "none"  # bitsandbytes QLoRA, for models that do not fit in bf16
    device_map: str | None = None  # "auto" spreads / offloads layers over GPU and CPU (slow; big MoE models)
    max_memory: dict[str, str] | None = None  # with device_map, e.g. {"0": "44GiB", "cpu": "200GiB"}
    optim: Literal["adamw", "adamw_8bit"] = "adamw"  # 8-bit states (bitsandbytes) cut optimizer memory ~4x
    grad_ckpt: bool = True
    group_by_length: bool = True
    num_workers: int = 2
    seed: int = 42
    # after training
    holdout: float = 0.02  # fraction of the mixture kept for calibration / validation
    calibrate: bool = True
    calibrate_per_type: bool = False
    eval_dataset: list[str] = []  # benchmark names run at the end
    eval_max_samples: int | None = None
    eval_every: int = 0  # steps; 0 = only at the end
    exclude_eval_overlap: bool = True
    exclude: list[str] = []  # extra benchmarks whose states are removed from training
    save_steps: int = 0
    save_state: bool = True  # optimizer / scheduler / RNG / data position with the newest step-N (older copies are deleted)
    resume: bool = False  # continue from the newest step-N/state in output_dir
    log_steps: int = 10
    report_to: Literal["none", "tensorboard", "wandb"] = "none"

    @field_validator("dataset", mode="before")
    @classmethod
    def _parse_dataset(cls, v):
        return _sources(v)

    @field_validator("eval_dataset", "exclude", mode="before")
    @classmethod
    def _parse_list(cls, v):
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return v or []

    @model_validator(mode="after")
    def _defaults(self):
        if self.lr is None:
            self.lr = 1e-4 if self.finetune == "lora" else 1e-5
        if self.lora.alpha is None:
            self.lora.alpha = 2 * self.lora.r
        if not self.dataset:
            raise ValueError("dataset: give at least one dataset name")
        return self


class EvalConfig(Strict):
    checkpoint: str | None = None
    model: str | None = None  # evaluate an untrained model zero-shot
    readout: str = "marker"
    family: str = "auto"
    benchmarks: list[str] = []
    max_samples: int | None = None
    batch_size: int = 16
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    output_dir: str | None = None
    temperature: float | None = None  # override the checkpoint's fitted temperature
    ablate: Literal["none", "mute", "black", "shuffle_audio"] = "none"  # modality ablation, see jevtrainer/eval/ablate.py
    readout_options: dict[str, Any] = {}  # override media options at eval time, e.g. {video_frames: 16}

    @field_validator("benchmarks", mode="before")
    @classmethod
    def _parse(cls, v):
        return [s.strip() for s in v.split(",") if s.strip()] if isinstance(v, str) else v


class ServeConfig(Strict):
    checkpoint: str
    host: str = "0.0.0.0"
    port: int = 8009
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    batch_size: int = 16
    model_name: str = "jev-latest"


def _set(d: dict, dotted: str, value: str) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = yaml.safe_load(value)


def load_config(cls, path: str | Path | None = None, overrides: list[str] | None = None, **extra):
    raw: dict = {}
    if path:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    raw.update(extra)
    for o in overrides or []:
        key, sep, val = o.partition("=")
        if not sep:
            raise ValueError(f"--set expects key=value, got {o!r}")
        _set(raw, key.strip(), val)
    try:
        return cls(**raw)
    except ValidationError as e:
        raise SystemExit(_explain(e, cls, path)) from None


def _explain(e: ValidationError, cls, path) -> str:
    fields = list(cls.model_fields)
    lines = [f"invalid config {path or ''}:"]
    for err in e.errors():
        loc = ".".join(str(x) for x in err["loc"])
        msg = err["msg"]
        if err["type"] == "extra_forbidden":
            close = difflib.get_close_matches(str(err["loc"][-1]), fields, n=3)
            msg = "unknown field" + (f"; did you mean {', '.join(close)}?" if close else "")
        lines.append(f"  {loc}: {msg}")
    return "\n".join(lines)
