"""`jt` command line. Everything takes a YAML file; `--set key=value` overrides any field."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import List, Optional

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Train Jev-like typed decision models.")
data_app = typer.Typer(no_args_is_help=True, help="Datasets.")
bench_app = typer.Typer(no_args_is_help=True, help="Benchmarks.")
model_app = typer.Typer(no_args_is_help=True, help="Model families.")
app.add_typer(data_app, name="data")
app.add_typer(bench_app, name="bench")
app.add_typer(model_app, name="model")

SET = typer.Option(None, "--set", "-s", help="override a config field, e.g. --set lr=5e-5 --set lora.r=32")


@app.command()
def train(config: Path, set: Optional[List[str]] = SET, dry_run: bool = typer.Option(False, help="print the plan, do not train")):
    """Train from one YAML file."""
    from jevtrainer.config import TrainConfig, load_config
    from jevtrainer.registry import TRAINERS

    cfg = load_config(TrainConfig, config, set)
    trainer = TRAINERS.get("sft")(cfg)
    if dry_run:
        typer.echo(json.dumps(trainer.dry_run(), indent=2, ensure_ascii=False, default=str))
        return
    result = trainer.run()
    typer.echo(json.dumps(result, indent=2, ensure_ascii=False, default=str))


@app.command("eval")
def evaluate(config: Optional[Path] = typer.Argument(None), set: Optional[List[str]] = SET):
    """Evaluate a checkpoint (or an untrained model zero-shot) on benchmarks."""
    import torch

    from jevtrainer.config import EvalConfig, load_config
    from jevtrainer.eval.runner import run_benchmarks
    from jevtrainer.model.load import build, load

    cfg = load_config(EvalConfig, config, set)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.checkpoint:
        b = load(cfg.checkpoint, cfg.dtype, device)
    elif cfg.model:
        b = build(cfg.model, cfg.readout, cfg.family, cfg.dtype)
        b.model.to(device).eval()
        b.readout.to(device).eval()
    else:
        raise typer.BadParameter("set checkpoint or model")
    if cfg.temperature is not None:
        b.temperature = {"default": cfg.temperature}
    if cfg.readout_options:  # e.g. video_frames: 16 for the long-video benchmarks
        b.readout.cfg.options.update(cfg.readout_options)
        b.readout._media_opts = None
    suffix = f"_ablate_{cfg.ablate}" if cfg.ablate != "none" else ""
    if cfg.output_dir:
        out = cfg.output_dir if (not suffix or "ablate" in cfg.output_dir) else cfg.output_dir.rstrip("/\\") + suffix
    else:
        out = str(Path(cfg.checkpoint) / f"eval{suffix}") if cfg.checkpoint else f"runs/eval{suffix}"
    run_benchmarks(b, cfg.benchmarks, out, cfg.max_samples, cfg.batch_size, ablate=cfg.ablate)
    typer.echo((Path(out) / "results.md").read_text(encoding="utf-8"))


@app.command()
def serve(checkpoint: str, port: int = 8009, host: str = "0.0.0.0"):
    """Serve POST /v1/systemone for a checkpoint."""
    from jevtrainer.config import ServeConfig
    from jevtrainer.serve.systemone import serve as run

    run(ServeConfig(checkpoint=checkpoint, port=port, host=host))


@app.command()
def init(
    readout: str = typer.Option("marker", help="marker | slot | pointer"),
    model: str = typer.Option("Qwen/Qwen3.5-0.8B"),
    finetune: str = typer.Option("lora", help="lora | full"),
    dataset: str = typer.Option("banking77,boolq,agnews"),
):
    """Print a starter YAML."""
    tmpl = resources.files("jevtrainer").joinpath("templates/train.yaml").read_text(encoding="utf-8")
    typer.echo(tmpl.format(readout=readout, model=model, finetune=finetune, dataset=dataset))


@data_app.command("list")
def data_list():
    from jevtrainer.registry import DATASETS

    for name, s in DATASETS.items():
        flags = ",".join(f for f, on in (("eval_only", s.eval_only), ("multimodal", s.multimodal)) if on)
        typer.echo(f"{name:32} {s.area:12} {'/'.join(s.splits):28} {s.source}  {flags}")


@data_app.command("show")
def data_show(name: str, split: str = "train", n: int = 2):
    from jevtrainer.data.base import load

    for r in load(name, split, n):
        typer.echo(json.dumps(r.to_dict(), ensure_ascii=False, indent=2)[:3000])


@data_app.command("prepare")
def data_prepare(names: str = typer.Argument("all", help="comma list of name or name:split; 'all' = every split of every dataset"),
                 cap: int = 100_000):
    """Download and convert datasets into the cache; failures are reported and skipped."""
    import traceback

    from jevtrainer.data.base import load
    from jevtrainer.registry import DATASETS

    if names == "all":
        items = [(n, sp) for n, s in DATASETS.items() for sp in s.splits]
    else:
        items = [(n.partition(":")[0], n.partition(":")[2] or "train") for n in names.split(",") if n.strip()]
    for name, split in items:
        try:
            typer.echo(f"OK   {name}:{split} -> {len(load(name, split, cap=cap))} records")
        except Exception as e:
            typer.echo(f"FAIL {name}:{split} -> {type(e).__name__}: {str(e)[:300]}")
            traceback.print_exc()


@data_app.command("fetch")
def data_fetch(what: str = typer.Argument("intern")):
    """Fetch evaluation bundles (currently: intern)."""
    from jevtrainer.data.converters.typed import fetch_intern

    typer.echo(str(fetch_intern()))


@bench_app.command("list")
def bench_list():
    from jevtrainer.registry import BENCHMARKS

    for name, s in BENCHMARKS.items():
        typer.echo(f"{name:28} {s.suite:20} {s.area:12} {s.dataset}:{s.split}  metric={s.metric}")


@model_app.command("list")
def model_list():
    from jevtrainer.registry import FAMILIES

    for name, cls in FAMILIES.items():
        typer.echo(f"{name:10} {', '.join(cls.model_types) or '(any model, inferred)'}")


@model_app.command("inspect")
def model_inspect(model: str, family: str = "auto"):
    """Show what the family infers for a checkpoint (without loading weights)."""
    import torch
    from transformers import AutoConfig

    from jevtrainer.model.families import resolve_family

    fam = resolve_family(model, family)
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    typer.echo(f"family={fam.name} model_type={config.model_type} multimodal={fam.is_multimodal(config)} causal={fam.causal}")
    with torch.device("meta"):
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

        cls = AutoModelForImageTextToText if fam.is_multimodal(config) else AutoModelForCausalLM
        m = cls.from_config(config, trust_remote_code=True)
    typer.echo(f"lora_targets={fam.lora_target_regex(m)}")
    typer.echo(f"linear_attention={fam.has_linear_attention(m)} vision_modules={len(fam.vision_modules(m))}")


if __name__ == "__main__":
    app()
