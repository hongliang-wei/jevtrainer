"""A local TypeSafe System One compatible endpoint: POST /v1/systemone, GET /v1/models."""

from __future__ import annotations

import json
import threading
import time

import torch

from jevtrainer.eval.runner import collect
from jevtrainer.model.load import load
from jevtrainer.predict import answer, probs, temperature_for
from jevtrainer.schema import Record


def _text(x) -> str:
    """Instructions may arrive as JSON (Decision Index); the readouts want text."""
    if x is None:
        return ""
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def make_app(cfg):
    from fastapi import FastAPI, HTTPException

    device = "cuda" if torch.cuda.is_available() else "cpu"
    b = load(cfg.checkpoint, cfg.dtype, device)
    app = FastAPI(title="jevtrainer System One server")
    lock = threading.Lock()  # one forward at a time on the one model

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": cfg.model_name, "object": "model", "owned_by": "jevtrainer"}]}

    @app.get("/v1/limits")
    def limits():
        return {"max_options": b.readout.max_options, "max_state_tokens": b.readout.cfg.max_state_tokens}

    @app.post("/v1/systemone")
    def systemone(req: dict):
        t0 = time.time()
        try:
            questions = {k: dict(q, instructions=_text(q.get("instructions"))) for k, q in req["questions"].items()}
            rec = Record.from_dict({"id": "request", "state": req.get("state", ""), "questions": questions}).validate()
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            raise HTTPException(422, str(e))
        with lock:
            outs = collect(b, [rec], batch_size=1)
        if len(outs) != len(rec.questions):
            # the wording matters: Decision Index reads these markers as a declared capacity limit, not an error
            raise HTTPException(413, "request exceeds the maximum context length (context window) or the options per choice limit")
        answers = {}
        for o in outs:
            q = rec.questions[o["name"]]
            answers[o["name"]] = answer(q, probs(o["logits"], o["k"], temperature_for(b.temperature, q.type)))
        rows = b.readout.encode(rec)
        usage = {"input_tokens": sum(len(r.input_ids) for r in rows), "output_tokens": 0}
        return {"model": req.get("model") or cfg.model_name, "usage": usage, "answers": answers, "latency_ms": round(1000 * (time.time() - t0), 1)}

    return app


def serve(cfg) -> None:
    import uvicorn

    uvicorn.run(make_app(cfg), host=cfg.host, port=cfg.port)
