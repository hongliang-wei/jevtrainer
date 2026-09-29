"""A local TypeSafe System One compatible endpoint: POST /v1/systemone, GET /v1/models."""

from __future__ import annotations

import time

import torch

from jevtrainer.eval.runner import collect
from jevtrainer.model.load import load
from jevtrainer.predict import answer, probs, temperature_for
from jevtrainer.schema import Record


def make_app(cfg):
    from fastapi import FastAPI, HTTPException

    device = "cuda" if torch.cuda.is_available() else "cpu"
    b = load(cfg.checkpoint, cfg.dtype, device)
    app = FastAPI(title="jevtrainer System One server")

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
            rec = Record.from_dict({"id": "request", "state": req.get("state", ""), "questions": req["questions"]}).validate()
        except (KeyError, ValueError, TypeError) as e:
            raise HTTPException(422, str(e))
        outs = collect(b, [rec], batch_size=1)
        if len(outs) != len(rec.questions):
            raise HTTPException(422, "request does not fit the model (too long or too many options)")
        answers = {}
        for o in outs:
            q = rec.questions[o["name"]]
            answers[o["name"]] = answer(q, probs(o["logits"], o["k"], temperature_for(b.temperature, q.type)))
        return {"model": req.get("model", cfg.model_name), "answers": answers, "latency_ms": round(1000 * (time.time() - t0), 1)}

    return app


def serve(cfg) -> None:
    import uvicorn

    uvicorn.run(make_app(cfg), host=cfg.host, port=cfg.port)
