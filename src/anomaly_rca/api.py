"""FastAPI service: score a record for anomalies and return ranked root causes.

Trains a demo model on synthetic sensor data at startup so the service is usable
out of the box; swap `_MODEL` for one fitted on real data in production.
"""
from __future__ import annotations

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from .data import make_sensor_data
from .detector import AnomalyDetector
from .rca import root_cause

FEATURES = [f"sensor_{i}" for i in range(8)]

_X, _, _ = make_sensor_data(n=3000, d=len(FEATURES), seed=1)
_MODEL = AnomalyDetector(contamination=0.05, random_state=1).fit(_X)


class Record(BaseModel):
    values: list[float]


class Cause(BaseModel):
    feature: str
    contribution: float


class Verdict(BaseModel):
    anomaly: bool
    score: float
    root_causes: list[Cause]


app = FastAPI(title="Anomaly RCA Service", version="0.1.0")


@app.get("/healthz")
def healthz():
    return {"status": "ok", "features": FEATURES}


@app.post("/detect", response_model=Verdict)
def detect(rec: Record):
    x = np.asarray(rec.values, dtype=float)[None, :]
    score = float(_MODEL.score(x)[0])
    is_anom = bool(_MODEL.predict(x)[0])
    causes = root_cause(_MODEL, x, feature_names=FEATURES, top_k=3)[0]
    return Verdict(
        anomaly=is_anom,
        score=round(score, 4),
        root_causes=[Cause(feature=f, contribution=round(c, 4)) for f, c in causes],
    )
