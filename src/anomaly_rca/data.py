"""Synthetic multivariate sensor data with injected anomalies (demo + tests).

Anomalies are made by spiking ONE random feature per anomalous row on top of a
correlated normal baseline, so a correct root-cause method should recover exactly
that feature. Returns (X, y, spiked): y is 1 for anomalies, `spiked` is the injected
feature index per row (or -1 for normal rows).
"""
from __future__ import annotations

import numpy as np


def make_sensor_data(n: int = 2000, d: int = 8, anomaly_frac: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, d))
    mix = rng.normal(size=(d, d))           # induce correlation between sensors
    X = base @ mix
    X = (X - X.mean(0)) / X.std(0)
    y = np.zeros(n, dtype=int)
    spiked = np.full(n, -1, dtype=int)
    n_anom = int(n * anomaly_frac)
    idx = rng.choice(n, size=n_anom, replace=False)
    for i in idx:
        f = int(rng.integers(d))
        X[i, f] += rng.choice([-1, 1]) * rng.uniform(6, 10)
        y[i] = 1
        spiked[i] = f
    return X, y, spiked
