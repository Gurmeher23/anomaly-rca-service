"""Multivariate anomaly detection: an Isolation Forest + robust z-score ensemble.

The Isolation Forest captures multivariate / interaction anomalies; the robust
per-feature z-score (median / MAD) captures sharp univariate spikes that a forest
can smooth over. The two calibrated scores are averaged, which is more reliable on
sensor data than either alone.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    def __init__(self, contamination: float = 0.05, random_state: int = 0):
        self.contamination = contamination
        self.random_state = random_state
        self._iforest = IsolationForest(
            contamination=contamination, random_state=random_state, n_estimators=200
        )

    def fit(self, X: np.ndarray) -> "AnomalyDetector":
        X = np.asarray(X, dtype=float)
        self._iforest.fit(X)
        self._median = np.median(X, axis=0)
        mad = np.median(np.abs(X - self._median), axis=0)
        self._mad = np.where(mad == 0, 1e-9, mad) * 1.4826  # scale MAD to ~std
        iso_raw = -self._iforest.score_samples(X)
        self._iso_min, self._iso_max = float(iso_raw.min()), float(iso_raw.max())
        z_raw = self._zscore(X).max(axis=1)
        self._z_max = float(z_raw.max())
        train = self._combine(iso_raw, z_raw)
        self._threshold = float(np.quantile(train, 1 - self.contamination))
        return self

    def _zscore(self, X: np.ndarray) -> np.ndarray:
        return np.abs(np.asarray(X, dtype=float) - self._median) / self._mad

    def _combine(self, iso_raw: np.ndarray, z_raw: np.ndarray) -> np.ndarray:
        iso = np.clip((iso_raw - self._iso_min) / (self._iso_max - self._iso_min + 1e-9), 0, 1)
        z = np.clip(z_raw / (self._z_max + 1e-9), 0, 1)
        return 0.5 * iso + 0.5 * z

    def score(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score in ~[0, 1]; higher is more anomalous. Works on single rows."""
        X = np.asarray(X, dtype=float)
        iso_raw = -self._iforest.score_samples(X)
        z_raw = self._zscore(X).max(axis=1)
        return self._combine(iso_raw, z_raw)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """1 = anomaly, 0 = normal (threshold calibrated at fit time)."""
        return (self.score(X) >= self._threshold).astype(int)
