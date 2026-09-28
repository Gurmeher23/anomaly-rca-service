import numpy as np

from anomaly_rca.data import make_sensor_data
from anomaly_rca.detector import AnomalyDetector


def test_detector_recall():
    X, y, _ = make_sensor_data(n=2000, seed=2)
    det = AnomalyDetector(contamination=0.05, random_state=2).fit(X)
    pred = det.predict(X)
    tp = int(((pred == 1) & (y == 1)).sum())
    recall = tp / max(int(y.sum()), 1)
    assert recall >= 0.7, f"recall too low: {recall:.2f}"


def test_score_single_row_is_bounded():
    X, _, _ = make_sensor_data(n=500, seed=3)
    det = AnomalyDetector().fit(X)
    s = det.score(X[:1])
    assert s.shape == (1,)
    assert 0.0 <= float(s[0]) <= 1.0
