import numpy as np

from anomaly_rca.data import make_sensor_data
from anomaly_rca.detector import AnomalyDetector
from anomaly_rca.rca import root_cause


def test_rca_recovers_spiked_feature():
    X, y, spiked = make_sensor_data(n=3000, d=8, seed=4)
    det = AnomalyDetector(random_state=4).fit(X)
    names = [f"f{i}" for i in range(8)]
    anom_idx = np.where(y == 1)[0]
    causes = root_cause(det, X[anom_idx], feature_names=names, top_k=1)
    hits = sum(1 for j, i in enumerate(anom_idx)
               if causes[j][0][0] == f"f{spiked[i]}")
    acc = hits / len(anom_idx)
    assert acc >= 0.8, f"rca top-1 accuracy too low: {acc:.2f}"
