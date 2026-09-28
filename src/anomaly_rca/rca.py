"""Root-cause attribution for flagged anomalies.

Contribution is the robust z-score deviation of each feature (how many MAD-scaled
units it sits from the training median), normalised to a share per row. This is a
transparent, model-agnostic attribution that answers the operator's real question:
"which signals are abnormal here, and by how much." (SHAP-style model attribution
is a drop-in alternative; this deviation view is cheaper and directly interpretable.)
"""
from __future__ import annotations

import numpy as np


def root_cause(detector, X, feature_names=None, top_k: int = 3):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    z = detector._zscore(X)  # (n_rows, n_features)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    out = []
    for row in z:
        share = row / (row.sum() + 1e-9)
        order = np.argsort(row)[::-1][:top_k]
        out.append([(feature_names[i], float(share[i])) for i in order])
    return out
