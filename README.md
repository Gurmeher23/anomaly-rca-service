# Anomaly RCA Service

Multivariate **anomaly detection** with transparent **root-cause attribution**, served over a FastAPI endpoint and containerised with Docker.

Detecting that *something* is wrong is only half the job. On real sensor / telemetry
data, the operator's next question is always **"which signal is abnormal, and by how
much?"** This service answers both: it flags anomalous records *and* ranks the
features that drove each anomaly.

## How it works

**Detection — an ensemble of two complementary views:**
- **Isolation Forest** catches multivariate / interaction anomalies (unusual *combinations* of otherwise-normal values).
- **Robust z-score** (median / MAD per feature) catches sharp univariate spikes that a forest tends to smooth over.

Both scores are calibrated at fit time and averaged, and the decision threshold is set from a training-data quantile (`contamination`). Because calibration stats are stored on the model, scoring works on a **single record** at inference time (important for a live API), not just on a batch.

**Root cause — transparent, model-agnostic attribution:**
Each feature's contribution is its robust z-score deviation (MAD-scaled distance from the training median), normalised to a per-record share. This directly answers "which signals are abnormal here" and needs no post-hoc explainer. (A SHAP-style attribution is a drop-in alternative; this deviation view is cheaper and more directly interpretable for operators.)

## Results

On synthetic sensor data where each anomaly is created by spiking exactly one feature (see `data.py`), the CI test-suite asserts:

- **Detection recall ≥ 0.70** on the injected anomalies.
- **Root-cause top-1 accuracy ≥ 0.80** — the attribution recovers the *exact* spiked feature the large majority of the time.

Run them yourself: `pytest`.

## Usage

```bash
pip install -r requirements.txt
pytest                                   # run the test suite
uvicorn anomaly_rca.api:app --reload     # start the API (PYTHONPATH=src)
```

Score a record:

```bash
curl -X POST localhost:8000/detect \
  -H "content-type: application/json" \
  -d '{"values": [0.1, 0.2, 9.5, -0.3, 0.0, 0.4, -0.1, 0.2]}'
```

```json
{
  "anomaly": true,
  "score": 0.87,
  "root_causes": [
    {"feature": "sensor_2", "contribution": 0.71},
    {"feature": "sensor_5", "contribution": 0.09},
    {"feature": "sensor_0", "contribution": 0.06}
  ]
}
```

### Docker

```bash
docker build -t anomaly-rca .
docker run -p 8000:8000 anomaly-rca
```

## Project layout

```
src/anomaly_rca/
  detector.py   # IsolationForest + robust z-score ensemble, calibrated for single-row scoring
  rca.py        # per-feature root-cause attribution
  data.py       # synthetic multivariate sensor data with injected, labelled anomalies
  api.py        # FastAPI: POST /detect, GET /healthz
tests/          # detection recall + root-cause accuracy tests
```

## Using it on real data

Replace the startup model in `api.py` with `AnomalyDetector(...).fit(X_train)` on your
own historical, mostly-normal data, and pass real `feature_names` to `root_cause`. The
detector is unsupervised, so no labels are required to fit.
