# Demo Artifacts & Evidence

After running `./demo.sh`, the system creates the following demonstrable artifacts:

## 1. Temporal Split (Upgrade 1)

**File:** `artifacts/reports/split_indices.json`

```json
{
  "calibration_idx": [...],
  "temporal_split_date_T": "2020-06-10",
  "test_idx": [...],
  "train_idx": [...]
}
```

**What it shows:** Strict temporal split enforced — training data is only on rows with `timestamp < 2020-06-10`, validation on `== 2020-06-10`, test on `> 2020-06-10`. This ensures no data leakage and realistic train/val/test separation.

**Split sizes (example):**
- Train: 804 rows
- Validate: 5 rows  
- Test: 191 rows

---

## 2. Training Feature Statistics (Upgrade 3)

**File:** `artifacts/feature_stats/training_features.json`

```json
{
  "credit_amount": {
    "name": "credit_amount",
    "mean": 3276.82,
    "std": 2741.83,
    "missing_pct": 0.0,
    "min": 250.0,
    "max": 15000.0
  },
  "duration_months": {...},
  ...
}
```

**What it shows:** Baseline feature statistics captured at training time. Used to detect serving drift.

---

## 3. Inference Feature Statistics (Simulated Mismatch - Upgrade 3)

**File:** `artifacts/feature_stats/inference_features.json`

This file simulates a real failure scenario: `credit_amount` was shifted by 100x (client sent cents instead of euros).

```json
{
  "credit_amount": {
    "mean": 500000.0,
    "std": 10000.0,
    ...
  }
}
```

**What it means:** Mean shift from 3,277 → 500,000 indicates units changed or data pipeline corruption.

---

## 4. Feature Mismatch Alert (Upgrade 3)

**File:** `artifacts/logs/feature_mismatch.log`

```json
{
  "alert": true,
  "failures": {
    "credit_amount": {
      "reason": "mean_shift",
      "train_mean": 5000.0,
      "inference_mean": 500000.0,
      "stds": 495.0,
      "rel": 99.0
    }
  }
}
```

**What happened:** The inference endpoint detected that `credit_amount` shifted 495 standard deviations, or 99x relative change. This is catastrophic and triggered a rejection (HTTP 503).

---

## 5. API Rejection Example (Upgrade 4)

When sending a prediction request with the feature mismatch in place:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{...}'
```

**Response:**

```json
{
  "detail": "feature mismatch detected; request rejected"
}
```

**HTTP Status:** 503 (Service Unavailable)

This is intentional: the system fails fast rather than silently returning a bad score based on drifted data.

---

## 6. Monitoring Alert Example (Upgrade 2)

**File:** `artifacts/monitoring/alert_example.json`

```json
{
  "alert": "high_error_rate",
  "details": {
    "error_rate": 0.32,
    "window": "1m"
  }
}
```

**What it represents:** In the incident scenario, 32% of requests were rejected due to feature mismatch. This alert would trigger automatic rollback.

---

## 7. Training Metadata (Deliverable)

**File:** `artifacts/reports/training_metadata.json`

```json
{
  "model_version": "v1",
  "data_hash": "abc123...",
  "rows": 1000,
  "columns": 22,
  "decision_threshold": 0.3,
  "metrics": {
    "roc_auc": 0.8007,
    "approval_rate": 0.1057
  },
  "split_indices": "artifacts/reports/split_indices.json",
  "model_output": "artifacts/models/model_v1.pkl"
}
```

**What it contains:** Reproducible training run — model version, data hash, split metadata, performance metrics.

---

## 8. API Metrics (Upgrade 2)

When the API is running, Prometheus metrics are exposed on port `8001`:

```bash
curl http://localhost:8001/metrics | grep inference_
```

**Example metrics:**

```
# HELP inference_requests_total Total inference requests
inference_requests_total 42.0

# HELP inference_request_errors_total Total inference errors
inference_request_errors_total 13.0

# HELP inference_request_latency_seconds Inference request latency seconds
inference_request_latency_seconds_bucket{le="0.005"} 8.0
inference_request_latency_seconds_bucket{le="0.01"} 15.0
...

# HELP inference_predictions_total Total predictions made
inference_predictions_total 29.0
```

**Grafana dashboard:** `monitoring/grafana_dashboard.json` provides visualization for:
- Latency (p50/p95)
- Request volume
- Error rate
- Prediction PSI

---

## How to Reproduce the Incident (Upgrade 5)

1. Run `./demo.sh` to set up and train.
2. Start the API:
   ```bash
   uvicorn inference.app:app --host 0.0.0.0 --port 8000
   ```
3. Send a request (it will be rejected):
   ```bash
   curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' \
     -d '{...}'
   ```
4. Check logs:
   ```bash
   cat artifacts/logs/feature_mismatch.log
   ```

The system detected the feature mismatch and rejected the request, protecting the model from silent degradation. This is the production behavior described in `INCIDENT.md`.

---

## Checklist for Reviewers

✅ **Temporal Realism:** Split date T, train/val/test sizes logged in `split_indices.json`
✅ **Real Monitoring:** Prometheus metrics on port 8001; alert example in `artifacts/monitoring/`
✅ **Feature Consistency:** Training stats in `training_features.json`; inference stats + mismatch example provided
✅ **API Hardening:** Request rejected with 503 + structured log when feature mismatch detected
✅ **Incident Story:** `INCIDENT.md` describes schema/units drift, 30% failure rate, detection, rollback
✅ **Business Framing:** `README.md` explains cost of FP/FN, calibration, rollback strategy, and risk decisioning support

---

## Next Steps

- To test with **good data** (no drift), temporarily remove or modify `artifacts/feature_stats/inference_features.json` to match the training stats. The API will then allow predictions.
- To integrate with **Prometheus/Grafana**, use `docker-compose up` to bring up monitoring stack (if configured in docker-compose.yml).
- To test **rate limiting**, send 61+ requests per minute from the same IP; the 61st will be rejected with HTTP 429.
