# Evidence Artifacts

Inference request examples and validation outputs.

## Files

### 1. `drift_alert_rollback.log`
Drift alerting and rollback triggered by PSI breach. Monitoring run with `--auto-rollback` enabled showing PSI metric, temporal PSI batches, breach detection, and active model pointer update.

---

### 2. `temporal_training.log`
Temporal split creation and leakage check. Temporal split log from training confirming non-overlapping train/validation/test ranges.

---

### 3. `inference_success.json`
Successful API response. Response payload from `/predict` call containing risk score, decision, and request ID.

---

### 4. `inference_failure.json`
Schema validation failure. Error response for request missing `credit_amount`. FastAPI/Pydantic validation error returned before inference.

---

### 5. `feature_mismatch.log`
Feature distribution mismatch detection. JSON output from feature consistency check when inference statistics deviate from training baselines.

---

## Summary

- Temporal split logging produced during training
- Drift monitoring emits PSI metrics and triggers rollback when enabled
- Inference API returns both success and validation error responses
- Feature consistency checks flag distribution shifts