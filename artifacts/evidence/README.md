# Evidence Artifacts

This directory contains outputs captured from validation runs for the credit risk ML demo.

## Files

### 1. `drift_alert_rollback.log`
**Demonstrates:** Drift alerting and rollback triggered by PSI breach

Snapshot of a monitoring run with `--auto-rollback` enabled. Shows PSI metric, temporal PSI batches, breach detection, and the active model pointer update.

**Key Evidence:**
- PSI monitoring output
- Drift threshold breach detection
- Rollback action recorded in logs

---

### 2. `temporal_training.log`
**Demonstrates:** Temporal split creation and leakage check

Snapshot of the temporal split log written during training. Confirms non-overlapping train/validation/test ranges.

**Key Evidence:**
- Time-ordered split boundaries
- No leakage check passed

---

### 3. `inference_success.json`
**Demonstrates:** Successful API response

Captured response payload from a real `/predict` call. Contains the risk score, decision, and request ID.

**Key Evidence:**
- 200 response with score + decision
- Model version reported in output

---

### 4. `inference_failure.json`
**Demonstrates:** Schema validation failure

Captured error response for a request missing `credit_amount`. This is a FastAPI/Pydantic validation error returned before inference.

**Key Evidence:**
- 422 response
- Field-level error details

---

### 5. `feature_mismatch.log`
**Demonstrates:** Feature distribution mismatch detection

JSON output written by the feature consistency check when inference statistics deviate from training baselines.

**Key Evidence:**
- Mean-shift detection for `credit_amount`
- Request rejection message emitted

---

## What This Proves

- Temporal split logging is produced during training
- Drift monitoring emits PSI metrics and can trigger rollback when enabled
- Inference API returns both success and validation error responses
- Feature consistency checks flag distribution shifts

## For Hiring Managers

These artifacts show a governed ML workflow with temporal discipline, drift checks, schema enforcement, and fail-fast protections. Each artifact is a snapshot of actual script output from this repo.