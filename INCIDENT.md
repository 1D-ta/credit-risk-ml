# Incident Response: Schema Drift Response

## Scenario Overview

The system detects and responds to a common production failure mode: upstream data format changes causing inference failures.

---

## Failure Mode

**What Would Fail:**
- Real-time inference endpoint would start rejecting requests with HTTP 422 validation errors due to feature schema mismatches
- Monitoring would detect increased error rates and feature distribution shifts

**Detection Mechanism:**
- Monitoring alerts on increased inference error rate (metric `inference_request_errors_total` / `inference_requests_total`)
- `feature_mismatch` entries appear in logs
- Comparison of `artifacts/feature_stats/inference_features.json` vs `artifacts/feature_stats/training_features.json` shows large mean shifts
- PSI calculation exceeds threshold (> 0.2) on key features like `credit_amount` and `duration_months`

---

## Root Cause Analysis

**Scenario:** A client-side change in the ingestion pipeline starts sending `credit_amount` values in cents instead of whole euros, increasing the numeric mean by ~100x.

**Why This Matters:**
- Feature validation compares training vs inference distributions
- Large deviations trigger rejection to prevent model degradation
- System correctly identifies data quality issue before making predictions

---

## Automated Response

**Rollback Trigger:**
- Governance automation detects sustained high error rate or feature mismatch alerts
- Active model marked as unhealthy
- Automated rollback script (`governance/rollback.py`) invoked
- `artifacts/models/active_model.json` pointer updated to previously approved model
- Service restored quickly without manual intervention

---

## System Response

### 1. Defensive Design
- Schema validation prevents silent model degradation
- System fails fast with clear error messages
- No incorrect predictions served on bad data

### 2. Monitoring Capabilities
- PSI-based drift detection
- Feature distribution comparison
- Alert generation on threshold breaches

### 3. Governance Controls
- Automated rollback on detection
- Model registry maintains approved versions
- Quick recovery path without manual intervention

### 4. Operational Patterns
- Clear error logging for debugging
- Artifact-based evidence trail
- Separation of detection and remediation

---

## Remediation Steps

**Short-term:**
- Rollback to previous approved model (automated)
- Identify root cause from logs and feature statistics
- Add data validation at ingestion point

**Medium-term:**
- Implement unit conversion in ingestion pipeline
- Add schema contract tests
- Update client validation examples

**Long-term:**
- Add CI tests validating production data format
- Implement adapter layer for unit normalization
- Enhance monitoring with unit-aware checks

---

## Evidence Artifacts

The following artifacts demonstrate the detection and response:

- `artifacts/logs/feature_mismatch.log` - Feature validation failures
- `artifacts/monitoring/alert_example.json` - Alert payload example
- `artifacts/reports/drift_report.json` - PSI calculations showing drift
- `artifacts/feature_stats/` - Training vs inference feature distributions

---

## Key Takeaways

1. **Automatic rejection protects model quality** - Better to reject requests than serve incorrect predictions
2. **Monitoring enables fast detection** - PSI and feature checks catch issues quickly
3. **Rollback provides safe recovery** - Automated remediation reduces downtime
4. **Clear logging aids debugging** - Artifact trail makes root cause analysis straightforward

This scenario implements production-ready patterns for handling data quality issues in ML systems.
