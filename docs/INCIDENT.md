# INCIDENT: Schema drift caused inference failures and model rollback

> **Note**: This is a **demo incident story** illustrating how the system detects and recovers from data issues. It demonstrates schema validation, drift monitoring, and automated rollback in action. See the [monitoring](../monitoring/) and [governance](../governance/) modules to understand how detection and remediation work.

Date: 2026-05-12

## Summary
- During routine operation a spike in inference errors and feature mismatch was observed causing 30% of incoming requests to be rejected. The system automatically flagged the issue and the most recent approved model was rolled back to the previous version.

## What failed
- Real-time inference endpoint started rejecting requests with HTTP 503 due to feature mismatch checks. The failure rate rose to ~30% of requests.

## How it was detected
- Monitoring alerted on increased inference error rate (metric `inference_request_errors_total` / `inference_requests_total`) and `feature_mismatch` entries in logs. A downstream check of `artifacts/feature_stats/inference_features.json` showed large mean shifts versus `artifacts/feature_stats/training_features.json` (PSI > 0.25 on `credit_amount` and `duration_months`).

## Business impact
- ~30% of loan applications could not be scored, causing manual queueing of applications. Estimated operational cost: ~10-20 hours of manual review per day and delayed revenue from loan originations.

## Why it happened
- A client-side change in the ingestion pipeline started sending `credit_amount` values in cents instead of whole euros, increasing the numeric mean by ~100x. Our feature-check compared training vs inference means and rejected requests when deviation exceeded thresholds.

## How rollback was triggered
- Governance automation detects sustained high error rate or feature mismatch alerts and marks the active model as unhealthy. An automated rollback script ([governance/rollback.py](../governance/rollback.py)) was invoked to point `artifacts/models/active_model.json` to the previously approved model, restoring service quickly.

## What was fixed to prevent recurrence
- **Short-term**: temporarily relaxed the feature-check threshold for `credit_amount` and added a conversion step in ingestion to normalize units.
- **Medium-term**: added a lightweight schema contract test at ingestion to assert units for sensitive numeric fields and an example client validation in the README.
- **Long-term**: plan to add unit tests in CI that validate the production ingestion format and add a small adapter layer to enforce expected units.

## Detection artifacts
- See `artifacts/logs/feature_mismatch.log` and `artifacts/monitoring/alert_example.json` for recorded evidence.

## Lessons learned
- Automatic rejection is valuable for protecting model performance, but must be paired with clear on-call runbooks and a safe rollback path. The system succeeded in preventing silent model degradation and enabled quick remediation.
