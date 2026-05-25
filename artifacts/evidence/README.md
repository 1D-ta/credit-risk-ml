# Evidence Samples

This folder contains a minimal public-safe evidence set.

Principles:
- No secrets, credentials, or private identifiers
- No raw runtime dumps beyond tiny curated samples
- Only small files that demonstrate behavior and controls

Files:
- drift_alert_rollback.log: sample alert + rollback trace
- temporal_training.log: temporal split no-leakage snapshot
- inference_success.json: sample successful inference response
- inference_failure.json: sample schema-validation failure response
