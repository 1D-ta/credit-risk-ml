# INCIDENT: Drift Injection and Rollback

This is the reproducible incident used to demonstrate failure detection and safe recovery in the credit-risk system.

## Incident summary
A numeric distribution shift was injected into the raw German Credit data. The shifted dataset still passed schema validation, but the drift monitor detected the shift and raised an alert.

## Reproduction
1. Create shifted input data:

```bash
python scripts/failure_injection/inject_shifted_numeric.py \
  --src data/raw/german_credit_raw.txt \
  --out data/failure_cases/shifted.txt \
  --offset 5.0
```

2. Run drift detection:

```bash
python monitoring/drift.py \
  --reference-data data/raw/german_credit_raw.txt \
  --current-data data/failure_cases/shifted.txt \
  --schema data/schemas/schema.json \
  --output artifacts/reports/drift_report.json
```

## Expected behavior
- The command prints `DRIFT DETECTED`.
- `artifacts/reports/drift_report.json` records `"alert": true` and per-feature PSI/KS values.

## Mitigation and rollback
- If a candidate model is rejected, the active model pointer is preserved.
- If a bad promotion must be undone manually, restore the latest approved model with:

```bash
python governance/rollback.py \
  --registry artifacts/reports/model_registry.json \
  --active-pointer artifacts/models/active_model.json
```

## Validation behavior
- Schema validation raises immediately on malformed rows.
- Approval rejection does not null the active model pointer.

