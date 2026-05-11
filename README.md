# Credit Risk ML

Production-shaped credit risk scoring system with strict schema validation, deterministic training, held-out calibration, approval and rollback governance, drift monitoring, and a minimal FastAPI inference service.

## Setup
Create a local virtual environment and install dependencies before running the workflow:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## What is implemented
- Raw German Credit data in `data/raw/`
- Schema contract in `data/schemas/schema.json`
- Training, evaluation, calibration, approval, rollback, inference, and drift monitoring
- Dockerfiles in `docker/` and orchestration in `docker-compose.yml`
- Failure-injection scripts in `scripts/failure_injection/`
- Unit tests in `tests/`

## Verified workflow
Run these from the repository root:

```bash
make train
make evaluate
make calibrate
make approve
make rollback
docker compose up --build
```

The governance policy in `governance/policy.json` requires an approved model to meet the metrics threshold. A rejected candidate does not clear or null the active pointer; the last approved model remains in `artifacts/models/active_model.json`.

## Failure injection
Schema mismatch:

```bash
python scripts/failure_injection/inject_schema_mismatch.py \
	--src data/raw/german_credit_raw.txt \
	--out data/failure_cases/schema_mismatch.txt
```

Numeric drift:

```bash
python scripts/failure_injection/inject_shifted_numeric.py \
	--src data/raw/german_credit_raw.txt \
	--out data/failure_cases/shifted.txt \
	--offset 5.0
python monitoring/drift.py \
	--reference-data data/raw/german_credit_raw.txt \
	--current-data data/failure_cases/shifted.txt \
	--schema data/schemas/schema.json
```

## Rollback story
If a candidate is rejected or a bad promotion is discovered, restore the last approved model with:

```bash
make rollback
```

The active pointer is designed to remain on the last approved artifact until a better candidate is successfully promoted.

## Testing
```bash
pytest tests/ -q
```

## Artifacts
- Trained model: `artifacts/models/model_v1.pkl`
- Calibrated candidate: `artifacts/models/calibrated_model_v1.pkl`
- Approved model: `artifacts/models/approved/calibrated_model_v1.pkl`
- Active model pointer: `artifacts/models/active_model.json`
- Metrics: `artifacts/reports/metrics.json`
- Calibration report: `artifacts/reports/calibration_report.json`
- Drift report: `artifacts/reports/drift_report.json`
