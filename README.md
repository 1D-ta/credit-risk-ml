# Credit Risk ML System

This repository contains a production-style credit risk scoring pipeline with data validation, model training, calibrated inference, governance controls, and monitoring.

Key highlights:

- End‑to‑end data validation and temporal splitting
- Model training with calibration and evaluation
- Versioned model registry with approval gating and safe rollback
- Monitoring: PSI‑based drift detection, Prometheus metrics, structured logging
- Reproducible artifacts and a focused test suite

Generated runtime artifacts are written under `artifacts/` and are mostly ignored by git. A minimal public evidence sample is retained in `artifacts/evidence/`.

## Quick Start (tests-first)

```bash
# create venv and install (see Makefile targets)
make setup
source .venv/bin/activate

# run the focused test-suite
make test

# local service (requires Docker installed):
# docker compose up --build
```

## Where evidence lives

- Audit logs and sample API responses: `artifacts/logs/`
- Model artifacts and approved models: `artifacts/models/`
- Reports and evaluation outputs: `artifacts/reports/`

## Repository Structure

```
credit-risk-ml/
├── training/            # Model training, evaluation, calibration
├── inference/           # FastAPI service with validation
├── monitoring/          # Drift detection and alerting
├── governance/          # Approval and rollback scripts
├── credit_risk_ml/      # Core data contracts and utilities
├── tests/               # Test suite
├── artifacts/           # Logs, reports, models
│   ├── logs/
│   ├── models/
│   └── reports/
├── data/                # Raw data, schemas, failure cases
└── scripts/             # Reproducibility and failure injection
```

## Public Repository Notes

- `.gitignore` excludes local environments, secrets, model binaries, and generated logs.
- Minimal example evidence for API behavior and monitoring is included in `artifacts/evidence/`.
- Full runtime outputs can be regenerated locally using Makefile targets.

