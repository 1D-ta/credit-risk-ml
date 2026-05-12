# Credit Risk ML

Production-shaped credit risk scoring system with strict schema validation, deterministic training, held-out calibration, approval and rollback governance, drift monitoring, and a minimal FastAPI inference service.

## What This Project Demonstrates

This is a **complete, runnable example** of a production ML system focused on **reliability and auditability** rather than pure accuracy. It illustrates:

- **Schema validation & data contracts**: Reject requests that don't match the training data schema
- **Deterministic, auditable training**: Temporal train/val/test splits recorded in artifacts for reproducibility
- **Model calibration**: Ensure predicted probabilities match observed rates for business decision-making
- **Governance**: Approval gates that enforce minimum quality standards before promotion
- **Automated rollback**: Detect data drift and performance degradation; restore the last known-good model
- **Monitoring & observability**: Compare training vs. inference data; alert on metric mismatches
- **Failure injection**: Test resilience with intentional schema mismatches, numeric drift, and categorical surprises

Use this as a template for building **safe, governed, observable ML systems** in production.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│  data/raw/                                                  │
│  german_credit_with_ts.txt                                  │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────────┐      ┌──────────────────┐             │
│  │ Schema Validation│─────▶│  Temporal Split  │             │
│  │ (data_contract.py)     │  (train/val/test)│             │
│  └──────────────────┘      └──────────────────┘             │
│         │                            │                       │
│         ▼                            ▼                       │
│  ┌──────────────────┐      ┌──────────────────┐             │
│  │  Feature Eng.    │─────▶│   Train Model    │             │
│  │ (normalize/scale)      │  (gradient boost)│             │
│  └──────────────────┘      └──────────────────┘             │
│                                    │                         │
│                                    ▼                         │
│                          ┌──────────────────┐               │
│                          │   Calibration    │               │
│                          │  (Platt scaling) │               │
│                          └──────────────────┘               │
│                                    │                         │
│                                    ▼                         │
│                          ┌──────────────────┐               │
│                          │  Approval Gate   │               │
│                          │ (policy.json)    │               │
│                          └──────────────────┘               │
│                                    │                         │
│                     ┌──────────────┴──────────────┐         │
│                     ▼                             ▼         │
│              ✅ APPROVED         ❌ REJECTED       │
│              └─────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│                  INFERENCE & MONITORING                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Active Model Pointer                                       │
│  artifacts/models/active_model.json                        │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────────┐      ┌──────────────────┐             │
│  │ FastAPI Service  │────▶ │ Schema Validation│             │
│  │ (inference/app.py)     │ (feature_check.py)             │
│  └──────────────────┘      └──────────────────┘             │
│         │                            │                       │
│         │                    ┌───────┴────────┐             │
│         │                    ▼                ▼             │
│         │             ✅ Request OK    ❌ Rejected         │
│         │                    │                │             │
│         ▼                    ▼                ▼             │
│  ┌──────────────────┐  ┌──────────────┐  ┌──────────┐     │
│  │  Score & Return  │  │  Log Error   │  │ Alert    │     │
│  │  Prediction      │  │  (error rate)│  │ on Drift │     │
│  └──────────────────┘  └──────────────┘  └──────────┘     │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────────┐                                       │
│  │  Drift Detection │                                       │
│  │  (monitoring/)   │                                       │
│  │  Compare train   │                                       │
│  │  vs inference    │                                       │
│  │  feature stats   │                                       │
│  └──────────────────┘                                       │
│         │                                                   │
│         ▼                                                   │
│  ┌──────────────────┐                                       │
│  │ High Error Rate? │                                       │
│  │ Feature Drift?   │                                       │
│  │ PSI > threshold? │                                       │
│  └──────────────────┘                                       │
│         │                                                   │
│    ┌────┴────┐                                             │
│    ▼         ▼                                             │
│   YES       NO                                              │
│    │         └─────────────────────┐                       │
│    ▼                               ▼                       │
│ ┌─────────────┐          ✅ Continue Serving              │
│ │  ALERT      │                                            │
│ │  Trigger    │                                            │
│ │  Rollback   │                                            │
│ │ (rollback.py)                                            │
│ └─────────────┘                                            │
│    │                                                       │
│    ▼                                                       │
│ ┌─────────────────────────────────┐                       │
│ │ Restore Previous Approved Model │                       │
│ │ Resume Service with Known-Good  │                       │
│ └─────────────────────────────────┘                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key design principle**: Model promotion is strict (requires approval), but rollback is automatic and fast. This prevents silent degradation.

---

## Quick Start (Demo)

Run the end-to-end demo to generate all artifacts and see the system in action:

```bash
./demo.sh
```

This will:
1. Set up the Python environment
2. Generate temporal data with timestamps
3. Train a model using strict temporal splits
4. Create training feature statistics and monitoring artifacts
5. Print all artifact locations and next steps

This generates all training and monitoring artifacts. See [DEMO-ARTIFACTS.md](DEMO-ARTIFACTS.md) for detailed explanations of what each artifact shows and how to interpret failures.

---

## Setup (Manual)

Create a local virtual environment and install dependencies before running the workflow:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project Structure

```
credit-risk-ml/
├── credit_risk_ml/          # Core business logic
│   ├── data_contract.py    # Schema validation
│   └── modeling.py         # Feature engineering & model training
│
├── training/               # Model training pipeline
│   ├── train.py           # Train XGBoost model
│   ├── evaluate.py        # Compute metrics (ROC-AUC, calibration error)
│   ├── calibration.py     # Apply Platt scaling for probability calibration
│
├── inference/              # FastAPI inference service
│   ├── app.py             # REST API for scoring
│   └── schema.py          # Request/response schemas
│
├── monitoring/             # Drift detection & observability
│   ├── drift.py           # Compute PSI & compare feature distributions
│   ├── feature_check.py   # Real-time feature validation
│   ├── metrics.py         # Prometheus metrics exporter
│   ├── prometheus.yml     # Prometheus config
│   └── grafana_dashboard.json # Grafana visualization
│
├── governance/             # Approval & rollback automation
│   ├── approve_model.py   # Promote candidate to active model
│   ├── rollback.py        # Restore previous approved model
│   └── policy.json        # Governance thresholds (ROC-AUC min, calibration gap max)
│
├── scripts/
│   ├── generate_temporal_data.py          # Add timestamps to raw data
│   ├── failure_injection/                 # Intentional data corruption for testing
│   │   ├── inject_schema_mismatch.py
│   │   ├── inject_shifted_numeric.py
│   │   └── inject_unexpected_categorical.py
│   └── demo_requests.py   # Example inference requests
│
├── data/
│   ├── raw/               # Public German Credit dataset (UCI)
│   │   ├── german_credit_raw.txt
│   │   └── german_credit_with_ts.txt  # Raw + synthetic timestamps
│   ├── schemas/           # Data contracts
│   │   └── schema.json
│   └── failure_cases/     # Test data with intentional issues
│
├── artifacts/             # Outputs (example artifacts tracked, runtime artifacts ignored)
│   ├── models/            # Trained models
│   │   ├── model_v1.pkl   # Initial trained model
│   │   ├── calibrated_model_v1.pkl
│   │   ├── approved/      # Approved models (git-ignored)
│   │   ├── candidate/     # Candidate models (git-ignored)
│   │   └── active_model.json  # Pointer to active model
│   ├── reports/           # Metrics and reports (example artifacts tracked)
│   │   ├── metrics.json   # Training metrics
│   │   ├── calibration_report.json
│   │   ├── data_validation_report.json
│   │   └── split_indices.json
│   ├── feature_stats/     # Feature statistics (example artifacts tracked)
│   │   ├── training_features.json
│   │   └── inference_features.json
│   ├── logs/              # Runtime logs (git-ignored, grows unbounded)
│   │   └── predictions.jsonl
│   └── monitoring/        # Monitoring alerts (example artifacts tracked)
│       └── alert_example.json
│
├── docker/                # Container configs
│   ├── training.Dockerfile
│   ├── inference.Dockerfile
│   └── monitoring.Dockerfile
│
├── tests/                 # Unit tests
│   ├── test_schema_validation.py
│   ├── test_drift_detection.py
│   ├── test_inference_validation.py
│   └── test_approval_rollback.py
│
├── docs/                  # Documentation
│   ├── INCIDENT.md        # Demo incident + rollback story
│   └── DEMO-ARTIFACTS.md  # Artifact interpretation guide
│
├── Makefile               # Workflow automation
├── docker-compose.yml     # Local development orchestration
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── PRD.md                 # Product requirements
```

### Artifacts

**Example artifacts (tracked in git for reference)**:
- [artifacts/reports/metrics.json](artifacts/reports/metrics.json) — Training metrics baseline
- [artifacts/reports/calibration_report.json](artifacts/reports/calibration_report.json) — Calibration quality
- [artifacts/feature_stats/training_features.json](artifacts/feature_stats/training_features.json) — Training data statistics
- [artifacts/monitoring/alert_example.json](artifacts/monitoring/alert_example.json) — Example drift alert

**Runtime artifacts (git-ignored)**:
- `artifacts/logs/` — Prediction logs (rotate and archive to S3 in production)
- `artifacts/models/*.pkl` — Binary model files (use git-lfs for version control if needed)
- `artifacts/reports/split_indices.json` — Training split metadata

## What is implemented
- Raw German Credit data in `data/raw/`
- Schema contract in `data/schemas/schema.json`
- Training, evaluation, calibration, approval, rollback, inference, and drift monitoring
- Dockerfiles in `docker/` and orchestration in `docker-compose.yml`
- Failure-injection scripts in `scripts/failure_injection/`
- Unit tests in `tests/`

## How to Verify Everything Works

### 1. **Test the training pipeline** (generates artifacts and metrics)
```bash
make train
```
**Expected output**: Trained model file and metrics report in `artifacts/reports/metrics.json`. Compare with tracked baseline.

### 2. **Run unit tests** (validates schema, drift detection, inference)
```bash
pytest tests/ -q
```
**Expected output**: All tests pass (✓). If any fail, check that relative paths (`data/`, `artifacts/`) are correct.

### 3. **Test approval & rollback** (governance automation)
```bash
make calibrate  # Generate calibrated candidate
make approve    # Check against policy.json thresholds
make rollback   # Restore previous approved model
```
**Expected output**: Active model pointer updated in `artifacts/models/active_model.json`.

### 4. **Run inference service + monitoring** (full system)
```bash
docker compose up --build
```
This starts:
- **Inference API** (http://localhost:8000) — Score applicants
- **Prometheus** (http://localhost:9090) — Metrics collection
- **Grafana** (http://localhost:3000) — Dashboards

**Test**: Send a sample request:
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"checking_account_status": "A11", "credit_history": "A34", "credit_amount": 1000, "duration_months": 12, "purpose": "A40", "savings_account_status": "A61", "employment_years": 4, "installment_rate": 4, "personal_status": "A93", "debtors_guarantors": 1, "residence_years": 2, "age": 40, "concurrent_credit": "A192", "job": "A174", "dependents": 1, "telephone": "A192", "foreign_worker": "A201"}' 
```

**Expected output**: HTTP 200 with predicted probability.

## Full Workflow (Makefile)
Run these from the repository root:

```bash
make train      # Train model
make evaluate   # Compute metrics
make calibrate  # Apply calibration
make approve    # Check governance policy & promote
make rollback   # Restore previous approved model (if needed)
```

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

## Why This System Exists

This service scores loan applicants so risk teams can make consistent, auditable decisions.

- **Cost sensitivity:** A false positive (declining a good applicant) hurts revenue and customer relations; a false negative (approving a bad applicant) causes credit loss. The system prioritizes correctness and conservative promotion rules to limit credit losses.
- **Probability calibration matters:** Business users use predicted probabilities to set thresholds and allocate capital. Calibrated probabilities allow consistent risk limits and automated decisioning.
- **Rollback over blind retrain:** Rolling back to a known-good model keeps production stable while incidents are investigated. Retraining when data is broken risks amplifying bad data.
- **Support for risk decisioning:** The pipeline provides auditable artifacts (feature stats, split indices, model metadata) so risk committees can review why a model was promoted or rolled back.

See [docs/INCIDENT.md](docs/INCIDENT.md) for a concrete example of how the system detects drift and performs a safe rollback.

## Temporal Validation Strategy

- We attach a `timestamp` (YYYY-MM-DD) to raw rows to simulate daily batch ingestion with occasional missing days and late arrivals. Use `scripts/generate_temporal_data.py` to create `data/raw/german_credit_with_ts.txt` from the original raw file.
- Training uses a strict temporal split: pick a split date `T` (80th percentile of observed dates). The model is trained on rows with `t < T`, validated on `t == T`, and tested on `t > T`. Split indices and `T` are recorded in `artifacts/reports/split_indices.json` so leakage is auditable.
- This ensures no lookahead leakage and makes the validation more realistic for production credit scoring.
