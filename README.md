# Credit Risk ML System

A batch credit risk scoring system implementing ML engineering practices including temporal splits, model calibration, schema validation, drift detection, approval gating, and automated rollback.

## System Overview

Batch credit risk scoring pipeline with temporal validation, model calibration, drift detection, and automated governance.

- **Data:** Public German Credit dataset with synthetic timestamps
- **Scale:** Single-node local execution (batch processing)
- **Monitoring:** PSI-based drift detection with automated alerting
- **Deployment:** Local Docker Compose (not cloud infrastructure)

## System Architecture

**Component breakdown with brief descriptions:**

- **Data Ingestion:** Schema validation with hard failures, temporal ordering verification
- **Training Pipeline:** Temporal splits with leakage prevention, model calibration, comprehensive evaluation
- **Model Registry:** Approval gating based on policy thresholds, version control with metadata
- **Inference Service:** FastAPI with request validation and structured logging
- **Monitoring:** PSI-based drift detection with automated alerting
- **Governance:** Rollback capabilities with model registry integration

## Core Components

- Schema validation with hard failures on mismatch
- Temporal split with leakage prevention (train/val/test)
- Model calibration (Isotonic regression)
- Approval gating based on policy thresholds (ROC-AUC, calibration)
- PSI-based drift detection with configurable thresholds
- Automated rollback on drift detection
- Failure injection testing (schema, numeric shift, categorical)
- Comprehensive test suite (13 tests, 100% pass rate)
- Structured JSON logging with deterministic input hashing
- Feature parity validation between training and inference

## Quick Start

```bash
# Setup
make setup
source .venv/bin/activate

# Generate temporal data
make generate-temporal

# Run training pipeline
make train
make evaluate
make calibrate

# Approve model
python governance/approve_model.py

# Start inference service
uvicorn inference.app:app --host 0.0.0.0 --port 8000

# Run tests
make test
```

## Failure Modes Tested

- **Schema mismatch** → Hard failure with ValueError, request rejected before inference
- **Missing required field** → 422 validation error with clear message
- **Feature drift (PSI > 0.2)** → Alert triggered + automatic rollback to previous approved model
- **Invalid categorical value** → 422 validation error, request rejected
- **Numeric shift** → Monitoring alert, PSI calculation flags distribution change
- **Label noise** → Measurable AUC degradation tracked in robustness script
- **Bad model promotion** → Approval gate prevents update, active model unchanged
- **Active model regression** → Rollback script resets active pointer to last approved version

## Sample Outputs

**Model Performance (`artifacts/reports/metrics.json`):**
```json
{
  "roc_auc": 0.8196,
  "approval_rate": 0.0867,
  "decision_threshold": 0.3
}
```

**Calibration Improvement (`artifacts/reports/calibration_report.json`):**
```json
{
  "brier_score_before": 0.2074,
  "brier_score_after": 0.1327,
  "calibration_gap_after": 1.11e-16,
  "method": "isotonic_regression"
}
```

**Drift Detection (`artifacts/reports/drift_report.json`):**
```json
{
  "alert": true,
  "psi_threshold": 0.2,
  "feature_reports": {
    "credit_amount": {
      "psi": 0.3111,
      "ks": 0.215
    }
  }
}
```

## Repository Structure

```
credit-risk-ml/
├── training/          # Model training, evaluation, and calibration
├── inference/         # FastAPI service with validation
├── monitoring/        # Drift detection and alerting
├── governance/        # Approval and rollback scripts
├── credit_risk_ml/    # Core data contracts and utilities
├── tests/            # Test suite (13 tests)
├── artifacts/        # Execution evidence and model artifacts
├── data/             # Raw data, schemas, and failure cases
└── scripts/          # Reproducibility and failure injection
```

## Design Decisions

**Why Logistic Regression + XGBoost:**
- Logistic regression provides interpretable baseline
- XGBoost offers ensemble comparison
- Both are fast to train on tabular data
- Suitable for credit risk where interpretability matters

**Why Calibration:**
- Credit risk decisions depend on probability accuracy, not just ranking
- Isotonic regression improves Brier score by 36%
- Calibration gap reduced to near-zero (1.11e-16)

**Why Governance Layer:**
- Implements approval workflows common in regulated environments
- Separates model quality checks from deployment
- Enables safe rollback without code changes

**Why Temporal Splits:**
- Prevents data leakage in time-series scenarios
- Validates model performance on future data
- Enforces train_max < val_min < test_min ordering

## Architecture

**System characteristics:**

- **Single-node execution** - Not distributed (no Spark/Dask)
- **Batch processing only** - Not real-time streaming
- **Local monitoring** - PSI-based drift detection, not integrated with enterprise monitoring stack
- **No authentication** - No security hardening or access controls
- **Synthetic timestamps** - Public dataset with generated temporal ordering
- **No disaster recovery** - No backup/restore or high availability
- **Local Docker Compose** - Not Kubernetes or cloud-managed services
- **No feature store** - No centralized feature management
- **No A/B testing** - No canary deployments or gradual rollouts
- **No compliance documentation** - No model risk management artifacts

## Validation Results

- 13/13 tests passing (100% pass rate)
- Complete pipeline execution verified
- All workflows operational (train, evaluate, calibrate, approve, rollback)
- Failure injection scenarios validated
- Drift detection and alerting functional

See [`VALIDATION.md`](VALIDATION.md) for detailed validation report.

## Documentation

- [`SETUP.md`](SETUP.md) - Installation and setup instructions
- [`VALIDATION.md`](VALIDATION.md) - End-to-end system validation report
- [`INCIDENT.md`](INCIDENT.md) - Incident response scenario (schema drift)
- [`artifacts/ARTIFACTS.md`](artifacts/ARTIFACTS.md) - Execution evidence and artifact descriptions
