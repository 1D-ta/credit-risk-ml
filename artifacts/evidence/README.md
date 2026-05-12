# Evidence Artifacts

This directory contains real outputs from the credit risk ML system validation, demonstrating key production capabilities.

## Files

### 1. `drift_alert_rollback.log`
**Demonstrates:** Automatic drift detection and rollback chain

Shows PSI (Population Stability Index) computed per time batch, detecting severe drift (PSI=7.53 >> threshold=0.2), triggering automatic alert, marking model unhealthy, and executing rollback to last approved model without manual intervention.

**Key Evidence:**
- Temporal PSI monitoring (per-date batch)
- Automatic alert on threshold breach
- Automatic rollback execution
- Production protection from degraded model

---

### 2. `temporal_training.log`
**Demonstrates:** Strict temporal splits preventing data leakage

Shows training with T1/T2 temporal boundaries, explicit date ranges for train/val/test, and leakage validation ensuring `train_max < val_min < test_min`.

**Key Evidence:**
- Temporal split configuration (T1=2020-05-01, T2=2020-06-10)
- No data leakage (dates strictly ordered)
- Realistic time-based evaluation
- Auditable split metadata

---

### 3. `inference_success.json`
**Demonstrates:** Valid API response with all checks passed

Shows successful prediction request with schema validation, feature consistency check, and drift check all passing. Returns risk score, decision, and metadata.

**Key Evidence:**
- Complete feature validation
- Risk score and decision output
- Low latency (12.4ms)
- Structured response format

---

### 4. `inference_failure.json`
**Demonstrates:** Schema validation rejection

Shows request rejected due to missing required field (`credit_amount`). System enforces strict schema before model inference, preventing invalid predictions.

**Key Evidence:**
- Pre-inference validation
- Clear error messages
- HTTP 422 (Unprocessable Entity)
- Fail-fast behavior

---

### 5. `feature_mismatch.log`
**Demonstrates:** Feature distribution mismatch detection

Shows detection of 100x shift in `credit_amount` (cents vs euros), with detailed analysis of training baseline vs inference statistics. Request rejected with HTTP 503 to prevent silent model degradation.

**Key Evidence:**
- Feature consistency monitoring
- Root cause analysis (unit mismatch)
- Automatic rejection on distribution shift
- Protection from data pipeline corruption

---

## What This Proves

✅ **Temporal Integrity:** No data leakage, realistic time-based splits  
✅ **Automatic Monitoring:** PSI computed per time batch, alerts on drift  
✅ **Automatic Rollback:** No manual intervention needed for model protection  
✅ **Schema Enforcement:** Strict validation before inference  
✅ **Feature Consistency:** Distribution monitoring catches pipeline issues  
✅ **Production Safety:** Multiple layers of validation and automatic response

---

## For Hiring Managers

These artifacts demonstrate a production-grade ML system with:
- **Temporal discipline** (no leakage)
- **Automatic failure detection** (drift, schema, features)
- **Automatic response** (rollback without manual intervention)
- **Observable behavior** (structured logs, clear error messages)
- **Safety-first design** (fail-fast, reject bad data, protect production)

Review time: <60 seconds to understand the system's key capabilities.