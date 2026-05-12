#!/bin/bash
set -e

echo "=========================================="
echo "Credit Risk ML: End-to-End Demo"
echo "=========================================="
echo ""

# Step 1: Setup
echo "Step 1: Setup Python environment"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✓ Created virtual environment"
fi
source .venv/bin/activate
pip install -q -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Step 2: Generate temporal data
echo "Step 2: Generate temporal data with timestamps"
python scripts/generate_temporal_data.py \
    --raw data/raw/german_credit_raw.txt \
    --out data/raw/german_credit_with_ts.txt
echo "✓ Temporal data created: data/raw/german_credit_with_ts.txt"
echo ""

# Step 3: Train model with temporal split
echo "Step 3: Train model with temporal validation split"
python training/train.py \
    --raw-data data/raw/german_credit_with_ts.txt \
    --schema data/schemas/schema.json \
    --model-output artifacts/models/model_v1.pkl \
    --training-metadata-output artifacts/reports/training_metadata.json
echo "✓ Model trained"
echo ""

# Step 4: Evaluate, calibrate, approve
echo "Step 4: Evaluate + calibrate + approve candidate"
python training/evaluate.py \
    --raw-data data/raw/german_credit_with_ts.txt \
    --schema data/schemas/schema.json \
    --model-artifact artifacts/models/model_v1.pkl
python training/calibration.py \
    --raw-data data/raw/german_credit_with_ts.txt \
    --schema data/schemas/schema.json \
    --model-artifact artifacts/models/model_v1.pkl \
    --output-artifact artifacts/models/calibrated_model_v1.pkl
python governance/approve_model.py \
    --model-artifact artifacts/models/calibrated_model_v1.pkl \
    --metrics artifacts/reports/metrics.json \
    --calibration-report artifacts/reports/calibration_report.json
echo "✓ Candidate approved and active model pointer set"
echo ""

# Step 5: Create drifted current data and run monitoring -> alert -> rollback
echo "Step 5: Monitoring loop (metric -> breach -> alert -> rollback)"
python scripts/failure_injection/inject_shifted_current.py \
    --src data/raw/german_credit_with_ts.txt \
    --out data/failure_cases/current_shifted_with_ts.txt \
    --offset 500
python monitoring/drift.py \
    --reference-data data/raw/german_credit_with_ts.txt \
    --current-data data/failure_cases/current_shifted_with_ts.txt \
    --schema data/schemas/schema.json \
    --psi-threshold 0.2 \
    --auto-rollback
echo "✓ Monitoring loop executed"
echo ""

# Step 6: Feature parity mismatch example
echo "Step 6: Feature parity enforcement example"
python - <<'PY'
import json
from pathlib import Path

schema = json.loads(Path("data/schemas/schema.json").read_text())
base_line = Path("data/raw/german_credit_with_ts.txt").read_text().splitlines()[0].split()
fields = [f["name"] for f in schema["fields"]]
record = {k: v for k, v in zip(fields, base_line)}

samples = Path("artifacts/feature_stats/inference_samples.jsonl")
samples.parent.mkdir(parents=True, exist_ok=True)
with samples.open("w", encoding="utf-8") as h:
    for _ in range(30):
        shifted = dict(record)
        shifted["credit_amount"] = int(float(shifted["credit_amount"]) * 1.35)
        shifted.pop("event_time", None)
        shifted.pop("credit_risk", None)
        h.write(json.dumps(shifted) + "\n")
PY
python - <<'PY'
from monitoring.feature_check import run_feature_check_and_maybe_fail
print(run_feature_check_and_maybe_fail())
PY
echo "✓ Feature parity check executed"
echo ""

# Step 7: Show key artifacts
echo "=========================================="
echo "📊 KEY ARTIFACTS CREATED"
echo "=========================================="
echo ""

if [ -f "artifacts/reports/split_indices.json" ]; then
    echo "📅 Temporal Split Information:"
    echo "  File: artifacts/reports/split_indices.json"
    python -c "import json; d = json.load(open('artifacts/reports/split_indices.json')); r=d.get('event_time_ranges',{}); print(f'  Split Date (T): {d.get(\"temporal_split_date_T\", \"N/A\")}'); print(f'  Train: {r.get(\"train\",{}).get(\"start\")}..{r.get(\"train\",{}).get(\"end\")} rows={r.get(\"train\",{}).get(\"rows\")}'); print(f'  Validation: {r.get(\"validation\",{}).get(\"start\")} rows={r.get(\"validation\",{}).get(\"rows\")}'); print(f'  Test: {r.get(\"test\",{}).get(\"start\")}..{r.get(\"test\",{}).get(\"end\")} rows={r.get(\"test\",{}).get(\"rows\")}')"
    echo ""
fi

if [ -f "artifacts/logs/temporal_split.log" ]; then
    echo "🧪 Temporal Split Log:"
    echo "  File: artifacts/logs/temporal_split.log"
    tail -n 4 artifacts/logs/temporal_split.log
    echo ""
fi

if [ -f "artifacts/feature_stats/training_features.json" ]; then
    echo "📈 Training Feature Stats:"
    echo "  File: artifacts/feature_stats/training_features.json"
    echo "  Sample (credit_amount):"
    python -c "import json; stats = json.load(open('artifacts/feature_stats/training_features.json')); ca = stats.get('credit_amount', {}); print(f'    Mean: {ca.get(\"mean\")}, Std: {ca.get(\"std\")}, Missing%: {ca.get(\"missing_pct\")}')"
    echo ""
fi

if [ -f "artifacts/feature_stats/inference_features.json" ]; then
    echo "⚠️  Inference Feature Stats (Simulated Mismatch - Example):"
    echo "  File: artifacts/feature_stats/inference_features.json"
    echo "  This shows a shifted credit_amount distribution for parity enforcement"
    echo ""
fi

if [ -f "artifacts/logs/feature_mismatch.log" ]; then
    echo "🚨 Feature Mismatch Detection Log (Example):"
    echo "  File: artifacts/logs/feature_mismatch.log"
    python -c "import json; data = json.load(open('artifacts/logs/feature_mismatch.log')); print(f'  Alert: {data.get(\"alert\")}'); [print(f'    {m}') for m in data.get('messages', [])[:2]]"
    echo ""
fi

if [ -f "artifacts/monitoring/alert_example.json" ]; then
    echo "📢 Monitoring Alert Example:"
    echo "  File: artifacts/monitoring/alert_example.json"
    python -c "import json; alert = json.load(open('artifacts/monitoring/alert_example.json')); print(f'  Alert: {alert.get(\"alert\")} (PSI: {alert.get(\"details\", {}).get(\"psi\")})')"
    echo ""
fi

if [ -f "artifacts/logs/monitoring_alerts.log" ]; then
    echo "🔁 Alert to Action Log Sequence:"
    echo "  File: artifacts/logs/monitoring_alerts.log"
    tail -n 12 artifacts/logs/monitoring_alerts.log
    echo ""
fi

if [ -f "artifacts/reports/training_metadata.json" ]; then
    echo "📋 Training Metadata:"
    echo "  File: artifacts/reports/training_metadata.json"
    python -c "import json; md = json.load(open('artifacts/reports/training_metadata.json')); print(f'  Model Version: {md.get(\"model_version\")}'); print(f'  Rows: {md.get(\"rows\")}'); print(f'  ROC AUC: {md.get(\"metrics\", {}).get(\"roc_auc\")}'); print(f'  Approval Rate: {md.get(\"metrics\", {}).get(\"approval_rate\")}')"
    echo ""
fi

echo "=========================================="
echo "📝 DOCUMENTATION"
echo "=========================================="
echo ""
echo "Business Context: README.md → 'Risk Tradeoffs' section"
echo "Temporal Strategy:  README.md → 'Temporal Validation' section"
echo "Failure Flow:       README.md → 'Failure Detection Flow' section"
echo "Incident Story:     INCIDENT.md (schema/units drift, detection, rollback)"
echo "Requirements:       requirements.txt (now includes prometheus_client)"
echo ""

echo "=========================================="
echo "🚀 NEXT STEPS"
echo "=========================================="
echo ""
echo "1. Start the inference API (opens on port 8000):"
echo "   $ uvicorn inference.app:app --host 0.0.0.0 --port 8000"
echo ""
echo "2. In another terminal, send a test request:"
echo "   $ curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' \\"
echo "     -d '{\"checking_account_status\":\"A11\",\"duration_months\":12,\"credit_history\":\"A34\",\"purpose\":\"A40\",\"credit_amount\":5000,\"savings_account\":\"A61\",\"employment_status\":\"A75\",\"installment_rate\":4,\"personal_status_sex\":\"A93\",\"other_debtors\":\"A101\",\"residence_since\":4,\"property\":\"A121\",\"age\":67,\"other_installment_plans\":\"A143\",\"housing\":\"A152\",\"existing_credits\":2,\"job\":\"A173\",\"num_liable\":1,\"telephone\":\"A192\",\"foreign_worker\":\"A201\"}'"
echo ""
echo "3. View Prometheus metrics (will be collected on port 8001):"
echo "   $ curl http://localhost:8001/metrics | grep inference_"
echo ""
echo "4. Check prediction logs:"
echo "   $ tail -f artifacts/logs/predictions.jsonl"
echo ""
echo "=========================================="
echo "✅ Demo setup complete!"
echo "=========================================="
