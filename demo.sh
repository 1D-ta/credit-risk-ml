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

# Step 4: Show key artifacts
echo "=========================================="
echo "📊 KEY ARTIFACTS CREATED"
echo "=========================================="
echo ""

if [ -f "artifacts/reports/split_indices.json" ]; then
    echo "📅 Temporal Split Information:"
    echo "  File: artifacts/reports/split_indices.json"
    python -c "import json; d = json.load(open('artifacts/reports/split_indices.json')); print(f'  Split Date (T): {d.get(\"temporal_split_date_T\", \"N/A\")}'); print(f'  Train rows: {len(d.get(\"train_idx\", []))}'); print(f'  Validate rows: {len(d.get(\"calibration_idx\", []))}'); print(f'  Test rows: {len(d.get(\"test_idx\", []))}')"
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
    echo "  This shows a 100x shift in credit_amount (units changed to cents)"
    echo ""
fi

if [ -f "artifacts/logs/feature_mismatch.log" ]; then
    echo "🚨 Feature Mismatch Detection Log (Example):"
    echo "  File: artifacts/logs/feature_mismatch.log"
    python -c "import json; data = json.load(open('artifacts/logs/feature_mismatch.log')); print(f'  Alert: {data.get(\"alert\")}'); failures = data.get('failures', {}); [print(f'    {k}: {v.get(\"reason\")} (train: {v.get(\"train_mean\")}, infer: {v.get(\"inference_mean\")})') for k, v in list(failures.items())[:2]]"
    echo ""
fi

if [ -f "artifacts/monitoring/alert_example.json" ]; then
    echo "📢 Monitoring Alert Example:"
    echo "  File: artifacts/monitoring/alert_example.json"
    python -c "import json; alert = json.load(open('artifacts/monitoring/alert_example.json')); print(f'  Alert: {alert.get(\"alert\")} (Error rate: {alert.get(\"details\", {}).get(\"error_rate\")})')"
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
echo "Business Context: README.md → 'Why This System Exists' section"
echo "Temporal Strategy:  README.md → 'Temporal Validation Strategy' section"
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
