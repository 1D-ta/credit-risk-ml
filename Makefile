PYTHON ?= $(shell if [ -x .venv/bin/python ]; then printf '%s' .venv/bin/python; else command -v python3; fi)
TEMPORAL_DATA ?= data/raw/german_credit_with_ts.txt

.PHONY: generate-temporal validate-data train evaluate calibrate approve rollback

generate-temporal:
	$(PYTHON) scripts/generate_temporal_data.py --raw data/raw/german_credit_raw.txt --out $(TEMPORAL_DATA)

validate-data:
	$(MAKE) generate-temporal
	$(PYTHON) training/train.py --raw-data $(TEMPORAL_DATA) --schema data/schemas/schema.json

train:
	$(MAKE) generate-temporal
	$(PYTHON) training/train.py --raw-data $(TEMPORAL_DATA) --schema data/schemas/schema.json

evaluate:
	$(PYTHON) training/evaluate.py --raw-data $(TEMPORAL_DATA) --schema data/schemas/schema.json --model-artifact artifacts/models/model_v1.pkl

calibrate:
	$(PYTHON) training/calibration.py --raw-data $(TEMPORAL_DATA) --schema data/schemas/schema.json --model-artifact artifacts/models/model_v1.pkl

approve:
	$(PYTHON) governance/approve_model.py --model-artifact artifacts/models/calibrated_model_v1.pkl --metrics artifacts/reports/metrics.json --calibration-report artifacts/reports/calibration_report.json

rollback:
	$(PYTHON) governance/rollback.py