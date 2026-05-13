# Detect Python interpreter (prefer virtual environment)
PYTHON ?= $(shell if [ -x .venv/bin/python ]; then printf '%s' .venv/bin/python; else command -v python3; fi)
TEMPORAL_DATA ?= data/raw/german_credit_with_ts.txt

.PHONY: help setup install validate-setup generate-temporal validate-data train evaluate calibrate approve rollback test clean

# Default target: show help
help:
	@echo "Credit Risk ML Pipeline - Available Commands:"
	@echo ""
	@echo "  make setup           - Create virtual environment and install dependencies"
	@echo "  make install         - Install dependencies (assumes venv exists)"
	@echo "  make validate-setup  - Validate environment setup (run after setup)"
	@echo "  make generate-temporal - Generate timestamped training data"
	@echo "  make validate-data   - Validate data schema and quality"
	@echo "  make train           - Train the credit risk model"
	@echo "  make evaluate        - Evaluate model on test set"
	@echo "  make calibrate       - Calibrate model probabilities"
	@echo "  make approve         - Approve model for production"
	@echo "  make rollback        - Rollback to previous approved model"
	@echo "  make test            - Run test suite"
	@echo "  make clean           - Remove generated artifacts"
	@echo ""

# Setup: create virtual environment and install dependencies
setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo ""
	@echo "Setup complete! Activate the virtual environment with:"
	@echo "  source .venv/bin/activate"

# Install dependencies into existing environment
install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

# Validate setup after installation
validate-setup:
	$(PYTHON) scripts/validate_setup.py

# Generate temporal data with timestamps
generate-temporal:
	$(PYTHON) scripts/generate_temporal_data.py --raw data/raw/german_credit_raw.txt --out $(TEMPORAL_DATA)

# Validate data schema and quality
validate-data:
	$(MAKE) generate-temporal
	$(PYTHON) training/train.py --raw-data $(TEMPORAL_DATA) --schema data/schemas/schema.json

# Train the model
train:
	$(MAKE) generate-temporal
	$(PYTHON) training/train.py --raw-data $(TEMPORAL_DATA) --schema data/schemas/schema.json

# Evaluate model performance
evaluate:
	$(PYTHON) training/evaluate.py --raw-data $(TEMPORAL_DATA) --schema data/schemas/schema.json --model-artifact artifacts/models/model_v1.pkl

# Calibrate model probabilities
calibrate:
	$(PYTHON) training/calibration.py --raw-data $(TEMPORAL_DATA) --schema data/schemas/schema.json --model-artifact artifacts/models/model_v1.pkl

# Approve model for production
approve:
	$(PYTHON) governance/approve_model.py --model-artifact artifacts/models/calibrated_model_v1.pkl --metrics artifacts/reports/metrics.json --calibration-report artifacts/reports/calibration_report.json

# Rollback to previous approved model
rollback:
	$(PYTHON) governance/rollback.py

# Run test suite
test:
	$(PYTHON) -m pytest tests/ -v

# Clean generated artifacts
clean:
	rm -rf artifacts/models/*.pkl
	rm -rf artifacts/reports/*.json
	rm -rf artifacts/logs/*.log
	rm -rf artifacts/logs/*.jsonl
	@echo "Cleaned generated artifacts"