# Setup Guide

This guide walks you through setting up the Credit Risk ML project on a fresh machine.

## Prerequisites

- **Python 3.9+** (Python 3.10 or 3.11 recommended)
- **Git** (for cloning the repository)
- **Make** (optional, but recommended for easier workflow)

### Platform-Specific Notes

- **macOS/Linux**: All commands should work out of the box
- **Windows**: Use Git Bash, WSL, or PowerShell. Some shell scripts may require WSL.

## Quick Start (Recommended)

```bash
# 1. Clone the repository
git clone <repository-url>
cd credit-risk-ml

# 2. Create virtual environment and install dependencies
make setup

# 3. Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate     # On Windows

# 4. Validate setup
make validate-setup

# 5. Generate temporal data and train model
make train

# 6. Run tests
make test
```

## Manual Setup (Without Make)

If you don't have `make` installed:

```bash
# 1. Create virtual environment
python3 -m venv .venv

# 2. Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate     # On Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Validate setup
python scripts/validate_setup.py

# 5. Generate temporal data
python scripts/generate_temporal_data.py \
    --raw data/raw/german_credit_raw.txt \
    --out data/raw/german_credit_with_ts.txt

# 6. Train model
python training/train.py \
    --raw-data data/raw/german_credit_with_ts.txt \
    --schema data/schemas/schema.json

# 7. Run tests
pytest tests/ -v
```

## Verifying Your Setup

After installation, run the validation script:

```bash
python scripts/validate_setup.py
```

This checks:
- ✓ Python version (3.9+)
- ✓ All required packages installed
- ✓ Project structure is correct
- ✓ Data files exist

## Common Issues

### Issue: `ModuleNotFoundError`

**Solution**: Make sure you've activated the virtual environment and installed dependencies:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### Issue: `FileNotFoundError` for data files

**Solution**: The raw data file needs to be present. If missing:
1. Download the German Credit dataset from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
2. Place `german.data` in `data/raw/` and rename to `german_credit_raw.txt`

### Issue: Permission denied on shell scripts

**Solution**: Make scripts executable:
```bash
chmod +x scripts/*.sh
```

### Issue: `make: command not found`

**Solution**: Either install `make` or use the manual setup commands above.

- **macOS**: `brew install make`
- **Ubuntu/Debian**: `sudo apt-get install build-essential`
- **Windows**: Use WSL or follow manual setup

## Docker Setup (Alternative)

If you prefer Docker:

```bash
# Build and run all services
docker-compose up --build

# The inference API will be available at http://localhost:8000
# Prometheus at http://localhost:9090
# Grafana at http://localhost:3000
```

## Development Workflow

Once set up, typical workflow:

```bash
# 1. Make code changes
# 2. Run tests
make test

# 3. Train model with new changes
make train

# 4. Evaluate model
make evaluate

# 5. Calibrate model
make calibrate

# 6. Approve for production
make approve
```

## Project Structure

```
credit-risk-ml/
├── credit_risk_ml/      # Core ML logic (data contracts, modeling)
├── training/            # Training pipeline scripts
├── inference/           # FastAPI inference service
├── monitoring/          # Drift detection and alerting
├── governance/          # Model approval and rollback
├── tests/               # Test suite
├── scripts/             # Utility scripts
├── data/                # Data files and schemas
├── artifacts/           # Generated models and reports
├── docker/              # Docker configurations
├── requirements.txt     # Python dependencies
├── Makefile            # Build automation
└── SETUP.md            # This file
```

## Next Steps

After successful setup:

1. **Read the README.md** for project overview
2. **Explore the code** starting with `credit_risk_ml/modeling.py`
3. **Run the full pipeline** with `make train`
4. **Start the inference API** with `uvicorn inference.app:app --reload`
5. **Send test requests** with `python scripts/demo_requests.py`

## Getting Help

If you encounter issues:

1. Check this SETUP.md for common issues
2. Run `make validate-setup` to diagnose problems
3. Ensure all paths are relative (no hardcoded absolute paths)
4. Check that you're in the project root directory

## Environment Variables

Optional environment variables:

- `API_URL`: Override inference API URL (default: `http://localhost:8000/predict`)
- `TMPDIR`: Override temporary directory (default: `/tmp` on Unix)

## Reproducibility Notes

This project is designed for reproducibility:

- ✓ All dependencies pinned in `requirements.txt`
- ✓ All paths are relative (no hardcoded absolute paths)
- ✓ Random seeds fixed where applicable
- ✓ Docker support for consistent environments
- ✓ Validation script to verify setup

## Troubleshooting Checklist

- [ ] Python 3.9+ installed
- [ ] Virtual environment activated
- [ ] Dependencies installed (`pip list` shows all packages)
- [ ] In project root directory (`pwd` shows `.../credit-risk-ml`)
- [ ] Data files present in `data/raw/`
- [ ] No import errors when running `python -c "import credit_risk_ml"`