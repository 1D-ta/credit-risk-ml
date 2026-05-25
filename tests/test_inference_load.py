#!/usr/bin/env python3
"""Test that inference can load the active model and make predictions."""
import sys
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from credit_risk_ml.data_contract import load_schema, rows_to_frame, validate_and_load_rows
from credit_risk_ml.modeling import load_artifact, predict_bad_probability_from_model
import json


def _run(args: list[str]) -> None:
    cp = subprocess.run([sys.executable] + args, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(args)}\nSTDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}")


def _ensure_active_model() -> None:
    active_model_path = Path("artifacts/models/active_model.json")
    if active_model_path.exists():
        return

    raw_with_ts = Path("data/raw/german_credit_with_ts.txt")
    if not raw_with_ts.exists():
        _run([
            "scripts/generate_temporal_data.py",
            "--raw",
            "data/raw/german_credit_raw.txt",
            "--out",
            "data/raw/german_credit_with_ts.txt",
        ])

    _run([
        "training/train.py",
        "--raw-data",
        "data/raw/german_credit_with_ts.txt",
        "--schema",
        "data/schemas/schema.json",
    ])
    _run([
        "training/evaluate.py",
        "--raw-data",
        "data/raw/german_credit_with_ts.txt",
        "--schema",
        "data/schemas/schema.json",
        "--model-artifact",
        "artifacts/models/model_v1.pkl",
    ])
    _run([
        "training/calibration.py",
        "--raw-data",
        "data/raw/german_credit_with_ts.txt",
        "--schema",
        "data/schemas/schema.json",
        "--model-artifact",
        "artifacts/models/model_v1.pkl",
    ])
    _run([
        "governance/approve_model.py",
        "--model-artifact",
        "artifacts/models/calibrated_model_v1.pkl",
        "--metrics",
        "artifacts/reports/metrics.json",
        "--calibration-report",
        "artifacts/reports/calibration_report.json",
    ])

def test_inference_load():
    """Test loading active model and making predictions."""
    print("=" * 80)
    print("INFERENCE LOAD TEST")
    print("=" * 80)
    
    # Load active model pointer
    _ensure_active_model()
    active_model_path = Path("artifacts/models/active_model.json")
    assert active_model_path.exists(), "active_model.json not found"
    
    with open(active_model_path) as f:
        active_model_config = json.load(f)
    
    model_path = Path(active_model_config["active_model"])
    print(f"Active model pointer: {model_path}")
    
    assert model_path.exists(), f"Model file not found: {model_path}"
    
    # Load the model
    artifact = load_artifact(model_path)
    print(f"Model loaded successfully: version={artifact.model_version}")
    
    # Load schema and test data
    schema = load_schema(Path("data/schemas/schema.json"))
    print(f"Schema loaded: {len(schema.fields)} fields")
    
    # Load a sample from the data
    rows = validate_and_load_rows(Path("data/raw/german_credit_with_ts.txt"), schema)
    frame = rows_to_frame(rows, schema)
    
    # Take first 5 samples
    test_frame = frame.head(5)
    print(f"Test data loaded: {len(test_frame)} samples")
    
    # Make predictions
    predictions = predict_bad_probability_from_model(artifact, test_frame)
    print(f"Predictions generated: {len(predictions)} values")
    print(f"  Sample predictions: {predictions[:3].tolist()}")
    # Check predictions are valid probabilities
    assert all(0 <= float(p) <= 1 for p in predictions), "Predictions outside [0,1] range"
    print("All predictions are valid probabilities")
    
    print("\n" + "=" * 80)
    print("INFERENCE LOAD TEST PASSED")
    print("=" * 80)

if __name__ == "__main__":
    test_inference_load()
    sys.exit(0)
