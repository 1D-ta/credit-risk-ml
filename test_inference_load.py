#!/usr/bin/env python3
"""Test that inference can load the active model and make predictions."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from credit_risk_ml.data_contract import load_schema, rows_to_frame, validate_and_load_rows
from credit_risk_ml.modeling import load_artifact, predict_bad_probability_from_model
import json

def test_inference_load():
    """Test loading active model and making predictions."""
    print("=" * 80)
    print("INFERENCE LOAD TEST")
    print("=" * 80)
    
    # Load active model pointer
    active_model_path = Path("artifacts/models/active_model.json")
    if not active_model_path.exists():
        print("❌ FAIL: active_model.json not found")
        return False
    
    with open(active_model_path) as f:
        active_model_config = json.load(f)
    
    model_path = Path(active_model_config["active_model"])
    print(f"✓ Active model pointer: {model_path}")
    
    if not model_path.exists():
        print(f"❌ FAIL: Model file not found: {model_path}")
        return False
    
    # Load the model
    try:
        artifact = load_artifact(model_path)
        print(f"✓ Model loaded successfully: version={artifact.model_version}")
    except Exception as e:
        print(f"❌ FAIL: Could not load model: {e}")
        return False
    
    # Load schema and test data
    schema = load_schema(Path("data/schemas/schema.json"))
    print(f"✓ Schema loaded: {len(schema.fields)} fields")
    
    # Load a sample from the data
    rows = validate_and_load_rows(Path("data/raw/german_credit_with_ts.txt"), schema)
    frame = rows_to_frame(rows, schema)
    
    # Take first 5 samples
    test_frame = frame.head(5)
    print(f"✓ Test data loaded: {len(test_frame)} samples")
    
    # Make predictions
    try:
        predictions = predict_bad_probability_from_model(artifact, test_frame)
        print(f"✓ Predictions generated: {len(predictions)} values")
        print(f"  Sample predictions: {predictions[:3].tolist()}")
        
        # Check predictions are valid probabilities
        if not all(0 <= p <= 1 for p in predictions):
            print("❌ FAIL: Predictions outside [0,1] range")
            return False
        
        print("✓ All predictions are valid probabilities")
        
    except Exception as e:
        print(f"❌ FAIL: Prediction failed: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("✓ INFERENCE LOAD TEST PASSED")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = test_inference_load()
    sys.exit(0 if success else 1)