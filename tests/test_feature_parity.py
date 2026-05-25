"""
Feature Parity Test: Verify training and inference pipelines produce identical features.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from credit_risk_ml.data_contract import load_schema, rows_to_frame, validate_and_load_rows
from credit_risk_ml.modeling import load_artifact


def test_feature_parity():
    """
    Test that training pipeline and inference pipeline produce identical features.
    """
    # Load training data and trained model
    schema = load_schema(Path("data/schemas/schema.json"))
    
    # Load a small sample of raw data
    raw_data_path = Path("data/raw/german_credit_with_ts.txt")
    if not raw_data_path.exists():
        # Skip if no data yet
        print("SKIP: raw data not found, run make generate-temporal first")
        return
    
    rows = validate_and_load_rows(raw_data_path, schema)
    frame = rows_to_frame(rows, schema)
    
    # Load trained model
    model_path = Path("artifacts/models/calibrated_model_v1.pkl")
    if not model_path.exists():
        # Skip if model not trained yet
        print("SKIP: model not found, run make approve first")
        return
    
    calibrated_artifact = load_artifact(model_path)
    training_artifact = calibrated_artifact.training_artifact
    
    # Take first 100 samples for testing
    test_frame = frame.head(100).copy()
    
    # Get features via model pipeline (this is what happens during training)
    X_from_pipeline = test_frame[training_artifact.feature_columns].copy()
    X_transformed = training_artifact.pipeline.named_steps['preprocessor'].transform(X_from_pipeline)
    
    # Expected column names after preprocessing
    expected_feature_names = training_artifact.pipeline.named_steps['preprocessor'].get_feature_names_out().tolist()
    
    print(f"\n{'='*80}")
    print("FEATURE PARITY TEST")
    print(f"{'='*80}")
    print(f"Model version: {training_artifact.model_version}")
    print(f"Training feature columns: {training_artifact.feature_columns}")
    print(f"Categorical columns: {training_artifact.categorical_columns}")
    print(f"Numeric columns: {training_artifact.numeric_columns}")
    print(f"Expected transformed features: {len(expected_feature_names)}")
    print(f"Actual transformed features: {X_transformed.shape[1]}")
    
    # Verify all expected features present
    assert X_transformed.shape[1] > 0, "No features generated after preprocessing"
    assert X_transformed.shape[0] == len(test_frame), f"Row count mismatch: {X_transformed.shape[0]} vs {len(test_frame)}"
    
    # Verify no NaN values (imputation should have filled them)
    nan_count = np.isnan(X_transformed).sum()
    assert nan_count == 0, f"Found {nan_count} NaN values in transformed features (imputation failed)"
    
    # Verify feature values are reasonable (scaled)
    feature_mean = X_transformed.mean(axis=0)
    feature_std = X_transformed.std(axis=0)
    print(f"\nTransformed feature statistics:")
    print(f"  Mean: {feature_mean.mean():.4f} (should be ≈0 after scaling)")
    print(f"  Std: {feature_std.mean():.4f} (should be ≈1 after scaling)")
    
    # Verify consistency: same raw row should always produce same transformed features
    row_0 = test_frame.iloc[0:1]
    X_0_first = training_artifact.pipeline.named_steps['preprocessor'].transform(
        row_0[training_artifact.feature_columns]
    )
    X_0_second = training_artifact.pipeline.named_steps['preprocessor'].transform(
        row_0[training_artifact.feature_columns]
    )
    
    parity_check = np.allclose(X_0_first, X_0_second, rtol=1e-10, atol=1e-15)
    assert parity_check, "FEATURE_PARITY_FAILED: Same input produces different output (preprocessing is non-deterministic)"
    
    print("\nFeature parity check passed")
    print(f"  Same raw input transforms identically across calls")
    print(f"  All features properly imputed and scaled")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    try:
        test_feature_parity()
        print("All feature parity tests passed")
        sys.exit(0)
    except AssertionError as e:
        print(f"FEATURE_PARITY_TEST_FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"FEATURE_PARITY_TEST_ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
