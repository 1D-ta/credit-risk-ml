#!/usr/bin/env python3
"""
Robustness testing: train models with and without label noise to measure impact.

This script demonstrates the effect of label noise on model performance,
which is crucial for understanding real-world risks.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from credit_risk_ml.data_contract import load_schema, rows_to_frame, validate_and_load_rows
from credit_risk_ml.label_noise import inject_label_noise, compare_model_performance
from credit_risk_ml.modeling import fit_artifact


def main():
    parser = argparse.ArgumentParser(description="Test model robustness to label noise")
    parser.add_argument("--raw-data", type=Path, required=True, help="Raw training data")
    parser.add_argument("--schema", type=Path, required=True, help="Schema file")
    parser.add_argument("--noise-fraction", type=float, default=0.05,
                        help="Fraction of labels to corrupt (default 0.05)")
    parser.add_argument("--output", type=Path, default=Path("artifacts/reports/label_noise_report.json"),
                        help="Output report file")
    
    args = parser.parse_args()
    
    print(f"Loading training data...")
    schema = load_schema(args.schema)
    rows = validate_and_load_rows(args.raw_data, schema)
    frame = rows_to_frame(rows, schema)
    
    # Temporal split
    import pandas as _pd
    if "event_time" not in frame.columns:
        raise RuntimeError("event_time column missing; run make generate-temporal first")
    
    frame["event_time"] = _pd.to_datetime(frame["event_time"], errors="coerce").dt.date
    unique_dates = sorted(frame["event_time"].unique())
    t1_pos = int(len(unique_dates) * 0.6)
    T1 = unique_dates[t1_pos]
    
    train_idx = frame.index[frame["event_time"] < T1].tolist()
    test_idx = frame.index[frame["event_time"] >= T1].tolist()
    
    train_frame = frame.loc[train_idx].reset_index(drop=True)
    test_frame = frame.loc[test_idx].reset_index(drop=True)
    
    print(f"Training data: {len(train_frame):,} samples")
    print(f"Test data: {len(test_frame):,} samples")
    print(f"Baseline target rate: {train_frame[schema.target_column].mean()*100:.2f}%")
    
    # Train model on clean data
    print(f"\n1. Training model on CLEAN labels...")
    artifact_clean, metrics_clean = fit_artifact(train_frame, schema)
    
    # For simplicity, use uncalibrated predictions
    X_test = test_frame[artifact_clean.feature_columns]
    y_test_binary = (test_frame[schema.target_column] == 2).astype(int)
    pred_clean = artifact_clean.pipeline.predict_proba(X_test)[:, 1]
    
    print(f"   AUC (clean): {metrics_clean['roc_auc']:.4f}")
    
    # Inject label noise
    print(f"\n2. Injecting {args.noise_fraction*100:.1f}% label noise...")
    y_train_clean = (train_frame[schema.target_column] == 2).astype(int)
    y_train_noisy, noise_metadata = inject_label_noise(
        y_train_clean,
        noise_fraction=args.noise_fraction,
        noise_type="flip"
    )
    
    print(f"   Noise injected: {noise_metadata['num_flipped']} labels flipped")
    print(f"   - 0→1 flips: {noise_metadata['num_0_to_1_flips']}")
    print(f"   - 1→0 flips: {noise_metadata['num_1_to_0_flips']}")
    
    # Retrain with noisy labels
    print(f"\n3. Training model on NOISY labels...")
    train_frame_noisy = train_frame.copy()
    train_frame_noisy[schema.target_column] = y_train_noisy
    
    artifact_noisy, metrics_noisy = fit_artifact(train_frame_noisy, schema)
    pred_noisy = artifact_noisy.pipeline.predict_proba(X_test)[:, 1]
    
    print(f"   AUC (noisy): {metrics_noisy['roc_auc']:.4f}")
    print(f"   AUC degradation: {metrics_clean['roc_auc'] - metrics_noisy['roc_auc']:.4f}")
    
    # Compare performance
    print(f"\n4. Comparing performance on clean test set...")
    perf_comparison = compare_model_performance(y_test_binary, pred_clean, pred_noisy)
    
    for metric, clean_val in perf_comparison["clean_metrics"].items():
        noisy_val = perf_comparison["noisy_metrics"][metric]
        delta = perf_comparison["performance_degradation"][f"delta_{metric}"]
        print(f"   {metric:10s}: {clean_val:.4f} (clean) vs {noisy_val:.4f} (noisy) → {delta:+.4f}")
    
    # Report
    report = {
        "test_name": "label_noise_robustness",
        "data": {
            "train_samples": len(train_frame),
            "test_samples": len(test_frame),
            "target_rate": float(y_train_clean.mean()),
        },
        "noise_injection": noise_metadata,
        "model_clean": {
            "version": artifact_clean.model_version,
            "metrics": metrics_clean,
        },
        "model_noisy": {
            "version": artifact_noisy.model_version,
            "metrics": metrics_noisy,
        },
        "performance_comparison": perf_comparison,
        "conclusion": f"Label noise of {args.noise_fraction*100:.1f}% causes "
                     f"{perf_comparison['performance_degradation']['delta_auc']*100:.2f}% AUC degradation",
    }
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    
    print(f"\n✓ Report saved to {args.output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
