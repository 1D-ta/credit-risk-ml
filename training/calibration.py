from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import brier_score_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from credit_risk_ml.data_contract import load_schema, rows_to_frame, target_series, validate_and_load_rows
from credit_risk_ml.modeling import CalibratedRiskArtifact, load_artifact, predict_bad_probability_from_model, save_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate the trained credit risk model.")
    parser.add_argument("--raw-data", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--model-artifact", required=True, type=Path)
    parser.add_argument("--output-artifact", type=Path, default=Path("artifacts/models/calibrated_model_v1.pkl"))
    parser.add_argument("--report-output", type=Path, default=Path("artifacts/reports/calibration_report.json"))
    parser.add_argument("--split-indices", type=Path, default=Path("artifacts/reports/split_indices.json"))
    return parser.parse_args()


def calibration_gap(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    gaps: list[float] = []
    for index in range(bins):
        lower_bound = bin_edges[index]
        upper_bound = bin_edges[index + 1]
        if index == bins - 1:
            mask = (probabilities >= lower_bound) & (probabilities <= upper_bound)
        else:
            mask = (probabilities >= lower_bound) & (probabilities < upper_bound)
        if not mask.any():
            continue
        observed = float(y_true[mask].mean())
        expected = float(probabilities[mask].mean())
        gaps.append(abs(observed - expected))
    return float(max(gaps) if gaps else 0.0)


def main() -> None:
    args = parse_args()
    schema = load_schema(args.schema)
    rows = validate_and_load_rows(args.raw_data, schema)
    frame = rows_to_frame(rows, schema)

    # load split indices produced by training
    split_indices = None
    if args.split_indices.exists():
        split_indices = json.loads(args.split_indices.read_text())

    if not split_indices:
        raise RuntimeError(f"Split indices not found at {args.split_indices}; run training to produce splits")

    calibration_idx = split_indices.get("calibration_idx", [])
    calibration_frame = frame.loc[calibration_idx].reset_index(drop=True)
    calibration_target = (target_series(calibration_frame, schema) == 2).astype(int)

    training_artifact = load_artifact(args.model_artifact)
    calibration_probabilities = predict_bad_probability_from_model(training_artifact, calibration_frame)
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(calibration_probabilities.to_numpy(), calibration_target.to_numpy())

    calibrated_probabilities = calibrator.transform(calibration_probabilities.to_numpy())
    report = {
        "method": "isotonic_regression",
        "calibration_size": int(len(calibration_frame)),
        "brier_score_before": float(brier_score_loss(calibration_target, calibration_probabilities)),
        "brier_score_after": float(brier_score_loss(calibration_target, calibrated_probabilities)),
        "calibration_gap_after": calibration_gap(calibration_target.to_numpy(), calibrated_probabilities),
    }

    calibrated_artifact = CalibratedRiskArtifact(
        model_version=training_artifact.model_version,
        training_artifact=training_artifact,
        calibrator=calibrator,
        calibration_method="isotonic_regression",
        decision_threshold=training_artifact.decision_threshold,
    )

    save_artifact(calibrated_artifact, args.output_artifact)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True))

    print(json.dumps({"status": "ok", "artifact": str(args.output_artifact), "report": str(args.report_output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()