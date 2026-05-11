from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from credit_risk_ml.data_contract import dataset_summary, load_schema, rows_to_frame, validate_and_load_rows
from credit_risk_ml.modeling import fit_artifact, save_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the German Credit raw dataset and train the model.")
    parser.add_argument("--raw-data", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--report-output", type=Path, default=Path("artifacts/reports/data_validation_report.json"))
    parser.add_argument("--model-output", type=Path, default=Path("artifacts/models/model_v1.pkl"))
    parser.add_argument("--training-metadata-output", type=Path, default=Path("artifacts/reports/training_metadata.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schema = load_schema(args.schema)
    rows = validate_and_load_rows(args.raw_data, schema)
    report = dataset_summary(rows, schema, args.raw_data)
    frame = rows_to_frame(rows, schema)

    # deterministic split into train/calibration/test (60/20/20)
    import numpy as _np

    rng = _np.random.RandomState(42)
    indices = rng.permutation(len(frame))
    n = len(frame)
    n_train = int(n * 0.6)
    n_cal = int(n * 0.2)
    train_idx = indices[:n_train].tolist()
    calibration_idx = indices[n_train : n_train + n_cal].tolist()
    test_idx = indices[n_train + n_cal :].tolist()

    split_indices = {
        "train_idx": train_idx,
        "calibration_idx": calibration_idx,
        "test_idx": test_idx,
    }

    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True))

    # save split indices for downstream steps
    split_path = Path("artifacts/reports/split_indices.json")
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split_indices, indent=2, sort_keys=True))

    train_frame = frame.loc[train_idx].reset_index(drop=True)
    artifact, metrics = fit_artifact(train_frame, schema)

    save_artifact(artifact, args.model_output)

    training_metadata = {
        "model_version": artifact.model_version,
        "data_hash": report["data_hash"],
        "rows": report["rows"],
        "columns": report["columns"],
        "split_indices": str(split_path),
        "decision_threshold": artifact.decision_threshold,
        "metrics": metrics,
        "model_output": str(args.model_output),
    }
    args.training_metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.training_metadata_output.write_text(json.dumps(training_metadata, indent=2, sort_keys=True))

    print(
        json.dumps(
            {
                "status": "ok",
                "report": str(args.report_output),
                "model": str(args.model_output),
                "training_metadata": str(args.training_metadata_output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()