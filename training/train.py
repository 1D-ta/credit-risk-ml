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
    parser.add_argument("--split-log-output", type=Path, default=Path("artifacts/logs/temporal_split.log"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schema = load_schema(args.schema)
    rows = validate_and_load_rows(args.raw_data, schema)
    report = dataset_summary(rows, schema, args.raw_data)
    frame = rows_to_frame(rows, schema)

    # Temporal split: train on t < T1, validate on T1 <= t < T2, test on t >= T2
    import pandas as _pd

    if "event_time" not in frame.columns:
        raise RuntimeError("event_time column missing from data; run scripts/generate_temporal_data.py first")

    # ensure event_time parsed as date
    frame["event_time"] = _pd.to_datetime(frame["event_time"], errors="coerce").dt.date
    if frame["event_time"].isna().any():
        raise RuntimeError("Some event_time values could not be parsed as dates")

    unique_dates = sorted(frame["event_time"].unique())
    if len(unique_dates) < 3:
        raise RuntimeError("Not enough distinct dates for temporal split")

    # choose T1 as ~60th percentile and T2 as ~80th percentile
    # train: t < T1, val: T1 <= t < T2, test: t >= T2
    t1_pos = int(len(unique_dates) * 0.6)
    t2_pos = int(len(unique_dates) * 0.8)
    t1_pos = max(1, min(t1_pos, len(unique_dates) - 2))
    t2_pos = max(t1_pos + 1, min(t2_pos, len(unique_dates) - 1))
    T1 = unique_dates[t1_pos]
    T2 = unique_dates[t2_pos]

    train_idx = frame.index[frame["event_time"] < T1].tolist()
    calibration_idx = frame.index[(frame["event_time"] >= T1) & (frame["event_time"] < T2)].tolist()
    test_idx = frame.index[frame["event_time"] >= T2].tolist()

    if not train_idx or not calibration_idx or not test_idx:
        raise RuntimeError("Temporal split produced an empty train/val/test partition")

    train_dates = frame.loc[train_idx, "event_time"]
    val_dates = frame.loc[calibration_idx, "event_time"]
    test_dates = frame.loc[test_idx, "event_time"]
    
    train_max = train_dates.max()
    val_min = val_dates.min()
    val_max = val_dates.max()
    test_min = test_dates.min()
    
    if not (train_max < val_min and val_max < test_min):
        raise RuntimeError("Temporal leakage detected: split ordering is invalid")

    # Explicit logging after split creation
    print(f"TRAIN_RANGE: {train_dates.min()} to {train_dates.max()}")
    print(f"VAL_RANGE: {val_dates.min()} to {val_dates.max()}")
    print(f"TEST_RANGE: {test_dates.min()} to {test_dates.max()}")

    split_indices = {
        "train_idx": train_idx,
        "calibration_idx": calibration_idx,
        "test_idx": test_idx,
        "temporal_split_T1": str(T1),
        "temporal_split_T2": str(T2),
        "event_time_ranges": {
            "train": {
                "start": str(train_dates.min()),
                "end": str(train_max),
                "rows": len(train_idx),
            },
            "validation": {
                "start": str(val_min),
                "end": str(val_max),
                "rows": len(calibration_idx),
            },
            "test": {
                "start": str(test_min),
                "end": str(test_dates.max()),
                "rows": len(test_idx),
            },
        },
        "no_leakage_check": "passed",
    }

    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True))

    # save split indices for downstream steps
    split_path = Path("artifacts/reports/split_indices.json")
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(json.dumps(split_indices, indent=2, sort_keys=True))

    args.split_log_output.parent.mkdir(parents=True, exist_ok=True)
    log_lines = [
        f"TEMPORAL_SPLIT: train t < T1 range={split_indices['event_time_ranges']['train']['start']}..{split_indices['event_time_ranges']['train']['end']} rows={split_indices['event_time_ranges']['train']['rows']}",
        f"TEMPORAL_SPLIT: val T1 <= t < T2 range={split_indices['event_time_ranges']['validation']['start']}..{split_indices['event_time_ranges']['validation']['end']} rows={split_indices['event_time_ranges']['validation']['rows']}",
        f"TEMPORAL_SPLIT: test t >= T2 range={split_indices['event_time_ranges']['test']['start']}..{split_indices['event_time_ranges']['test']['end']} rows={split_indices['event_time_ranges']['test']['rows']}",
        "TEMPORAL_SPLIT: NO_LEAKAGE_CHECK=passed",
    ]
    args.split_log_output.write_text("\n".join(log_lines) + "\n")
    for line in log_lines:
        print(line)

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