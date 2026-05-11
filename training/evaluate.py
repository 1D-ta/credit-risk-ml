from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from credit_risk_ml.data_contract import load_schema, rows_to_frame, validate_and_load_rows
from credit_risk_ml.modeling import evaluate_predictions, load_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the credit risk model.")
    parser.add_argument("--raw-data", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--model-artifact", required=True, type=Path)
    parser.add_argument("--metrics-output", type=Path, default=Path("artifacts/reports/metrics.json"))
    parser.add_argument("--split-indices", type=Path, default=Path("artifacts/reports/split_indices.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schema = load_schema(args.schema)
    rows = validate_and_load_rows(args.raw_data, schema)
    frame = rows_to_frame(rows, schema)

    if not args.split_indices.exists():
        raise RuntimeError(f"Split indices not found at {args.split_indices}; run training first")
    split_indices = json.loads(args.split_indices.read_text())
    test_idx = split_indices.get("test_idx", [])
    test_frame = frame.loc[test_idx].reset_index(drop=True)

    artifact = load_artifact(args.model_artifact)
    metrics = evaluate_predictions(artifact, test_frame, schema)

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps({"status": "ok", "metrics": str(args.metrics_output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()