from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from credit_risk_ml.data_contract import DatasetSchema, load_schema, rows_to_frame, validate_and_load_rows


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    reference = reference.dropna()
    current = current.dropna()
    if reference.empty or current.empty:
        return 0.0

    if pd.api.types.is_numeric_dtype(reference):
        unique_values = reference.nunique(dropna=True)
        if unique_values <= 1:
            return 0.0
        quantile_count = min(bins, unique_values)
        edges = np.unique(np.quantile(reference.to_numpy(), np.linspace(0.0, 1.0, quantile_count + 1)))
        if len(edges) <= 2:
            edges = np.linspace(float(reference.min()), float(reference.max()), 2)
        reference_buckets = pd.cut(reference, bins=edges, include_lowest=True, duplicates="drop")
        current_buckets = pd.cut(current, bins=edges, include_lowest=True, duplicates="drop")
        categories = reference_buckets.cat.categories
        reference_distribution = reference_buckets.value_counts(normalize=True).reindex(categories, fill_value=1e-6)
        current_distribution = current_buckets.value_counts(normalize=True).reindex(categories, fill_value=1e-6)
    else:
        categories = sorted(set(reference.astype(str).unique()) | set(current.astype(str).unique()))
        reference_distribution = reference.astype(str).value_counts(normalize=True).reindex(categories, fill_value=1e-6)
        current_distribution = current.astype(str).value_counts(normalize=True).reindex(categories, fill_value=1e-6)

    reference_distribution = reference_distribution.replace(0, 1e-6)
    current_distribution = current_distribution.replace(0, 1e-6)
    psi_values = (current_distribution - reference_distribution) * np.log(current_distribution / reference_distribution)
    return float(psi_values.sum())


def kolmogorov_smirnov_statistic(reference: pd.Series, current: pd.Series) -> float:
    reference = pd.to_numeric(reference.dropna(), errors="coerce").dropna().to_numpy()
    current = pd.to_numeric(current.dropna(), errors="coerce").dropna().to_numpy()
    if reference.size == 0 or current.size == 0:
        return 0.0

    points = np.unique(np.concatenate([reference, current]))
    reference_cdf = np.searchsorted(np.sort(reference), points, side="right") / reference.size
    current_cdf = np.searchsorted(np.sort(current), points, side="right") / current.size
    return float(np.max(np.abs(reference_cdf - current_cdf)))


def build_drift_report(reference_frame: pd.DataFrame, current_frame: pd.DataFrame, schema: DatasetSchema) -> dict[str, object]:
    features = [field for field in schema.fields if field.name != schema.target_column]
    feature_reports: dict[str, dict[str, float | None]] = {}
    alert = False

    for field in features:
        reference_series = reference_frame[field.name]
        current_series = current_frame[field.name]
        psi_value = population_stability_index(reference_series, current_series)
        ks_value = None
        if field.field_type == "integer":
            ks_value = kolmogorov_smirnov_statistic(reference_series, current_series)
        feature_reports[field.name] = {"psi": psi_value, "ks": ks_value}
        if psi_value > 0.2 or (ks_value is not None and ks_value > 0.1):
            alert = True

    return {
        "alert": alert,
        "feature_reports": feature_reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute PSI/KS drift between two raw German Credit datasets.")
    parser.add_argument("--reference-data", required=True, type=Path)
    parser.add_argument("--current-data", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/reports/drift_report.json"))
    parser.add_argument("--reference-predictions", type=Path, default=None)
    parser.add_argument("--current-predictions", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schema = load_schema(args.schema)
    reference_rows = validate_and_load_rows(args.reference_data, schema)
    current_rows = validate_and_load_rows(args.current_data, schema)
    reference_frame = rows_to_frame(reference_rows, schema)
    current_frame = rows_to_frame(current_rows, schema)
    report = build_drift_report(reference_frame, current_frame, schema)

    # optional prediction-distribution comparison
    prediction_psi = None
    if args.reference_predictions and args.reference_predictions.exists() and args.current_predictions and args.current_predictions.exists():
        def _load_preds(path: Path):
            text = path.read_text()
            try:
                data = json.loads(text)
                if isinstance(data, list):
                    return pd.Series(data)
            except Exception:
                pass
            # fallback: one float per line
            return pd.Series([float(line.strip()) for line in text.splitlines() if line.strip()])

        ref_preds = _load_preds(args.reference_predictions)
        cur_preds = _load_preds(args.current_predictions)
        prediction_psi = population_stability_index(ref_preds, cur_preds)
        report["prediction_psi"] = prediction_psi

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))

    if report["alert"] or (prediction_psi is not None and prediction_psi > 0.2):
        print("DRIFT DETECTED")
    print(json.dumps({"status": "ok", "report": str(args.output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()