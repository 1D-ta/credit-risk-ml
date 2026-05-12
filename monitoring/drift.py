from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from credit_risk_ml.data_contract import DatasetSchema, load_schema, rows_to_frame, validate_and_load_rows
from governance.rollback import rollback_to_latest_approved
from monitoring.metrics import save_alert_example, set_psi


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


def build_temporal_psi_trend(
    reference_frame: pd.DataFrame,
    current_frame: pd.DataFrame,
    schema: DatasetSchema,
    time_column: str,
) -> list[dict[str, object]]:
    if time_column not in current_frame.columns:
        return []

    current = current_frame.copy()
    current[time_column] = pd.to_datetime(current[time_column], errors="coerce").dt.date
    current = current.dropna(subset=[time_column])
    if current.empty:
        return []

    features = [field for field in schema.fields if field.name not in {schema.target_column, time_column}]
    trend: list[dict[str, object]] = []
    for day in sorted(current[time_column].unique()):
        day_frame = current[current[time_column] == day]
        if day_frame.empty:
            continue

        day_feature_psi: dict[str, float] = {}
        for field in features:
            day_feature_psi[field.name] = population_stability_index(reference_frame[field.name], day_frame[field.name])
        max_feature = max(day_feature_psi, key=day_feature_psi.get)
        trend.append(
            {
                "event_time": str(day),
                "rows": int(len(day_frame)),
                "max_psi": float(day_feature_psi[max_feature]),
                "max_psi_feature": max_feature,
            }
        )
    return trend


def append_monitoring_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().isoformat() + "Z"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{stamp} {message}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute PSI/KS drift between two raw German Credit datasets.")
    parser.add_argument("--reference-data", required=True, type=Path)
    parser.add_argument("--current-data", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/reports/drift_report.json"))
    parser.add_argument("--reference-predictions", type=Path, default=None)
    parser.add_argument("--current-predictions", type=Path, default=None)
    parser.add_argument("--psi-threshold", type=float, default=0.2)
    parser.add_argument("--time-column", type=str, default="event_time")
    parser.add_argument("--auto-rollback", action="store_true")
    parser.add_argument("--registry", type=Path, default=Path("artifacts/reports/model_registry.json"))
    parser.add_argument("--active-pointer", type=Path, default=Path("artifacts/models/active_model.json"))
    parser.add_argument("--monitoring-log", type=Path, default=Path("artifacts/logs/monitoring_alerts.log"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    schema = load_schema(args.schema)
    reference_rows = validate_and_load_rows(args.reference_data, schema)
    current_rows = validate_and_load_rows(args.current_data, schema)
    reference_frame = rows_to_frame(reference_rows, schema)
    current_frame = rows_to_frame(current_rows, schema)
    report = build_drift_report(reference_frame, current_frame, schema)
    report["psi_threshold"] = args.psi_threshold

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

    trend = build_temporal_psi_trend(reference_frame, current_frame, schema, args.time_column)
    report["temporal_psi_trend"] = trend

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))

    # Export a single PSI value for Prometheus/Grafana panels.
    psi_for_metrics = float(prediction_psi) if prediction_psi is not None else float(
        max((item["psi"] for item in report["feature_reports"].values()), default=0.0)
    )
    set_psi(psi_for_metrics)

    print(f"METRIC: psi={psi_for_metrics:.4f} threshold={args.psi_threshold:.4f}")
    append_monitoring_log(args.monitoring_log, f"METRIC: psi={psi_for_metrics:.4f} threshold={args.psi_threshold:.4f}")

    if trend:
        for item in trend[: min(10, len(trend))]:
            msg = (
                "TEMPORAL_PSI: "
                f"event_time={item['event_time']} max_psi={item['max_psi']:.4f} "
                f"feature={item['max_psi_feature']} rows={item['rows']}"
            )
            print(msg)
            append_monitoring_log(args.monitoring_log, msg)

    breached = psi_for_metrics > args.psi_threshold
    if report["alert"] or (prediction_psi is not None and prediction_psi > args.psi_threshold) or breached:
        breach_msg = f"BREACH: psi={psi_for_metrics:.4f} > threshold={args.psi_threshold:.4f}"
        print(breach_msg)
        append_monitoring_log(args.monitoring_log, breach_msg)

        alert_msg = "ALERT: drift threshold breached"
        print(alert_msg)
        append_monitoring_log(args.monitoring_log, alert_msg)

        save_alert_example(
            "high_psi",
            {
                "psi": psi_for_metrics,
                "prediction_psi": prediction_psi,
                "window": "latest_batch",
            },
        )

        if args.auto_rollback:
            result = rollback_to_latest_approved(args.registry, args.active_pointer)
            action_msg = f"ACTION: rollback_executed active_model={result['active_model']}"
            print(action_msg)
            append_monitoring_log(args.monitoring_log, action_msg)
            report["rollback_action"] = result

    if report["alert"] or (prediction_psi is not None and prediction_psi > args.psi_threshold) or breached:
        print("DRIFT DETECTED")
    print(json.dumps({"status": "ok", "report": str(args.output)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()