from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

TRAINING_STATS = Path("artifacts/feature_stats/training_features.json")
INFERENCE_SAMPLE = Path("artifacts/feature_stats/inference_samples.jsonl")


def load_training_stats() -> Dict[str, dict]:
    if not TRAINING_STATS.exists():
        raise RuntimeError("Training feature stats not found")
    return json.loads(TRAINING_STATS.read_text())


def compute_inference_stats_from_samples() -> tuple[Dict[str, dict], int]:
    # read recent inference samples (one JSON object per line)
    if not INFERENCE_SAMPLE.exists():
        return {}, 0
    rows = [json.loads(l) for l in INFERENCE_SAMPLE.read_text().splitlines() if l.strip()]
    if not rows:
        return {}, 0
    df = pd.DataFrame(rows)
    stats = {}
    for col in df.columns:
        ser = df[col]
        stats[col] = {
            "name": col,
            "mean": None if not pd.api.types.is_numeric_dtype(ser) else float(ser.dropna().astype(float).mean()),
            "std": None if not pd.api.types.is_numeric_dtype(ser) else float(ser.dropna().astype(float).std()),
            "missing_pct": float(ser.isna().mean()),
            "min": None if ser.dropna().empty else (float(ser.dropna().min()) if pd.api.types.is_numeric_dtype(ser) else str(ser.dropna().min())),
            "max": None if ser.dropna().empty else (float(ser.dropna().max()) if pd.api.types.is_numeric_dtype(ser) else str(ser.dropna().max())),
        }
    return stats, len(rows)


def compare_stats(training: Dict[str, dict], inference: Dict[str, dict], mean_tol: float = 3.0, rel_tol: float = 0.3) -> Dict[str, dict]:
    """Compare training vs inference stats. Return dict of failures with magnitude."""
    failures = {}
    for name, train in training.items():
        inf = inference.get(name)
        if not inf:
            continue
        # only compare numeric means/std
        t_mean = train.get("mean")
        t_std = train.get("std")
        i_mean = inf.get("mean")
        if t_mean is None or t_std is None or i_mean is None:
            # compare missing percentage
            t_m = train.get("missing_pct", 0)
            i_m = inf.get("missing_pct", 0)
            if abs(t_m - i_m) > 0.2:
                failures[name] = {"reason": "missing_pct_change", "train": t_m, "inference": i_m}
            continue

        # absolute deviation in units of train std
        if t_std == 0 or t_std is None:
            dev = abs(i_mean - t_mean)
        else:
            dev = abs(i_mean - t_mean) / (t_std if t_std else 1.0)
        rel = abs(i_mean - t_mean) / (abs(t_mean) if t_mean else 1.0)
        if dev > mean_tol or rel > rel_tol:
            failures[name] = {"reason": "mean_shift", "train_mean": t_mean, "inference_mean": i_mean, "stds": dev, "rel": rel}
    return failures


def run_feature_check_and_maybe_fail(thresholds=None) -> dict:
    training = load_training_stats()
    inference, sample_count = compute_inference_stats_from_samples()
    # write inference aggregate for auditing
    out_inf = Path("artifacts/feature_stats/inference_features.json")
    out_inf.parent.mkdir(parents=True, exist_ok=True)
    out_inf.write_text(json.dumps(inference, indent=2, sort_keys=True))

    min_samples = 20
    failures = {} if sample_count < min_samples else compare_stats(training, inference)
    # log failures
    if failures:
        msg = {"alert": True, "failures": failures}
        Path("artifacts/logs/feature_mismatch.log").write_text(json.dumps(msg, indent=2))
    return {"failures": failures, "sample_count": sample_count, "min_samples": min_samples}
