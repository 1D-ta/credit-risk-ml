from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

TRAINING_STATS = Path("artifacts/feature_stats/training_features.json")
INFERENCE_SAMPLE = Path("artifacts/feature_stats/inference_samples.jsonl")
FEATURE_MISMATCH_LOG = Path("artifacts/logs/feature_mismatch.log")

STD_THRESHOLD = 3.0
RELATIVE_MEAN_THRESHOLD = 0.30
MISSING_PCT_THRESHOLD = 0.20


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


def compare_stats(training: Dict[str, dict], inference: Dict[str, dict], mean_tol: float = STD_THRESHOLD, rel_tol: float = RELATIVE_MEAN_THRESHOLD) -> Dict[str, dict]:
    """Compare training vs inference stats. Return dict of failures with magnitude."""
    failures = {}
    for name, train in training.items():
        inf = inference.get(name)
        if not inf:
            failures[name] = {"reason": "missing_feature", "detail": "feature absent in inference batch"}
            continue
        # only compare numeric means/std
        t_mean = train.get("mean")
        t_std = train.get("std")
        i_mean = inf.get("mean")
        if t_mean is None or t_std is None or i_mean is None:
            # compare missing percentage
            t_m = train.get("missing_pct", 0)
            i_m = inf.get("missing_pct", 0)
            if abs(t_m - i_m) > MISSING_PCT_THRESHOLD:
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

    min_samples = 1
    failures = {} if sample_count < min_samples else compare_stats(training, inference)
    # log failures
    if failures:
        sample_messages = []
        for feature, detail in failures.items():
            if detail.get("reason") == "mean_shift":
                train_mean = float(detail.get("train_mean", 0.0))
                inference_mean = float(detail.get("inference_mean", 0.0))
                deviation_pct = (
                    abs(inference_mean - train_mean) / (abs(train_mean) if train_mean else 1.0)
                ) * 100.0
                sample_messages.append(
                    f"FEATURE_MISMATCH: {feature} mean deviation {deviation_pct:.0f}% REQUEST_REJECTED"
                )
            elif detail.get("reason") == "missing_feature":
                sample_messages.append(f"FEATURE_MISMATCH: {feature} missing in inference batch REQUEST_REJECTED")

        msg = {
            "alert": True,
            "thresholds": {
                "std_threshold": STD_THRESHOLD,
                "relative_mean_threshold": RELATIVE_MEAN_THRESHOLD,
                "missing_pct_threshold": MISSING_PCT_THRESHOLD,
            },
            "failures": failures,
            "messages": sample_messages,
        }
        FEATURE_MISMATCH_LOG.parent.mkdir(parents=True, exist_ok=True)
        FEATURE_MISMATCH_LOG.write_text(json.dumps(msg, indent=2))
    return {"failures": failures, "sample_count": sample_count, "min_samples": min_samples}
