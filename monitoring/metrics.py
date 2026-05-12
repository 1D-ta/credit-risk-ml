from __future__ import annotations

from typing import Iterable

import numpy as np

import json
from pathlib import Path
from prometheus_client import Gauge


def summarize_prediction_batch(probabilities: Iterable[float], threshold: float) -> dict[str, float]:
    values = np.asarray(list(probabilities), dtype=float)
    if values.size == 0:
        return {"batch_size": 0, "approval_rate": 0.0, "mean_risk_score": 0.0}

    approvals = values < threshold
    return {
        "batch_size": float(values.size),
        "approval_rate": float(approvals.mean()),
        "mean_risk_score": float(values.mean()),
    }


METRICS_DIR = Path("artifacts/monitoring")
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# simple PSI gauge
PSI_GAUGE = Gauge("model_prediction_psi", "Prediction PSI between reference and current")


def save_alert_example(name: str, details: dict):
    out = METRICS_DIR / "alert_example.json"
    out.write_text(json.dumps({"alert": name, "details": details}, indent=2))


def set_psi(value: float):
    try:
        PSI_GAUGE.set(value)
    except Exception:
        pass
    (METRICS_DIR / "psi_value.txt").write_text(str(value))