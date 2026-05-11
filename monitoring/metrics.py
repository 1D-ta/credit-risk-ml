from __future__ import annotations

from typing import Iterable

import numpy as np


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