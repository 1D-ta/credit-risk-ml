from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


@dataclass(frozen=True)
class LabelNoiseMetadata:
    noise_fraction_requested: float
    noise_fraction_actual: float
    total_labels: int
    num_flipped: int
    num_0_to_1_flips: int
    num_1_to_0_flips: int
    noise_type: str
    seed: int


def inject_label_noise(
    labels: np.ndarray | pd.Series,
    noise_fraction: float = 0.05,
    noise_type: str = "flip",
    seed: int = 42,
) -> tuple[np.ndarray, dict[str, Any]]:
    np.random.seed(seed)
    labels_array = np.asarray(labels).astype(int)
    noisy_labels = labels_array.copy()

    n_to_flip = int(len(labels_array) * noise_fraction)
    flip_indices = np.random.choice(len(labels_array), size=n_to_flip, replace=False)

    if noise_type == "flip":
        noisy_labels[flip_indices] = 1 - noisy_labels[flip_indices]
    elif noise_type == "majority":
        majority_class = int(np.bincount(labels_array).argmax())
        noisy_labels[flip_indices] = majority_class
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    num_changes = int((labels_array != noisy_labels).sum())
    metadata = {
        "noise_fraction_requested": noise_fraction,
        "noise_fraction_actual": num_changes / len(labels_array),
        "total_labels": int(len(labels_array)),
        "num_flipped": num_changes,
        "num_0_to_1_flips": int(((labels_array == 0) & (noisy_labels == 1)).sum()),
        "num_1_to_0_flips": int(((labels_array == 1) & (noisy_labels == 0)).sum()),
        "noise_type": noise_type,
        "seed": seed,
    }
    return noisy_labels, metadata


def inject_delayed_labels(
    labels: pd.Series,
    delay_days: int = 30,
    missing_fraction: float = 0.1,
) -> tuple[pd.Series, dict[str, Any]]:
    labels_with_delay = labels.copy()
    n_missing = int(len(labels) * missing_fraction)
    missing_indices = np.random.choice(len(labels), size=n_missing, replace=False)
    labels_with_delay.iloc[missing_indices] = np.nan

    metadata = {
        "total_labels": len(labels),
        "missing_labels": int(n_missing),
        "available_labels": int(len(labels) - n_missing),
        "missing_fraction": missing_fraction,
        "simulated_delay_days": delay_days,
    }
    return labels_with_delay, metadata


def compare_model_performance(
    y_true: np.ndarray,
    y_pred_clean: np.ndarray,
    y_pred_noisy: np.ndarray,
) -> dict[str, Any]:
    metrics_clean = {
        "accuracy": accuracy_score(y_true, (y_pred_clean > 0.5).astype(int)),
        "auc": roc_auc_score(y_true, y_pred_clean),
        "precision": precision_score(y_true, (y_pred_clean > 0.5).astype(int), zero_division=0),
        "recall": recall_score(y_true, (y_pred_clean > 0.5).astype(int), zero_division=0),
    }
    metrics_noisy = {
        "accuracy": accuracy_score(y_true, (y_pred_noisy > 0.5).astype(int)),
        "auc": roc_auc_score(y_true, y_pred_noisy),
        "precision": precision_score(y_true, (y_pred_noisy > 0.5).astype(int), zero_division=0),
        "recall": recall_score(y_true, (y_pred_noisy > 0.5).astype(int), zero_division=0),
    }

    return {
        "clean_metrics": {k: round(float(v), 4) for k, v in metrics_clean.items()},
        "noisy_metrics": {k: round(float(v), 4) for k, v in metrics_noisy.items()},
        "performance_degradation": {
            f"delta_{metric}": round(float(metrics_clean[metric] - metrics_noisy[metric]), 4)
            for metric in metrics_clean
        },
    }
