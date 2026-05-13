"""
Temporal split validator: enforces no leakage, logs feature statistics.

PURPOSE (Interview):
- Prevents accidental data leakage (common root cause of overfitting)
- Validates that train/test splits respect time ordering
- Logs feature distributions to detect drift across splits
- Fails loudly if random split is detected (dates not in order)
"""

from pathlib import Path
from typing import Any

import pandas as pd


def validate_temporal_split_integrity(
    frame: pd.DataFrame,
    train_idx: list[int],
    test_idx: list[int],
    timestamp_column: str = "timestamp",
) -> dict[str, Any]:
    """
    Validate that temporal split has no leakage.
    
    Args:
        frame: DataFrame with timestamp column
        train_idx: Training set indices
        test_idx: Test set indices
        timestamp_column: Name of timestamp column
        
    Returns:
        Validation report dict
        
    Raises:
        RuntimeError if leakage detected or random split inferred
    """
    if timestamp_column not in frame.columns:
        raise RuntimeError(f"Missing timestamp column: {timestamp_column}")
    
    train_df = frame.loc[train_idx]
    test_df = frame.loc[test_idx]
    
    train_ts = pd.to_datetime(train_df[timestamp_column])
    test_ts = pd.to_datetime(test_df[timestamp_column])
    
    train_max = train_ts.max()
    test_min = test_ts.min()
    
    # CRITICAL: Reject if split is not temporal
    if not (train_max < test_min):
        raise RuntimeError(
            f"TEMPORAL_LEAKAGE_DETECTED: train_max={train_max} >= test_min={test_min}. "
            f"Random split detected. Must use strict time ordering."
        )
    
    report = {
        "split_integrity": "passed",
        "train_range": (str(train_ts.min()), str(train_max)),
        "test_range": (str(test_ts.min()), str(test_ts.max())),
        "temporal_order_check": f"train_max {train_max} < test_min {test_min}",
    }
    
    return report


def log_split_statistics(
    frame: pd.DataFrame,
    train_idx: list[int],
    test_idx: list[int],
    schema: Any = None,
) -> None:
    """
    Log feature statistics for training vs test to detect drift.
    
    Args:
        frame: Full dataframe
        train_idx: Training indices
        test_idx: Test indices
        schema: Dataset schema (optional, for feature names)
    """
    train_df = frame.loc[train_idx]
    test_df = frame.loc[test_idx]
    
    print(f"\n{'='*80}")
    print("SPLIT STATISTICS")
    print(f"{'='*80}")
    print(f"Train samples: {len(train_df):,} ({len(train_df)/len(frame)*100:.1f}%)")
    print(f"Test samples: {len(test_df):,} ({len(test_df)/len(frame)*100:.1f}%)")
    
    # Target distribution (if exists)
    if "target" in frame.columns:
        train_target_rate = train_df["target"].mean()
        test_target_rate = test_df["target"].mean()
        print(f"\nTarget Distribution:")
        print(f"  Train: {train_target_rate*100:.2f}% positive ({train_df['target'].sum()} defaults)")
        print(f"  Test: {test_target_rate*100:.2f}% positive ({test_df['target'].sum()} defaults)")
        if abs(train_target_rate - test_target_rate) > 0.05:
            print(f"  ⚠️  WARNING: Target shift > 5% detected")
    
    # Feature statistics (numeric columns only)
    print(f"\nFeature Statistics (mean ± std):")
    print(f"{'':<20} {'TRAIN':<25} {'TEST':<25} {'SHIFT':<10}")
    print(f"{'-'*80}")
    
    numeric_cols = frame.select_dtypes(include=['int64', 'float64']).columns
    numeric_cols = [c for c in numeric_cols if c != "target"]
    
    for col in numeric_cols:
        train_mean = train_df[col].mean()
        train_std = train_df[col].std()
        test_mean = test_df[col].mean()
        test_std = test_df[col].std()
        
        # Jensen-Shannon divergence approximation (for drift detection)
        if train_std > 0 and test_std > 0:
            shift_pct = abs(test_mean - train_mean) / (train_std + 1e-8) * 100
        else:
            shift_pct = 0
        
        print(f"{col:<20} {train_mean:.2f} ± {train_std:.2f}     "
              f"{test_mean:.2f} ± {test_std:.2f}     {shift_pct:+.1f}%")
    
    print(f"{'='*80}\n")


def infer_split_type(frame: pd.DataFrame, timestamp_column: str = "timestamp") -> str:
    """
    Infer whether split is temporal or random.
    
    Returns:
        "temporal" or "random"
    """
    if timestamp_column not in frame.columns:
        return "unknown"
    
    ts = pd.to_datetime(frame[timestamp_column])
    is_sorted = ts.is_monotonic_increasing or ts.is_monotonic_decreasing
    
    return "temporal" if is_sorted else "random"
