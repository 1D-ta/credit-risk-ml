"""
Time-aware synthetic credit risk dataset generator with drift injection.

PURPOSE (Interview):
- Enables reproducible evaluation of drift detection and temporal integrity
- Simulates real-world distribution shift: early performance, later degradation
- Validates that train/test split respects time boundaries (no leakage)
- Tests business decision calibration across regime changes

DRIFT MECHANISM:
After 70% of timeline, feature distributions shift:
- Income increases (→ lower risk, model underestimates)
- Loan size increases (→ higher risk, misses signals)
This creates gap between training and serving distributions.
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


def generate_credit_risk_dataset(
    n_rows: int = 750_000,
    drift_point_pct: float = 0.70,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate temporal credit risk dataset with drift injection.
    
    Args:
        n_rows: Total rows to generate
        drift_point_pct: When to inject drift (0-1, default 70% of timeline)
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns: timestamp, features..., target
    """
    np.random.seed(seed)
    
    # Timeline: 24 months
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=int(i / n_rows * 730)) for i in range(n_rows)]
    drift_idx = int(n_rows * drift_point_pct)
    
    # Pre-drift features (training regime)
    income_pre = np.random.lognormal(mean=10.5, sigma=0.6, size=drift_idx)  # mean ~$32k
    loan_size_pre = np.random.lognormal(mean=9.8, sigma=0.7, size=drift_idx)  # mean ~$18k
    age_pre = np.random.normal(loc=40, scale=12, size=drift_idx)
    age_pre = np.clip(age_pre, 18, 75)
    existing_credit_pre = np.random.exponential(scale=2, size=drift_idx)
    existing_credit_pre = np.clip(existing_credit_pre, 0, 50)
    employment_months_pre = np.random.exponential(scale=48, size=drift_idx)
    employment_months_pre = np.clip(employment_months_pre, 1, 600)
    
    # Post-drift features (serving regime: economic shift)
    # Income increases (inflation, economy stronger), loan sizes increase
    income_post = np.random.lognormal(mean=10.8, sigma=0.6, size=n_rows - drift_idx)
    income_post = income_post * 1.15  # 15% increase
    loan_size_post = np.random.lognormal(mean=10.1, sigma=0.7, size=n_rows - drift_idx)
    loan_size_post = loan_size_post * 1.20  # 20% increase
    age_post = np.random.normal(loc=41, scale=11, size=n_rows - drift_idx)
    age_post = np.clip(age_post, 18, 75)
    existing_credit_post = np.random.exponential(scale=2, size=n_rows - drift_idx)
    existing_credit_post = np.clip(existing_credit_post, 0, 50)
    employment_months_post = np.random.exponential(scale=50, size=n_rows - drift_idx)
    employment_months_post = np.clip(employment_months_post, 1, 600)
    
    # Concatenate
    income = np.concatenate([income_pre, income_post])
    loan_size = np.concatenate([loan_size_pre, loan_size_post])
    age = np.concatenate([age_pre, age_post])
    existing_credit = np.concatenate([existing_credit_pre, existing_credit_post])
    employment_months = np.concatenate([employment_months_pre, employment_months_post])
    
    # Target: correlated with income (negative) and loan size (positive)
    # Base probability modulated by features
    base_prob_bad = 0.035  # ~3.5% positive class
    log_odds = np.log(base_prob_bad / (1 - base_prob_bad))
    log_odds = log_odds - 0.0008 * (income - income.mean()) / income.std()
    log_odds = log_odds + 0.0012 * (loan_size - loan_size.mean()) / loan_size.std()
    log_odds = log_odds - 0.002 * (age - age.mean()) / age.std()
    prob_bad = 1 / (1 + np.exp(-log_odds))
    target = (np.random.uniform(size=n_rows) < prob_bad).astype(int)
    
    # Build dataframe
    df = pd.DataFrame({
        "timestamp": dates,
        "income": np.round(income, 0).astype(int),
        "loan_size": np.round(loan_size, 0).astype(int),
        "age": np.round(age, 0).astype(int),
        "existing_credit": np.round(existing_credit, 1),
        "employment_months": np.round(employment_months, 0).astype(int),
        "target": target,
    })
    
    return df, drift_idx


def compute_distribution_stats(df: pd.DataFrame, label: str) -> None:
    """Log distribution statistics for interview-relevant analysis."""
    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"{'='*70}")
    print(f"Samples: {len(df):,}")
    print(f"Target (1=default): {df['target'].sum():,} ({df['target'].mean()*100:.2f}%)")
    for col in ["income", "loan_size", "age", "existing_credit", "employment_months"]:
        print(f"{col:20s} | mean={df[col].mean():10.1f} std={df[col].std():8.1f} "
              f"p25={df[col].quantile(0.25):10.1f} p75={df[col].quantile(0.75):10.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate time-aware synthetic credit risk dataset with drift injection."
    )
    parser.add_argument("--output", type=Path, default=Path("data/credit_risk_time.csv"),
                        help="Output CSV path")
    parser.add_argument("--rows", type=int, default=750_000,
                        help="Number of rows to generate (default 750K)")
    parser.add_argument("--drift-pct", type=float, default=0.70,
                        help="Drift injection point (0-1, default 0.70)")
    args = parser.parse_args()
    
    print(f"Generating {args.rows:,} rows of synthetic credit risk data...")
    df, drift_idx = generate_credit_risk_dataset(n_rows=args.rows, drift_point_pct=args.drift_pct)
    
    # Split and log
    train_cutoff = int(len(df) * 0.70)
    train_df = df[df.index < train_cutoff]
    test_df = df[df.index >= train_cutoff]
    
    compute_distribution_stats(train_df, "TRAINING SET")
    compute_distribution_stats(test_df, "TEST SET")
    
    # Drift impact
    drift_start_idx = drift_idx
    pre_drift_df = df[df.index < drift_start_idx]
    post_drift_df = df[df.index >= drift_start_idx]
    print(f"\n{'='*70}")
    print("DRIFT INJECTION IMPACT (timeline 70% mark)")
    print(f"{'='*70}")
    print(f"Pre-drift samples: {len(pre_drift_df):,}")
    print(f"Post-drift samples: {len(post_drift_df):,}")
    print(f"Income shift: {pre_drift_df['income'].mean():.0f} → {post_drift_df['income'].mean():.0f} "
          f"({(post_drift_df['income'].mean() / pre_drift_df['income'].mean() - 1)*100:+.1f}%)")
    print(f"Loan size shift: {pre_drift_df['loan_size'].mean():.0f} → {post_drift_df['loan_size'].mean():.0f} "
          f"({(post_drift_df['loan_size'].mean() / pre_drift_df['loan_size'].mean() - 1)*100:+.1f}%)")
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\n✓ Saved to {args.output}")


if __name__ == "__main__":
    main()
