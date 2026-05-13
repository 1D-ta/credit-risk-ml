"""
Business metrics computation and reporting for credit risk decisions.

PURPOSE (Interview):
- Translates model predictions to business impact: approval rate, loss estimates
- Enables stakeholder communication: "X% approved, Y% manual review, Z% rejected"
- Justifies decision thresholds: shows sensitivity to different thresholds
- Supports cost-benefit analysis: links model decisions to financial outcomes
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd


def compute_business_metrics(
    predictions: list[dict[str, Any]] | None = None,
    log_file: Path | None = None,
    default_loss_per_default: float = 1000.0,
    default_processing_cost: float = 50.0,
) -> dict[str, Any]:
    """
    Compute business metrics from predictions or log file.
    
    Args:
        predictions: List of prediction dicts with 'prediction' and 'target' fields
        log_file: Path to inference log JSONL file (alternative to predictions)
        default_loss_per_default: Loss in dollars per default (default: $1000)
        default_processing_cost: Processing cost per request (default: $50)
        
    Returns:
        Dict with business metrics
    """
    if log_file and log_file.exists():
        logs = []
        with log_file.open("r") as f:
            for line in f:
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        df = pd.DataFrame(logs)
    elif predictions:
        df = pd.DataFrame(predictions)
    else:
        return {"error": "No data provided"}
    
    if df.empty:
        return {"error": "No predictions found"}
    
    # Decision distribution
    decision_counts = df["decision"].value_counts()
    total = len(df)
    
    approval_count = decision_counts.get("approve", 0)
    review_count = decision_counts.get("review", 0)
    reject_count = decision_counts.get("reject", 0)
    
    approval_rate = approval_count / total if total > 0 else 0
    review_rate = review_count / total if total > 0 else 0
    rejection_rate = reject_count / total if total > 0 else 0
    
    # Prediction distribution
    pred_stats = df["prediction"].describe().to_dict()
    
    # Expected loss estimation (simplified)
    # Assumes: approved accounts have default rate based on prediction
    # Rejected accounts have 0% default rate
    # Reviewed accounts have 50% default rate (manual screening fails 50%)
    
    approved_preds = df[df["decision"] == "approve"]["prediction"]
    reviewed_preds = df[df["decision"] == "review"]["prediction"]
    rejected_preds = df[df["decision"] == "reject"]["prediction"]
    
    # Estimate loss
    approval_expected_loss = (approved_preds.sum() * default_loss_per_default) / total if not approved_preds.empty else 0
    review_expected_loss = (reviewed_preds.mean() * 0.5 * default_loss_per_default * review_count) / total if not reviewed_preds.empty and review_count > 0 else 0
    rejection_expected_loss = 0  # Rejected applicants don't get credit
    
    total_expected_loss = approval_expected_loss + review_expected_loss + rejection_expected_loss
    
    # Revenue calculation (simplified)
    # Assume: interest rate = 15% annual
    approval_revenue = approval_count * default_processing_cost * 1.15
    review_revenue = review_count * default_processing_cost * 0.8  # Lower revenue due to manual cost
    
    net_profit = (approval_revenue + review_revenue) - total_expected_loss
    
    return {
        "total_requests": total,
        "approval_rate": round(approval_rate, 4),
        "review_rate": round(review_rate, 4),
        "rejection_rate": round(rejection_rate, 4),
        "approval_count": int(approval_count),
        "review_count": int(review_count),
        "rejection_count": int(reject_count),
        "prediction_distribution": {
            "mean": round(float(pred_stats.get("mean", 0)), 4),
            "std": round(float(pred_stats.get("std", 0)), 4),
            "min": round(float(pred_stats.get("min", 0)), 4),
            "max": round(float(pred_stats.get("max", 0)), 4),
            "p25": round(float(df["prediction"].quantile(0.25)), 4),
            "p50": round(float(df["prediction"].quantile(0.50)), 4),
            "p75": round(float(df["prediction"].quantile(0.75)), 4),
        },
        "financial_impact": {
            "expected_loss_per_request_usd": round(total_expected_loss / total, 2),
            "approval_expected_loss_usd": round(approval_expected_loss, 2),
            "review_expected_loss_usd": round(review_expected_loss, 2),
            "approval_revenue_usd": round(approval_revenue, 2),
            "review_revenue_usd": round(review_revenue, 2),
            "net_profit_usd": round(net_profit, 2),
            "net_profit_per_request_usd": round(net_profit / total if total > 0 else 0, 2),
        },
        "assumptions": {
            "loss_per_default_usd": default_loss_per_default,
            "processing_cost_usd": default_processing_cost,
            "review_screening_success_rate": 0.5,
            "interest_rate": 0.15,
        },
    }


def generate_business_metrics_report(
    log_file: Path,
    output_file: Path | None = None,
) -> str:
    """
    Generate markdown report of business metrics.
    
    Args:
        log_file: Path to inference logs
        output_file: Optional output markdown file
        
    Returns:
        Markdown string
    """
    metrics = compute_business_metrics(log_file=log_file)
    
    if "error" in metrics:
        return f"# Business Metrics Report\n\n{metrics['error']}\n"
    
    md = f"""# Business Metrics Report

**Generated:** {pd.Timestamp.now().isoformat()}

## Decision Distribution

| Decision | Count | Rate |
|----------|-------|------|
| Approve | {metrics['approval_count']} | {metrics['approval_rate']*100:.2f}% |
| Review | {metrics['review_count']} | {metrics['review_rate']*100:.2f}% |
| Reject | {metrics['rejection_count']} | {metrics['rejection_rate']*100:.2f}% |
| **Total** | **{metrics['total_requests']}** | **100.00%** |

## Risk Score Distribution

- **Mean:** {metrics['prediction_distribution']['mean']}
- **Std Dev:** {metrics['prediction_distribution']['std']}
- **Min:** {metrics['prediction_distribution']['min']}
- **Max:** {metrics['prediction_distribution']['max']}
- **Median (p50):** {metrics['prediction_distribution']['p50']}
- **p95:** {metrics['prediction_distribution']['p75']}

## Financial Impact (Estimated)

### Loss Calculation

- **Expected loss per approved request:** ${metrics['financial_impact']['approval_expected_loss_usd'] / max(metrics['approval_count'], 1):.2f}
- **Total expected loss (all approvals):** ${metrics['financial_impact']['approval_expected_loss_usd']:,.2f}
- **Total expected loss (manual review):** ${metrics['financial_impact']['review_expected_loss_usd']:,.2f}
- **Total expected loss (all decisions):** ${metrics['financial_impact']['approval_expected_loss_usd'] + metrics['financial_impact']['review_expected_loss_usd']:,.2f}

### Revenue & Profit

- **Approval revenue:** ${metrics['financial_impact']['approval_revenue_usd']:,.2f}
- **Review revenue:** ${metrics['financial_impact']['review_revenue_usd']:,.2f}
- **Net profit:** ${metrics['financial_impact']['net_profit_usd']:,.2f}
- **Profit per request:** ${metrics['financial_impact']['net_profit_per_request_usd']:.2f}

## Model Assumptions

- **Loss per default:** ${metrics['assumptions']['loss_per_default_usd']}
- **Processing cost:** ${metrics['assumptions']['processing_cost_usd']}
- **Manual review success rate:** {metrics['assumptions']['review_screening_success_rate']*100:.0f}%
- **Interest rate:** {metrics['assumptions']['interest_rate']*100:.0f}%

## Decision Thresholds

- **Approve:** Risk score < 0.30
- **Review:** 0.30 ≤ Risk score ≤ 0.70
- **Reject:** Risk score > 0.70

**Interpretation:** These thresholds balance false positives (approving bad loans) and false negatives (rejecting good customers).
"""
    
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(md)
    
    return md
