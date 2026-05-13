"""
Structured JSON logging for inference: enables post-hoc debugging and auditability.

PURPOSE (Interview):
- Deterministic request hashing: identify duplicate requests
- Audit trail: every prediction is logged with timestamp and model version
- No raw sensitive data: hash input instead of storing PII
- Enables label-lag handling: join with true labels later for retraining
- Supports data drift analysis: compare inference features to training
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd


@dataclass(frozen=True)
class InferencePredictionLog:
    """Structured log entry for a single prediction."""
    timestamp: str
    request_id: str
    input_hash: str  # deterministic SHA256 of input dict
    model_version: str
    prediction: float
    decision: str
    client_id: Optional[str] = None
    features_count: int = 0
    latency_ms: float = 0.0
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def compute_input_hash(payload_dict: dict[str, Any]) -> str:
    """
    Compute deterministic hash of input payload.
    
    Args:
        payload_dict: Input features dict
        
    Returns:
        SHA256 hex string
        
    NOTE: Same input always produces same hash (deterministic).
    This enables:
    - Deduplication detection
    - Request replay detection
    - Data contamination auditing
    """
    # Sort keys to ensure determinism
    serialized = json.dumps(payload_dict, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def create_inference_log_entry(
    request_id: str,
    payload_dict: dict[str, Any],
    prediction: float,
    decision: str,
    model_version: str,
    client_id: Optional[str] = None,
    latency_ms: float = 0.0,
) -> InferencePredictionLog:
    """
    Create structured log entry for an inference request.
    
    Args:
        request_id: Unique request ID
        payload_dict: Input features dict
        prediction: Model prediction (probability)
        decision: Derived business decision (approve/review/decline)
        model_version: Version of model used
        client_id: Optional client identifier
        latency_ms: Request latency in milliseconds
        
    Returns:
        InferencePredictionLog entry (not yet written)
    """
    return InferencePredictionLog(
        timestamp=datetime.utcnow().isoformat() + "Z",
        request_id=request_id,
        input_hash=compute_input_hash(payload_dict),
        model_version=model_version,
        prediction=round(prediction, 6),  # 6 decimals for probability
        decision=decision,
        client_id=client_id,
        features_count=len(payload_dict),
        latency_ms=round(latency_ms, 2),
    )


def write_log_entry(log_entry: InferencePredictionLog, log_file: Path) -> None:
    """
    Append log entry to JSONL file (thread-safe append).
    
    Args:
        log_entry: Entry to log
        log_file: Path to JSONL log file
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        f.write(log_entry.to_json() + "\n")


def read_inference_logs(log_file: Path, limit: Optional[int] = None) -> list[dict[str, Any]]:
    """
    Read inference logs back from JSONL file.
    
    Args:
        log_file: Path to JSONL log file
        limit: Max records to read (None = all)
        
    Returns:
        List of log dicts
    """
    if not log_file.exists():
        return []
    
    logs = []
    with log_file.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                logs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    
    return logs


def analyze_inference_logs(log_file: Path) -> dict[str, Any]:
    """
    Analyze inference logs for observability.
    
    Returns:
        Dict with statistics: prediction distribution, decision rates, error counts
    """
    logs = read_inference_logs(log_file)
    
    if not logs:
        return {"total_requests": 0}
    
    df = pd.DataFrame(logs)
    
    return {
        "total_requests": len(df),
        "unique_inputs": df["input_hash"].nunique(),
        "model_versions": df["model_version"].unique().tolist(),
        "decision_distribution": df["decision"].value_counts().to_dict(),
        "prediction_stats": {
            "mean": float(df["prediction"].mean()),
            "std": float(df["prediction"].std()),
            "min": float(df["prediction"].min()),
            "max": float(df["prediction"].max()),
            "p50": float(df["prediction"].quantile(0.5)),
            "p95": float(df["prediction"].quantile(0.95)),
            "p99": float(df["prediction"].quantile(0.99)),
        },
        "latency_stats": {
            "mean_ms": float(df["latency_ms"].mean()),
            "p50_ms": float(df["latency_ms"].quantile(0.5)),
            "p95_ms": float(df["latency_ms"].quantile(0.95)),
            "p99_ms": float(df["latency_ms"].quantile(0.99)),
        },
    }
