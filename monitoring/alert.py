"""Alert system for model health monitoring."""
import json
from pathlib import Path

def trigger_alert(metric: str, value: float, threshold: float):
    """Trigger alert when metric breaches threshold."""
    print(f"ALERT: {metric} breach - value={value:.4f}, threshold={threshold}")
    mark_model_unhealthy(metric, value)
    
    # Import here to avoid circular dependency
    from governance.rollback import trigger_rollback
    trigger_rollback()

def mark_model_unhealthy(metric: str, value: float):
    """Mark current model as unhealthy."""
    print(f"MODEL marked unhealthy due to {metric}={value:.4f}")
    
    # Write health status to file for tracking
    health_file = Path("artifacts/monitoring/model_health.json")
    health_file.parent.mkdir(parents=True, exist_ok=True)
    
    health_status = {
        "status": "unhealthy",
        "reason": f"{metric} breach",
        "metric": metric,
        "value": value
    }
    
    health_file.write_text(json.dumps(health_status, indent=2))
