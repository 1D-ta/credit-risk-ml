from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException

from credit_risk_ml.data_contract import load_schema
from credit_risk_ml.modeling import load_artifact, predict_bad_probability_from_model

from inference.schema import CreditRiskRequest, load_inference_schema, validate_request


app = FastAPI(title="Credit Risk Scoring API", version="1.0.0")

SCHEMA = load_inference_schema(Path("data/schemas/schema.json"))
ACTIVE_MODEL_POINTER = Path("artifacts/models/active_model.json")
ACTIVE_MODEL = None
PREDICTION_LOG = Path("artifacts/logs/predictions.jsonl")


def load_active_model() -> None:
    global ACTIVE_MODEL
    if not ACTIVE_MODEL_POINTER.exists():
        raise RuntimeError(f"Active model pointer not found at {ACTIVE_MODEL_POINTER}")
    pointer = json.loads(ACTIVE_MODEL_POINTER.read_text())
    active_model_path = pointer.get("active_model")
    if not active_model_path:
        raise RuntimeError("No approved model is active")
    ACTIVE_MODEL = load_artifact(Path(active_model_path))


@app.on_event("startup")
def startup_event() -> None:
    load_active_model()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: CreditRiskRequest) -> dict[str, object]:
    if ACTIVE_MODEL is None:
        raise HTTPException(status_code=503, detail="No approved model is loaded")

    try:
        frame = validate_request(payload, SCHEMA)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    probability = float(predict_bad_probability_from_model(ACTIVE_MODEL, frame).iloc[0])
    decision = "decline" if probability >= ACTIVE_MODEL.decision_threshold else "approve"
    request_hash = hashlib.sha256(json.dumps(payload.model_dump(), sort_keys=True).encode("utf-8")).hexdigest()
    # structured prediction logging (append JSON lines)
    from datetime import datetime

    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event": "predict",
        "model_version": ACTIVE_MODEL.model_version,
        "request_hash": request_hash,
        "prob_bad": probability,
        "decision": decision,
    }
    PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PREDICTION_LOG.open("a", encoding="utf-8") as _handle:
        _handle.write(json.dumps(log_entry, sort_keys=True) + "\n")

    return {
        "risk_score": probability,
        "decision": decision,
        "model_version": ACTIVE_MODEL.model_version,
    }