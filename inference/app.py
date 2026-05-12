from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
import asyncio
import time
import uuid
import logging

from prometheus_client import Counter, Histogram, start_http_server

from credit_risk_ml.data_contract import load_schema
from credit_risk_ml.modeling import load_artifact, predict_bad_probability_from_model

from inference.schema import CreditRiskRequest, load_inference_schema, validate_request


app = FastAPI(title="Credit Risk Scoring API", version="1.0.0")

# Prometheus metrics (exported on port 8001)
REQUEST_COUNT = Counter("inference_requests_total", "Total inference requests")
REQUEST_ERRORS = Counter("inference_request_errors_total", "Total inference errors")
REQUEST_LATENCY = Histogram("inference_request_latency_seconds", "Inference request latency seconds")
PREDICTIONS = Counter("inference_predictions_total", "Total predictions made")

start_http_server(8001)

# simple in-memory rate limiter: max requests per minute per client
_RATE_LIMIT = {"window_seconds": 60, "max_per_window": 60, "clients": {}}

logger = logging.getLogger("inference")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

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
async def predict(request: Request, payload: CreditRiskRequest) -> dict[str, object]:
    if ACTIVE_MODEL is None:
        raise HTTPException(status_code=503, detail="No approved model is loaded")

    # structured logging + metrics + rate limit + timeout
    request_id = str(uuid.uuid4())
    client = request.client.host if request.client else "unknown"

    # rate limit
    now = time.time()
    win = _RATE_LIMIT["window_seconds"]
    client_record = _RATE_LIMIT["clients"].setdefault(client, [])
    # purge old
    client_record[:] = [ts for ts in client_record if ts > now - win]
    if len(client_record) >= _RATE_LIMIT["max_per_window"]:
        REQUEST_ERRORS.inc()
        logger.info(json.dumps({"request_id": request_id, "event": "rate_limited", "client": client}))
        raise HTTPException(status_code=429, detail="rate limit exceeded")
    client_record.append(now)

    REQUEST_COUNT.inc()
    start = time.perf_counter()

    try:
        # validate input
        try:
            frame = validate_request(payload, SCHEMA)
        except ValueError as exc:
            REQUEST_ERRORS.inc()
            logger.info(json.dumps({"request_id": request_id, "event": "validation_error", "detail": str(exc)}))
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        # record input sample for feature stats
        sample_path = Path("artifacts/feature_stats/inference_samples.jsonl")
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.open("a", encoding="utf-8").write(json.dumps(frame.iloc[0].to_dict(), default=str) + "\n")

        # optional feature check - fail fast if drift/mismatch
        try:
            from monitoring.feature_check import run_feature_check_and_maybe_fail

            fc = run_feature_check_and_maybe_fail()
            if fc.get("failures"):
                mismatch_log = Path("artifacts/logs/feature_mismatch.log")
                if mismatch_log.exists():
                    try:
                        payload = json.loads(mismatch_log.read_text())
                        for message in payload.get("messages", []):
                            logger.info(message)
                    except Exception:
                        pass
                # log and fail
                REQUEST_ERRORS.inc()
                logger.info(json.dumps({"request_id": request_id, "event": "feature_mismatch", "failures": fc.get("failures")}))
                raise HTTPException(status_code=503, detail="feature mismatch detected; request rejected")
        except HTTPException:
            raise
        except Exception:
            # don't block inference on check failures
            logger.info(json.dumps({"request_id": request_id, "event": "feature_check_error"}))

        # prediction with timeout
        try:
            async def _predict():
                prob = float(predict_bad_probability_from_model(ACTIVE_MODEL, frame).iloc[0])
                return prob

            probability = await asyncio.wait_for(_predict(), timeout=2.0)
        except asyncio.TimeoutError:
            REQUEST_ERRORS.inc()
            logger.info(json.dumps({"request_id": request_id, "event": "timeout"}))
            raise HTTPException(status_code=504, detail="prediction timed out")

        decision = "decline" if probability >= ACTIVE_MODEL.decision_threshold else "approve"
        request_hash = hashlib.sha256(json.dumps(payload.model_dump(), sort_keys=True).encode("utf-8")).hexdigest()

        # structured prediction logging (append JSON lines)
        from datetime import datetime

        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event": "predict",
            "request_id": request_id,
            "client": client,
            "model_version": ACTIVE_MODEL.model_version,
            "request_hash": request_hash,
            "prob_bad": probability,
            "decision": decision,
        }
        PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
        with PREDICTION_LOG.open("a", encoding="utf-8") as _handle:
            _handle.write(json.dumps(log_entry, sort_keys=True) + "\n")

        PREDICTIONS.inc()
        return {"risk_score": probability, "decision": decision, "model_version": ACTIVE_MODEL.model_version, "request_id": request_id}

    finally:
        elapsed = time.perf_counter() - start
        REQUEST_LATENCY.observe(elapsed)
        logger.info(json.dumps({"request_id": request_id, "event": "request_complete", "latency": elapsed}))