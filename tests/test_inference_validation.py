from pathlib import Path

import pytest

from credit_risk_ml.data_contract import load_schema, validate_and_load_rows, rows_to_frame
from inference.schema import CreditRiskRequest, validate_request


SCHEMA = Path("data/schemas/schema.json")
RAW = Path("data/raw/german_credit_raw.txt")


def _first_row_payload():
    schema = load_schema(SCHEMA)
    rows = validate_and_load_rows(RAW, schema)
    frame = rows_to_frame(rows, schema)
    row = frame.iloc[0].to_dict()
    # drop target
    row.pop(schema.target_column, None)
    return row


def test_valid_request_passes():
    payload = _first_row_payload()
    req = CreditRiskRequest(**payload)
    schema = load_schema(SCHEMA)
    df = validate_request(req, schema)
    assert df.shape[0] == 1


def test_extra_field_rejected():
    payload = _first_row_payload()
    payload["extra"] = "x"
    with pytest.raises(Exception):
        CreditRiskRequest(**payload)


def test_missing_field_rejected():
    payload = _first_row_payload()
    payload.pop("age", None)
    with pytest.raises(Exception):
        CreditRiskRequest(**payload)


def test_invalid_categorical_rejected():
    payload = _first_row_payload()
    # pick a categorical field from schema
    payload["checking_account_status"] = "BAD"
    req = CreditRiskRequest(**{k: v for k, v in payload.items() if k in CreditRiskRequest.model_fields})
    schema = load_schema(SCHEMA)
    with pytest.raises(ValueError):
        validate_request(req, schema)
