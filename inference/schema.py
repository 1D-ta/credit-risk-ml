from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict

from credit_risk_ml.data_contract import DatasetSchema, load_schema


class CreditRiskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checking_account_status: str
    duration_months: int
    credit_history: str
    purpose: str
    credit_amount: int
    savings_account: str
    employment_status: str
    installment_rate: int
    personal_status_sex: str
    other_debtors: str
    residence_since: int
    property: str
    age: int
    other_installment_plans: str
    housing: str
    existing_credits: int
    job: str
    num_liable: int
    telephone: str
    foreign_worker: str


def load_inference_schema(schema_path: Path | None = None) -> DatasetSchema:
    resolved_path = schema_path or Path("data/schemas/schema.json")
    return load_schema(resolved_path)


def validate_request(payload: CreditRiskRequest, schema: DatasetSchema) -> pd.DataFrame:
    data = payload.model_dump()
    for field in schema.fields:
        if field.name == schema.target_column:
            continue
        value = data[field.name]
        if field.field_type == "categorical" and str(value) not in field.allowed_values:
            raise ValueError(f"Field {field.name} has invalid value {value!r}")
        if field.field_type == "integer" and not isinstance(value, int):
            raise ValueError(f"Field {field.name} must be an integer")

    return pd.DataFrame([data])