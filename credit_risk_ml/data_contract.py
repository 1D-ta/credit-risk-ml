from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


INTEGER_PATTERN = re.compile(r"^-?\d+$")


@dataclass(frozen=True)
class FieldSpec:
    name: str
    field_type: str
    allowed_values: tuple[Any, ...] = ()


@dataclass(frozen=True)
class DatasetSchema:
    dataset_name: str
    source_file: str
    expected_rows: int
    expected_columns: int
    target_column: str
    target_encoding: dict[str, str]
    cost_matrix: dict[str, Any]
    fields: tuple[FieldSpec, ...]


def load_schema(schema_path: Path) -> DatasetSchema:
    payload = json.loads(schema_path.read_text())
    fields = tuple(
        FieldSpec(
            name=item["name"],
            field_type=item["type"],
            allowed_values=tuple(str(value) for value in item.get("allowed_values", ())),
        )
        for item in payload["fields"]
    )
    return DatasetSchema(
        dataset_name=payload["dataset_name"],
        source_file=payload["source_file"],
        expected_rows=payload["expected_rows"],
        expected_columns=payload["expected_columns"],
        target_column=payload["target_column"],
        target_encoding=dict(payload["target_encoding"]),
        cost_matrix=dict(payload["cost_matrix"]),
        fields=fields,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_and_load_rows(raw_path: Path, schema: DatasetSchema) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(raw_path.read_text().splitlines(), start=1):
        if not raw_line.strip():
            raise ValueError(f"Empty row found at line {line_number}")
        tokens = raw_line.split()
        if len(tokens) != schema.expected_columns:
            raise ValueError(
                f"Line {line_number} has {len(tokens)} columns; expected {schema.expected_columns}"
            )

        record: dict[str, Any] = {}
        for field_spec, token in zip(schema.fields, tokens):
            if field_spec.field_type == "integer":
                if not INTEGER_PATTERN.fullmatch(token):
                    raise ValueError(
                        f"Field {field_spec.name} on line {line_number} must be an integer, got {token!r}"
                    )
                record[field_spec.name] = int(token)
            elif field_spec.field_type in {"categorical", "target"}:
                if token not in field_spec.allowed_values:
                    raise ValueError(
                        f"Field {field_spec.name} on line {line_number} has invalid value {token!r}"
                    )
                record[field_spec.name] = int(token) if field_spec.field_type == "target" else token
            else:
                raise ValueError(f"Unknown field type {field_spec.field_type!r} for {field_spec.name}")

        rows.append(record)

    if len(rows) != schema.expected_rows:
        raise ValueError(f"Dataset has {len(rows)} rows; expected {schema.expected_rows}")

    return rows


def dataset_summary(rows: list[dict[str, Any]], schema: DatasetSchema, raw_path: Path) -> dict[str, Any]:
    target_counts: dict[int, int] = {}
    for row in rows:
        target = int(row[schema.target_column])
        target_counts[target] = target_counts.get(target, 0) + 1

    return {
        "dataset_name": schema.dataset_name,
        "rows": len(rows),
        "columns": schema.expected_columns,
        "data_hash": file_sha256(raw_path),
        "target_counts": target_counts,
    }


def rows_to_frame(rows: list[dict[str, Any]], schema: DatasetSchema) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    ordered_columns = [field.name for field in schema.fields]
    return frame[ordered_columns]


def target_series(frame: pd.DataFrame, schema: DatasetSchema) -> pd.Series:
    target = frame[schema.target_column].astype(int)
    return target


def feature_frame(frame: pd.DataFrame, schema: DatasetSchema) -> pd.DataFrame:
    return frame.drop(columns=[schema.target_column])