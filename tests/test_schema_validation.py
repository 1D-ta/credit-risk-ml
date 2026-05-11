from pathlib import Path

import pytest

from credit_risk_ml.data_contract import load_schema, validate_and_load_rows


DATA_DIR = Path("data/raw")
RAW = DATA_DIR / "german_credit_raw.txt"
SCHEMA = Path("data/schemas/schema.json")


def test_valid_raw_passes_validation():
    schema = load_schema(SCHEMA)
    rows = validate_and_load_rows(RAW, schema)
    assert len(rows) == schema.expected_rows


def test_extra_column_fails(tmp_path):
    text = RAW.read_text().splitlines()
    bad = tmp_path / "bad.txt"
    # append extra token to first line
    first = text[0] + " EXTRA"
    bad.write_text("\n".join([first] + text[1:]))
    schema = load_schema(SCHEMA)
    with pytest.raises(ValueError):
        validate_and_load_rows(bad, schema)


def test_unknown_categorical_fails(tmp_path):
    lines = RAW.read_text().splitlines()
    parts = lines[0].split()
    parts[0] = "UNKNOWN_CAT"
    bad = tmp_path / "bad2.txt"
    bad.write_text("\n".join([" ".join(parts)] + lines[1:]))
    schema = load_schema(SCHEMA)
    with pytest.raises(ValueError):
        validate_and_load_rows(bad, schema)


def test_wrong_row_count_fails(tmp_path):
    # write a file with only one line
    one = tmp_path / "one.txt"
    one.write_text(RAW.read_text().splitlines()[0])
    schema = load_schema(SCHEMA)
    with pytest.raises(ValueError):
        validate_and_load_rows(one, schema)
