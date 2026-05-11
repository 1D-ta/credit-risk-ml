#!/usr/bin/env python3
"""
Shift only numeric (integer/float) columns based on the schema so that
categorical tokens and the target are preserved. This produces a current
dataset that still validates but has distributional shifts on numeric fields.
"""
import argparse
import json
from pathlib import Path


def load_integer_field_indices(schema_path: Path):
    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    indices = []
    for i, field in enumerate(schema.get("fields", [])):
        if field.get("type") in ("integer", "number"):
            indices.append(i)
    return indices


def main(src: Path, out: Path, schema: Path, offset: float):
    int_indices = load_integer_field_indices(schema)
    out.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as f_in, out.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split()
            # guard against unexpected column counts
            if len(parts) < max(int_indices) + 1:
                f_out.write(line + "\n")
                continue
            for idx in int_indices:
                try:
                    parts[idx] = str(int(parts[idx]) + int(offset))
                except Exception:
                    try:
                        parts[idx] = str(float(parts[idx]) + offset)
                    except Exception:
                        pass
            f_out.write(" ".join(parts) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--schema", type=Path, required=True)
    p.add_argument("--offset", type=float, default=2.0)
    args = p.parse_args()
    main(args.src, args.out, args.schema, args.offset)
