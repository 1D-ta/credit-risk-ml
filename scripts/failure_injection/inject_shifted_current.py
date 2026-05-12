#!/usr/bin/env python3
"""
Create a shifted version of the raw dataset by adding a numeric offset to
tokens that parse as numbers. Useful for triggering PSI/KS drift alerts.
"""
import argparse
from pathlib import Path


def shift_token(tok: str, offset: float):
    try:
        # try integer first, then float
        if tok.isdigit():
            return str(int(tok) + int(offset))
        f = float(tok)
        return str(f + offset)
    except Exception:
        return tok


def main(src: Path, out: Path, offset: float):
    out.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as f_in, out.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split()
            # Keep the final token (credit_risk target) unchanged.
            if len(parts) > 1:
                shifted_prefix = [shift_token(p, offset) for p in parts[:-1]]
                parts = shifted_prefix + [parts[-1]]
            else:
                parts = [shift_token(p, offset) for p in parts]
            f_out.write(" ".join(parts) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--offset", type=float, default=2.0, help="Numeric offset to add")
    args = p.parse_args()
    main(args.src, args.out, args.offset)
