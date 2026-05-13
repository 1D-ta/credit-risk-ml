#!/usr/bin/env python3
"""
Create a dataset with an unexpected categorical feature value.

This script corrupts the first categorical feature after the timestamp
so validation fails on an out-of-schema value without changing the target.
"""
import argparse
from pathlib import Path


def main(src: Path, out: Path, n: int):
    out.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as f_in, out.open("w", encoding="utf-8") as f_out:
        for i, line in enumerate(f_in):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split()
            if i < n:
                # Replace checking_account_status with an invalid category token.
                if len(parts) > 1:
                    parts[1] = "A999"
            f_out.write(" ".join(parts) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rows", type=int, default=50, help="Number of rows to mutate")
    args = p.parse_args()
    main(args.src, args.out, args.rows)
