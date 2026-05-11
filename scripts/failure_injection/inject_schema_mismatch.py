#!/usr/bin/env python3
"""
Create a schema-mismatch version of the raw dataset by adding an extra token
at the end of each line (extra column).
"""
import argparse
from pathlib import Path


def main(src: Path, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as f_in, out.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.rstrip("\n")
            if not line:
                continue
            f_out.write(line + " X\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True, help="Path to source raw file")
    p.add_argument("--out", type=Path, required=True, help="Path for output mutated file")
    args = p.parse_args()
    main(args.src, args.out)
