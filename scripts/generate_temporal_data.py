from __future__ import annotations

import argparse
import random
from datetime import date, timedelta
from pathlib import Path


def generate_timestamped(raw_input: Path, output: Path, start_date: str = "2020-01-01", days: int = 200, missing_day_prob: float = 0.05, late_arrival_prob: float = 0.05):
    lines = [l for l in raw_input.read_text().splitlines() if l.strip()]
    rng = random.Random(42)
    base = date.fromisoformat(start_date)
    # build list of candidate ingestion dates with some missing days
    all_dates = [base + timedelta(days=i) for i in range(days)]
    dates = [d for d in all_dates if rng.random() > missing_day_prob]

    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, line in enumerate(lines):
        # pick an ingestion date roughly in order but allow out-of-order for late arrivals
        idx = i % len(dates)
        ts = dates[idx]
        # late arrival: occasionally assign a timestamp older or newer than ingestion order
        if rng.random() < late_arrival_prob:
            shift = rng.randint(-10, 10)
            ts = ts + timedelta(days=shift)
        # timestamp as ISO date
        rows.append(f"{ts.isoformat()} {line}")

    output.write_text("\n".join(rows) + "\n")
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw/german_credit_raw.txt"))
    parser.add_argument("--out", type=Path, default=Path("data/raw/german_credit_with_ts.txt"))
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--days", type=int, default=200)
    parser.add_argument("--missing-day-prob", type=float, default=0.05)
    parser.add_argument("--late-arrival-prob", type=float, default=0.05)
    args = parser.parse_args()
    generate_timestamped(args.raw, args.out, args.start, args.days, args.missing_day_prob, args.late_arrival_prob)
