from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path


def generate_timestamped(raw_input: Path, output: Path, start_date: str = "2020-01-01", days: int = 200, missing_day_prob: float = 0.05, late_arrival_prob: float = 0.05):
    lines = [l for l in raw_input.read_text().splitlines() if l.strip()]
    rng = random.Random(42)
    base = date.fromisoformat(start_date)
    # Build daily batches with explicit missing days.
    all_dates = [base + timedelta(days=i) for i in range(days)]
    dates = [d for d in all_dates if rng.random() > missing_day_prob]
    missing_days = sorted(set(all_dates) - set(dates))

    if not dates:
        raise RuntimeError("All generated days were missing; reduce missing_day_prob")

    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    late_events = 0
    for i, line in enumerate(lines):
        # Pick ingestion dates in daily batches and occasionally back/forward-date event_time.
        idx = i % len(dates)
        ts = dates[idx]
        if rng.random() < late_arrival_prob:
            shift = rng.randint(-10, 10)
            ts = ts + timedelta(days=shift)
            late_events += 1
        rows.append(f"{ts.isoformat()} {line}")

    output.write_text("\n".join(rows) + "\n")
    manifest = {
        "rows": len(rows),
        "days_requested": days,
        "days_emitted": len(dates),
        "missing_days": len(missing_days),
        "late_event_rows": late_events,
        "first_day": dates[0].isoformat(),
        "last_day": dates[-1].isoformat(),
    }

    manifest_path = Path("artifacts/reports/temporal_data_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    print(f"TEMPORAL_DATA: wrote {len(rows)} rows to {output}")
    print(
        "TEMPORAL_DATA: "
        f"daily_batches={len(dates)} missing_days={len(missing_days)} late_data_rows={late_events} "
        f"range={dates[0].isoformat()}..{dates[-1].isoformat()}"
    )


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
