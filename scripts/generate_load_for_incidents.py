#!/usr/bin/env python3
"""
Generate inference traffic for incident testing.
Used in Phase 1B to create real traffic for monitoring validation.

Run AFTER docker compose is up:
    python scripts/generate_load_for_incidents.py --scenario=normal --requests=100
    python scripts/generate_load_for_incidents.py --scenario=schema_mismatch --requests=5
"""

import argparse
import json
import random
import time
from pathlib import Path

import requests


BASE_URL = "http://localhost:8000"


def send_request(payload: dict, scenario: str = "normal") -> dict:
    """Send a request and return (status_code, elapsed_ms, error)."""
    start = time.perf_counter()
    try:
        resp = requests.post(f"{BASE_URL}/predict", json=payload, timeout=5)
        elapsed = (time.perf_counter() - start) * 1000
        return {
            "status": resp.status_code,
            "elapsed_ms": elapsed,
            "response": resp.json() if resp.ok else resp.text,
            "error": None,
        }
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        return {"status": None, "elapsed_ms": elapsed, "response": None, "error": str(e)}


def normal_payload() -> dict:
    """Valid request."""
    return {
        "age": random.randint(18, 80),
        "sex": random.choice(["M", "F"]),
        "job": random.randint(0, 3),
        "housing": random.choice(["own", "rent", "free"]),
        "existing_credits": random.randint(1, 4),
        "num_dependents": random.randint(1, 2),
        "duration_months": random.randint(4, 72),
        "purpose": random.choice(["car", "furniture", "education", "other"]),
        "amount": random.randint(500, 20000),
        "rate": random.randint(1, 4),
        "status": random.choice(["0<X<200", "no checking", ">=200", "<0"]),
        "savings": random.choice(["<100", "100<=X<500", "500<=X<1000", ">=1000", "unknown"]),
        "employment": random.choice(["<1", "1<=X<4", "4<=X<7", ">=7", "unemployed"]),
    }


def schema_mismatch_payload() -> dict:
    """Invalid: missing required field 'age'."""
    payload = normal_payload()
    del payload["age"]  # Intentional error
    return payload


def shifted_numeric_payload() -> dict:
    """Valid schema, but shifted distribution (high age, high amount)."""
    payload = normal_payload()
    payload["age"] = random.randint(70, 90)  # Older than training
    payload["amount"] = random.randint(15000, 30000)  # Much higher than training
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=["normal", "schema_mismatch", "shifted"],
        default="normal",
        help="Load scenario",
    )
    parser.add_argument("--requests", type=int, default=10, help="Number of requests")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between requests (seconds)")
    parser.add_argument("--output", default=None, help="Output JSONL file for results")

    args = parser.parse_args()

    payload_fn = {
        "normal": normal_payload,
        "schema_mismatch": schema_mismatch_payload,
        "shifted": shifted_numeric_payload,
    }[args.scenario]

    print(f"🚀 Generating {args.requests} {args.scenario} requests...")
    results = []

    for i in range(args.requests):
        payload = payload_fn()
        result = send_request(payload, args.scenario)
        results.append(result)

        status_str = f"{result['status']}" if result['status'] else "ERROR"
        print(
            f"  [{i+1}/{args.requests}] {status_str} | "
            f"{result['elapsed_ms']:.1f}ms | "
            f"{result.get('error', 'OK')}"
        )

        if i < args.requests - 1:
            time.sleep(args.delay)

    # Summary
    success = sum(1 for r in results if r["status"] == 200)
    errors = sum(1 for r in results if r["error"])
    avg_latency = sum(r["elapsed_ms"] for r in results) / len(results)

    print(f"\n📊 Summary:")
    print(f"   Success: {success}/{args.requests}")
    print(f"   Errors: {errors}")
    print(f"   Avg latency: {avg_latency:.1f}ms")

    # Save results if requested
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")
        print(f"   Saved to: {out_path}")


if __name__ == "__main__":
    main()
