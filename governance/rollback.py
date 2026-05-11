from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rollback to the latest approved model.")
    parser.add_argument("--registry", type=Path, default=Path("artifacts/reports/model_registry.json"))
    parser.add_argument("--active-pointer", type=Path, default=Path("artifacts/models/active_model.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.registry.exists():
        raise FileNotFoundError(f"Registry not found at {args.registry}")

    registry = json.loads(args.registry.read_text())
    approved_entries = [entry for entry in registry if entry.get("approved")]
    if not approved_entries:
        raise ValueError("No approved model available for rollback")

    latest_approved = approved_entries[-1]
    args.active_pointer.parent.mkdir(parents=True, exist_ok=True)
    args.active_pointer.write_text(json.dumps({"active_model": latest_approved["artifact"]}, indent=2, sort_keys=True))
    print(json.dumps({"status": "rolled_back", "active_model": latest_approved["artifact"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()