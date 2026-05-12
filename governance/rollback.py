from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rollback to the latest approved model.")
    parser.add_argument("--registry", type=Path, default=Path("artifacts/reports/model_registry.json"))
    parser.add_argument("--active-pointer", type=Path, default=Path("artifacts/models/active_model.json"))
    return parser.parse_args()


def rollback_to_latest_approved(registry_path: Path, active_pointer_path: Path) -> dict[str, str]:
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry not found at {registry_path}")

    registry = json.loads(registry_path.read_text())
    approved_entries = [entry for entry in registry if entry.get("approved")]
    if not approved_entries:
        raise ValueError("No approved model available for rollback")

    latest_approved = approved_entries[-1]
    active_pointer_path.parent.mkdir(parents=True, exist_ok=True)
    active_pointer_path.write_text(json.dumps({"active_model": latest_approved["artifact"]}, indent=2, sort_keys=True))
    return {"status": "rolled_back", "active_model": latest_approved["artifact"]}


def trigger_rollback():
    """Trigger automatic rollback to last approved model."""
    print("ROLLBACK triggered automatically")
    result = rollback_to_latest_approved(
        Path("artifacts/reports/model_registry.json"),
        Path("artifacts/models/active_model.json"),
    )
    print(f"ROLLBACK executed → {result['active_model']} active")
    return result


def main() -> None:
    args = parse_args()
    result = rollback_to_latest_approved(args.registry, args.active_pointer)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()