from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Approve or reject a calibrated model artifact.")
    parser.add_argument("--model-artifact", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--calibration-report", required=True, type=Path)
    parser.add_argument("--policy", type=Path, default=Path("governance/policy.json"))
    parser.add_argument("--approved-dir", type=Path, default=Path("artifacts/models/approved"))
    parser.add_argument("--candidate-dir", type=Path, default=Path("artifacts/models/candidate"))
    parser.add_argument("--registry-output", type=Path, default=Path("artifacts/reports/model_registry.json"))
    parser.add_argument("--active-pointer", type=Path, default=Path("artifacts/models/active_model.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = json.loads(args.metrics.read_text())
    calibration_report = json.loads(args.calibration_report.read_text())
    policy = json.loads(args.policy.read_text())

    approved = (
        metrics["roc_auc"] >= policy["minimum_roc_auc"]
        and calibration_report["calibration_gap_after"] <= policy["maximum_calibration_gap"]
    )

    args.candidate_dir.mkdir(parents=True, exist_ok=True)
    args.approved_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.candidate_dir / args.model_artifact.name
    shutil.copy2(args.model_artifact, candidate_path)
    approved_path = args.approved_dir / args.model_artifact.name

    registry = []
    if args.registry_output.exists():
        registry = json.loads(args.registry_output.read_text())

    entry = {
        "artifact": str(approved_path if approved else candidate_path),
        "approved": approved,
        "roc_auc": metrics["roc_auc"],
        "calibration_gap_after": calibration_report["calibration_gap_after"],
    }
    registry.append(entry)

    # preserve existing active pointer on rejection
    current_active = None
    if args.active_pointer.exists():
        try:
            current_active = json.loads(args.active_pointer.read_text()).get("active_model")
        except Exception:
            current_active = None

    if approved:
        shutil.copy2(args.model_artifact, approved_path)
        args.active_pointer.write_text(json.dumps({"active_model": str(approved_path)}, indent=2, sort_keys=True))
    else:
        # do not null out active pointer; keep previous active_model if present
        if current_active:
            # re-write same pointer to keep file consistent
            args.active_pointer.write_text(json.dumps({"active_model": current_active}, indent=2, sort_keys=True))
        else:
            # leave active pointer absent when there was none
            pass

    args.registry_output.parent.mkdir(parents=True, exist_ok=True)
    args.registry_output.write_text(json.dumps(registry, indent=2, sort_keys=True))
    out = {"status": "approved" if approved else "rejected", "artifact": str(args.model_artifact)}
    if not approved and current_active:
        out["active_model_unchanged"] = current_active
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()