import json
import subprocess
import sys
from pathlib import Path


def run_script(args, cwd: Path):
    cp = subprocess.run([sys.executable] + args, cwd=cwd, capture_output=True, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"Script failed: {' '.join(args)}\nSTDOUT:\n{cp.stdout}\nSTDERR:\n{cp.stderr}")
    return cp.stdout


def test_approve_and_reject_preserves_active(tmp_path):
    project_root = Path.cwd()
    # create dummy artifact
    model = tmp_path / "model.pkl"
    model.write_text("dummy")

    metrics = tmp_path / "metrics.json"
    calibration = tmp_path / "calibration.json"
    # first: approved
    metrics.write_text(json.dumps({"roc_auc": 0.95}))
    calibration.write_text(json.dumps({"calibration_gap_after": 0.02}))

    registry = tmp_path / "registry.json"
    active = tmp_path / "active.json"

    args = [
        "governance/approve_model.py",
        "--model-artifact",
        str(model),
        "--metrics",
        str(metrics),
        "--calibration-report",
        str(calibration),
        "--registry-output",
        str(registry),
        "--active-pointer",
        str(active),
        "--approved-dir",
        str(tmp_path / "approved"),
        "--candidate-dir",
        str(tmp_path / "candidate"),
    ]
    run_script(args, cwd=project_root)

    assert active.exists()
    payload = json.loads(active.read_text())
    assert payload.get("active_model")

    # now a rejection should NOT null out active pointer
    metrics.write_text(json.dumps({"roc_auc": 0.1}))
    calibration.write_text(json.dumps({"calibration_gap_after": 0.9}))

    run_script(args, cwd=project_root)

    payload2 = json.loads(active.read_text())
    assert payload2.get("active_model") == payload.get("active_model")

    # rollback should set active to latest approved
    run_script(["governance/rollback.py", "--registry", str(registry), "--active-pointer", str(active)], cwd=project_root)
    rolled = json.loads(active.read_text())
    assert rolled.get("active_model")
