"""Replay archived SOTA gates and inventory saved camera-LiDAR development inputs.

Reads committed scores and snapshot metadata only, never sensor arrays or new data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from calibrex.core.provenance import git_commit
from calibrex.core.result import StrictModel
from calibrex.core.sota_audit import load_sota_audit_result
from calibrex.evaluation.sota_audit import audit_sota_claim

ROOT = Path(__file__).resolve().parents[1]


class BaselineReport(StrictModel):
    schema_version: Literal["calibrex.sota_development_baseline/v0.1"] = (
        "calibrex.sota_development_baseline/v0.1")
    provenance: dict[str, Any]
    audit_replays: list[dict[str, Any]]
    camera_lidar_inputs: list[dict[str, Any]]
    limitations: list[str] = Field(default_factory=lambda: [
        "Replays archived benchmark scores; does not rerun sensor calibration.",
        "Supported is scoped to each frozen claim; not universal superiority.",
        "An available arrays path is not a digest/shape check; use load_snapshot before fitting.",
        "No new held-out data, audit thresholds, verdicts or estimator defaults are changed.",
    ])


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report(root: Path = ROOT) -> BaselineReport:
    """Reproduce every archived audit and report expected local array availability."""
    audits = []
    for protocol in sorted((root / "docs/assets").rglob("*sota_protocol.yaml")):
        previous_path = protocol.with_name(protocol.name.replace("sota_protocol", "sota_audit"))
        previous = load_sota_audit_result(previous_path)
        replay = audit_sota_claim(
            protocol, command=["python", "-m", "tools.sota_development_baseline"])
        before = [(r.requirement_id, r.status) for r in previous.requirements]
        after = [(r.requirement_id, r.status) for r in replay.requirements]
        audits.append({
            "protocol": protocol.relative_to(root).as_posix(), "protocol_sha256": sha(protocol),
            "archived_audit_sha256": sha(previous_path), "verdict": replay.verdict,
            "matches_archive": replay.verdict == previous.verdict and before == after,
            "result": replay.model_dump(mode="json"),
        })
    inputs = []
    for path in sorted((root / "docs/assets/kitti_camera_lidar_v3").rglob("snapshot.json")):
        meta = json.loads(path.read_text())
        inputs.append({
            "snapshot": path.relative_to(root).as_posix(), "snapshot_sha256": sha(path),
            "drive": meta["drive"], "run": meta["run"],
            "expected_arrays_sha256": meta["arrays_sha256"],
            "arrays_present_beside_snapshot": (path.parent / "arrays.npz").is_file(),
        })
    return BaselineReport(
        provenance={"created_at": datetime.now(timezone.utc).isoformat(),
                    "git_commit": git_commit(), "generator_sha256": sha(Path(__file__)),
                    "command": ["python", "-m", "tools.sota_development_baseline"],
                    "sensor_data_read": False, "new_evaluation_started": False},
        audit_replays=audits, camera_lidar_inputs=inputs,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    report = build_report()
    (args.output / "report.json").write_text(report.model_dump_json(indent=2) + "\n")
    (args.output / "report.schema.json").write_text(
        json.dumps(BaselineReport.model_json_schema(), indent=2) + "\n")
    matched = report.audit_replays and all(r["matches_archive"] for r in report.audit_replays)
    return 0 if matched else 2


if __name__ == "__main__":
    raise SystemExit(main())
