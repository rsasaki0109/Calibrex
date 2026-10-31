#!/usr/bin/env python3
"""Measure the opt-in cross-start disagreement warning of ``calibrex check`` on spent KITTI.

Reads the 55 drives' published agreement-rule edge artifacts (7 runs each), restores
``start_disagreement_deg`` from each withheld axis's recorded reason, and judges every
run with the shipped ``camera_lidar_axis_estimates`` and ``judge_pair``. Reports baseline
false-warning and run-level turn detection rates. The data are spent and motivated the
rule; the numbers describe the feature, not held-out accuracy.

    python tools/kitti_camera_lidar_v3_disagreement_warning.py RESULTS_DIR OUT.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calibrex.check.estimators import camera_lidar_axis_estimates, transform_matrix  # noqa: E402
from calibrex.check.verdict import VerdictOptions, judge_pair  # noqa: E402
from calibrex.core.camera_lidar_edge import (  # noqa: E402
    CameraLidarEdgeArtifact,
    load_camera_lidar_edge,
)

SCHEMA = "calibrex.camera_lidar_disagreement_warning/v0.1"
RUNS = ("baseline", *(f"inject_{a}+{d}" for d in (3, 1) for a in "xyz"))
WITHHELD = re.compile(
    r"^(roll|pitch|yaw): optima from different search starts differ on it by up to "
    r"([0-9.eE+-]+) deg, above"
)


def restore(artifact: CameraLidarEdgeArtifact) -> CameraLidarEdgeArtifact:
    """The artifact as current code writes it: withheld axes also listed in a field."""

    found = {m[1]: float(m[2]) for r in artifact.policy_reasons if (m := WITHHELD.match(r))}
    return artifact.model_copy(update={"start_disagreement_deg": found or None})


def judge(artifact: CameraLidarEdgeArtifact) -> dict[str, Any]:
    assert artifact.start_transform is not None
    candidate = transform_matrix(
        artifact.start_transform.translation_m, artifact.start_transform.rotation_quat_xyzw
    )
    pair = judge_pair(camera_lidar_axis_estimates(artifact, candidate), VerdictOptions())
    return {
        "verdict": pair.verdict,
        "judged": {axis.name: axis.status for axis in pair.axes},
        "disputed": sorted(
            item.name for item in pair.unchecked if item.reason_code == "start_disagreement"
        ),
    }


def drive_dirs(results: Path) -> dict[str, Path]:
    """Each spent drive's proposed-rule artifact directory, latest attempt for re-captures."""

    dirs = {p.parent.name: p for p in (results / "cached").glob("*/proposed")}
    for drive_dir in sorted((results / "recaptured").iterdir()):
        if not drive_dir.is_dir():
            continue
        latest = max(drive_dir.glob("attempt_*"), key=lambda p: int(p.name.split("_")[1]))
        dirs[drive_dir.name] = latest / "comparison" / drive_dir.name / "proposed"
    return dirs


def measure(results: Path, counted: dict[str, dict[str, bool]]) -> dict[str, Any]:
    runs: dict[str, dict[str, Any]] = {}
    for drive, directory in sorted(drive_dirs(results).items()):
        runs[drive] = {
            run: judge(restore(load_camera_lidar_edge(directory / f"{run}.json"))) for run in RUNS
        }
    baselines = {d: r["baseline"]["verdict"] for d, r in runs.items()}
    detection: dict[str, Any] = {}
    for angle in ("3", "1"):
        trials = [
            (drive, run)
            for drive, by_run in runs.items()
            for run in by_run
            if run.endswith(f"+{angle}") and counted.get(drive, {}).get(f"{run[7]}:{angle}", False)
        ]
        flagged = [t for t in trials if runs[t[0]][t[1]]["verdict"] in ("warn", "fail")]
        detection[f"{angle}deg"] = {
            "counted": len(trials),
            "run_flagged": len(flagged),
            "run_flagged_fraction": len(flagged) / len(trials) if trials else None,
            "unflagged": sorted(map(list, set(trials) - set(flagged))),
        }
    return {
        "drives": len(runs),
        "baseline_verdicts": baselines,
        "baselines_warned": sorted(d for d, v in baselines.items() if v in ("warn", "fail")),
        "detection": detection,
        "runs": runs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("refuse overwrite")
    screen = json.loads((args.results / "screen.json").read_text())
    counted = {
        drive: {name: trial["counted"] for name, trial in item.get("trials", {}).items()}
        for drive, item in _scores(args.results).items()
    }
    result = {"schema_version": SCHEMA, **measure(args.results, counted)}
    result["scope"] = (
        "spent KITTI development drives that motivated the rule; describes the opt-in feature, "
        "not held-out accuracy or a SOTA claim"
    )
    result["screen_proposed_options"] = screen["proposed_options"]
    commit = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    result["provenance"] = {
        "generator": "tools/kitti_camera_lidar_v3_disagreement_warning.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_commit": commit,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "screen_sha256": hashlib.sha256((args.results / "screen.json").read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("baselines_warned", "detection")})[:600])
    return 0


def _scores(results: Path) -> dict[str, Any]:
    """Proposed-rule drive scores from the cached and re-captured comparisons."""

    scores = dict(
        json.loads((results / "cached" / "comparison.json").read_text())["scores"]["proposed"]
    )
    for drive, directory in drive_dirs(results).items():
        comparison = directory.parent.parent / "comparison.json"
        if comparison.is_file():
            scores[drive] = json.loads(comparison.read_text())["scores"]["proposed"][drive]
    return scores


if __name__ == "__main__":
    sys.exit(main())
