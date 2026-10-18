#!/usr/bin/env python3
"""Post hoc development inventory of the spent KITTI camera-lidar v2 runs.

Reads committed scores and final-fit evidence only. It does not fetch data, replay
the estimator, score a new rule, derive audit gates or change a v2 verdict::

    python tools/kitti_camera_lidar_v3_development.py --output REPORT.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from calibrex.core.camera_lidar_edge import load_camera_lidar_edge
from calibrex.evaluation.camera_lidar_edge import EdgeAlignmentOptions

PHASES = ("development", "evaluation")
RUNS = (
    "baseline",
    "inject_x+3",
    "inject_y+3",
    "inject_z+3",
    "inject_x+1",
    "inject_y+1",
    "inject_z+1",
)
ROTATION_AXES = ("roll", "pitch", "yaw")


def summarize_scores(scores: dict[str, Any]) -> dict[str, Any]:
    """Descriptive metrics only; no audit thresholds, requirements or verdict."""

    scored = {drive: item for drive, item in scores.items() if item["state"] == "scored"}
    judged_errors = [
        abs(axis["error_deg"])
        for item in scored.values()
        for axis in item["axes"].values()
        if axis["judged"]
    ]
    turned_errors = [
        abs(error)
        for item in scored.values()
        for trial in item["trials"].values()
        for error in trial["judged_errors_deg"].values()
    ]
    trials: dict[str, Any] = {}
    for angle in (3, 1):
        counted = [
            trial
            for item in scored.values()
            for name, trial in item["trials"].items()
            if name.endswith(f":{angle}") and trial["counted"]
        ]
        flagged = sum(trial["flagged"] for trial in counted)
        trials[f"{angle}deg"] = {
            "counted": len(counted),
            "flagged": flagged,
            "flagged_fraction": flagged / len(counted) if counted else None,
            "judged_right": sum(trial["judged_right"] for trial in counted),
        }
    return {
        "drives": len(scores),
        "scored_drives": len(scored),
        "failed_drives": [d for d, s in scores.items() if s["state"] == "failed"],
        "inconclusive_drives": [d for d, s in scores.items() if s["state"] == "inconclusive"],
        "flagged_baselines": [d for d, s in scored.items() if s["verdict"] in ("warn", "fail")],
        "max_baseline_judged_error_deg": max(judged_errors, default=None),
        "max_turned_judged_error_deg": max(turned_errors, default=None),
        "judged_axes": len(judged_errors),
        "scored_axes": 3 * len(scored),
        "all_axes": 3 * len(scores),
        "judged_fraction_scored": len(judged_errors) / (3 * len(scored)) if scored else None,
        "judged_fraction_all": len(judged_errors) / (3 * len(scores)) if scores else None,
        "trials": trials,
    }


def development_report(root: Path, *, min_far_control_t: float = 3.5) -> dict[str, Any]:
    """Aggregate old scores and screen saved final far fits; never infer new metrics."""

    proposal = EdgeAlignmentOptions(start_scan_min_control_t=min_far_control_t)
    source = root / "docs/assets/kitti_camera_lidar_v2"
    inputs: dict[str, str] = {}

    def pin(path: Path) -> None:
        inputs[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()

    scores: dict[str, Any] = {}
    far_fits: list[dict[str, Any]] = []
    missing: dict[str, list[str]] = {}
    evidence_count = 0
    for phase in PHASES:
        score_path = source / phase / "scores.json"
        pin(score_path)
        document = json.loads(score_path.read_text(encoding="utf-8"))
        if set(document["drives"]) != set(document["scores"]):
            raise ValueError(f"{score_path}: drive list differs from scores")
        for drive in document["drives"]:
            if drive in scores:
                raise ValueError(f"drive appears twice: {drive}")
            scores[drive] = document["scores"][drive]
            for run in RUNS:
                path = (
                    source
                    / phase
                    / "runs"
                    / drive
                    / f"{run}_evidence"
                    / "camera-lidar_cam2_optical_velo_link.yaml"
                )
                if not path.is_file():
                    missing.setdefault(drive, []).append(run)
                    continue
                pin(path)
                evidence = load_camera_lidar_edge(path)
                evidence_count += 1
                if not any(
                    "search started at the profile maxima" in r for r in evidence.policy_reasons
                ):
                    continue
                judged = {
                    record.name: record.known_bad_control.min_t_statistic
                    if record.known_bad_control is not None
                    else None
                    for record in evidence.dofs
                    if record.name in ROTATION_AXES and record.status == "estimated"
                }
                below_margin = [
                    axis
                    for axis, t in judged.items()
                    if t is None or not math.isfinite(t) or t < min_far_control_t
                ]
                far_fits.append(
                    {
                        "drive": drive,
                        "run": run,
                        "saved_judged_axis_control_t": judged,
                        "axes_failing_proposed_margin": below_margin,
                        "saved_final_fit_clears_margin": not below_margin,
                    }
                )

    metrics = summarize_scores(scores)
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "analysis": "post hoc",
        "status": "development_incomplete",
        "current_v2_metrics": metrics,
        "proposed_rule": {
            "option": "start_scan_min_control_t",
            "value": proposal.start_scan_min_control_t,
            "default_changed": False,
            "scope": "all observed rotation axes of a profile-start optimum before adoption",
        },
        "proposed_rule_metrics": None,
        "coverage_change_percentage_points": None,
        "saved_fit_diagnostics": {
            "expected_evidence_runs": len(scores) * len(RUNS),
            "available_evidence_runs": evidence_count,
            "missing_evidence_runs": missing,
            "selected_far_fits": far_fits,
            "selected_far_fits_failing_margin": sum(
                not item["saved_final_fit_clears_margin"] for item in far_fits
            ),
        },
        "limitations": [
            "Only saved v2 final fits are screened. Rejected fits and estimator inputs are absent.",
            "Rejecting a saved fit does not establish which fit a replay would select.",
            "Passing the margin at a saved fit does not guarantee the same replay result: "
            "earlier adoptions can change.",
            "New-rule errors, flags, coverage, detection rates and audit thresholds "
            "require development replays.",
            "No evaluation data was fetched or read; no previous verdict is changed.",
        ],
        "provenance": {
            "generator": "tools/kitti_camera_lidar_v3_development.py",
            "generator_version": "1",
            "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "source_files_sha256": inputs,
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "estimator_sha256": hashlib.sha256(
                (root / "src/calibrex/evaluation/camera_lidar_edge.py").read_bytes()
            ).hexdigest(),
            "reference": "each KITTI day's published calibration, not ground truth",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-far-control-t", type=float, default=3.5)
    args = parser.parse_args(argv)
    report = development_report(args.root.resolve(), min_far_control_t=args.min_far_control_t)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
