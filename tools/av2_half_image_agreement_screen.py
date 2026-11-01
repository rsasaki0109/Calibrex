#!/usr/bin/env python3
"""One-shot development screen of a half-image agreement rule on saved Argoverse 2 inputs.

Rule (``docs/benchmarks/camera_lidar_v4_candidates.md``, committed before this ran):
for every run, split each frame's LiDAR points by the image row they project to under
that run's candidate, above the principal point (``v < cy``) or not, and re-estimate
the rotation from each half with the unchanged estimator and options. A rotation axis
the full run judges is **withheld** (unchecked, no forced warning) when the two halves'
estimates of it differ by more than 0.5 deg. The halves' raw estimates are compared
whether or not a half alone passes the evidence rules: on 0fb7276f neither half alone
observes tilt, yet their raw tilt estimates differ by 3.7 deg (+2.95 vs -0.75), while
on 02678d04 they differ by 0.1 deg (the two baselines inspected before this rule was
fixed). A real calibration error
moves both halves alike, so it is meant to survive. A false optimum that only one half
supports does not. Scored with the frozen v2 scorer on the run's own judgements.

    python tools/av2_half_image_agreement_screen.py WORK_ROOT OUT.json
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tools import kitti_camera_lidar_v3_replay as replay  # noqa: E402

from calibrex.check.estimators import camera_lidar_axis_estimates  # noqa: E402
from calibrex.check.verdict import VerdictOptions, judge_pair  # noqa: E402
from calibrex.core.camera_lidar_edge import load_camera_lidar_edge  # noqa: E402
from calibrex.evaluation import camera_lidar_edge as edge  # noqa: E402

SCHEMA = "calibrex.av2_half_image_agreement_screen/v0.1"
LIMIT_DEG = 0.5
AXES = ("roll", "pitch", "yaw")
CRITERIA = {
    "fewer_wrong_judgements": "fewer judged baseline axes beyond 0.5 deg of the vendor "
    "calibration than the default rule",
    "no_more_vendor_fails": "no more baseline fail verdicts than the default rule",
    "detection_3deg_kept": "counted 3 deg trials flagged fraction drops by at most 0.05",
    "detection_1deg_kept": "counted 1 deg trials flagged fraction drops by at most 0.05",
    "no_more_large_turned_errors": "no more judged turned-run axes beyond 1.0 deg",
    "coverage": "at least 0.5 of scored baseline axes judged",
}


def halves(frames: list[Any], camera: Any, candidate: np.ndarray) -> tuple[list[Any], list[Any]]:
    upper, lower = [], []
    for frame in frames:
        points = np.asarray(frame.points, dtype=np.float64)
        moved = points @ candidate[:3, :3].T + candidate[:3, 3]
        _u, v, _valid = camera.project(moved)
        above = v < camera.cy
        for keep, out in ((above, upper), (~above, lower)):
            times = frame.point_time_offsets_s
            out.append(
                dataclasses.replace(
                    frame,
                    points=frame.points[keep],
                    point_time_offsets_s=None if times is None else times[keep],
                )
            )
    return upper, lower


def half_estimates(path: Path) -> dict[str, float | None]:
    """Each half's raw rotation estimate minus the candidate (deg, camera axes)."""

    meta, frames, camera, candidate, options = replay.load_snapshot(path)
    result: dict[str, Any] = {}
    for name, part in zip(("upper", "lower"), halves(frames, camera, candidate), strict=True):
        try:
            artifact = edge.estimate_from_frames(
                part,
                camera,
                candidate,
                options=options,
                provenance=meta.input_provenance,
                parent_frame=meta.parent_frame,
                child_frame=meta.child_frame,
                reference_label=meta.reference_label,
                start_note=meta.start_note,
                motion_compensation=meta.motion_compensation,
            )
        except Exception as exc:  # a half with too few edges cannot speak
            result[name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        values = {}
        if artifact.transform is not None:
            estimate = Rotation.from_quat(artifact.transform.rotation_quat_xyzw).as_matrix()
            delta = Rotation.from_matrix(estimate @ candidate[:3, :3].T).as_rotvec(degrees=True)
            values = {axis: float(delta[i]) for i, axis in enumerate(AXES)}
        result[name] = values
    return result


def run_input(captured: Path, withheld: set[str]) -> Any:
    """The run as the v2 scorer reads it, with withheld axes unchecked."""

    scorer = replay._tool("score_kitti_camera_lidar_v2")
    artifact = load_camera_lidar_edge(captured)
    assert artifact.start_transform is not None
    from calibrex.check.estimators import transform_matrix

    candidate = transform_matrix(
        artifact.start_transform.translation_m, artifact.start_transform.rotation_quat_xyzw
    )
    estimates = [
        dataclasses.replace(item, estimated=False, unchecked_code="unobservable")
        if item.name in withheld and item.estimated
        else item
        for item in camera_lidar_axis_estimates(artifact, candidate)
    ]
    judgement = judge_pair(estimates, VerdictOptions())
    rotation = (
        None
        if artifact.transform is None
        else Rotation.from_quat(artifact.transform.rotation_quat_xyzw).as_matrix()
    )
    return scorer.RunInput(
        verdict=judgement.verdict,
        solver_status=artifact.solver_status,
        candidate=candidate[:3, :3],
        estimate=rotation,
        judged={axis.name: axis.status for axis in judgement.axes},
    )


def screen(work: Path) -> dict[str, Any]:
    scorer = replay._tool("score_kitti_camera_lidar_v2")
    summarize = replay._tool("kitti_camera_lidar_v3_development").summarize_scores
    scores: dict[str, dict[str, Any]] = {"default": {}, "half_agreement": {}}
    withheld_log: dict[str, Any] = {}
    for root in sorted(work.glob("*/attempt_1/snapshots/*")):
        log = root.name
        if not all((root / run / "arrays.npz").is_file() for run in replay.RUNS):
            continue
        inputs: dict[str, dict[str, Any]] = {"default": {}, "half_agreement": {}}
        withheld_log[log] = {}
        for run in replay.RUNS:
            parts = half_estimates(root / run)
            upper, lower = parts.get("upper", {}), parts.get("lower", {})
            withheld = {
                axis
                for axis in AXES
                if axis in upper
                and axis in lower
                and isinstance(upper, dict)
                and isinstance(lower, dict)
                and abs(upper[axis] - lower[axis]) > LIMIT_DEG
            }
            withheld_log[log][run] = {"upper": upper, "lower": lower, "withheld": sorted(withheld)}
            inputs["default"][run] = run_input(root / run / "captured_edge.json", set())
            inputs["half_agreement"][run] = run_input(root / run / "captured_edge.json", withheld)
        for rule in scores:
            drive = scorer.DriveInput(
                baseline=inputs[rule]["baseline"],
                turned={spec: inputs[rule][run] for run, spec in replay.RUNS.items() if spec},
            )
            scores[rule][log] = dataclasses.asdict(scorer.score_drive(drive))
        print(log[:8], withheld_log[log]["baseline"]["withheld"], flush=True)
    metrics = {}
    for rule, values in scores.items():
        m = summarize(values)
        wrong = sum(
            1
            for s in values.values()
            if s["state"] == "scored"
            for a in s["axes"].values()
            if a["judged"] and abs(a["error_deg"]) > 0.5
        )
        fails = sum(1 for s in values.values() if s["state"] == "scored" and s["verdict"] == "fail")
        large_turned = sum(
            1
            for s in values.values()
            if s["state"] == "scored"
            for t in s["trials"].values()
            for e in t["judged_errors_deg"].values()
            if abs(e) > 1.0
        )
        metrics[rule] = m | {
            "wrong_judged_baseline_axes": wrong,
            "baseline_fails": fails,
            "large_turned_errors": large_turned,
        }
    d, h = metrics["default"], metrics["half_agreement"]

    def flagged(m: dict[str, Any], key: str) -> float:
        return m["trials"][key]["flagged_fraction"] or 0.0

    checks = {
        "fewer_wrong_judgements": h["wrong_judged_baseline_axes"] < d["wrong_judged_baseline_axes"],
        "no_more_vendor_fails": h["baseline_fails"] <= d["baseline_fails"],
        "detection_3deg_kept": flagged(h, "3deg") >= flagged(d, "3deg") - 0.05,
        "detection_1deg_kept": flagged(h, "1deg") >= flagged(d, "1deg") - 0.05,
        "no_more_large_turned_errors": h["large_turned_errors"] <= d["large_turned_errors"],
        "coverage": (h["judged_fraction_scored"] or 0.0) >= 0.5,
    }
    return {
        "logs": sorted(scores["default"]),
        "metrics": metrics,
        "criteria": checks,
        "carried_forward": all(checks.values()),
        "withheld": withheld_log,
        "scores": scores,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("work", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("refuse overwrite: the screen runs once")
    result = {"schema_version": SCHEMA, **screen(args.work), "criteria_text": CRITERIA}
    result["provenance"] = {
        "generator": "tools/av2_half_image_agreement_screen.py",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_commit": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "post hoc development on the AV2 logs that motivated the rule; not a claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("criteria", "carried_forward")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
