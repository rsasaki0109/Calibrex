"""Spent-data held-out controls with training-only nuisance reoptimization.

Research diagnostic: compare a raw axis shift with the same shift after the
other rotation coordinates have been refitted on training frames only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field

from calibrex.core.camera_lidar_edge import CameraLidarEdgeControl
from calibrex.core.provenance import git_commit
from calibrex.core.result import StrictModel
from calibrex.evaluation import camera_lidar_edge as edge


class ProfileFold(StrictModel):
    base_theta: list[float] = Field(min_length=6, max_length=6)
    shifted_theta: list[list[float]]
    profiled_theta: list[list[float]]
    direct_drops: list[float] = Field(min_length=2, max_length=2)
    profiled_drops: list[float] = Field(min_length=2, max_length=2)
    nuisance_at_bound: bool


class ProfileAxis(StrictModel):
    axis: int = Field(ge=0, le=2)
    direct_control: CameraLidarEdgeControl
    profiled_control: CameraLidarEdgeControl
    folds: list[ProfileFold]


class ProfileReport(StrictModel):
    schema_version: Literal["calibrex.camera_lidar_profile_probe/v0.1"] = (
        "calibrex.camera_lidar_profile_probe/v0.1"
    )
    status: Literal["running", "complete", "failed"]
    error: str | None = None
    provenance: dict[str, Any]
    axes: list[ProfileAxis] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=lambda: [
        "Training-only local coordinate searches may miss distant nuisance optima.",
        "Initialization uses the pooled conditional fit; these are not nested validation folds.",
        "Held-out drops need not decrease after training-only nuisance refitting.",
        "Control detection is not accuracy, joint observability or a native verdict.",
        "Dataset calibration is not ground truth; no new audit or default change.",
    ])


def _control(drops: np.ndarray, options: edge.EdgeAlignmentOptions) -> CameraLidarEdgeControl:
    means = drops.mean(axis=1)
    errors = drops.std(axis=1, ddof=1) / math.sqrt(drops.shape[1])
    statistics = np.where(errors > 0, means / np.where(errors > 0, errors, 1), 0)
    worst = int(np.argmin(statistics))
    return CameraLidarEdgeControl(
        amount=options.rotation_control_deg, unit="deg", held_out_blocks=drops.shape[1],
        detected_blocks=int(np.sum(drops.min(axis=0) > 0)),
        mean_objective_drop=float(means[worst]), min_t_statistic=float(statistics[worst]),
        detected=bool(statistics.min() >= options.min_control_t and means.min() > 0),
    )


def refit_nuisance(
    objective: edge.EdgeObjective, theta: np.ndarray, axis: int,
    options: edge.EdgeAlignmentOptions, *, multi_start: bool = False,
) -> np.ndarray:
    """Keep the tested coordinate fixed; optionally search training profile maxima too."""
    if multi_start and options.start_scan_deg is None:
        raise ValueError("multi-start nuisance search requires a profile range")
    class FixedAxisObjective(edge.EdgeObjective):
        def __init__(self) -> None:
            pass

        def score(self, coordinates: np.ndarray) -> float:
            actual = coordinates.copy()
            actual[axis] = theta[axis]
            return objective.score(actual)

    start = theta.copy()
    start[axis] = 0.0  # Dummy coordinate; only nuisance bounds constrain this search.
    fixed_objective = FixedAxisObjective()
    nuisance_axes = tuple(i for i in range(3) if i != axis)
    fitted, best_score = edge.search(
        fixed_objective, start, options, axes=nuisance_axes,
        first_level=options.jackknife_first_level,
    )
    if multi_start:
        assert options.start_scan_deg is not None
        offsets = np.arange(-options.start_scan_deg,
                            options.start_scan_deg + 0.5 * options.start_scan_step_deg,
                            options.start_scan_step_deg)
        starts = []
        combined = start.copy()
        for nuisance_axis in nuisance_axes:
            poses = np.repeat(start[None, :], len(offsets), axis=0)
            poses[:, nuisance_axis] = offsets
            index, _ = edge._drive_edge_workflow(edge._edge_maximum(fixed_objective, poses))
            starts.append(poses[index].copy())
            combined[nuisance_axis] = poses[index, nuisance_axis]
        starts.append(combined)
        seen = {start.tobytes()}
        for profile_start in starts:
            if profile_start.tobytes() in seen:
                continue
            seen.add(profile_start.tobytes())
            trial, score = edge.search(
                fixed_objective, profile_start, options, axes=nuisance_axes,
            )
            # Keep the original local optimum on ties; held-out scores never select starts.
            if score > best_score:
                fitted, best_score = trial, score
    fitted[axis] = theta[axis]
    return fitted


def profile_axis(
    training: Sequence[edge.EdgeObjective], held_out: Sequence[edge.EdgeObjective],
    axis: int, conditional_start_deg: float, options: edge.EdgeAlignmentOptions,
    *, multi_start: bool = False,
) -> ProfileAxis:
    """Optimize on training only; score fixed poses on the held-out blocks."""
    if axis not in (0, 1, 2) or len(training) != len(held_out) or len(training) < 3:
        raise ValueError("need a rotation axis and at least three paired folds")
    if not math.isfinite(conditional_start_deg):
        raise ValueError("conditional start must be finite")
    if abs(conditional_start_deg) > options.rotation_bound_deg:
        raise ValueError("conditional start exceeds rotation bound")
    nuisance_axes = tuple(i for i in range(3) if i != axis)
    folds = []
    for train, held in zip(training, held_out, strict=True):
        start = np.zeros(6)
        start[axis] = conditional_start_deg
        base, _ = edge.search(train, start, options, axes=(0, 1, 2),
                              first_level=options.jackknife_first_level)
        shifted, profiled = [], []
        for sign in (-1, 1):
            shift = base.copy()
            shift[axis] += sign * options.rotation_control_deg
            fitted = refit_nuisance(train, shift, axis, options, multi_start=multi_start)
            shifted.append(shift)
            profiled.append(fitted)
        # No held-out score is consulted until all parameters of this fold are fixed.
        base_score = held.score(base)
        folds.append(ProfileFold(
            base_theta=base.tolist(), shifted_theta=[p.tolist() for p in shifted],
            profiled_theta=[p.tolist() for p in profiled],
            direct_drops=[base_score - held.score(p) for p in shifted],
            profiled_drops=[base_score - held.score(p) for p in profiled],
            nuisance_at_bound=any(
                abs(p[i]) >= options.rotation_bound_deg - 1e-9
                for p in [base, *profiled] for i in nuisance_axes
            ),
        ))
    return ProfileAxis(
        axis=axis, folds=folds,
        direct_control=_control(np.array([f.direct_drops for f in folds]).T, options),
        profiled_control=_control(np.array([f.profiled_drops for f in folds]).T, options),
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile_snapshot(
    snapshot: Path, conditional_report: Path, output: Path, *, multi_start: bool = False,
    spatial_balance: bool = False,
) -> ProfileReport:
    """Verify a previously captured input and conditional receipt; retain partial failures."""
    from tools import kitti_camera_lidar_axis_probe as conditional
    from tools import kitti_camera_lidar_v3_replay as replay

    if (output.resolve().is_relative_to(snapshot.resolve())
            or snapshot.resolve().is_relative_to(output.resolve())
            or conditional_report.resolve().is_relative_to(output.resolve())):
        raise ValueError("output must be separate from source inputs")
    meta, frames, camera, candidate, options = replay.load_snapshot(snapshot)
    previous = conditional.ProbeReport.model_validate_json(conditional_report.read_bytes())
    if (previous.status != "complete" or [p.axis for p in previous.probes] != [0, 1, 2]
            or previous.provenance["snapshot_sha256"] != _sha(snapshot / "snapshot.json")
            or previous.provenance["arrays_sha256"] != meta.arrays_sha256
            or previous.provenance["drive"] != meta.drive
            or previous.provenance["run"] != meta.run):
        raise ValueError("conditional receipt does not match the verified snapshot")
    output.mkdir(parents=True, exist_ok=False)
    report = ProfileReport(status="running", provenance={
        "created_at": datetime.now(timezone.utc).isoformat(), "git_commit": git_commit(),
        "generator_sha256": _sha(Path(__file__)),
        "estimator_sha256": _sha(Path(edge.__file__)),
        "replay_sha256": _sha(Path(replay.__file__)),
        "snapshot_sha256": _sha(snapshot / "snapshot.json"), "arrays_sha256": meta.arrays_sha256,
        "conditional_report_sha256": _sha(conditional_report), "drive": meta.drive, "run": meta.run,
        "options": options.as_dict(), "analysis": "post hoc development",
        "new_accuracy_claim": False, "new_evaluation_started": False,
        "fit_scope": "training frames only, fixed translation; held-out frames score fixed poses",
        "start_scope": "previous full-input conditional correction; folds are local jackknife fits",
        "nuisance_search": "profile_multistart" if multi_start else "local",
        "nuisance_selection": "maximum training objective; original local fit retained on ties",
        "spatial_weights": "candidate_fixed_2x2" if spatial_balance else "original",
    })

    def write() -> None:
        (output / "report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")

    (output / "report.schema.json").write_text(
        json.dumps(ProfileReport.model_json_schema(), indent=2), encoding="utf-8")
    write()
    try:
        prepared, dropped = edge.prepare_frames(frames, camera, candidate, options)
        report.provenance.update({"prepared_frames": len(prepared), "dropped": dropped})
        if len(prepared) < options.min_frames:
            raise ValueError("not enough prepared frames")
        if spatial_balance:
            from tools import kitti_camera_lidar_spatial_weights as spatial

            balanced = [spatial.balance_frame(frame, camera, candidate) for frame in prepared]
            prepared = [frame for frame, _ in balanced]
            report.provenance.update({
                "spatial_source_sha256": _sha(Path(spatial.__file__)),
                "tile_support": [support.model_dump(mode="json") for _, support in balanced],
                "spatial_scope": "per-frame fixed preprocessing on train and held-out frames",
            })
        blocks = edge._block_slices(len(prepared), options.blocks)
        held = [edge._fixed_objective([prepared[i] for i in block], camera, candidate)
                for block in blocks]
        training = [edge._fixed_objective(
            [frame for i, frame in enumerate(prepared) if i not in block], camera, candidate,
        ) for block in blocks]
        for p in previous.probes:
            report.axes.append(profile_axis(training, held, p.axis, p.correction_deg, options,
                                            multi_start=multi_start))
            write()
        report.status = "complete"
    except Exception as exc:
        report.status, report.error = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--conditional-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--multi-start", action="store_true",
                        help="also refine training-only nuisance profile maxima")
    parser.add_argument("--spatial-balance", action="store_true",
                        help="use fixed candidate-image 2x2 tile weights (research only)")
    args = parser.parse_args(argv)
    profile_snapshot(args.snapshot, args.conditional_report, args.output,
                     multi_start=args.multi_start, spatial_balance=args.spatial_balance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
