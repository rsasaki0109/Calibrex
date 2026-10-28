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
from pydantic import Field, model_serializer

from calibrex.core.camera_lidar_edge import CameraLidarEdgeControl
from calibrex.core.provenance import git_commit
from calibrex.core.result import StrictModel
from calibrex.evaluation import camera_lidar_edge as edge


class ProfileFold(StrictModel):
    training_start_theta: list[float] | None = Field(default=None, min_length=6, max_length=6)
    base_theta: list[float] = Field(min_length=6, max_length=6)
    shifted_theta: list[list[float]]
    profiled_theta: list[list[float]]
    direct_drops: list[float] = Field(min_length=2, max_length=2)
    profiled_drops: list[float] = Field(min_length=2, max_length=2)
    nuisance_at_bound: bool

    @model_serializer(mode="wrap")
    def omit_absent_training_start(self, handler: Any) -> dict[str, Any]:
        """Keep archived/default fold dumps compatible with their frozen schemas."""
        result: dict[str, Any] = handler(self)
        if self.training_start_theta is None:
            result.pop("training_start_theta", None)
        return result


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


def refine_baseline_from_controls(
    objective: edge.EdgeObjective, baseline: np.ndarray, controls: Sequence[np.ndarray],
    options: edge.EdgeAlignmentOptions,
) -> np.ndarray:
    """One training-only restart per signed control; preserve the baseline on ties."""
    best = baseline.copy()
    score = objective.score(best)
    for control in controls:
        seed = control.copy()
        seed[:3] = np.clip(seed[:3], -options.rotation_bound_deg, options.rotation_bound_deg)
        fitted, _ = edge.search(objective, seed, options, axes=(0, 1, 2),
                                first_level=options.jackknife_first_level)
        value = objective.score(fitted)
        if value > score:
            best, score = fitted, value
    return best


def profile_axis(
    training: Sequence[edge.EdgeObjective], held_out: Sequence[edge.EdgeObjective],
    axis: int, conditional_start_deg: float, options: edge.EdgeAlignmentOptions,
    *, multi_start: bool = False, training_only_start: bool = False,
    control_seeded_baseline: bool = False,
) -> ProfileAxis:
    """Optimize on training only; score fixed poses on the held-out blocks."""
    if axis not in (0, 1, 2) or len(training) != len(held_out) or len(training) < 3:
        raise ValueError("need a rotation axis and at least three paired folds")
    if not math.isfinite(conditional_start_deg):
        raise ValueError("conditional start must be finite")
    if abs(conditional_start_deg) > options.rotation_bound_deg:
        raise ValueError("conditional start exceeds rotation bound")
    if training_only_start and conditional_start_deg != 0:
        raise ValueError("training-only initialization cannot use a pooled conditional start")
    nuisance_axes = tuple(i for i in range(3) if i != axis)
    folds = []
    for train, held in zip(training, held_out, strict=True):
        start = np.zeros(6)
        start[axis] = conditional_start_deg
        if training_only_start:
            start = training_profile_start(train, axis, options)
        base, _ = edge.search(train, start, options, axes=(0, 1, 2),
                              first_level=options.jackknife_first_level)
        def controls_for(
            pose: np.ndarray, objective: edge.EdgeObjective = train,
        ) -> tuple[list[np.ndarray], list[np.ndarray]]:
            shifted, profiled = [], []
            for sign in (-1, 1):
                shift = pose.copy()
                shift[axis] += sign * options.rotation_control_deg
                fitted = refit_nuisance(objective, shift, axis, options, multi_start=multi_start)
                shifted.append(shift)
                profiled.append(fitted)
            return shifted, profiled

        shifted, profiled = controls_for(base)
        if control_seeded_baseline:
            refined = refine_baseline_from_controls(train, base, profiled, options)
            if not np.array_equal(refined, base):
                base = refined
                shifted, profiled = controls_for(base)
        # No held-out score is consulted until all parameters of this fold are fixed.
        base_score = held.score(base)
        folds.append(ProfileFold(
            training_start_theta=start.tolist() if training_only_start else None,
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


def training_profile_start(
    objective: edge.EdgeObjective, axis: int, options: edge.EdgeAlignmentOptions,
) -> np.ndarray:
    """Choose and refine an axis-profile start using only this fold's training frames."""
    if axis not in (0, 1, 2) or options.start_scan_deg is None:
        raise ValueError("training-only initialization requires an axis and profile range")
    offsets = np.arange(-options.start_scan_deg,
                        options.start_scan_deg + 0.5 * options.start_scan_step_deg,
                        options.start_scan_step_deg)
    # Zero is first so a flat profile retains the supplied candidate on ties.
    poses = np.zeros((1 + len(offsets), 6))
    poses[1:, axis] = offsets
    best, _ = edge._drive_edge_workflow(edge._edge_maximum(objective, poses))
    fitted, _ = edge.search(objective, poses[best], options, axes=(axis,))
    return fitted


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile_snapshot(
    snapshot: Path, conditional_report: Path | None, output: Path, *, multi_start: bool = False,
    spatial_balance: bool = False,
    azimuth_orientation: bool = False,
    boundary_tangent: bool = False,
    training_only_start: bool = False,
    control_seeded_baseline: bool = False,
) -> ProfileReport:
    """Verify a previously captured input and conditional receipt; retain partial failures."""
    from tools import kitti_camera_lidar_axis_probe as conditional
    from tools import kitti_camera_lidar_v3_replay as replay

    if sum((spatial_balance, azimuth_orientation, boundary_tangent)) > 1:
        raise ValueError("choose one objective ablation at a time")
    if training_only_start == (conditional_report is not None):
        raise ValueError("choose either training-only starts or a conditional report")
    if (output.resolve().is_relative_to(snapshot.resolve())
            or snapshot.resolve().is_relative_to(output.resolve())
            or (conditional_report is not None
                and conditional_report.resolve().is_relative_to(output.resolve()))):
        raise ValueError("output must be separate from source inputs")
    meta, frames, camera, candidate, options = replay.load_snapshot(snapshot)
    previous = None
    if conditional_report is not None:
        previous = conditional.ProbeReport.model_validate_json(conditional_report.read_bytes())
    if previous is not None and (previous.status != "complete"
            or [p.axis for p in previous.probes] != [0, 1, 2]
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
        "conditional_report_sha256": _sha(conditional_report) if conditional_report else None,
        "drive": meta.drive, "run": meta.run,
        "options": options.as_dict(), "analysis": "post hoc development",
        "new_accuracy_claim": False, "new_evaluation_started": False,
        "fit_scope": "training frames only, fixed translation; held-out frames score fixed poses",
        "start_scope": (
            "per-fold training-only axis profile and refinement; no pooled conditional input"
            if training_only_start else
            "previous full-input conditional correction; folds are local jackknife fits"
        ),
        "training_only_start": training_only_start,
        "control_seeded_baseline": control_seeded_baseline,
        "baseline_restart_policy": (
            "one pass, two signed controls, training score; ties keep original"
            if control_seeded_baseline else "original"
        ),
        "nuisance_search": "profile_multistart" if multi_start else "local",
        "nuisance_selection": "maximum training objective; original local fit retained on ties",
        "spatial_weights": "candidate_fixed_2x2" if spatial_balance else "original",
        "orientation_model": ("local_boundary_tangent" if boundary_tangent else
                              "projected_azimuth_gradient" if azimuth_orientation else "none"),
    })
    if training_only_start:
        report.limitations = [item for item in report.limitations
                              if not item.startswith("Initialization uses")]
        report.limitations.append(
            "Training-only initialization removes pooled-fit reuse, but does not establish "
            "independent data: preprocessing, candidate, settings and development drives are fixed."
        )

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
        orientation_features = None
        if azimuth_orientation or boundary_tangent:
            from tools import kitti_camera_lidar_orientation as oriented

            images = {frame.time_s: frame.image for frame in frames}
            if len(images) != len(frames):
                raise ValueError("orientation image lookup requires unique frame timestamps")
            orientation_features = []
            for frame in prepared:
                feature, factor = oriented.prepare_orientation_image(images[frame.time_s], options)
                if factor != frame.scale:
                    raise ValueError("orientation image scale differs from prepared frame")
                orientation_features.append(feature)
            report.provenance.update({
                "orientation_source_sha256": _sha(Path(oriented.__file__)),
                "orientation_scope": "projected azimuth proxy from compensated point coordinates",
                "orientation_limit": "scan direction is not a recovered physical boundary normal",
                "orientation_penalty": "attenuate nonnegative response; retain background penalty",
                "orientation_support": "organized frames and undistorted pinhole only",
                "orientation_unknown": "flat gradients or zero projected direction are neutral",
                "orientation_confidence": "existing scalar 99th-percentile gradient normalization",
            })

        tangents = None
        if boundary_tangent:
            from tools import kitti_camera_lidar_boundary_orientation as boundary

            tangents = [boundary.boundary_tangents(frame.points) for frame in prepared]
            report.provenance.update({
                "boundary_source_sha256": _sha(Path(boundary.__file__)),
                "boundary_settings": boundary.SETTINGS,
                "orientation_scope": "local PCA of prepared depth-discontinuity points",
                "orientation_limit": "local lines may mix surfaces; not exact physical boundaries",
                "boundary_support": [
                    {"points": len(t),
                     "supported": int(np.count_nonzero(np.linalg.norm(t, axis=1)))}
                    for t in tangents],
            })

        def objective_for(indices: Sequence[int]) -> edge.EdgeObjective:
            selected = [prepared[i] for i in indices]
            if orientation_features is not None and tangents is not None:
                return boundary.BoundaryObjective(
                    selected, camera, candidate, [orientation_features[i] for i in indices],
                    [tangents[i] for i in indices])
            if orientation_features is not None:
                return oriented.OrientationObjective(
                    selected, camera, candidate, [orientation_features[i] for i in indices],
                )
            return edge._fixed_objective(selected, camera, candidate)

        blocks = edge._block_slices(len(prepared), options.blocks)
        held = [objective_for(block) for block in blocks]
        training = [objective_for([i for i in range(len(prepared)) if i not in block])
                    for block in blocks]
        starts = [(p.axis, p.correction_deg) for p in previous.probes] if previous else [
            (axis, 0.0) for axis in range(3)]
        for axis, correction in starts:
            report.axes.append(profile_axis(training, held, axis, correction, options,
                                            multi_start=multi_start,
                                            training_only_start=training_only_start,
                                            control_seeded_baseline=control_seeded_baseline))
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
    starts = parser.add_mutually_exclusive_group(required=True)
    starts.add_argument("--conditional-report", type=Path)
    starts.add_argument("--training-only-start", action="store_true",
                        help="initialize each fold from training frames only; no pooled receipt")
    parser.add_argument("--control-seeded-baseline", action="store_true",
                        help="one training-only baseline restart from each signed control")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--multi-start", action="store_true",
                        help="also refine training-only nuisance profile maxima")
    parser.add_argument("--spatial-balance", action="store_true",
                        help="use fixed candidate-image 2x2 tile weights (research only)")
    parser.add_argument("--azimuth-orientation", action="store_true",
                        help="penalize gradient/azimuth-direction mismatch (research only)")
    parser.add_argument("--boundary-tangent", action="store_true")
    args = parser.parse_args(argv)
    profile_snapshot(args.snapshot, args.conditional_report, args.output,
                     multi_start=args.multi_start, spatial_balance=args.spatial_balance,
                     azimuth_orientation=args.azimuth_orientation,
                     boundary_tangent=args.boundary_tangent,
                     training_only_start=args.training_only_start,
                     control_seeded_baseline=args.control_seeded_baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
