"""Evaluate a GNSS antenna lever arm and clock offset against LiDAR odometry.

The LiDAR stream is read once.  LiDAR odometry runs only while the GNSS track
covers the scans without gaps (RTK-fixed epochs by default) and restarts after
every gap; each odometry segment is split at unreliable registrations and then
cut into fixed-length windows, each with its own ENU alignment.

Evidence mirrors the INS-LiDAR evaluation:

* every third window is held out of the fit;
* the reported std is the larger of the analytic std and a grouped window
  jackknife, because odometry errors are correlated in time;
* each estimated quantity is shifted by a known-bad amount, and the held-out
  chi-square must detect the shift; and
* a reference (for example a CAD offset) is compared only after the fit.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex import __version__
from calibrex.core.gnss_lidar_lever_arm import (
    GnssLidarDofRecord,
    GnssLidarKnownBadControl,
    GnssLidarLeverArmArtifact,
    GnssLidarLeverArmProvenance,
    GnssLidarPolicyStatus,
    GnssLidarSegmentSummary,
)
from calibrex.core.provenance import git_commit
from calibrex.data.rtk_slam import (
    iter_livox_scans,
    load_rtk_track,
    rtk_slam_input_digest,
    rtk_slam_reference_lever_arm,
)
from calibrex.solvers.gnss_lever_arm_solver import (
    LEVER_ARM_DOFS,
    GnssLeverArmSolver,
    GnssTrackModel,
    LeverArmOptions,
    LeverArmResult,
    OdometryWindow,
    WindowPairs,
    build_pairs,
    window_residuals,
)
from calibrex.solvers.scan_to_scan_odometry import (
    IncrementalScanOdometry,
    ScanOdometryOptions,
    ScanRegistration,
)

FloatArray: TypeAlias = NDArray[np.float64]

RTK_SLAM_LIMITATIONS: tuple[str, ...] = (
    "The reference lever arm combines the MID360 manual IMU position with a CAD "
    "antenna offset; it is not metrology.",
    "Only RTK-fixed epochs are used, so GNSS-degraded stretches of each sequence "
    "contribute no windows.",
    "LiDAR odometry has no gravity or absolute heading; each window's alignment to "
    "ENU is estimated and profiled out.",
    "The clock offset is observable only through velocity changes within a window; "
    "steady motion leaves it weakly constrained.",
)


@dataclass(frozen=True)
class GnssLidarRunOptions:
    """Segmentation, holdout, controls, and solver settings."""

    window_duration_s: float = 10.0
    holdout_every: int = 3
    jackknife_groups: int = 8
    lever_arm_control_m: float = 0.05
    time_control_s: float = 0.02
    detection_delta_chi2: float = 9.0
    coverage_margin_s: float = 0.3
    min_correspondences: int = 200
    max_registration_rmse_m: float = 0.3
    solver: LeverArmOptions = field(default_factory=LeverArmOptions)
    odometry: ScanOdometryOptions = field(
        default_factory=lambda: ScanOdometryOptions(
            voxel_size_m=0.3, min_range_m=1.0, max_range_m=60.0, local_map_scans=5
        )
    )


@dataclass(frozen=True)
class GnssLidarEvaluation:
    """Dataset-independent evaluation output."""

    result: LeverArmResult
    records: tuple[GnssLidarDofRecord, ...]
    train_windows: int
    holdout_windows: int
    jackknife_fits: int
    train_median_residual_m: float | None
    holdout_median_residual_m: float | None
    policy_status: GnssLidarPolicyStatus
    policy_reasons: tuple[str, ...]


@dataclass
class _Segmenter:
    """Streams scans into odometry segments and fixed-length windows."""

    track: GnssTrackModel
    options: GnssLidarRunOptions
    windows: list[OdometryWindow] = field(default_factory=list)
    scans_read: int = 0
    scans_covered: int = 0
    segments: int = 0
    unreliable: int = 0
    _odometry: IncrementalScanOdometry | None = None
    _times: list[float] = field(default_factory=list)
    _prefix: str = ""

    def add(self, time_s: float, scan: FloatArray) -> None:
        self.scans_read += 1
        margin = self.options.coverage_margin_s
        if not self.track.continuous(time_s - margin, time_s + margin):
            self.flush()
            return
        self.scans_covered += 1
        if self._odometry is None:
            self._odometry = IncrementalScanOdometry(self.options.odometry)
            self._times = []
        self._odometry.add(scan, time_s)
        self._times.append(time_s)

    def flush(self) -> None:
        if self._odometry is None:
            return
        result = self._odometry.result()
        self._odometry = None
        if len(result.poses) < 2:
            return
        self.segments += 1
        poses = np.stack(result.poses)
        times = np.array(self._times)
        start = 0
        for index, registration in enumerate(result.registrations, start=1):
            if not self._reliable(registration):
                self.unreliable += 1
                self._cut(times[start:index], poses[start:index])
                start = index
        self._cut(times[start:], poses[start:])

    def _reliable(self, registration: ScanRegistration) -> bool:
        return (
            registration.correspondences >= self.options.min_correspondences
            and math.isfinite(registration.rmse_m)
            and registration.rmse_m <= self.options.max_registration_rmse_m
        )

    def _cut(self, times: FloatArray, poses: FloatArray) -> None:
        if len(times) < 2:
            return
        edges = np.arange(times[0], times[-1] + 1e-9, self.options.window_duration_s)
        for begin in edges:
            mask = (times >= begin) & (times < begin + self.options.window_duration_s)
            if np.count_nonzero(mask) >= 20:
                block = len(self.windows)
                self.windows.append(
                    OdometryWindow(
                        window_id=f"{self._prefix}w{block}",
                        block=block,
                        times_s=times[mask],
                        poses=poses[mask],
                    )
                )


def collect_windows(
    track: GnssTrackModel,
    scans: Iterable[tuple[float, FloatArray]],
    options: GnssLidarRunOptions,
    *,
    prefix: str = "",
    max_scans: int | None = None,
    into: _Segmenter | None = None,
) -> _Segmenter:
    """Stream scans once and return the segmenter holding every window."""

    segmenter = into or _Segmenter(track, options)
    segmenter._prefix = prefix
    for count, (time_s, scan) in enumerate(scans):
        if max_scans is not None and count >= max_scans:
            break
        segmenter.add(time_s, scan)
    segmenter.flush()
    return segmenter


def evaluate_gnss_lidar_lever_arm(
    track: GnssTrackModel,
    windows: Sequence[OdometryWindow],
    options: GnssLidarRunOptions | None = None,
    *,
    reference_lever_arm: FloatArray | None = None,
) -> GnssLidarEvaluation:
    """Fit on train windows and collect jackknife, control, and holdout evidence."""

    opts = options or GnssLidarRunOptions()
    holdout = [w for position, w in enumerate(windows) if position % opts.holdout_every == 1]
    train = [w for position, w in enumerate(windows) if position % opts.holdout_every != 1]
    solver = GnssLeverArmSolver()
    result = solver.solve(track, train, opts.solver)
    if result.status != "converged" or result.lever_arm_m is None:
        return GnssLidarEvaluation(
            result=result,
            records=(),
            train_windows=len(train),
            holdout_windows=len(holdout),
            jackknife_fits=0,
            train_median_residual_m=None,
            holdout_median_residual_m=None,
            policy_status="fail",
            policy_reasons=("too few GNSS-covered windows to solve",),
        )
    jackknife, fits = _jackknife(solver, track, train, opts)
    epoch = float(track.times_s[0])
    rebased = GnssTrackModel(track.times_s - epoch, track.enu_m, track.sigma_m, track.max_gap_s)
    train_pairs = build_pairs(rebased, train, opts.solver, epoch=epoch)
    holdout_pairs = build_pairs(rebased, holdout, opts.solver, epoch=epoch)
    records = tuple(
        _record(
            dof.name,
            jackknife.get(dof.name),
            result,
            holdout_pairs,
            rebased,
            reference_lever_arm,
            opts,
        )
        for dof in result.dofs
    )
    train_residual = _median_residual(rebased, train_pairs, result, opts)
    holdout_residual = _median_residual(rebased, holdout_pairs, result, opts)
    status, reasons = _policy(records, train_residual, holdout_residual, len(holdout_pairs))
    return GnssLidarEvaluation(
        result=result,
        records=records,
        train_windows=len(train),
        holdout_windows=len(holdout),
        jackknife_fits=fits,
        train_median_residual_m=train_residual,
        holdout_median_residual_m=holdout_residual,
        policy_status=status,
        policy_reasons=reasons,
    )


def run_rtk_slam_lever_arm(
    sequences: Sequence[tuple[str | Path, str | Path]],
    calib_path: str | Path,
    options: GnssLidarRunOptions | None = None,
    *,
    topic: str = "/livox/points",
    max_scans: int | None = None,
    command: list[str] | None = None,
) -> GnssLidarLeverArmArtifact:
    """Run the evaluation on RTK-SLAM ``(bag directory, rtk.txt)`` sequences.

    Sequences of one rig are pooled; their windows keep separate alignments.
    """

    opts = options or GnssLidarRunOptions()
    if not sequences:
        raise ValueError("at least one RTK-SLAM sequence is required")
    first = load_rtk_track(sequences[0][1])
    tracks = [first] + [
        load_rtk_track(rtk, origin=first.origin_lat_lon_height) for _, rtk in sequences[1:]
    ]
    times = np.concatenate([item.times_s for item in tracks])
    order = np.argsort(times)
    if np.any(np.diff(times[order]) <= 0.0):
        raise ValueError("pooled RTK tracks overlap in time")
    track = GnssTrackModel(
        times[order],
        np.concatenate([item.enu_m for item in tracks])[order],
        np.concatenate([item.sigma_m for item in tracks])[order],
    )
    segmenter: _Segmenter | None = None
    digests: list[str] = []
    scope = ""
    for bag_dir, rtk in sequences:
        segmenter = collect_windows(
            track,
            iter_livox_scans(bag_dir, topic),
            opts,
            prefix=f"{Path(bag_dir).name}/",
            max_scans=max_scans,
            into=segmenter,
        )
        digest, scope = rtk_slam_input_digest(bag_dir, rtk, calib_path)
        digests.append(digest)
    assert segmenter is not None
    reference = rtk_slam_reference_lever_arm(calib_path)
    evaluation = evaluate_gnss_lidar_lever_arm(
        track, segmenter.windows, opts, reference_lever_arm=reference
    )
    return build_gnss_lidar_artifact(
        evaluation,
        opts,
        segments=GnssLidarSegmentSummary(
            scans_read=segmenter.scans_read,
            scans_in_gnss_coverage=segmenter.scans_covered,
            odometry_segments=segmenter.segments,
            unreliable_registrations=segmenter.unreliable,
            windows=len(segmenter.windows),
            gnss_epochs_used=len(track.times_s),
            gnss_epochs_rejected=sum(item.rejected_epochs for item in tracks),
        ),
        provenance=GnssLidarLeverArmProvenance(
            generator=__name__,
            generator_version=__version__,
            git_commit=git_commit(),
            command=command or [],
            dataset_family="rtk_slam",
            sequence_ids=[Path(bag_dir).name for bag_dir, _ in sequences],
            input_sha256=hashlib.sha256("".join(digests).encode("ascii")).hexdigest(),
            input_digest_scope=scope,
            dataset_license="see https://huggingface.co/datasets/Willyzw/rtk-slam-dataset",
        ),
        reference="calib.yaml: MID360 manual IMU position + CAD GNSS antenna offset",
        limitations=RTK_SLAM_LIMITATIONS,
    )


def build_gnss_lidar_artifact(
    evaluation: GnssLidarEvaluation,
    options: GnssLidarRunOptions,
    *,
    segments: GnssLidarSegmentSummary,
    provenance: GnssLidarLeverArmProvenance,
    reference: str | None,
    limitations: Sequence[str],
) -> GnssLidarLeverArmArtifact:
    """Assemble the schema-valid artifact."""

    result = evaluation.result
    solver = options.solver
    return GnssLidarLeverArmArtifact(
        solver_status=result.status,
        policy_status=evaluation.policy_status,
        policy_reasons=list(evaluation.policy_reasons),
        calibrated_dofs=[item.name for item in evaluation.records if item.status == "estimated"],
        lever_arm_frame="lidar",
        lever_arm_m=None
        if result.lever_arm_m is None
        else [float(value) for value in result.lever_arm_m],
        time_offset_s=result.time_offset_s,
        dofs=list(evaluation.records),
        options={
            "window_duration_s": options.window_duration_s,
            "holdout_every": options.holdout_every,
            "jackknife_groups": options.jackknife_groups,
            "lever_arm_control_m": options.lever_arm_control_m,
            "time_control_s": options.time_control_s,
            "detection_delta_chi2": options.detection_delta_chi2,
            "pair_steps": list(solver.pair_steps),
            "odometry_pose_sigma_m": solver.odometry_pose_sigma_m,
            "odometry_sigma_per_m": solver.odometry_sigma_per_m,
            "observable_translation_std_m": solver.observable_translation_std_m,
            "observable_time_offset_std_s": solver.observable_time_offset_std_s,
            "local_map_scans": options.odometry.local_map_scans,
            "voxel_size_m": options.odometry.voxel_size_m,
        },
        segments=segments,
        train_windows=evaluation.train_windows,
        holdout_windows=evaluation.holdout_windows,
        jackknife_fits=evaluation.jackknife_fits,
        train_median_residual_m=evaluation.train_median_residual_m,
        holdout_median_residual_m=evaluation.holdout_median_residual_m,
        reference=reference,
        limitations=list(limitations),
        provenance=provenance,
    )


def _jackknife(
    solver: GnssLeverArmSolver,
    track: GnssTrackModel,
    train: Sequence[OdometryWindow],
    opts: GnssLidarRunOptions,
) -> tuple[dict[str, float], int]:
    groups = min(opts.jackknife_groups, len(train))
    if groups < 3:
        return {}, 0
    assignment = np.array_split(np.arange(len(train)), groups)
    samples: dict[str, list[float]] = {name: [] for name in LEVER_ARM_DOFS}
    for members in assignment:
        excluded = set(members.tolist())
        subset = [window for index, window in enumerate(train) if index not in excluded]
        fit = solver.solve(track, subset, opts.solver)
        if fit.status != "converged":
            return {}, 0
        for dof in fit.dofs:
            samples[dof.name].append(dof.value)
    spread: dict[str, float] = {}
    for name, values in samples.items():
        if len(values) == groups:
            array = np.array(values)
            spread[name] = float(
                math.sqrt((groups - 1) / groups * np.sum((array - array.mean()) ** 2))
            )
    return spread, groups


def _record(
    name: Literal["x", "y", "z", "time_offset"],
    jackknife: float | None,
    result: LeverArmResult,
    holdout: Sequence[WindowPairs],
    track: GnssTrackModel,
    reference: FloatArray | None,
    opts: GnssLidarRunOptions,
) -> GnssLidarDofRecord:
    estimate = result.dof(name)
    reported = max(estimate.std, jackknife or 0.0)
    threshold = (
        opts.solver.observable_time_offset_std_s
        if name == "time_offset"
        else opts.solver.observable_translation_std_m
    )
    reference_value = None
    if reference is not None and name != "time_offset":
        reference_value = float(reference["xyz".index(name)])
    return GnssLidarDofRecord(
        name=name,
        unit="s" if name == "time_offset" else "m",
        value=estimate.value,
        std_analytic=estimate.std,
        std_jackknife=jackknife,
        std_reported=reported,
        status="estimated" if reported <= threshold else "unobservable",
        reference_value=reference_value,
        error_to_reference=None if reference_value is None else estimate.value - reference_value,
        known_bad_control=_control(name, result, holdout, track, opts),
    )


def _control(
    name: str,
    result: LeverArmResult,
    holdout: Sequence[WindowPairs],
    track: GnssTrackModel,
    opts: GnssLidarRunOptions,
) -> GnssLidarKnownBadControl | None:
    if result.lever_arm_m is None or len(holdout) < 2:
        return None
    lever = result.lever_arm_m.copy()
    offset = result.time_offset_s
    if name == "time_offset":
        amount, unit = opts.time_control_s, "s"
        moved_offset = offset + amount
        moved_lever = lever
    else:
        amount, unit = opts.lever_arm_control_m, "m"
        moved_lever = lever.copy()
        moved_lever["xyz".index(name)] += amount
        moved_offset = offset
    baseline = _chi2(track, holdout, lever, offset, result, opts)
    moved = _chi2(track, holdout, moved_lever, moved_offset, result, opts)
    delta = moved - baseline
    return GnssLidarKnownBadControl(
        amount=amount,
        unit=unit,  # type: ignore[arg-type]
        holdout_delta_chi2=float(delta),
        detected=bool(delta >= opts.detection_delta_chi2),
    )


def _chi2(
    track: GnssTrackModel,
    pairs: Sequence[WindowPairs],
    lever: FloatArray,
    offset: float,
    result: LeverArmResult,
    opts: GnssLidarRunOptions,
) -> float:
    residuals = window_residuals(track, pairs, lever, offset, opts.solver)
    variance = result.variance_factor or 1.0
    return float(np.sum(np.minimum(residuals**2 / variance, 25.0)))


def _median_residual(
    track: GnssTrackModel,
    pairs: Sequence[WindowPairs],
    result: LeverArmResult,
    opts: GnssLidarRunOptions,
) -> float | None:
    if not pairs or result.lever_arm_m is None:
        return None
    errors = window_residuals(
        track, pairs, result.lever_arm_m, result.time_offset_s, opts.solver, normalize=False
    ).reshape(-1, 3)
    return float(np.median(np.linalg.norm(errors, axis=1)))


def _policy(
    records: Sequence[GnssLidarDofRecord],
    train_residual: float | None,
    holdout_residual: float | None,
    holdout_windows: int,
) -> tuple[GnssLidarPolicyStatus, tuple[str, ...]]:
    if holdout_windows < 2 or holdout_residual is None:
        return "inconclusive", ("fewer than two held-out windows; holdout evidence is missing",)
    failures: list[str] = []
    warnings: list[str] = []
    if train_residual is not None and holdout_residual > max(3.0 * train_residual, 0.1):
        failures.append(
            f"held-out median residual {holdout_residual:.3f} m exceeds three times the "
            f"training residual {train_residual:.3f} m"
        )
    for record in records:
        control = record.known_bad_control
        if record.status == "estimated" and control is not None and not control.detected:
            warnings.append(
                f"{record.name} is reported as estimated but a known-bad {control.amount:g} "
                f"{control.unit} shift is not detected on held-out windows "
                f"(delta chi-square {control.holdout_delta_chi2:.1f})"
            )
    unobservable = [record.name for record in records if record.status == "unobservable"]
    notes = (
        ["not constrained by the data, so not calibrated: " + ", ".join(unobservable)]
        if unobservable
        else []
    )
    if failures:
        return "fail", tuple(failures + warnings + notes)
    if warnings:
        return "warn", tuple(warnings + notes)
    if any(name in {"x", "y", "z"} for name in unobservable):
        return "inconclusive", tuple(notes)
    reasons = ["every lever-arm component is estimated and held-out windows detect every control"]
    return "pass", tuple(reasons + notes)
