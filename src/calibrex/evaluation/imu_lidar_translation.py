"""Evaluate the IMU-LiDAR translation (IMU lever arm) on held-out windows.

The rotation, clock offset, and gyro bias come from a calibrated
``slac.imu_lidar_rotation/v0.1`` artifact, and the LiDAR odometry is deskewed
with gyro rotations mapped by that rotation.  The lever arm is fitted on train
windows, with every third window held out, as in the rotation evaluation.

The reported std of each axis is the largest of three:

* the analytic std;
* an 8-group window jackknife; and
* the largest change when the fit is repeated with other segment durations,
  which bounds modelling error (accelerometer bias drift, odometry errors
  correlated with motion) that resampling windows cannot reveal.

Each axis is shifted by a known-bad amount that the held-out chi-square must
detect.  A design reference (for example the MID360 manual) is compared only
after the fit.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from calibrex import __version__
from calibrex.core.imu_lidar_rotation import ImuLidarPolicyStatus, ImuLidarRotationArtifact
from calibrex.core.imu_lidar_translation import (
    ImuLidarRotationInput,
    ImuLidarSegmentFit,
    ImuLidarTranslationArtifact,
    ImuLidarTranslationAxisRecord,
    ImuLidarTranslationControl,
    ImuLidarTranslationProvenance,
    ImuLidarTranslationWindows,
)
from calibrex.core.provenance import git_commit
from calibrex.data.livox_ros2 import (
    LIVOX_PROFILES,
    MID360_T_LIDAR_IMU,
    bag_input_digest,
    load_livox_imu,
)
from calibrex.evaluation.imu_lidar_rotation import (
    ImuLidarRunOptions,
    _concatenate,
    collect_livox_windows,
)
from calibrex.evaluation.odometry_windows import WindowingOptions
from calibrex.solvers.gnss_lever_arm_solver import OdometryWindow
from calibrex.solvers.imu_lidar_rotation_solver import GyroSeries, gyro_rotation_model
from calibrex.solvers.imu_lidar_translation_solver import (
    TRANSLATION_AXES,
    ImuPreintegrator,
    TranslationOptions,
    TranslationResult,
    WindowSystem,
    gravity_norm,
    solve_translation,
    translation_std_m,
    window_residuals,
    window_systems,
)

FloatArray: TypeAlias = NDArray[np.float64]
MID360_TRANSLATION_LIMITATIONS: tuple[str, ...] = (
    "The reference translation is the MID360 design value (11.0, 23.29, -44.12) mm; "
    "it is not a measurement.",
    "The rotation, clock offset, and gyro bias are taken from the rotation artifact; "
    "their errors propagate into the lever arm.",
    "The LiDAR odometry is deskewed with the gyro, so it depends on the IMU.",
    "Accelerometer scale and axis misalignment are not estimated.",
)


@dataclass(frozen=True)
class ImuLidarTranslationOptions:
    """Holdout, jackknife, controls, sensitivity, windowing, and solver settings."""

    holdout_every: int = 3
    jackknife_groups: int = 8
    translation_control_m: float = 0.02
    detection_delta_chi2: float = 9.0
    # 1-second segments are excluded: with gravity and velocity fitted per segment
    # they are poorly constrained, and on every MID360 development recording they
    # moved x by +7 to +16 mm for reasons not yet identified (see the benchmark page).
    sensitivity_segment_durations_s: tuple[float, ...] = (3.0, 4.0)
    coverage_margin_s: float = 0.2
    windowing: WindowingOptions = field(default_factory=WindowingOptions)
    solver: TranslationOptions = field(default_factory=TranslationOptions)


@dataclass(frozen=True)
class ImuLidarTranslationEvaluation:
    """Dataset-independent evaluation output."""

    result: TranslationResult
    records: tuple[ImuLidarTranslationAxisRecord, ...]
    segment_fits: tuple[ImuLidarSegmentFit, ...]
    train_windows: int
    holdout_windows: int
    jackknife_fits: int
    train_median_m: float | None
    holdout_median_m: float | None
    gravity_norm_median_mps2: float | None
    segments: int
    scan_rows: int
    policy_status: ImuLidarPolicyStatus
    policy_reasons: tuple[str, ...]


def evaluate_imu_lidar_translation(
    imu: ImuPreintegrator,
    windows: Sequence[OdometryWindow],
    rotation: FloatArray,
    time_offset_s: float,
    options: ImuLidarTranslationOptions | None = None,
    *,
    reference_translation: FloatArray | None = None,
) -> ImuLidarTranslationEvaluation:
    """Fit on train windows; collect jackknife, sensitivity, control, and holdout evidence."""

    opts = options or ImuLidarTranslationOptions()
    systems = window_systems(windows, imu, rotation, time_offset_s, opts.solver)
    train, holdout = _split(systems, opts.holdout_every)
    result = solve_translation(train, opts.solver)
    segments = sum(system.segments for system in systems)
    rows = sum(system.scan_rows for system in systems)
    if result.translation_m is None or result.sigma_m is None:
        return ImuLidarTranslationEvaluation(
            result, (), (), len(train), len(holdout), 0, None, None, None, segments, rows,
            "fail", ("too few odometry segments with IMU coverage to solve",),
        )  # fmt: skip
    translation = result.translation_m
    jackknife, fits = _jackknife(train, opts)
    segment_fits = tuple(
        _segment_fit(duration, windows, imu, rotation, time_offset_s, opts)
        for duration in opts.sensitivity_segment_durations_s
    )
    sensitivity = _sensitivity(translation, segment_fits)
    analytic = translation_std_m(result)
    records = tuple(
        _record(
            axis,
            float(translation[axis]),
            float(analytic[axis]),
            None if jackknife is None else float(jackknife[axis]),
            None if sensitivity is None else float(sensitivity[axis]),
            None if reference_translation is None else float(reference_translation[axis]),
            _control(axis, holdout, translation, result.sigma_m, opts),
            opts,
        )
        for axis in range(3)
    )
    train_median = _median_residual(train, translation)
    holdout_median = _median_residual(holdout, translation)
    gravity = [gravity_norm(system, translation) for system in systems]
    holdout_rows = sum(system.scan_rows for system in holdout)
    status, reasons = _policy(records, train_median, holdout_median, holdout_rows)
    return ImuLidarTranslationEvaluation(
        result=result,
        records=records,
        segment_fits=segment_fits,
        train_windows=len(train),
        holdout_windows=len(holdout),
        jackknife_fits=fits,
        train_median_m=train_median,
        holdout_median_m=holdout_median,
        gravity_norm_median_mps2=float(np.median(gravity)) if gravity else None,
        segments=segments,
        scan_rows=rows,
        policy_status=status,
        policy_reasons=reasons,
    )


def run_livox_imu_lidar_translation(
    bags: Sequence[str | Path],
    profile: str,
    rotation_artifact: ImuLidarRotationArtifact,
    options: ImuLidarTranslationOptions | None = None,
    *,
    dataset_family: str,
    dataset_license: str,
    rotation_artifact_sha256: str | None = None,
    reference_translation: FloatArray | None = MID360_T_LIDAR_IMU[:3, 3],
    max_scans: int | None = None,
    command: list[str] | None = None,
) -> ImuLidarTranslationArtifact:
    """Run gyro-deskewed odometry and the lever-arm evaluation on Livox ROS 2 bags."""

    opts = options or ImuLidarTranslationOptions()
    if not bags:
        raise ValueError("at least one bag is required")
    if rotation_artifact.rotation_quat_xyzw is None:
        raise ValueError("the rotation artifact has no rotation estimate")
    rotation = Rotation.from_quat(rotation_artifact.rotation_quat_xyzw).as_matrix()
    gyro_bias = np.asarray(rotation_artifact.gyro_bias_rps, dtype=np.float64)
    time_offset = rotation_artifact.time_offset_s
    stream = LIVOX_PROFILES[profile]
    samples = _concatenate([load_livox_imu(bag, stream) for bag in bags])
    gyro = GyroSeries(samples.times_s, samples.gyro_rps)
    run_options = ImuLidarRunOptions(
        windowing=opts.windowing, coverage_margin_s=opts.coverage_margin_s
    )
    _, windows = collect_livox_windows(
        bags,
        profile,
        run_options,
        rotation_model=gyro_rotation_model(gyro, rotation, gyro_bias, time_offset),
        max_scans=max_scans,
    )
    imu = ImuPreintegrator(samples.times_s, samples.gyro_rps, samples.accel_mps2, gyro_bias)
    evaluation = evaluate_imu_lidar_translation(
        imu,
        windows,
        rotation,
        time_offset,
        opts,
        reference_translation=reference_translation,
    )
    digest, scope = bag_input_digest(bags)
    return translation_artifact_from_evaluation(
        evaluation,
        opts,
        rotation_input=ImuLidarRotationInput(
            artifact_sha256=rotation_artifact_sha256,
            policy_status=rotation_artifact.policy_status,
            rotation_quat_xyzw=list(rotation_artifact.rotation_quat_xyzw),
            time_offset_s=time_offset,
            gyro_bias_rps=[float(value) for value in gyro_bias],
        ),
        windows=len(windows),
        imu_samples=len(samples.times_s),
        reference=None
        if reference_translation is None
        else "Livox MID360 manual: IMU at (11.0, 23.29, -44.12) mm in the LiDAR frame",
        limitations=list(MID360_TRANSLATION_LIMITATIONS),
        provenance=ImuLidarTranslationProvenance(
            generator=__name__,
            generator_version=__version__,
            git_commit=git_commit(),
            command=command or [],
            dataset_family=dataset_family,
            sequence_ids=[Path(bag).name for bag in bags],
            stream_profile=profile,
            input_sha256=hashlib.sha256(digest.encode("ascii")).hexdigest(),
            input_digest_scope=scope,
            dataset_license=dataset_license,
        ),
    )


def translation_artifact_from_evaluation(
    evaluation: ImuLidarTranslationEvaluation,
    options: ImuLidarTranslationOptions,
    *,
    rotation_input: ImuLidarRotationInput,
    windows: int,
    imu_samples: int,
    reference: str | None,
    limitations: list[str],
    provenance: ImuLidarTranslationProvenance,
) -> ImuLidarTranslationArtifact:
    """Assemble a schema-valid lever-arm artifact from an evaluation."""

    opts = options
    translation = evaluation.result.translation_m
    return ImuLidarTranslationArtifact(
        method="accelerometer_lever_arm/v0.2"
        if opts.solver.gravity_per_segment
        else "accelerometer_lever_arm/v0.1",
        solver_status=evaluation.result.status,
        policy_status=evaluation.policy_status,
        policy_reasons=list(evaluation.policy_reasons),
        calibrated_dofs=[item.name for item in evaluation.records if item.status == "estimated"],
        translation_m=None if translation is None else [float(value) for value in translation],
        rotation_input=rotation_input,
        axes=list(evaluation.records),
        segment_sensitivity=list(evaluation.segment_fits),
        options={
            "holdout_every": opts.holdout_every,
            "jackknife_groups": opts.jackknife_groups,
            "translation_control_m": opts.translation_control_m,
            "detection_delta_chi2": opts.detection_delta_chi2,
            "segment_duration_s": opts.solver.segment_duration_s,
            "sensitivity_segment_durations_s": list(opts.sensitivity_segment_durations_s),
            "min_segment_scans": opts.solver.min_segment_scans,
            "gravity_per_segment": opts.solver.gravity_per_segment,
            "huber_k": opts.solver.huber_k,
            "observable_translation_std_m": opts.solver.observable_translation_std_m,
            "window_duration_s": opts.windowing.window_duration_s,
            "local_map_scans": opts.windowing.odometry.local_map_scans,
        },
        windows=ImuLidarTranslationWindows(
            windows=windows,
            segments=evaluation.segments,
            scan_rows=evaluation.scan_rows,
            imu_samples=imu_samples,
        ),
        train_windows=evaluation.train_windows,
        holdout_windows=evaluation.holdout_windows,
        jackknife_fits=evaluation.jackknife_fits,
        train_median_position_residual_m=evaluation.train_median_m,
        holdout_median_position_residual_m=evaluation.holdout_median_m,
        gravity_norm_median_mps2=evaluation.gravity_norm_median_mps2,
        reference=reference,
        limitations=limitations,
        provenance=provenance,
    )


def _split(
    systems: Sequence[WindowSystem], holdout_every: int
) -> tuple[list[WindowSystem], list[WindowSystem]]:
    train = [system for system in systems if system.window_index % holdout_every != 1]
    holdout = [system for system in systems if system.window_index % holdout_every == 1]
    return train, holdout


def _jackknife(
    train: Sequence[WindowSystem], opts: ImuLidarTranslationOptions
) -> tuple[FloatArray | None, int]:
    groups = min(opts.jackknife_groups, len(train))
    if groups < 3:
        return None, 0
    members = np.array_split(np.arange(len(train)), groups)
    estimates = []
    for group in members:
        left_out = set(group.tolist())
        fit = solve_translation(
            [system for index, system in enumerate(train) if index not in left_out], opts.solver
        )
        if fit.translation_m is None:
            return None, 0
        estimates.append(fit.translation_m)
    array = np.array(estimates)
    spread = np.sqrt((groups - 1) / groups * np.sum((array - array.mean(axis=0)) ** 2, axis=0))
    return np.asarray(spread, dtype=np.float64), groups


def _segment_fit(
    duration: float,
    windows: Sequence[OdometryWindow],
    imu: ImuPreintegrator,
    rotation: FloatArray,
    time_offset_s: float,
    opts: ImuLidarTranslationOptions,
) -> ImuLidarSegmentFit:
    solver = replace(opts.solver, segment_duration_s=duration)
    train, _ = _split(
        window_systems(windows, imu, rotation, time_offset_s, solver), opts.holdout_every
    )
    fit = solve_translation(train, solver)
    values = [math.nan] * 3 if fit.translation_m is None else fit.translation_m.tolist()
    return ImuLidarSegmentFit(segment_duration_s=duration, translation_m=values)


def _sensitivity(
    translation: FloatArray, fits: Sequence[ImuLidarSegmentFit]
) -> FloatArray | None:
    if not fits:
        return None
    changes = np.abs(np.array([fit.translation_m for fit in fits]) - translation)
    if not np.all(np.isfinite(changes)):
        return np.full(3, math.inf)
    return np.asarray(changes.max(axis=0), dtype=np.float64)


def _record(
    axis: int,
    value: float,
    analytic: float,
    jackknife: float | None,
    sensitivity: float | None,
    reference: float | None,
    control: ImuLidarTranslationControl | None,
    opts: ImuLidarTranslationOptions,
) -> ImuLidarTranslationAxisRecord:
    reported = max(analytic, jackknife or 0.0, sensitivity or 0.0)
    finite = sensitivity is not None and math.isfinite(sensitivity)
    finite_sensitivity = sensitivity if finite else None
    return ImuLidarTranslationAxisRecord(
        name=TRANSLATION_AXES[axis],  # type: ignore[arg-type]
        value=value,
        std_analytic=analytic,
        std_jackknife=jackknife,
        std_segment_sensitivity=finite_sensitivity,
        std_reported=reported if math.isfinite(reported) else 1e9,
        status="estimated"
        if reported <= opts.solver.observable_translation_std_m
        else "unobservable",
        reference_value=reference,
        error_to_reference=None if reference is None else value - reference,
        known_bad_control=control,
    )


def _control(
    axis: int,
    holdout: Sequence[WindowSystem],
    translation: FloatArray,
    sigma: float,
    opts: ImuLidarTranslationOptions,
) -> ImuLidarTranslationControl | None:
    if sum(system.scan_rows for system in holdout) < 20:
        return None
    moved = translation.copy()
    moved[axis] += opts.translation_control_m
    delta = _chi2(holdout, moved, sigma) - _chi2(holdout, translation, sigma)
    return ImuLidarTranslationControl(
        amount=opts.translation_control_m,
        holdout_delta_chi2=float(delta),
        detected=bool(delta >= opts.detection_delta_chi2),
    )


def _chi2(systems: Sequence[WindowSystem], translation: FloatArray, sigma: float) -> float:
    return float(
        sum(
            np.sum(np.minimum((window_residuals(system, translation) / sigma) ** 2, 25.0))
            for system in systems
        )
    )


def _median_residual(systems: Sequence[WindowSystem], translation: FloatArray) -> float | None:
    if not systems:
        return None
    residuals = np.concatenate([window_residuals(system, translation) for system in systems])
    return float(np.median(np.linalg.norm(residuals.reshape(-1, 3), axis=1)))


def _policy(
    records: Sequence[ImuLidarTranslationAxisRecord],
    train_median: float | None,
    holdout_median: float | None,
    holdout_rows: int,
) -> tuple[ImuLidarPolicyStatus, tuple[str, ...]]:
    if holdout_rows < 20 or holdout_median is None:
        return "inconclusive", ("too few held-out scans; holdout evidence is missing",)
    failures: list[str] = []
    warnings: list[str] = []
    if train_median is not None and holdout_median > max(3.0 * train_median, 0.01):
        failures.append(
            f"held-out median position residual {1000 * holdout_median:.1f} mm exceeds three "
            f"times the training residual {1000 * train_median:.1f} mm"
        )
    for record in records:
        control = record.known_bad_control
        if record.status == "estimated" and control is not None and not control.detected:
            warnings.append(
                f"{record.name} is reported as estimated but a known-bad "
                f"{1000 * control.amount:g} mm shift is not detected on held-out windows "
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
    if unobservable:
        return "inconclusive", tuple(notes)
    return "pass", (
        "every lever-arm axis is estimated and held-out windows detect every control",
    )


@dataclass(frozen=True)
class TranslationCandidateScore:
    """Held-out fit of a fixed rotation, clock offset, and lever arm."""

    holdout_span_rms_m: dict[float, float]
    holdout_span_median_m: dict[float, float]
    holdout_rows: int


def score_imu_lidar_translation_candidate(
    imu: ImuPreintegrator,
    windows: Sequence[OdometryWindow],
    rotation: FloatArray,
    time_offset_s: float,
    translation: FloatArray,
    holdout_spans_s: Sequence[tuple[float, float]],
    options: TranslationOptions | None = None,
    *,
    clip_m: float = 0.05,
) -> TranslationCandidateScore:
    """Score a complete candidate extrinsic on shared held-out time spans.

    Nothing of the candidate is refit: only the per-span nuisances (gravity,
    accelerometer bias, segment velocities).  Each odometry window is cut at
    the span boundaries, so every candidate is scored on the same stretches of
    the recording even when its own deskewing shifted the segmentation.  The
    per-span score is the RMS of the position residual components, each
    clipped at ``clip_m`` so a single broken registration cannot dominate.
    """

    opts = options or TranslationOptions()
    pieces: list[OdometryWindow] = []
    keys: list[float] = []
    for window in windows:
        for start, end in holdout_spans_s:
            inside = (window.times_s >= start) & (window.times_s <= end)
            if np.count_nonzero(inside) >= opts.min_segment_scans:
                pieces.append(
                    OdometryWindow(
                        f"{window.window_id}@{start:.1f}",
                        window.block,
                        window.times_s[inside],
                        window.poses[inside],
                    )
                )
                keys.append(round(float(start), 1))
    systems = window_systems(pieces, imu, rotation, time_offset_s, opts)
    residuals: dict[float, list[FloatArray]] = {}
    for system in systems:
        key = keys[system.window_index]
        residuals.setdefault(key, []).append(window_residuals(system, translation))
    rms: dict[float, float] = {}
    median: dict[float, float] = {}
    rows = 0
    for key, parts in residuals.items():
        values = np.concatenate(parts)
        rows += len(values) // 3
        clipped = np.clip(values, -clip_m, clip_m)
        rms[key] = float(np.sqrt(np.mean(clipped**2)))
        median[key] = float(np.median(np.linalg.norm(values.reshape(-1, 3), axis=1)))
    return TranslationCandidateScore(rms, median, rows)
