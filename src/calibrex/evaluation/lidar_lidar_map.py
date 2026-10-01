"""Evaluate a LiDAR-LiDAR extrinsic from map registration on held-out time blocks.

The reference LiDAR's odometry places its scans in one frame.  Every
``sample_every_scans``-th target scan becomes a sample, registered to the
reference scans within ``map_half_window_scans`` of its time (see
:mod:`calibrex.solvers.lidar_lidar_map_solver`).  Samples are grouped into
``block_duration_s`` blocks and every third block is held out.  The reported
std of each DoF is the larger of the analytic std and an 8-group block
jackknife.  Shifting each rotation axis by 1 deg and each translation axis by
5 cm must raise the held-out chi-square by at least 9.  A reference transform,
if given, is compared only after the fit, both as a per-DoF difference and as
the held-out chi-square it would cost.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from calibrex import __version__
from calibrex.core.lidar_lidar_extrinsic import (
    LidarLidarControl,
    LidarLidarDofRecord,
    LidarLidarExtrinsicArtifact,
    LidarLidarPolicyStatus,
    LidarLidarProvenance,
    LidarLidarSamples,
    LidarLidarTransform,
)
from calibrex.core.provenance import git_commit
from calibrex.data.livox_ros2 import LivoxStreamProfile, bag_input_digest, iter_livox_points
from calibrex.solvers.lidar_lidar_map_solver import (
    EXTRINSIC_DOFS,
    MapExtrinsicResult,
    MapSample,
    MapSampleOptions,
    chi_square,
    perturb_extrinsic,
    solve_map_extrinsic,
)
from calibrex.solvers.scan_to_scan_odometry import (
    IncrementalScanOdometry,
    ScanOdometryOptions,
    preprocess_scan,
)

FloatArray: TypeAlias = NDArray[np.float64]
PointTimeEncoding: TypeAlias = Literal["offset_s", "absolute_ns", "absolute_s"]
_ROTATIONS = ("roll", "pitch", "yaw")
LIMITATIONS: tuple[str, ...] = (
    "The reference LiDAR's odometry supplies the map; its drift over the map window "
    "adds noise to every sample.",
    "The clock offset between the two LiDARs is not estimated; both clouds are "
    "matched at their header times.",
    "The reference transform, if given, is a comparison, not ground truth.",
)


@dataclass(frozen=True)
class LidarLidarMapOptions:
    """Sampling, blocking, evidence, and observability settings."""

    sample_every_scans: int = 10
    map_half_window_scans: int = 5
    max_time_gap_s: float = 0.06
    min_target_points: int = 300
    block_duration_s: float = 10.0
    holdout_every: int = 3
    jackknife_groups: int = 8
    rotation_control_deg: float = 1.0
    translation_control_m: float = 0.05
    detection_delta_chi2: float = 9.0
    observable_rotation_std_deg: float = 0.1
    observable_translation_std_m: float = 0.02
    holdout_residual_ratio: float = 2.0
    preprocess: ScanOdometryOptions = field(
        default_factory=lambda: ScanOdometryOptions(
            voxel_size_m=0.2, min_range_m=1.0, max_range_m=40.0
        )
    )
    odometry: ScanOdometryOptions = field(
        default_factory=lambda: ScanOdometryOptions(
            voxel_size_m=0.3, min_range_m=1.0, max_range_m=60.0, local_map_scans=5
        )
    )
    solver: MapSampleOptions = field(default_factory=lambda: MapSampleOptions(iterations=30))


@dataclass(frozen=True)
class LidarLidarEvaluation:
    """Dataset-independent evaluation output."""

    result: MapExtrinsicResult
    records: tuple[LidarLidarDofRecord, ...]
    train_blocks: tuple[int, ...]
    holdout_blocks: tuple[int, ...]
    jackknife_fits: int
    train_samples: int
    holdout_samples: int
    holdout_correspondences: int
    holdout_median_m: float | None
    reference_delta_chi2: float | None
    policy_status: LidarLidarPolicyStatus
    policy_reasons: tuple[str, ...]


def evaluate_lidar_lidar_map(
    samples: Sequence[MapSample],
    initial: FloatArray,
    options: LidarLidarMapOptions | None = None,
    *,
    reference_transform: FloatArray | None = None,
) -> LidarLidarEvaluation:
    """Fit on train blocks; collect jackknife, control, and holdout evidence."""

    opts = options or LidarLidarMapOptions()
    blocks = sorted({sample.block for sample in samples})
    holdout_blocks = tuple(
        block for position, block in enumerate(blocks) if position % opts.holdout_every == 1
    )
    train_blocks = tuple(block for block in blocks if block not in holdout_blocks)
    train = [sample for sample in samples if sample.block in train_blocks]
    holdout = [sample for sample in samples if sample.block in holdout_blocks]
    result = solve_map_extrinsic(train, initial, opts.solver)
    if result.transform is None or result.covariance is None or result.sigma_m is None:
        return LidarLidarEvaluation(
            result, (), train_blocks, holdout_blocks, 0, len(train), len(holdout), 0, None, None,
            "fail", ("too few samples with map correspondences to solve",),
        )  # fmt: skip
    transform = result.transform
    jackknife, fits = _jackknife(train, train_blocks, transform, opts)
    base, holdout_median, holdout_count = chi_square(
        holdout, transform, result.sigma_m, opts.solver
    )
    analytic = np.sqrt(np.clip(np.diag(result.covariance), 0.0, None))
    records = tuple(
        _record(
            dof,
            index,
            transform,
            float(analytic[index]),
            None if jackknife is None else float(jackknife[index]),
            _control(dof, holdout, transform, result.sigma_m, base, opts),
            reference_transform,
            opts,
        )
        for index, dof in enumerate(EXTRINSIC_DOFS)
    )
    reference_delta = None
    if reference_transform is not None and holdout:
        moved, _, _ = chi_square(holdout, reference_transform, result.sigma_m, opts.solver)
        reference_delta = moved - base
    train_median = _median_residual(train, transform, result.sigma_m, opts)
    status, reasons = _policy(records, train_median, holdout_median, holdout_count, opts)
    return LidarLidarEvaluation(
        result=result,
        records=records,
        train_blocks=train_blocks,
        holdout_blocks=holdout_blocks,
        jackknife_fits=fits,
        train_samples=len(train),
        holdout_samples=len(holdout),
        holdout_correspondences=holdout_count,
        holdout_median_m=holdout_median if holdout_count else None,
        reference_delta_chi2=reference_delta,
        policy_status=status,
        policy_reasons=reasons,
    )


def collect_map_samples(
    bag: str | Path,
    *,
    reference_topic: str,
    target_topic: str,
    point_time_field: str | None,
    options: LidarLidarMapOptions | None = None,
    max_seconds: float | None = None,
    point_time_encoding: PointTimeEncoding = "offset_s",
    target_point_time_field: str | None = None,
    target_point_time_encoding: PointTimeEncoding | None = None,
) -> tuple[list[MapSample], int]:
    """Run the reference odometry, then build one sample per strided target scan.

    ``point_time_field``/``point_time_encoding`` describe the reference LiDAR's
    per-point time; the target uses the same unless ``target_point_time_field``
    is given (LiDARs of different models).
    """

    opts = options or LidarLidarMapOptions()
    same_target_time = target_point_time_field is None and target_point_time_encoding is None

    def profile(topic: str, target: bool = False) -> LivoxStreamProfile:
        if target and not same_target_time:
            return LivoxStreamProfile(
                "lidar",
                topic,
                "",
                target_point_time_field,
                target_point_time_encoding or point_time_encoding,
                "mps2",
            )
        return LivoxStreamProfile("lidar", topic, "", point_time_field, point_time_encoding, "mps2")

    odometry = IncrementalScanOdometry(opts.odometry)
    scans: list[NDArray[np.float32]] = []
    times: list[float] = []
    first: float | None = None
    reference_scan_stream = iter_livox_points(
        bag, profile(reference_topic), max_seconds=max_seconds
    )
    for time_s, points, offsets in reference_scan_stream:
        first = time_s if first is None else first
        finite = np.isfinite(points).all(axis=1)
        odometry.add(points[finite], time_s, None if offsets is None else offsets[finite])
        pose = odometry.poses[-1]
        local = preprocess_scan(points[finite], opts.preprocess)
        scans.append((local @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32))
        times.append(time_s)
    if first is None:
        raise ValueError(f"{bag} has no scans on {reference_topic}")
    poses = odometry.poses
    reference_times = np.asarray(times)
    half = opts.map_half_window_scans
    samples: list[MapSample] = []
    target_scans = iter_livox_points(
        bag, profile(target_topic, target=True), max_seconds=max_seconds
    )
    for index, (time_s, points, _) in enumerate(target_scans):
        if index % opts.sample_every_scans:
            continue
        nearest = int(np.argmin(np.abs(reference_times - time_s)))
        if abs(reference_times[nearest] - time_s) > opts.max_time_gap_s:
            continue
        if nearest < half or nearest >= len(scans) - half:
            continue
        target = preprocess_scan(points[np.isfinite(points).all(axis=1)], opts.preprocess)
        if len(target) < opts.min_target_points:
            continue
        samples.append(
            MapSample(
                time_s=float(time_s),
                block=int((time_s - first) // opts.block_duration_s),
                reference_pose=np.asarray(poses[nearest], dtype=np.float64),
                target_points=target,
                map_points=np.concatenate(scans[nearest - half : nearest + half + 1]).astype(
                    np.float64
                ),
            )
        )
    return samples, len(scans)


def run_ros2_lidar_lidar_map(
    bag: str | Path,
    *,
    reference_topic: str,
    target_topic: str,
    initial: FloatArray,
    point_time_field: str | None,
    dataset_family: str,
    dataset_license: str,
    reference_transform: FloatArray | None = None,
    reference: str | None = None,
    options: LidarLidarMapOptions | None = None,
    max_seconds: float | None = None,
    point_time_encoding: PointTimeEncoding = "offset_s",
    target_point_time_field: str | None = None,
    target_point_time_encoding: PointTimeEncoding | None = None,
    command: list[str] | None = None,
) -> LidarLidarExtrinsicArtifact:
    """Calibrate ``T_reference_target`` for two LiDARs recorded in one ROS 2 bag."""

    opts = options or LidarLidarMapOptions()
    samples, reference_scans = collect_map_samples(
        bag,
        reference_topic=reference_topic,
        target_topic=target_topic,
        point_time_field=point_time_field,
        options=opts,
        max_seconds=max_seconds,
        point_time_encoding=point_time_encoding,
        target_point_time_field=target_point_time_field,
        target_point_time_encoding=target_point_time_encoding,
    )
    evaluation = evaluate_lidar_lidar_map(
        samples, initial, opts, reference_transform=reference_transform
    )
    digest, scope = bag_input_digest([bag])
    result = evaluation.result
    transform = None
    if result.transform is not None:
        transform = LidarLidarTransform(
            parent_frame=reference_topic,
            child_frame=target_topic,
            translation_m=[float(value) for value in result.transform[:3, 3]],
            rotation_quat_xyzw=[
                float(value) for value in Rotation.from_matrix(result.transform[:3, :3]).as_quat()
            ],
        )
    return LidarLidarExtrinsicArtifact(
        solver_status=result.status,
        policy_status=evaluation.policy_status,
        policy_reasons=list(evaluation.policy_reasons),
        calibrated_dofs=[item.name for item in evaluation.records if item.status == "estimated"],
        transform=transform,
        dofs=list(evaluation.records),
        residual_sigma_m=result.sigma_m,
        holdout_median_residual_m=evaluation.holdout_median_m,
        reference_holdout_delta_chi2=evaluation.reference_delta_chi2,
        samples=LidarLidarSamples(
            reference_scans=reference_scans,
            samples=len(samples),
            train_samples=evaluation.train_samples,
            holdout_samples=evaluation.holdout_samples,
            train_correspondences=result.correspondences,
            holdout_correspondences=evaluation.holdout_correspondences,
        ),
        train_blocks=list(evaluation.train_blocks),
        holdout_blocks=list(evaluation.holdout_blocks),
        jackknife_fits=evaluation.jackknife_fits,
        options={
            "sample_every_scans": opts.sample_every_scans,
            "map_half_window_scans": opts.map_half_window_scans,
            "block_duration_s": opts.block_duration_s,
            "holdout_every": opts.holdout_every,
            "jackknife_groups": opts.jackknife_groups,
            "rotation_control_deg": opts.rotation_control_deg,
            "translation_control_m": opts.translation_control_m,
            "detection_delta_chi2": opts.detection_delta_chi2,
            "observable_rotation_std_deg": opts.observable_rotation_std_deg,
            "observable_translation_std_m": opts.observable_translation_std_m,
            "voxel_size_m": opts.preprocess.voxel_size_m,
            "max_correspondence_m": opts.solver.max_correspondence_m,
            "huber_m": opts.solver.huber_m,
            "iterations": opts.solver.iterations,
        },
        reference=reference,
        limitations=list(LIMITATIONS),
        provenance=LidarLidarProvenance(
            generator=__name__,
            generator_version=__version__,
            git_commit=git_commit(),
            command=command or [],
            dataset_family=dataset_family,
            sequence_ids=[Path(bag).name],
            reference_topic=reference_topic,
            target_topic=target_topic,
            input_sha256=hashlib.sha256(digest.encode("ascii")).hexdigest(),
            input_digest_scope=scope,
            dataset_license=dataset_license,
        ),
    )


def _jackknife(
    train: Sequence[MapSample],
    train_blocks: Sequence[int],
    transform: FloatArray,
    opts: LidarLidarMapOptions,
) -> tuple[FloatArray | None, int]:
    groups = min(opts.jackknife_groups, len(train_blocks))
    if groups < 3:
        return None, 0
    deltas = []
    for members in np.array_split(np.asarray(train_blocks), groups):
        left_out = set(members.tolist())
        fit = solve_map_extrinsic(
            [sample for sample in train if sample.block not in left_out], transform, opts.solver
        )
        if fit.transform is None:
            return None, 0
        rotation = Rotation.from_matrix(fit.transform[:3, :3] @ transform[:3, :3].T).as_rotvec()
        deltas.append(np.r_[rotation, fit.transform[:3, 3] - transform[:3, 3]])
    array = np.asarray(deltas)
    spread = np.sqrt((groups - 1) / groups * np.sum((array - array.mean(axis=0)) ** 2, axis=0))
    return np.asarray(spread, dtype=np.float64), groups


def _record(
    dof: str,
    index: int,
    transform: FloatArray,
    analytic: float,
    jackknife: float | None,
    control: LidarLidarControl | None,
    reference_transform: FloatArray | None,
    opts: LidarLidarMapOptions,
) -> LidarLidarDofRecord:
    rotation = dof in _ROTATIONS
    convert = math.degrees if rotation else (lambda value: value)
    reported = max(analytic, jackknife or 0.0)
    threshold = (
        math.radians(opts.observable_rotation_std_deg)
        if rotation
        else opts.observable_translation_std_m
    )
    if rotation:
        value = float(Rotation.from_matrix(transform[:3, :3]).as_euler("xyz")[index])
    else:
        value = float(transform[index - 3, 3])
    reference_value = None
    error = None
    if reference_transform is not None:
        if rotation:
            reference_value = float(
                Rotation.from_matrix(reference_transform[:3, :3]).as_euler("xyz")[index]
            )
            local = Rotation.from_matrix(
                transform[:3, :3] @ reference_transform[:3, :3].T
            ).as_rotvec()
            error = float(local[index])
        else:
            reference_value = float(reference_transform[index - 3, 3])
            error = value - reference_value
    return LidarLidarDofRecord(
        name=dof,  # type: ignore[arg-type]
        unit="deg" if rotation else "m",
        value=convert(value),
        std_analytic=convert(analytic),
        std_jackknife=None if jackknife is None else convert(jackknife),
        std_reported=convert(reported),
        status="estimated" if reported <= threshold else "unobservable",
        reference_value=None if reference_value is None else convert(reference_value),
        error_to_reference=None if error is None else convert(error),
        known_bad_control=control,
    )


def _control(
    dof: str,
    holdout: Sequence[MapSample],
    transform: FloatArray,
    sigma: float,
    base: float,
    opts: LidarLidarMapOptions,
) -> LidarLidarControl | None:
    if not holdout:
        return None
    rotation = dof in _ROTATIONS
    amount = opts.rotation_control_deg if rotation else opts.translation_control_m
    moved, _, _ = chi_square(
        holdout,
        perturb_extrinsic(transform, dof, math.radians(amount) if rotation else amount),
        sigma,
        opts.solver,
    )
    delta = moved - base
    return LidarLidarControl(
        amount=amount,
        unit="deg" if rotation else "m",
        holdout_delta_chi2=float(delta),
        detected=bool(delta >= opts.detection_delta_chi2),
    )


def _median_residual(
    samples: Sequence[MapSample], transform: FloatArray, sigma: float, opts: LidarLidarMapOptions
) -> float | None:
    if not samples:
        return None
    _, median, count = chi_square(samples, transform, sigma, opts.solver)
    return median if count else None


def _policy(
    records: Sequence[LidarLidarDofRecord],
    train_median: float | None,
    holdout_median: float,
    holdout_count: int,
    opts: LidarLidarMapOptions,
) -> tuple[LidarLidarPolicyStatus, tuple[str, ...]]:
    if holdout_count < 100 or not math.isfinite(holdout_median):
        return "inconclusive", ("too few held-out correspondences; holdout evidence is missing",)
    failures: list[str] = []
    warnings: list[str] = []
    if train_median is not None and holdout_median > opts.holdout_residual_ratio * train_median:
        failures.append(
            f"held-out median residual {1000 * holdout_median:.1f} mm exceeds "
            f"{opts.holdout_residual_ratio:g} times the training residual "
            f"{1000 * train_median:.1f} mm"
        )
    for record in records:
        control = record.known_bad_control
        if record.status == "estimated" and control is not None and not control.detected:
            warnings.append(
                f"{record.name} is reported as estimated but a known-bad {control.amount:g} "
                f"{control.unit} shift is not detected on held-out blocks "
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
    return "pass", ("every DoF is estimated and held-out blocks detect every control",)


def read_transform(path: str | Path) -> FloatArray:
    """Read a 4x4 transform from YAML/JSON (``matrix``, ``T``, or ``transform``) or OpenCV YAML.

    OpenCV files (``%YAML:1.0`` with ``!!opencv-matrix`` entries holding
    ``rows``, ``cols``, ``data``) are accepted as written by many datasets;
    the first 4x4 matrix in the file is used.
    """

    import yaml

    text = Path(path).read_text(encoding="utf-8")
    if text.lstrip().startswith("%YAML:1.0"):
        text = text.split("\n", 1)[1].replace("!!opencv-matrix", "")
    payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not hold a mapping")
    for key in ("matrix", "T", "transform"):
        if key in payload:
            matrix = np.asarray(payload[key], dtype=np.float64)
            break
    else:
        candidates = [
            value
            for value in payload.values()
            if isinstance(value, dict) and value.get("rows") == 4 and value.get("cols") == 4
        ]
        if not candidates:
            raise ValueError(f"{path} has no 4x4 matrix")
        matrix = np.asarray(candidates[0]["data"], dtype=np.float64).reshape(4, 4)
    if matrix.shape != (4, 4) or not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError(f"{path} does not hold a homogeneous 4x4 transform")
    return matrix
