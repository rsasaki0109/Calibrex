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

With ``target_motion="reference_trajectory"`` every target point is placed by
the reference trajectory interpolated at its own capture time, shifted by the
target-to-reference clock offset, and ``target_accumulate_scans`` consecutive
target scans form one sample.  That deskews the target along the reference
trajectory and densifies a sparse target LiDAR (the Unitree L1 of a Go2).  The
clock offset can be given or estimated by a profile search over train blocks
only (:func:`estimate_time_offset`); held-out blocks never choose it.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Collection, Iterator, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, TypeAlias

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
    LidarLidarTimeOffset,
    LidarLidarTimeOffsetProfilePoint,
    LidarLidarTransform,
)
from calibrex.core.progress import emit_stage
from calibrex.core.provenance import git_commit
from calibrex.data.livox_ros2 import LivoxStreamProfile, bag_input_digest, iter_livox_points
from calibrex.solvers.lidar_lidar_map_solver import (
    EXTRINSIC_DOFS,
    MapBackend,
    MapExtrinsicResult,
    MapSample,
    MapSampleOptions,
    _map_runtime,
    _MapRuntime,
    chi_square,
    perturb_extrinsic,
    solve_map_extrinsic,
)
from calibrex.solvers.scan_to_scan_odometry import (
    IncrementalScanOdometry,
    OdometryBackend,
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

TRAJECTORY_LIMITATIONS: tuple[str, ...] = (
    "The reference LiDAR's odometry supplies the map and the trajectory every target "
    "point is placed by; its drift over the map window adds noise to every sample.",
    "Target points are placed by the reference trajectory interpolated at their capture "
    "times (linear in translation, geodesic in rotation between reference scans); motion "
    "faster than the reference scan rate, such as gait vibration above half that rate, "
    "is not resolved.",
    "The reference transform, if given, is a comparison, not ground truth.",
)


RIGID_SCAN_LIMITATIONS: tuple[str, ...] = (
    "The reference LiDAR's scans were treated as rigid snapshots (deskew: none): its "
    "odometry, and so the map every target scan is registered to, is not corrected for "
    "motion during the sweep; the target scans never are. The extrinsic is biased by an "
    "amount that grows with the platform's speed.",
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
    # "constant_velocity": the reference odometry deskews each sweep with the cloud's
    # per-point time (the target scans are registered as given).  "none": rigid
    # scans, for reference clouds without a per-point time field.
    deskew: Literal["constant_velocity", "none"] = "constant_velocity"
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
    # "rigid_nearest": each target scan is one rigid snapshot at the nearest reference
    # pose.  "reference_trajectory": every target point is placed by the reference
    # trajectory interpolated at its capture time plus ``target_time_offset_s``.
    target_motion: Literal["rigid_nearest", "reference_trajectory"] = "rigid_nearest"
    # Consecutive target scans per sample (reference_trajectory only).
    target_accumulate_scans: int = 1
    # t_reference = t_target + offset: the target clock's lag behind the reference.
    target_time_offset_s: float = 0.0
    # Target preprocessing; None reuses ``preprocess``.  A sparse target (L1) keeps
    # more of its points with a finer voxel.
    target_preprocess: ScanOdometryOptions | None = None
    # Refit on the same train blocks with this map half window and count each DoF's
    # shift in its reported std.  None (or equal to map_half_window_scans) disables it.
    # 0 maps a target scan onto the nearest reference scan alone, where the reference
    # odometry cancels: the shift bounds the odometry's bias over the map window.
    sensitivity_half_window_scans: int | None = None

    def __post_init__(self) -> None:
        if self.target_accumulate_scans < 1:
            raise ValueError("target_accumulate_scans must be at least 1")
        if self.target_motion == "rigid_nearest" and (
            self.target_accumulate_scans != 1 or self.target_time_offset_s != 0.0
        ):
            raise ValueError(
                "target accumulation and a target time offset need "
                "target_motion='reference_trajectory'"
            )
        if not math.isfinite(self.target_time_offset_s):
            raise ValueError("target_time_offset_s must be finite")


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
    backend: MapBackend = "cpu",
    sensitivity_samples: Sequence[MapSample] | None = None,
) -> LidarLidarEvaluation:
    """Fit on train blocks; collect jackknife, control, and holdout evidence.

    ``sensitivity_samples`` are the same target scans with another map window (keyed by
    the same blocks).  The extrinsic refit on their train blocks, from the estimate,
    gives each DoF's systematic shift, which the reported std must cover.
    """

    opts = options or LidarLidarMapOptions()
    runtime = _map_runtime(samples, opts.solver, backend)
    blocks = sorted({sample.block for sample in samples})
    holdout_blocks = tuple(
        block for position, block in enumerate(blocks) if position % opts.holdout_every == 1
    )
    train_blocks = tuple(block for block in blocks if block not in holdout_blocks)
    train = [sample for sample in samples if sample.block in train_blocks]
    holdout = [sample for sample in samples if sample.block in holdout_blocks]
    result = solve_map_extrinsic(train, initial, opts.solver, _runtime=runtime)
    if result.transform is None or result.covariance is None or result.sigma_m is None:
        return LidarLidarEvaluation(
            result, (), train_blocks, holdout_blocks, 0, len(train), len(holdout), 0, None, None,
            "fail", ("too few samples with map correspondences to solve",),
        )  # fmt: skip
    transform = result.transform
    jackknife, fits = _jackknife(train, train_blocks, transform, opts, _runtime=runtime)
    base, holdout_median, holdout_count = chi_square(
        holdout, transform, result.sigma_m, opts.solver, _runtime=runtime
    )
    analytic = np.sqrt(np.clip(np.diag(result.covariance), 0.0, None))
    systematic = None
    if sensitivity_samples:
        refit = solve_map_extrinsic(
            [sample for sample in sensitivity_samples if sample.block in train_blocks],
            transform, opts.solver,
        )  # fmt: skip
        if refit.transform is not None:
            systematic = np.abs(_local_difference(refit.transform, transform))
    records = tuple(
        _record(
            dof,
            index,
            transform,
            float(analytic[index]),
            None if jackknife is None else float(jackknife[index]),
            _control(dof, holdout, transform, result.sigma_m, base, opts, _runtime=runtime),
            reference_transform,
            opts,
            None if systematic is None else float(systematic[index]),
        )
        for index, dof in enumerate(EXTRINSIC_DOFS)
    )
    reference_delta = None
    if reference_transform is not None and holdout:
        moved, _, _ = chi_square(
            holdout, reference_transform, result.sigma_m, opts.solver, _runtime=runtime
        )
        reference_delta = moved - base
    train_median = _median_residual(train, transform, result.sigma_m, opts, _runtime=runtime)
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
    odometry_backend: OdometryBackend = "cpu",
    _odometry_execution: dict[str, Any] | None = None,
    _sensitivity: list[MapSample] | None = None,
) -> tuple[list[MapSample], int]:
    """Run the reference odometry, then build one sample per strided target scan.

    With ``options.sensitivity_half_window_scans`` set, ``_sensitivity`` receives the
    same samples with that map half window.

    ``point_time_field``/``point_time_encoding`` describe the reference LiDAR's
    per-point time; the target uses the same unless ``target_point_time_field``
    is given (LiDARs of different models).
    """

    opts = options or LidarLidarMapOptions()
    if opts.deskew == "none":
        # Rigid scans: no per-point time is decoded or used.
        point_time_field = None
        target_point_time_field = None
        target_point_time_encoding = None
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

    emit_stage(f"reference LiDAR odometry + map ({reference_topic})")
    odometry = IncrementalScanOdometry(opts.odometry, backend=odometry_backend)
    if _odometry_execution is not None:
        _odometry_execution["odometry_execution"] = odometry.execution_backend
        _odometry_execution["odometry_options"] = asdict(opts.odometry)
    scans: list[NDArray[np.float32]] = []
    times: list[float] = []
    first: float | None = None
    reference_scan_stream = iter_livox_points(
        bag, profile(reference_topic), max_seconds=max_seconds
    )
    for time_s, points, offsets in reference_scan_stream:
        first = time_s if first is None else first
        finite = np.isfinite(points).all(axis=1)
        use_time = offsets is not None and opts.deskew != "none"
        odometry.add(
            points[finite], time_s, offsets[finite] if use_time and offsets is not None else None
        )
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
    emit_stage(f"target scans against the reference map ({target_topic})")
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
        target = preprocess_scan(
            points[np.isfinite(points).all(axis=1)], opts.target_preprocess or opts.preprocess
        )
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
        other = opts.sensitivity_half_window_scans
        if _sensitivity is not None and other is not None and other != half:
            _sensitivity.append(
                MapSample(
                    time_s=float(time_s),
                    block=samples[-1].block,
                    reference_pose=samples[-1].reference_pose,
                    target_points=target,
                    map_points=np.concatenate(
                        scans[max(0, nearest - other) : nearest + other + 1]
                    ).astype(np.float64),
                )
            )
    return samples, len(scans)


@dataclass(frozen=True)
class ReferenceTrajectory:
    """The reference LiDAR's odometry: poses at scan times and scans in the odometry frame."""

    times_s: FloatArray  # (S,) increasing reference header times
    poses: FloatArray  # (S, 4, 4) T_odom_reference at those times
    scans: tuple[NDArray[np.float32], ...]  # preprocessed scans in the odometry frame
    first_time_s: float

    def __post_init__(self) -> None:
        if len(self.times_s) != len(self.poses) or len(self.times_s) != len(self.scans):
            raise ValueError("reference times, poses, and scans must have one entry per scan")
        if len(self.times_s) > 1 and np.any(np.diff(self.times_s) <= 0.0):
            raise ValueError("reference scan times must increase strictly")

    def interpolate(self, query_s: FloatArray) -> FloatArray:
        """``T_odom_reference`` at each query time; NaN outside the trajectory.

        Translation is linear and rotation geodesic between the two bracketing scans.
        """

        query = np.asarray(query_s, dtype=np.float64).reshape(-1)
        out = np.full((len(query), 4, 4), np.nan)
        times = self.times_s
        inside = (query >= times[0]) & (query <= times[-1])
        if len(times) < 2 or not inside.any():
            return out
        q = query[inside]
        lower = np.clip(np.searchsorted(times, q, side="right") - 1, 0, len(times) - 2)
        alpha = (q - times[lower]) / (times[lower + 1] - times[lower])
        start = self.poses[lower]
        end = self.poses[lower + 1]
        relative = Rotation.from_matrix(
            np.einsum("nji,njk->nik", start[:, :3, :3], end[:, :3, :3])
        ).as_rotvec()
        rotation = np.einsum(
            "nij,njk->nik",
            start[:, :3, :3],
            Rotation.from_rotvec(relative * alpha[:, None]).as_matrix(),
        )
        block = np.zeros((len(q), 4, 4))
        block[:, :3, :3] = rotation
        block[:, :3, 3] = start[:, :3, 3] + alpha[:, None] * (end[:, :3, 3] - start[:, :3, 3])
        block[:, 3, 3] = 1.0
        out[inside] = block
        return out


@dataclass(frozen=True)
class TargetScan:
    """One preprocessed target scan with its per-point capture times."""

    time_s: float  # header time on the target clock
    points: FloatArray  # (N, 3) in the target LiDAR frame
    offsets_s: FloatArray | None  # (N,) seconds after ``time_s``, or None (all at time_s)


def preprocess_timed_scan(
    points: FloatArray, offsets_s: FloatArray | None, options: ScanOdometryOptions
) -> tuple[FloatArray, FloatArray | None]:
    """Crop by range and voxel-average points, averaging their capture times too."""

    xyz = np.asarray(points, dtype=np.float64)[:, :3]
    finite = np.isfinite(xyz).all(axis=1)
    if offsets_s is not None:
        finite &= np.isfinite(offsets_s)
    xyz = xyz[finite]
    times = None if offsets_s is None else np.asarray(offsets_s, dtype=np.float64)[finite]
    ranges = np.sqrt(np.einsum("ij,ij->i", xyz, xyz))
    keep = (ranges >= options.min_range_m) & (ranges <= options.max_range_m)
    xyz = xyz[keep]
    times = None if times is None else times[keep]
    if xyz.size == 0:
        return xyz.reshape(0, 3), None if times is None else times
    keys = np.floor(xyz / options.voxel_size_m).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    sums = np.stack(
        [np.bincount(inverse, weights=xyz[:, axis], minlength=counts.size) for axis in range(3)],
        axis=1,
    )
    averaged = None
    if times is not None:
        averaged = np.bincount(inverse, weights=times, minlength=counts.size) / counts
    return sums / counts[:, None], averaged


def read_reference_trajectory(
    bag: str | Path,
    *,
    reference_topic: str,
    point_time_field: str | None,
    options: LidarLidarMapOptions | None = None,
    max_seconds: float | None = None,
    point_time_encoding: PointTimeEncoding = "offset_s",
    odometry_backend: OdometryBackend = "cpu",
    _odometry_execution: dict[str, Any] | None = None,
) -> ReferenceTrajectory:
    """Run the reference LiDAR odometry and keep its poses and map scans."""

    opts = options or LidarLidarMapOptions()
    if opts.deskew == "none":
        point_time_field = None
    emit_stage(f"reference LiDAR odometry + map ({reference_topic})")
    odometry = IncrementalScanOdometry(opts.odometry, backend=odometry_backend)
    if _odometry_execution is not None:
        _odometry_execution["odometry_execution"] = odometry.execution_backend
        _odometry_execution["odometry_options"] = asdict(opts.odometry)
    profile = LivoxStreamProfile(
        "lidar", reference_topic, "", point_time_field, point_time_encoding, "mps2"
    )
    scans: list[NDArray[np.float32]] = []
    times: list[float] = []
    for time_s, points, offsets in iter_livox_points(bag, profile, max_seconds=max_seconds):
        if times and time_s <= times[-1]:
            continue  # a repeated or out-of-order stamp cannot be interpolated
        finite = np.isfinite(points).all(axis=1)
        use_time = offsets is not None and opts.deskew != "none"
        odometry.add(
            points[finite], time_s, offsets[finite] if use_time and offsets is not None else None
        )
        pose = odometry.poses[-1]
        local = preprocess_scan(points[finite], opts.preprocess)
        scans.append((local @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32))
        times.append(time_s)
    if not times:
        raise ValueError(f"{bag} has no scans on {reference_topic}")
    return ReferenceTrajectory(
        times_s=np.asarray(times, dtype=np.float64),
        poses=np.asarray(odometry.poses[: len(times)], dtype=np.float64),
        scans=tuple(scans),
        first_time_s=times[0],
    )


def read_target_scans(
    bag: str | Path,
    *,
    target_topic: str,
    point_time_field: str | None,
    options: LidarLidarMapOptions | None = None,
    max_seconds: float | None = None,
    point_time_encoding: PointTimeEncoding = "offset_s",
) -> list[TargetScan]:
    """Read and preprocess every target scan, keeping per-point capture times."""

    opts = options or LidarLidarMapOptions()
    preprocess = opts.target_preprocess or opts.preprocess
    emit_stage(f"target scans ({target_topic})")
    profile = LivoxStreamProfile(
        "lidar", target_topic, "", point_time_field, point_time_encoding, "mps2"
    )
    scans: list[TargetScan] = []
    for time_s, points, offsets in iter_livox_points(bag, profile, max_seconds=max_seconds):
        local, local_offsets = preprocess_timed_scan(points, offsets, preprocess)
        scans.append(TargetScan(float(time_s), local, local_offsets))
    return scans


def build_trajectory_samples(
    trajectory: ReferenceTrajectory,
    targets: Sequence[TargetScan],
    options: LidarLidarMapOptions | None = None,
    *,
    time_offset_s: float | None = None,
    times_s: Collection[float] | None = None,
) -> list[MapSample]:
    """One sample per ``sample_every_scans`` target scans, each point at its own pose.

    A sample accumulates ``target_accumulate_scans`` consecutive target scans.  Its
    map holds the reference scans spanning those points' (offset) capture times,
    widened by ``map_half_window_scans`` on each side.  Samples whose points fall
    outside the trajectory, or that keep fewer than ``min_target_points``, are skipped.
    The sample's ``time_s`` and ``block`` stay on the target clock, so blocks do not
    move with the offset.  ``times_s`` keeps only the samples with those ``time_s``.
    """

    opts = options or LidarLidarMapOptions()
    offset = opts.target_time_offset_s if time_offset_s is None else time_offset_s
    first = trajectory.first_time_s
    samples: list[MapSample] = []
    for middle, points, capture, low, high in _trajectory_windows(
        trajectory, targets, opts, offset
    ):
        if times_s is not None and middle not in times_s:
            continue
        poses = trajectory.interpolate(capture)
        if not np.isfinite(poses[:, 0, 0]).all():
            continue
        reference_pose = trajectory.interpolate(np.array([middle + offset]))[0]
        if not np.isfinite(reference_pose).all():
            reference_pose = poses[len(poses) // 2]
        samples.append(
            MapSample(
                time_s=middle,
                block=int((middle - first) // opts.block_duration_s),
                reference_pose=reference_pose,
                target_points=points,
                map_points=np.concatenate(trajectory.scans[low : high + 1]).astype(np.float64),
                point_poses=poses,
            )
        )
    return samples


def _trajectory_windows(
    trajectory: ReferenceTrajectory,
    targets: Sequence[TargetScan],
    opts: LidarLidarMapOptions,
    offset: float,
) -> Iterator[tuple[float, FloatArray, FloatArray, int, int]]:
    """``(time_s, points, capture times, first and last map scan)`` of each valid sample."""

    half = opts.map_half_window_scans
    times = trajectory.times_s
    count = opts.target_accumulate_scans
    for start in range(0, len(targets) - count + 1, opts.sample_every_scans):
        group = [scan for scan in targets[start : start + count] if len(scan.points)]
        if not group:
            continue
        points = np.concatenate([scan.points for scan in group])
        if len(points) < opts.min_target_points:
            continue
        capture = np.concatenate(
            [
                scan.time_s + (scan.offsets_s if scan.offsets_s is not None
                               else np.zeros(len(scan.points)))
                for scan in group
            ]
        ) + offset  # fmt: skip
        if capture.min() < times[0] or capture.max() > times[-1]:
            continue
        low = int(np.searchsorted(times, capture.min(), side="right")) - 1 - half
        high = int(np.searchsorted(times, capture.max(), side="left")) + half
        if low < 0 or high >= len(times):
            continue
        middle = float(targets[start + len(targets[start : start + count]) // 2].time_s)
        yield middle, points, capture, low, high


@dataclass(frozen=True)
class TimeOffsetEstimate:
    """Profile search result for the target-to-reference clock offset."""

    value_s: float
    std_jackknife_s: float | None
    at_search_edge: bool
    candidates_s: tuple[float, ...]
    train_cost_per_point: tuple[float, ...]
    common_samples: int


def estimate_time_offset(
    trajectory: ReferenceTrajectory,
    targets: Sequence[TargetScan],
    initial: FloatArray,
    options: LidarLidarMapOptions | None = None,
    *,
    half_width_s: float = 0.1,
    step_s: float = 0.01,
    center_s: float = 0.0,
) -> TimeOffsetEstimate:
    """Choose the clock offset that best fits the train blocks.

    For every candidate offset on the grid ``center ± half_width`` the extrinsic
    is refit on the train blocks (the same every-``holdout_every`` rule as
    :func:`evaluate_lidar_lidar_map`, on target-clock blocks) and scored by its
    clipped chi-square at the fixed Huber scale, over the samples every candidate
    shares.  A parabola through the best grid point and its neighbours refines the
    minimum; a block jackknife of that refinement gives the std.  Held-out blocks
    are never read here.
    """

    opts = options or LidarLidarMapOptions()
    if half_width_s <= 0.0 or step_s <= 0.0:
        raise ValueError("time offset search half width and step must be positive")
    steps = round(half_width_s / step_s)
    candidates = center_s + step_s * np.arange(-steps, steps + 1)
    # Sample times valid at every candidate offset (cheap: no maps are built here).
    keys = [
        {window[0] for window in _trajectory_windows(trajectory, targets, opts, float(value))}
        for value in candidates
    ]
    common = set.intersection(*keys) if keys else set()
    first = trajectory.first_time_s

    def block_of(time_s: float) -> int:
        return int((time_s - first) // opts.block_duration_s)

    blocks = sorted({block_of(time_s) for time_s in common})
    holdout_blocks = {
        block for position, block in enumerate(blocks) if position % opts.holdout_every == 1
    }
    train_blocks = [block for block in blocks if block not in holdout_blocks]
    train_times = {time_s for time_s in common if block_of(time_s) not in holdout_blocks}
    costs = np.full((len(candidates), len(train_blocks)), np.nan)
    points = np.zeros(len(train_blocks))
    for row, value in enumerate(candidates):
        # One candidate's samples at a time: the maps and per-point poses are large.
        train = build_trajectory_samples(
            trajectory, targets, opts, time_offset_s=float(value), times_s=train_times
        )
        fit = solve_map_extrinsic(train, initial, opts.solver)
        if fit.transform is None:
            continue
        for column, block in enumerate(train_blocks):
            members = [sample for sample in train if sample.block == block]
            cost, _, _ = chi_square(members, fit.transform, opts.solver.huber_m, opts.solver)
            costs[row, column] = cost
            points[column] = sum(len(sample.target_points) for sample in members)
    if not np.isfinite(costs).all() or len(train_blocks) == 0 or points.sum() == 0:
        raise ValueError("too few shared train samples to estimate the time offset")
    total = costs.sum(axis=1) / points.sum()
    value, edge = _parabolic_minimum(candidates, total)
    jackknife = None
    groups = min(opts.jackknife_groups, len(train_blocks))
    if groups >= 3:
        estimates = []
        for members in np.array_split(np.arange(len(train_blocks)), groups):
            keep = np.setdiff1d(np.arange(len(train_blocks)), members)
            estimates.append(
                _parabolic_minimum(candidates, costs[:, keep].sum(axis=1) / points[keep].sum())[0]
            )
        array = np.asarray(estimates)
        jackknife = float(np.sqrt((groups - 1) / groups * np.sum((array - array.mean()) ** 2)))
    return TimeOffsetEstimate(
        value_s=value,
        std_jackknife_s=jackknife,
        at_search_edge=edge,
        candidates_s=tuple(float(item) for item in candidates),
        train_cost_per_point=tuple(float(item) for item in total),
        common_samples=len(common),
    )


def _parabolic_minimum(candidates: FloatArray, costs: FloatArray) -> tuple[float, bool]:
    best = int(np.argmin(costs))
    if best == 0 or best == len(costs) - 1:
        return float(candidates[best]), True
    left, middle, right = costs[best - 1 : best + 2]
    curvature = left - 2.0 * middle + right
    step = candidates[best + 1] - candidates[best]
    shift = 0.0 if curvature <= 0.0 else 0.5 * (left - right) / curvature
    return float(candidates[best] + np.clip(shift, -0.5, 0.5) * step), False


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
    backend: MapBackend = "cpu",
    odometry_backend: OdometryBackend = "cpu",
    time_offset_search: tuple[float, float] | None = None,
) -> LidarLidarExtrinsicArtifact:
    """Calibrate ``T_reference_target`` for two LiDARs recorded in one ROS 2 bag.

    ``time_offset_search`` is ``(half_width_s, step_s)`` around
    ``options.target_time_offset_s``; it needs ``target_motion="reference_trajectory"``.
    """

    opts = options or LidarLidarMapOptions()
    odometry_execution: dict[str, Any] = {}
    time_offset: LidarLidarTimeOffset | None = None
    if time_offset_search is not None and opts.target_motion != "reference_trajectory":
        raise ValueError("a time offset search needs target_motion='reference_trajectory'")
    sensitivity: list[MapSample] = []
    other = opts.sensitivity_half_window_scans
    if opts.target_motion == "rigid_nearest":
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
            odometry_backend=odometry_backend,
            _odometry_execution=odometry_execution,
            _sensitivity=sensitivity,
        )
    else:
        trajectory = read_reference_trajectory(
            bag,
            reference_topic=reference_topic,
            point_time_field=point_time_field,
            options=opts,
            max_seconds=max_seconds,
            point_time_encoding=point_time_encoding,
            odometry_backend=odometry_backend,
            _odometry_execution=odometry_execution,
        )
        same_time = target_point_time_field is None and target_point_time_encoding is None
        targets = read_target_scans(
            bag,
            target_topic=target_topic,
            point_time_field=point_time_field if same_time else target_point_time_field,
            options=opts,
            max_seconds=max_seconds,
            point_time_encoding=target_point_time_encoding or point_time_encoding,
        )
        reference_scans = len(trajectory.times_s)
        if time_offset_search is not None:
            half_width, step = time_offset_search
            emit_stage("target clock offset profile (train blocks)")
            estimate = estimate_time_offset(
                trajectory,
                targets,
                initial,
                opts,
                half_width_s=half_width,
                step_s=step,
                center_s=opts.target_time_offset_s,
            )
            time_offset = LidarLidarTimeOffset(
                value_s=estimate.value_s,
                source="estimated",
                std_jackknife_s=estimate.std_jackknife_s,
                at_search_edge=estimate.at_search_edge,
                search_center_s=opts.target_time_offset_s,
                search_half_width_s=half_width,
                search_step_s=step,
                profile=[
                    LidarLidarTimeOffsetProfilePoint(offset_s=offset, train_cost_per_point=cost)
                    for offset, cost in zip(
                        estimate.candidates_s, estimate.train_cost_per_point, strict=True
                    )
                ],
            )
            opts = replace(opts, target_time_offset_s=estimate.value_s)
        else:
            time_offset = LidarLidarTimeOffset(value_s=opts.target_time_offset_s, source="given")
        samples = build_trajectory_samples(trajectory, targets, opts)
        if other is not None and other != opts.map_half_window_scans:
            sensitivity = build_trajectory_samples(
                trajectory, targets, replace(opts, map_half_window_scans=other),
                times_s={sample.time_s for sample in samples},
            )  # fmt: skip
    evaluation = evaluate_lidar_lidar_map(
        samples, initial, opts, reference_transform=reference_transform, backend=backend,
        sensitivity_samples=sensitivity or None,
    )  # fmt: skip
    policy_status = evaluation.policy_status
    policy_reasons = list(evaluation.policy_reasons)
    if time_offset is not None and time_offset.at_search_edge:
        policy_reasons.append(
            "the clock offset profile is lowest at the edge of its search window "
            f"({time_offset.value_s:+.3f} s); widen or move the search"
        )
        if policy_status == "pass":
            policy_status = "warn"
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
        policy_status=policy_status,
        policy_reasons=policy_reasons,
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
        time_offset=time_offset,
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
            **(
                {"registration_query_workers": opts.solver.query_workers}
                if opts.solver.query_workers != 1
                else {}
            ),
            **({"deskew": "none"} if opts.deskew == "none" else {}),
            **({"sensitivity_half_window_scans": other} if sensitivity else {}),
            **(
                {
                    "target_motion": opts.target_motion,
                    "target_accumulate_scans": opts.target_accumulate_scans,
                    "target_voxel_size_m": (opts.target_preprocess or opts.preprocess).voxel_size_m,
                }
                if opts.target_motion == "reference_trajectory"
                else {}
            ),
            **(
                {"registration_backend": "cuda", "registration_execution": result.execution_backend}
                if backend == "cuda"
                else {}
            ),
            **(
                {"odometry_backend": "cuda", **odometry_execution}
                if odometry_backend == "cuda"
                else {}
            ),
        },
        reference=reference,
        limitations=[
            *(LIMITATIONS if opts.target_motion == "rigid_nearest" else TRAJECTORY_LIMITATIONS),
            *(RIGID_SCAN_LIMITATIONS if opts.deskew == "none" else ()),
        ],
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


def _local_difference(moved: FloatArray, base: FloatArray) -> FloatArray:
    """Rotation about the reference axes (rad) and translation (m) from ``base`` to ``moved``."""

    rotation = Rotation.from_matrix(moved[:3, :3] @ base[:3, :3].T).as_rotvec()
    return np.asarray(np.r_[rotation, moved[:3, 3] - base[:3, 3]], dtype=np.float64)


def _jackknife(
    train: Sequence[MapSample],
    train_blocks: Sequence[int],
    transform: FloatArray,
    opts: LidarLidarMapOptions,
    *,
    _runtime: _MapRuntime | None = None,
) -> tuple[FloatArray | None, int]:
    groups = min(opts.jackknife_groups, len(train_blocks))
    if groups < 3:
        return None, 0
    deltas = []
    for members in np.array_split(np.asarray(train_blocks), groups):
        left_out = set(members.tolist())
        fit = solve_map_extrinsic(
            [sample for sample in train if sample.block not in left_out],
            transform,
            opts.solver,
            _runtime=_runtime,
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
    systematic: float | None = None,
) -> LidarLidarDofRecord:
    rotation = dof in _ROTATIONS
    convert = math.degrees if rotation else (lambda value: value)
    reported = max(analytic, jackknife or 0.0, systematic or 0.0)
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
        std_systematic=None if systematic is None else convert(systematic),
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
    *,
    _runtime: _MapRuntime | None = None,
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
        _runtime=_runtime,
    )
    delta = moved - base
    return LidarLidarControl(
        amount=amount,
        unit="deg" if rotation else "m",
        holdout_delta_chi2=float(delta),
        detected=bool(delta >= opts.detection_delta_chi2),
    )


def _median_residual(
    samples: Sequence[MapSample],
    transform: FloatArray,
    sigma: float,
    opts: LidarLidarMapOptions,
    *,
    _runtime: _MapRuntime | None = None,
) -> float | None:
    if not samples:
        return None
    _, median, count = chi_square(samples, transform, sigma, opts.solver, _runtime=_runtime)
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


def receive_time_clock_offset(
    bag: str | Path, *, reference_topic: str, target_topic: str, max_messages: int = 2000
) -> float:
    """Coarse ``t_reference - t_target`` from header stamps against bag receive times.

    Each topic's median ``header - receive`` is its clock's offset from the
    recorder's; their difference is a coarse target-to-reference offset, good
    to the transport jitter (milliseconds to tens of milliseconds).  It catches
    a LiDAR stamped by its own clock, such as the Go2 L1 on the robot's RTC.
    Use it as the centre of :func:`estimate_time_offset`, never as the answer.
    """

    import struct

    from calibrex.data.rosbag2 import iter_messages

    lags: dict[str, list[float]] = {reference_topic: [], target_topic: []}
    for connection, timestamp_ns, payload in iter_messages(
        bag, topics={reference_topic, target_topic}
    ):
        bucket = lags[connection.topic]
        if len(bucket) >= max_messages:
            if all(len(values) >= max_messages for values in lags.values()):
                break
            continue
        endian = "<" if payload[1] == 1 else ">"
        seconds, nanoseconds = struct.unpack_from(endian + "iI", payload, 4)
        bucket.append(seconds + nanoseconds * 1.0e-9 - timestamp_ns * 1.0e-9)
    for topic, values in lags.items():
        if not values:
            raise ValueError(f"{bag} has no messages on {topic}")
    return float(np.median(lags[reference_topic]) - np.median(lags[target_topic]))
