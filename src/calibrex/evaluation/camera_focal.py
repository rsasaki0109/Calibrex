"""Check a camera's focal lengths against the gyro, without a calibration target.

Camera rotations from tracked features scale with the focal length the
features were normalized with.  A rotation by ``theta`` about the camera's y
axis moves the image horizontally by about ``fx_true theta``, which the
tracker reads as ``theta fx_true / fx_used``.  About the x axis the ratio is
``fy_true / fy_used``, and about the optical axis there is no focal
dependence at all.  With the camera-IMU rotation, clock offset, and gyro
bias known, regressing the camera's angular rates on the gyro's, axis by
axis, therefore gives ``fx_est = fx_used k_y`` and ``fy_est = fy_used k_x``.
The optical-axis ratio ``k_z`` is a control: it should be 1, and when it is
not, something other than the focal length (for example tracking
attenuation) is scaling the rotations.

Evidence: blocks of windows with every third held out, the train and held-out
ratios must agree, and an 8-group block jackknife sets the std.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.camera_focal_scale import CameraFocalScaleArtifact
from calibrex.core.imu_lidar_rotation import ImuLidarRotationArtifact
from calibrex.evaluation.visual_rotation import CameraModel
from calibrex.solvers.gnss_lever_arm_solver import OdometryWindow
from calibrex.solvers.imu_lidar_rotation_solver import (
    GyroSeries,
    RotationOptions,
    rate_intervals,
)

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class FocalScaleOptions:
    """Blocking, outlier gate, and policy."""

    holdout_every: int = 3
    jackknife_groups: int = 8
    outlier_mad: float = 5.0
    min_rate_rps: float = 0.05
    max_holdout_sigma: float = 3.0
    max_optical_axis_deviation_sigma: float = 3.0
    observable_ratio_std: float = 0.005


@dataclass(frozen=True)
class FocalScaleResult:
    """Rate ratios about the camera x, y, z axes and their evidence."""

    ratio: FloatArray  # camera/gyro rate ratio about x, y, z
    ratio_std: FloatArray
    holdout_ratio: FloatArray
    intervals: int
    status: str
    reasons: tuple[str, ...]


def _rates(
    gyro: GyroSeries,
    windows: Sequence[OdometryWindow],
    rotation: FloatArray,
    time_offset_s: float,
    gyro_bias_rps: FloatArray,
) -> tuple[FloatArray, FloatArray, NDArray[np.int64]]:
    intervals = rate_intervals(windows, gyro, RotationOptions())
    start = intervals.start_s + time_offset_s
    end = intervals.end_s + time_offset_s
    gyro_rate = (gyro.integral_at(end) - gyro.integral_at(start)) / (end - start)[:, None]
    predicted = (gyro_rate - gyro_bias_rps) @ rotation.T
    return intervals.lidar_rate_rps, predicted, intervals.window_ids


def _ratio(camera: FloatArray, predicted: FloatArray, opts: FocalScaleOptions) -> FloatArray:
    ratios = []
    for axis in range(3):
        g = predicted[:, axis]
        c = camera[:, axis]
        keep = np.abs(g) >= opts.min_rate_rps
        if int(keep.sum()) < 10:
            ratios.append(math.nan)
            continue
        g, c = g[keep], c[keep]
        k = float(np.sum(c * g) / max(float(np.sum(g * g)), 1e-12))
        for _ in range(3):  # drop gross outliers (failed frame pairs) and refit
            residual = c - k * g
            spread = max(1.4826 * float(np.median(np.abs(residual))), 1e-9)
            inlier = np.abs(residual) <= opts.outlier_mad * spread
            k = float(np.sum(c[inlier] * g[inlier]) / max(float(np.sum(g[inlier] ** 2)), 1e-12))
        ratios.append(k)
    return np.asarray(ratios, dtype=np.float64)


def estimate_focal_scale(
    gyro: GyroSeries,
    windows: Sequence[OdometryWindow],
    rotation: FloatArray,
    time_offset_s: float,
    gyro_bias_rps: FloatArray,
    options: FocalScaleOptions | None = None,
) -> FocalScaleResult:
    """Per-axis camera/gyro rate ratios with a held-out and jackknife check."""

    opts = options or FocalScaleOptions()
    camera, predicted, window_ids = _rates(gyro, windows, rotation, time_offset_s, gyro_bias_rps)
    positions = np.unique(window_ids)
    holdout_windows = positions[np.arange(len(positions)) % opts.holdout_every == 1]
    holdout = np.isin(window_ids, holdout_windows)
    ratio = _ratio(camera[~holdout], predicted[~holdout], opts)
    holdout_ratio = _ratio(camera[holdout], predicted[holdout], opts)
    train_windows = positions[~np.isin(positions, holdout_windows)]
    groups = min(opts.jackknife_groups, len(train_windows))
    samples = []
    for members in np.array_split(train_windows, max(groups, 1)):
        keep = ~holdout & ~np.isin(window_ids, members)
        samples.append(_ratio(camera[keep], predicted[keep], opts))
    array = np.asarray(samples)
    std = (
        np.sqrt((groups - 1) / groups * np.sum((array - array.mean(axis=0)) ** 2, axis=0))
        if groups >= 3
        else np.full(3, math.inf)
    )
    reasons: list[str] = []
    status = "pass"
    if not (np.all(np.isfinite(ratio)) and np.all(np.isfinite(std))):
        return FocalScaleResult(
            ratio=ratio,
            ratio_std=np.asarray(std, dtype=np.float64),
            holdout_ratio=holdout_ratio,
            intervals=len(camera),
            status="inconclusive",
            reasons=("too few intervals rotate fast enough about every camera axis",),
        )
    disagreement = np.abs(holdout_ratio - ratio) / np.maximum(std * math.sqrt(1.5), 1e-9)
    if np.any(disagreement > opts.max_holdout_sigma):
        status = "fail"
        reasons.append(
            "held-out ratios disagree with the training ratios by "
            + ", ".join(f"{value:.1f}" for value in disagreement)
            + " std about x, y, z"
        )
    optical = abs(ratio[2] - 1.0) / max(float(std[2]), 1e-9)
    if status == "pass" and optical > opts.max_optical_axis_deviation_sigma:
        status = "warn"
        reasons.append(
            f"the optical-axis ratio {ratio[2]:.4f} differs from 1 by {optical:.1f} std: "
            "something besides the focal length scales the rotations"
        )
    if status == "pass" and np.any(std[:2] > opts.observable_ratio_std):
        status = "inconclusive"
        reasons.append("the x or y ratio std exceeds the bound")
    if status == "pass":
        reasons.append(
            "ratios are consistent on held-out windows and the optical-axis control is 1"
        )
    return FocalScaleResult(
        ratio=ratio,
        ratio_std=np.asarray(std, dtype=np.float64),
        holdout_ratio=holdout_ratio,
        intervals=len(camera),
        status=status,
        reasons=tuple(reasons),
    )


def run_ros2_camera_focal_check(
    bag: str,
    *,
    image_topic: str,
    imu_topic: str,
    camera_entry: dict[str, object],
    rotation_artifact_path: str,
    dataset_family: str,
    dataset_license: str,
    frame_stride: int = 4,
    options: FocalScaleOptions | None = None,
    command: list[str] | None = None,
) -> CameraFocalScaleArtifact:
    """Track the camera, then check its focal lengths against the gyro."""

    from pathlib import Path

    from calibrex.core.imu_lidar_rotation import load_imu_lidar_rotation
    from calibrex.core.provenance import sha256_path
    from calibrex.evaluation.visual_rotation import CameraModel

    rotation_digest = sha256_path(Path(rotation_artifact_path))
    assert rotation_digest is not None
    return run_camera_focal_check(
        bag,
        image_topic=image_topic,
        imu_topic=imu_topic,
        camera=CameraModel.from_kalibr(camera_entry),
        rotation_artifact=load_imu_lidar_rotation(rotation_artifact_path),
        rotation_artifact_sha256=rotation_digest,
        dataset_family=dataset_family,
        dataset_license=dataset_license,
        frame_stride=frame_stride,
        options=options,
        command=command,
    )


def run_camera_focal_check(
    bag: str,
    *,
    image_topic: str,
    imu_topic: str,
    camera: CameraModel,
    rotation_artifact: ImuLidarRotationArtifact,
    rotation_artifact_sha256: str,
    dataset_family: str,
    dataset_license: str,
    frame_stride: int = 4,
    max_seconds: float | None = None,
    options: FocalScaleOptions | None = None,
    command: list[str] | None = None,
) -> CameraFocalScaleArtifact:
    """Track the camera with ``camera``'s focal lengths, then regress its rates on the gyro's.

    The rotation, clock offset and gyro bias come from ``rotation_artifact``
    (digest ``rotation_artifact_sha256``); ``max_seconds`` limits the images
    tracked (the whole IMU stream is kept: only the tracked windows are read).
    """

    import hashlib
    from pathlib import Path

    from scipy.spatial.transform import Rotation

    from calibrex import __version__
    from calibrex.core.camera_focal_scale import CameraFocalProvenance, CameraFocalScaleArtifact
    from calibrex.core.provenance import git_commit
    from calibrex.data.livox_ros2 import bag_input_digest
    from calibrex.data.ros2_camera_imu import iter_gray_images, load_ros2_imu
    from calibrex.evaluation.visual_rotation import track_camera_rotations

    opts = options or FocalScaleOptions()
    if rotation_artifact.rotation_quat_xyzw is None:
        raise ValueError("the rotation artifact has no rotation estimate")
    track = track_camera_rotations(
        iter_gray_images(bag, image_topic, stride=frame_stride, max_seconds=max_seconds), camera
    )
    imu = load_ros2_imu(bag, imu_topic)
    result = estimate_focal_scale(
        GyroSeries(imu.times_s, imu.gyro_rps),
        track.windows(),
        Rotation.from_quat(rotation_artifact.rotation_quat_xyzw).as_matrix(),
        rotation_artifact.time_offset_s,
        np.asarray(rotation_artifact.gyro_bias_rps, dtype=np.float64),
        opts,
    )
    digest, _ = bag_input_digest([bag])
    return CameraFocalScaleArtifact(
        policy_status=result.status,  # type: ignore[arg-type]
        policy_reasons=list(result.reasons),
        rate_ratio=[float(value) for value in result.ratio],
        rate_ratio_std=[float(value) for value in result.ratio_std],
        holdout_rate_ratio=[float(value) for value in result.holdout_ratio],
        fx_used_px=camera.fx,
        fy_used_px=camera.fy,
        fx_estimate_px=camera.fx * float(result.ratio[1]),
        fy_estimate_px=camera.fy * float(result.ratio[0]),
        fx_std_px=camera.fx * float(result.ratio_std[1]),
        fy_std_px=camera.fy * float(result.ratio_std[0]),
        intervals=result.intervals,
        options={
            "frame_stride": frame_stride,
            "holdout_every": opts.holdout_every,
            "jackknife_groups": opts.jackknife_groups,
            "outlier_mad": opts.outlier_mad,
            "min_rate_rps": opts.min_rate_rps,
            "max_holdout_sigma": opts.max_holdout_sigma,
            "max_optical_axis_deviation_sigma": opts.max_optical_axis_deviation_sigma,
            "observable_ratio_std": opts.observable_ratio_std,
            **({"max_seconds": max_seconds} if max_seconds is not None else {}),
        },
        limitations=[
            "The ratio also absorbs anything else that scales tracked rotations, such as "
            "distortion errors or tracking attenuation; the optical-axis ratio is the check.",
            "Only fx and fy are checked; the principal point and distortion are taken as given.",
        ],
        provenance=CameraFocalProvenance(
            generator=__name__,
            generator_version=__version__,
            git_commit=git_commit(),
            command=command or [],
            dataset_family=dataset_family,
            sequence_ids=[Path(bag).name],
            image_topic=image_topic,
            rotation_artifact_sha256=rotation_artifact_sha256,
            input_sha256=hashlib.sha256(digest.encode("ascii")).hexdigest(),
            dataset_license=dataset_license,
        ),
    )
