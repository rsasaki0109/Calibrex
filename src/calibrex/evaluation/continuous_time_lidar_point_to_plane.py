"""Synthetic recovery for continuous-time LiDAR point-to-plane factors."""

from __future__ import annotations

from typing import Literal

import numpy as np

from calibrex.core.continuous_time_lidar_point_to_plane import (
    ContinuousTimeLidarPointToPlaneArtifact,
    ContinuousTimeLidarPointToPlaneProvenance,
)
from calibrex.core.continuous_time_sparse import (
    ContinuousTimeTrajectoryFitOptions,
    ContinuousTimeTrajectoryFitProblem,
    TrajectoryPointToPlaneMeasurement,
    fit_continuous_trajectory,
    interpolate_pose_at,
    point_to_plane_rmse,
)
from calibrex.core.geometry import SE3
from calibrex.core.provenance import git_commit
from calibrex.core.se3_manifold import se3_exp, se3_log

_SYNTHETIC_SEED = 20260820
_KNOWN_BAD_TRANSLATION_M = (0.0, 0.0, 0.15)
_PLANES: tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...] = (
    ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ((0.0, 0.8, 0.0), (0.0, 1.0, 0.0)),
    ((1.2, 0.0, 0.0), (1.0, 0.0, 0.0)),
)


def run_synthetic_lidar_point_to_plane_recovery(
    *,
    seed: int = _SYNTHETIC_SEED,
    command: list[str] | None = None,
    recovery_id: str = "ct-lidar-p2p-synthetic",
) -> ContinuousTimeLidarPointToPlaneArtifact:
    """Recover knots from plane residuals with holdout and a signed control."""

    rng = np.random.default_rng(seed)
    timestamps = tuple(float(index) for index in range(6))
    truth = _truth_knots(timestamps)
    samples: list[tuple[float, int]] = []
    for sample_index in range(90):
        timestamp = float(rng.uniform(0.05, 4.95))
        plane_index = int(sample_index % len(_PLANES))
        samples.append((timestamp, plane_index))
    split_start = 0.35 * timestamps[-1]
    split_end = 0.65 * timestamps[-1]
    train_samples = [
        item for item in samples if item[0] <= split_start or item[0] > split_end
    ]
    holdout_samples = [
        item for item in samples if split_start < item[0] <= split_end
    ]
    train = _measurements(truth, timestamps, train_samples, rng, prefix="train")
    holdout = _measurements(truth, timestamps, holdout_samples, rng, prefix="holdout")
    initial = tuple(
        se3_exp(knot, rng.normal(0.0, 0.04, 6)) for knot in truth
    )
    result = fit_continuous_trajectory(
        ContinuousTimeTrajectoryFitProblem(
            knot_timestamps=timestamps,
            initial_knot_poses=initial,
            point_to_plane_measurements=train,
        ),
        ContinuousTimeTrajectoryFitOptions(max_iterations=80),
    )
    train_rmse = result.final_point_to_plane_rmse
    if train_rmse is None:
        raise ValueError("point-to-plane recovery produced no train RMSE")
    holdout_rmse = point_to_plane_rmse(holdout, timestamps, result.knot_poses)
    if holdout_rmse is None:
        raise ValueError("point-to-plane recovery produced no holdout RMSE")
    known_bad = tuple(
        se3_exp(
            knot,
            np.concatenate(
                [np.asarray(_KNOWN_BAD_TRANSLATION_M, dtype=float), np.zeros(3)]
            ),
        )
        for knot in result.knot_poses
    )
    known_bad_rmse = point_to_plane_rmse(holdout, timestamps, known_bad)
    if known_bad_rmse is None:
        raise ValueError("point-to-plane recovery produced no known-bad RMSE")
    knot_errors = [
        float(np.max(np.abs(se3_log(estimate, knot))))
        for estimate, knot in zip(result.knot_poses, truth, strict=True)
    ]
    max_knot_error = max(knot_errors)
    known_bad_delta = known_bad_rmse - holdout_rmse
    policy_status, policy_reason = _policy(
        max_knot_error=max_knot_error,
        holdout_rmse=holdout_rmse,
        known_bad_delta=known_bad_delta,
    )
    return ContinuousTimeLidarPointToPlaneArtifact(
        recovery_id=recovery_id,
        interpolation="screw_linear",
        seed=seed,
        knot_count=len(truth),
        train_measurement_count=len(train),
        holdout_measurement_count=len(holdout),
        fit_status=result.status,
        max_knot_error=max_knot_error,
        train_point_to_plane_rmse_m=train_rmse,
        holdout_point_to_plane_rmse_m=holdout_rmse,
        known_bad_holdout_rmse_m=known_bad_rmse,
        known_bad_rmse_delta_m=known_bad_delta,
        known_bad_translation_m=_KNOWN_BAD_TRANSLATION_M,
        policy_status=policy_status,
        policy_reason=policy_reason,
        provenance=ContinuousTimeLidarPointToPlaneProvenance(
            generator=__name__,
            generator_version="0.1",
            git_commit=git_commit(),
            command=command or [],
            seed=seed,
        ),
    )


def _truth_knots(timestamps: tuple[float, ...]) -> tuple[SE3, ...]:
    knots = []
    for index, timestamp in enumerate(timestamps):
        translation = np.array([0.25 * timestamp, 0.05 * timestamp, 0.02 * timestamp])
        rotation = np.array([0.0, 0.0, 0.04 * index])
        knots.append(se3_exp(SE3.identity(), np.concatenate([translation, rotation])))
    return tuple(knots)


def _measurements(
    knots: tuple[SE3, ...],
    timestamps: tuple[float, ...],
    samples: list[tuple[float, int]],
    rng: np.random.Generator,
    *,
    prefix: str,
) -> tuple[TrajectoryPointToPlaneMeasurement, ...]:
    measurements: list[TrajectoryPointToPlaneMeasurement] = []
    for index, (timestamp, plane_index) in enumerate(samples):
        plane_point, plane_normal = _PLANES[plane_index]
        pose = interpolate_pose_at(knots, timestamps, timestamp)
        world = _sample_plane_point(plane_point, plane_normal, rng)
        body = pose.inverse().transform_point(world)
        noisy_body = np.asarray(body, dtype=float) + rng.normal(0.0, 0.003, 3)
        measurements.append(
            TrajectoryPointToPlaneMeasurement(
                measurement_id=f"{prefix}-{index:04d}",
                timestamp_sec=timestamp,
                point_body_m=tuple(float(value) for value in noisy_body),
                plane_point_world_m=plane_point,
                plane_normal_world=plane_normal,
            )
        )
    return tuple(measurements)


def _sample_plane_point(
    plane_point: tuple[float, float, float],
    plane_normal: tuple[float, float, float],
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    normal = np.asarray(plane_normal, dtype=float)
    origin = np.asarray(plane_point, dtype=float)
    tangent_a = np.cross(normal, np.array([0.0, 0.0, 1.0]))
    if float(np.linalg.norm(tangent_a)) < 1.0e-8:
        tangent_a = np.cross(normal, np.array([0.0, 1.0, 0.0]))
    tangent_a /= float(np.linalg.norm(tangent_a))
    tangent_b = np.cross(normal, tangent_a)
    point = origin + rng.uniform(-0.4, 0.4) * tangent_a + rng.uniform(-0.4, 0.4) * tangent_b
    return tuple(float(value) for value in point)


def _policy(
    *,
    max_knot_error: float,
    holdout_rmse: float,
    known_bad_delta: float,
) -> tuple[Literal["pass", "fail"], str]:
    if max_knot_error >= 0.08:
        return (
            "fail",
            f"knot recovery error {max_knot_error:.4f} exceeds the 0.08 budget",
        )
    if holdout_rmse >= 0.03:
        return (
            "fail",
            f"holdout point-to-plane RMSE {holdout_rmse:.4f} m is too large",
        )
    if known_bad_delta <= 0.05:
        return (
            "fail",
            (
                f"known-bad +{_KNOWN_BAD_TRANSLATION_M[2]:.2f} m z control only "
                f"increased holdout RMSE by {known_bad_delta:.4f} m"
            ),
        )
    return (
        "pass",
        (
            "synthetic knot recovery meets the error budget, holdout RMSE stays "
            "tight, and the signed z-translation control increases holdout error"
        ),
    )
