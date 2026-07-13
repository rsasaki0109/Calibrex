import math

import numpy as np
import pytest

from calibrex.core.geometry import SE3
from calibrex.solvers import (
    HornPointLidarCameraSolver,
    HornPointObservation,
    HornPointSolverOptions,
)


def _truth() -> SE3:
    axis = np.asarray([0.3, -0.7, 0.2], dtype=float)
    axis /= np.linalg.norm(axis)
    half_angle = math.radians(41.0) / 2.0
    sine = math.sin(half_angle)
    return SE3(
        (0.24, -0.17, 0.31),
        (
            float(axis[0] * sine),
            float(axis[1] * sine),
            float(axis[2] * sine),
            math.cos(half_angle),
        ),
    )


def _observations(count: int = 36) -> list[HornPointObservation]:
    rng = np.random.default_rng(29)
    truth = _truth()
    return [
        HornPointObservation(
            frame_id=f"capture-{index:03d}",
            camera_point_m=truth.transform_point(tuple(float(value) for value in point)),
            lidar_point_m=tuple(float(value) for value in point),
        )
        for index, point in enumerate(rng.normal(size=(count, 3)))
    ]


def _transform_errors(estimated: SE3, truth: SE3) -> tuple[float, float]:
    relative = truth.inverse().compose(estimated)
    translation = float(np.linalg.norm(relative.translation_m))
    rotation = math.degrees(2.0 * math.acos(min(1.0, abs(relative.rotation_quat_xyzw[3]))))
    return translation, rotation


def test_horn_recovers_truth_with_holdout_observability_and_probes() -> None:
    result = HornPointLidarCameraSolver().solve(
        _observations(),
        HornPointSolverOptions(holdout_ratio=0.25, split_seed=7),
    )

    assert result.status == "converged"
    assert result.transform_camera_lidar is not None
    translation_error, rotation_error = _transform_errors(result.transform_camera_lidar, _truth())
    assert translation_error < 1.0e-12
    assert rotation_error < 1.0e-6
    assert result.train_evaluation.point_rmse_m is not None
    assert result.train_evaluation.point_rmse_m < 1.0e-12
    assert result.holdout_evaluation.point_rmse_m is not None
    assert result.holdout_evaluation.point_rmse_m < 1.0e-12
    assert set(result.train_frame_ids).isdisjoint(result.holdout_frame_ids)
    assert len(result.train_frame_ids) + len(result.holdout_frame_ids) == 36
    assert result.centered_geometry_rank == 3
    assert result.joint_rank == 6
    assert result.joint_condition_number is not None
    assert result.quaternion_normalized_eigengap is not None
    assert result.quaternion_normalized_eigengap > 0.0
    assert result.rms_scale_ratio_camera_over_lidar == pytest.approx(1.0)
    assert len(result.probes) == 12
    assert all(probe.detectable is True for probe in result.probes)
    document = result.as_dict()
    assert document["method"] == "horn_unit_quaternion_point_alignment_irls/v0.1"
    assert document["paper"]["doi"] == "10.1364/JOSAA.4.000629"
    assert document["scale_policy"].startswith("rigid extrinsic scale fixed to one")


def test_horn_is_symmetric_when_correspondence_direction_is_reversed() -> None:
    observations = _observations()
    options = HornPointSolverOptions(holdout_ratio=0.2, split_seed=11)
    forward = HornPointLidarCameraSolver().solve(observations, options)
    reverse = HornPointLidarCameraSolver().solve(
        [
            HornPointObservation(
                frame_id=item.frame_id,
                camera_point_m=item.lidar_point_m,
                lidar_point_m=item.camera_point_m,
                weight=item.weight,
            )
            for item in observations
        ],
        options,
    )

    assert forward.transform_camera_lidar is not None
    assert reverse.transform_camera_lidar is not None
    translation_error, rotation_error = _transform_errors(
        reverse.transform_camera_lidar,
        forward.transform_camera_lidar.inverse(),
    )
    assert translation_error < 1.0e-12
    assert rotation_error < 1.0e-6
    assert forward.train_frame_ids == reverse.train_frame_ids
    assert forward.holdout_frame_ids == reverse.holdout_frame_ids


def test_horn_irls_is_order_invariant_and_limits_one_gross_outlier() -> None:
    observations = _observations(48)
    outlier = observations[17]
    observations[17] = HornPointObservation(
        frame_id=outlier.frame_id,
        camera_point_m=(
            outlier.camera_point_m[0] + 2.0,
            outlier.camera_point_m[1] - 1.5,
            outlier.camera_point_m[2] + 1.0,
        ),
        lidar_point_m=outlier.lidar_point_m,
    )
    options = HornPointSolverOptions(holdout_ratio=0.0, huber_delta_m=0.02)

    forward = HornPointLidarCameraSolver().solve(observations, options)
    reversed_result = HornPointLidarCameraSolver().solve(list(reversed(observations)), options)

    assert forward.transform_camera_lidar is not None
    assert reversed_result.transform_camera_lidar is not None
    translation_error, rotation_error = _transform_errors(forward.transform_camera_lidar, _truth())
    assert translation_error < 0.002
    assert rotation_error < 0.05
    order_translation, order_rotation = _transform_errors(
        reversed_result.transform_camera_lidar,
        forward.transform_camera_lidar,
    )
    assert order_translation < 1.0e-12
    assert order_rotation < 1.0e-6


def test_horn_rejects_collinear_geometry_and_duplicate_ids() -> None:
    truth = _truth()
    collinear = [
        HornPointObservation(
            frame_id=f"line-{index}",
            camera_point_m=truth.transform_point((float(index), 0.0, 0.0)),
            lidar_point_m=(float(index), 0.0, 0.0),
        )
        for index in range(8)
    ]

    result = HornPointLidarCameraSolver().solve(
        collinear, HornPointSolverOptions(holdout_ratio=0.0)
    )

    assert result.status == "degenerate_geometry"
    assert result.transform_camera_lidar is None
    assert result.centered_geometry_rank == 1
    duplicate = _observations(4)
    duplicate[1] = HornPointObservation(
        duplicate[0].frame_id,
        duplicate[1].camera_point_m,
        duplicate[1].lidar_point_m,
    )
    with pytest.raises(ValueError, match="frame IDs must be unique"):
        HornPointLidarCameraSolver().solve(duplicate)


def test_horn_accepts_noncollinear_planar_centres() -> None:
    truth = _truth()
    points = [
        (-1.0, -0.5, 0.0),
        (-0.2, 0.8, 0.0),
        (0.4, -0.7, 0.0),
        (1.2, 0.6, 0.0),
        (0.7, 1.4, 0.0),
        (-1.3, 1.1, 0.0),
    ]
    observations = [
        HornPointObservation(
            frame_id=f"plane-{index}",
            camera_point_m=truth.transform_point(point),
            lidar_point_m=point,
        )
        for index, point in enumerate(points)
    ]

    result = HornPointLidarCameraSolver().solve(
        observations, HornPointSolverOptions(holdout_ratio=0.0)
    )

    assert result.status == "converged"
    assert result.centered_geometry_rank == 2
    assert result.joint_rank == 6
    assert result.transform_camera_lidar is not None
    translation_error, rotation_error = _transform_errors(result.transform_camera_lidar, truth)
    assert translation_error < 1.0e-12
    assert rotation_error < 1.0e-6


@pytest.mark.parametrize(
    "options",
    [
        HornPointSolverOptions(min_train_observations=2),
        HornPointSolverOptions(holdout_ratio=1.0),
        HornPointSolverOptions(huber_delta_m=0.0),
        HornPointSolverOptions(max_condition_number=1.0),
    ],
)
def test_horn_rejects_invalid_options(options: HornPointSolverOptions) -> None:
    with pytest.raises(ValueError):
        HornPointLidarCameraSolver().solve(_observations(), options)
