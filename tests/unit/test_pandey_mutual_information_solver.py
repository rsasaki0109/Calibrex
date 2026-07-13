from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3, quaternion_xyzw_from_rotation_matrix
from calibrex.solvers.pandey_mutual_information_solver import (
    MutualInformationCameraModel,
    MutualInformationObservation,
    PandeyMutualInformationOptions,
    PandeyMutualInformationSolver,
    evaluate_mutual_information,
)


def test_pandey_mutual_information_truth_scores_above_known_bad_pose() -> None:
    observations, truth = _synthetic_observations()
    options = _options(max_iterations=3)
    truth_score = evaluate_mutual_information(observations, truth, options)
    bad = SE3(
        (truth.translation_m[0] + 0.08, truth.translation_m[1], truth.translation_m[2]),
        truth.rotation_quat_xyzw,
    )
    bad_score = evaluate_mutual_information(observations, bad, options)

    assert truth_score.projected_point_count > 1_000
    assert truth_score.normalized_mutual_information > bad_score.normalized_mutual_information
    assert truth_score.raw_mutual_information > bad_score.raw_mutual_information


def test_pandey_mutual_information_solver_improves_train_and_keeps_holdout() -> None:
    observations, truth = _synthetic_observations()
    initial_rotation = _euler_matrix(
        math.radians(1.3), math.radians(-1.7), math.radians(2.25)
    )
    initial = SE3(
        (truth.translation_m[0] + 0.01, truth.translation_m[1] - 0.01, truth.translation_m[2]),
        quaternion_xyzw_from_rotation_matrix(initial_rotation.reshape(-1)),
    )
    options = _options(max_iterations=60)
    initial_score = evaluate_mutual_information(observations, initial, options)

    result = PandeyMutualInformationSolver().solve(observations, initial, options)

    assert result.status in {"converged", "max_iterations"}
    assert result.transform_camera_lidar is not None
    assert result.iterations
    assert result.train_evaluation.raw_mutual_information > initial_score.raw_mutual_information
    assert result.holdout_evaluation.projected_point_count > 100
    assert len(result.probes) == 12
    assert sum(probe.detectable for probe in result.probes) >= 6
    assert result.curvature is not None
    assert result.curvature.parameter_dimension == 6
    assert math.dist(
        result.transform_camera_lidar.translation_m, truth.translation_m
    ) < 0.011
    assert _rotation_error_deg(result.transform_camera_lidar, truth) < 0.1
    payload = result.as_dict()
    assert payload["paper_doi"] == "10.1609/aaai.v26i1.8379"
    assert "not covariance" in str(payload["curvature_claim"])


def test_pandey_mutual_information_rejects_too_few_training_frames() -> None:
    observations, truth = _synthetic_observations(frame_count=1)
    result = PandeyMutualInformationSolver().solve(
        observations,
        truth,
        PandeyMutualInformationOptions(min_train_observations=2, holdout_ratio=0.0),
    )

    assert result.status == "insufficient_observations"
    assert result.transform_camera_lidar is None
    assert result.curvature is None


def _options(*, max_iterations: int) -> PandeyMutualInformationOptions:
    return PandeyMutualInformationOptions(
        holdout_ratio=0.2,
        split_seed=3,
        histogram_bins=24,
        min_projected_points=80,
        max_iterations=max_iterations,
        initial_step_size=0.01,
        max_step_size=0.025,
        gradient_steps=(
            0.003,
            0.003,
            0.003,
            math.radians(0.08),
            math.radians(0.08),
            math.radians(0.08),
        ),
    )


def _synthetic_observations(
    *, frame_count: int = 5
) -> tuple[list[MutualInformationObservation], SE3]:
    camera = MutualInformationCameraModel(
        width=128,
        height=96,
        fx=92.0,
        fy=90.0,
        cx=63.5,
        cy=47.5,
    )
    rotation = _euler_matrix(math.radians(1.0), math.radians(-1.5), math.radians(2.0))
    truth = SE3(
        (0.08, -0.04, 0.03),
        quaternion_xyzw_from_rotation_matrix(rotation.reshape(-1)),
    )
    inverse = truth.inverse()
    observations: list[MutualInformationObservation] = []
    for frame_index in range(frame_count):
        rng = np.random.default_rng(7100 + frame_index)
        image = _smooth_random_image(rng, camera.height, camera.width)
        u = rng.uniform(8.0, camera.width - 9.0, 900)
        v = rng.uniform(8.0, camera.height - 9.0, 900)
        depth = rng.uniform(3.0, 14.0, 900)
        camera_points = np.column_stack(
            (
                (u - camera.cx) * depth / camera.fx,
                (v - camera.cy) * depth / camera.fy,
                depth,
            )
        )
        lidar_points = np.asarray(
            [inverse.transform_point(point) for point in camera_points], dtype=float
        )
        reflectivity = _bilinear(image, u, v) + rng.normal(0.0, 0.01, u.shape)
        observations.append(
            MutualInformationObservation(
                frame_id=f"frame-{frame_index:02d}",
                luminance=image,
                lidar_points=lidar_points,
                reflectivity=reflectivity,
                camera=camera,
            )
        )
    return observations, truth


def _smooth_random_image(
    rng: np.random.Generator, height: int, width: int
) -> NDArray[np.float64]:
    image = rng.random((height, width))
    for _ in range(4):
        padded = np.pad(image, 1, mode="reflect")
        image = sum(
            padded[row : row + height, column : column + width]
            for row in range(3)
            for column in range(3)
        ) / 9.0
    return image


def _bilinear(
    image: NDArray[np.float64], u: NDArray[np.float64], v: NDArray[np.float64]
) -> NDArray[np.float64]:
    left = np.floor(u).astype(int)
    top = np.floor(v).astype(int)
    du = u - left
    dv = v - top
    return (
        (1.0 - du) * (1.0 - dv) * image[top, left]
        + du * (1.0 - dv) * image[top, left + 1]
        + (1.0 - du) * dv * image[top + 1, left]
        + du * dv * image[top + 1, left + 1]
    )


def _euler_matrix(roll: float, pitch: float, yaw: float) -> NDArray[np.float64]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        )
    )


def _rotation_error_deg(left: SE3, right: SE3) -> float:
    dot = abs(float(np.dot(left.rotation_quat_xyzw, right.rotation_quat_xyzw)))
    return math.degrees(2.0 * math.acos(min(1.0, dot)))
