import math

import numpy as np

from calibrex.core.geometry import SE3, rotate_vector_xyzw
from calibrex.solvers.planar_board_lidar_camera_solver import (
    OrientedPlane,
    PlanarBoardLidarCameraSolver,
    PlanarBoardObservation,
    PlanarBoardSolverOptions,
)


def _observations(transform: SE3, count: int = 15) -> list[PlanarBoardObservation]:
    rng = np.random.default_rng(12)
    result: list[PlanarBoardObservation] = []
    for index in range(count):
        normal = rng.normal(size=3)
        normal /= np.linalg.norm(normal)
        lidar_offset = float(rng.uniform(-4.0, -1.0))
        camera_normal = rotate_vector_xyzw(transform.rotation_quat_xyzw, tuple(normal))
        camera_offset = lidar_offset - sum(
            value * translation
            for value, translation in zip(camera_normal, transform.translation_m, strict=True)
        )
        result.append(
            PlanarBoardObservation(
                f"frame-{index:03d}",
                OrientedPlane(camera_normal, camera_offset),
                OrientedPlane(tuple(normal), lidar_offset),
            )
        )
    return result


def _rotation_error_deg(left: SE3, right: SE3) -> float:
    dot = abs(
        sum(
            a * b
            for a, b in zip(
                left.rotation_quat_xyzw,
                right.rotation_quat_xyzw,
                strict=True,
            )
        )
    )
    return math.degrees(2.0 * math.acos(min(1.0, dot)))


def test_recovers_six_dof_and_holdout() -> None:
    truth = SE3((0.42, -0.17, 0.09), (0.12, -0.08, 0.18, 0.972))
    result = PlanarBoardLidarCameraSolver().solve(_observations(truth))

    assert result.status == "converged"
    assert result.transform_camera_lidar is not None
    assert _rotation_error_deg(result.transform_camera_lidar, truth) < 1.0e-5
    assert (
        np.linalg.norm(
            np.asarray(result.transform_camera_lidar.translation_m)
            - np.asarray(truth.translation_m)
        )
        < 1.0e-8
    )
    assert result.holdout_evaluation.offset_rmse_m is not None
    assert result.holdout_evaluation.offset_rmse_m < 1.0e-8
    assert result.normal_rank == 3
    assert len(result.probes) == 12
    assert all(probe.detectable is True for probe in result.probes)


def test_known_bad_probes_are_serialized_with_provenance() -> None:
    truth = SE3((0.42, -0.17, 0.09), (0.12, -0.08, 0.18, 0.972))
    result = PlanarBoardLidarCameraSolver().solve(_observations(truth))

    payload = result.as_dict()
    assert payload["method"] == "zhang_pless_plane_correspondence_irls/v0.1"
    assert payload["paper_doi"] == "10.1109/IROS.2004.1389752"
    probes = payload["known_bad_probes"]
    assert isinstance(probes, list)
    assert {probe["dof"] for probe in probes} == {
        "x", "y", "z", "roll", "pitch", "yaw"
    }


def test_normal_sign_is_resolved_and_recorded() -> None:
    truth = SE3((0.2, 0.1, -0.3), (0.03, 0.08, -0.1, 0.991))
    observations = _observations(truth)
    chosen = observations[2]
    observations[2] = PlanarBoardObservation(
        chosen.frame_id,
        chosen.camera_plane,
        OrientedPlane(
            tuple(-value for value in chosen.lidar_plane.normal),
            -chosen.lidar_plane.offset_m,
        ),
    )
    result = PlanarBoardLidarCameraSolver().solve(observations, initial_transform=truth)

    assert result.status == "converged"
    assert chosen.frame_id in result.flipped_lidar_normal_frame_ids


def test_parallel_board_poses_are_rejected() -> None:
    observations = [
        PlanarBoardObservation(
            f"parallel-{index}",
            OrientedPlane((0.0, 0.0, 1.0), -2.0 - index),
            OrientedPlane((0.0, 0.0, 1.0), -2.0 - index),
        )
        for index in range(8)
    ]
    result = PlanarBoardLidarCameraSolver().solve(observations)

    assert result.status == "degenerate_normals"
    assert result.normal_rank == 1


def test_too_few_training_poses_are_rejected() -> None:
    truth = SE3.identity()
    result = PlanarBoardLidarCameraSolver().solve(
        _observations(truth, count=3),
        options=PlanarBoardSolverOptions(holdout_ratio=0.34),
    )

    assert result.status == "insufficient_observations"
