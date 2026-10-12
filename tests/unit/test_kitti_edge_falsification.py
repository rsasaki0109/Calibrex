"""Data-only KITTI camera-LiDAR falsification: geometry and decision rule."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from calibrex.core.geometry import SE3
from calibrex.data.kitti import (
    find_camera_lidar_pairs,
    read_camera_projection_model,
    read_velodyne_to_camera_transform,
)
from calibrex.evaluation.kitti_edge_falsification import (
    KITTIEdgeTrial,
    _decision,
    kitti_edge_candidates,
    select_kitti_pairs,
)
from calibrex.evaluation.kitti_falsification_benchmark import kitti_benchmark_candidates

FIXTURE = (
    "examples/public_datasets/kitti_lidar_camera_evidence/2011_09_26/2011_09_26_drive_0005_sync"
)


# KITTI's 2011_09_26 calibration (calib_velo_to_cam.txt and calib_cam_to_cam.txt).
VELO_R = "7.533745e-03 -9.999714e-01 -6.166020e-04 1.480249e-02 7.280733e-04 -9.998902e-01"
VELO_R += " 9.998621e-01 7.523790e-03 1.480755e-02"
VELO_T = "-4.069766e-03 -7.631618e-02 -2.717806e-01"
R_RECT = "9.999239e-01 9.837760e-03 -7.445048e-03 -9.869795e-03 9.999421e-01 -4.278459e-03"
R_RECT += " 7.402527e-03 4.351614e-03 9.999631e-01"
P_RECT = "7.215377e+02 0 6.095593e+02 4.485728e+01 0 7.215377e+02 1.728540e+02 2.163791e-01"
P_RECT += " 0 0 1 2.745884e-03"


def test_candidates_project_like_the_protocols_rectified_projection(tmp_path: Path) -> None:
    dataset = tmp_path / "2011_09_26_drive_0005_sync"
    dataset.mkdir()
    (tmp_path / "calib_velo_to_cam.txt").write_text(f"R: {VELO_R}\nT: {VELO_T}\n", encoding="utf-8")
    (tmp_path / "calib_cam_to_cam.txt").write_text(
        f"S_rect_02: 1242 375\nR_rect_00: {R_RECT}\nP_rect_02: {P_RECT}\n", encoding="utf-8"
    )
    reference = read_velodyne_to_camera_transform(dataset)
    model = read_camera_projection_model(dataset)
    assert reference is not None and model is not None
    candidates = kitti_benchmark_candidates(reference)
    camera, transforms = kitti_edge_candidates(dataset, candidates)
    projection = np.asarray(model.projection_matrix).reshape(3, 4)
    rectification = np.asarray(model.rectification_matrix).reshape(3, 3)
    lidar = np.asarray([[12.0, 1.5, -0.8], [25.0, -4.0, 0.5], [6.0, 2.0, -1.2]])

    for name, candidate in candidates.items():
        # the protocol: T_camera0_lidar, then R_rect_00, then P_rect_02
        camera0 = np.asarray([candidate.transform_point(tuple(point)) for point in lidar])
        homogeneous = np.c_[camera0 @ rectification.T, np.ones(len(lidar))] @ projection.T
        expected = homogeneous[:, :2] / homogeneous[:, 2:3]

        transform = transforms[name]
        u, v, valid = camera.project(lidar @ transform[:3, :3].T + transform[:3, 3])

        assert np.all(valid)
        np.testing.assert_allclose(np.c_[u, v], expected, atol=1e-6)


def test_frame_selection_takes_the_first_or_spreads_over_the_drive() -> None:
    every = find_camera_lidar_pairs(FIXTURE, max_pairs=1000)
    assert len(every) >= 2
    first = select_kitti_pairs(FIXTURE, 1, "first")
    assert [pair.camera_index for pair in first] == [every[0].camera_index]
    spread = select_kitti_pairs(FIXTURE, 2, "spread")
    assert [pair.camera_index for pair in spread] == [
        every[0].camera_index,
        every[-1].camera_index,
    ]


def _trial(candidate: str, verdict: str) -> KITTIEdgeTrial:
    return KITTIEdgeTrial(
        candidate_id=candidate,  # type: ignore[arg-type]
        verdict=verdict,  # type: ignore[arg-type]
        frames=50,
        edge_points=1,
    )


@pytest.mark.parametrize(
    ("reference", "known_bad", "status"),
    [
        ("pass", "fail", "pass"),
        ("pass", "warn", "inconclusive"),
        ("warn", "fail", "inconclusive"),
        ("inconclusive", "fail", "inconclusive"),
        ("fail", "fail", "fail"),
        ("pass", "pass", "fail"),
    ],
)
def test_the_reference_must_pass_and_the_known_bad_must_fail(
    reference: str, known_bad: str, status: str
) -> None:
    decided, reason = _decision(
        [_trial("dataset_reference", reference), _trial("known_bad", known_bad)]
    )
    assert decided == status and reason


def test_known_bad_candidate_is_reproduced_from_the_reference() -> None:
    reference = SE3((0.1, -0.2, 0.3), (0.0, 0.0, 0.0, 1.0))
    candidates = kitti_benchmark_candidates(reference)
    assert candidates["dataset_reference"] == reference
    assert candidates["known_bad"].translation_m == pytest.approx((0.35, -0.2, 0.3))
