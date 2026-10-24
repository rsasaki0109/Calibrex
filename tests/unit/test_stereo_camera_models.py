"""Wide-angle model geometry and Kalibr ingestion for stereo rectification."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage
from scipy.spatial.transform import Rotation

from calibrex.evaluation.fisheye_stereo import (
    EquidistantCamera,
    read_kalibr_fisheye_pair,
    rectify_fisheye_pair,
)
from calibrex.evaluation.stereo_camera_models import DoubleSphereCamera, MeiCamera

CAMERAS = [
    EquidistantCamera(180, 180, 159.5, 119.5, (-0.02, 0.001, 0, 0), 320, 240),
    MeiCamera(180, 180, 159.5, 119.5, 0.8, (-0.01, 0.001, 0.0002, -0.0001), 320, 240),
    DoubleSphereCamera(180, 180, 159.5, 119.5, 0.5, 0.6, 320, 240),
]


@pytest.mark.parametrize("camera", CAMERAS)
def test_projected_rays_round_trip(camera: object) -> None:
    rng = np.random.default_rng(5)
    points = rng.normal(size=(200, 3))
    points[:, 2] = np.abs(points[:, 2]) + 1
    u, v, valid = camera.project(points)  # type: ignore[attr-defined]
    assert valid.all()
    rays = camera.unproject(u, v)  # type: ignore[attr-defined]
    np.testing.assert_allclose(rays, points / np.linalg.norm(points, axis=1)[:, None], atol=1e-9)


@pytest.mark.parametrize("camera", CAMERAS[1:])
def test_wide_rays_and_invalid_inverse_domains(camera: object) -> None:
    theta = np.radians([0, 30, 60, 100, 120])
    points = np.c_[np.sin(theta), np.zeros(len(theta)), np.cos(theta)]
    u, v, valid = camera.project(points)  # type: ignore[attr-defined]
    assert valid.all()
    np.testing.assert_allclose(camera.unproject(u, v), points, atol=1e-8)  # type: ignore[attr-defined]
    _, _, valid = camera.project(np.array([[0.0, 0.0, 0.0], [np.nan, 1, 0]]))  # type: ignore[attr-defined]
    assert not valid.any()
    rays = camera.unproject(np.array([np.inf]), np.array([0.0]))  # type: ignore[attr-defined]
    assert np.isnan(rays).all()


@pytest.mark.parametrize("camera", CAMERAS)
def test_rectification_maps_and_disparity_recover_metric_depth(camera: object) -> None:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", [2, 7, 0], degrees=True).as_matrix()
    transform[:3, 3] = -transform[:3, :3] @ np.array([0.2, 0, 0])
    rect = rectify_fisheye_pair(camera, camera, transform)  # type: ignore[arg-type]
    points = np.array([[x, y, z] for x in [-0.5, 0.2] for y in [-0.3, 0.3] for z in [3, 6, 12]])
    p = points @ rect.rectified_from_left.T
    u, v = rect.fx * p[:, 0] / p[:, 2] + rect.cx, rect.fx * p[:, 1] / p[:, 2] + rect.cy
    disparity = rect.fx * 0.2 / p[:, 2]
    np.testing.assert_allclose(rect.stereo.base_px_m / disparity, p[:, 2], atol=1e-12)
    for mapping, raw, columns in [
        (rect.left_map, points, u),
        (rect.right_map, points @ transform[:3, :3].T + transform[:3, 3], u - disparity),
    ]:
        raw_u, raw_v, _ = camera.project(raw)  # type: ignore[attr-defined]
        np.testing.assert_allclose(
            ndimage.map_coordinates(mapping[1], [v, columns], order=1), raw_u, atol=0.03
        )
        np.testing.assert_allclose(
            ndimage.map_coordinates(mapping[0], [v, columns], order=1), raw_v, atol=0.03
        )


@pytest.mark.parametrize(
    "model,intrinsics,distortion,coeffs,kind",
    [
        ("pinhole", [180, 180, 160, 120], "equidistant", [0, 0, 0, 0], EquidistantCamera),
        ("omni", [0.8, 180, 180, 160, 120], "radtan", [0, 0, 0, 0], MeiCamera),
        ("ds", [0.5, 0.6, 180, 180, 160, 120], "none", [], DoubleSphereCamera),
    ],
)
def test_kalibr_models_are_read_without_reinterpreting_intrinsic_order(
    model: str,
    intrinsics: list[float],
    distortion: str,
    coeffs: list[float],
    kind: type,
) -> None:
    entry = {
        "camera_model": model,
        "intrinsics": intrinsics,
        "distortion_model": distortion,
        "distortion_coeffs": coeffs,
        "resolution": [320, 240],
    }
    tf = np.eye(4)
    tf[0, 3] = -0.2
    chain = {"cam0": entry, "cam1": {**entry, "T_cn_cnm1": tf.tolist()}}
    left, right, got = read_kalibr_fisheye_pair(chain)
    assert isinstance(left, kind) and isinstance(right, kind)
    np.testing.assert_array_equal(got, tf)
    with pytest.raises(ValueError, match="immediately follow"):
        read_kalibr_fisheye_pair({**chain, "cam2": chain["cam1"]}, "cam0", "cam2")
    with pytest.raises(ValueError, match="rigid"):
        bad = tf.copy()
        bad[0, 0] = 2
        read_kalibr_fisheye_pair({**chain, "cam1": {**entry, "T_cn_cnm1": bad.tolist()}})


@pytest.mark.parametrize("camera", CAMERAS[1:])
def test_front_projection_agrees_with_existing_depth_camera_engine(camera: object) -> None:
    from calibrex.solvers.borer_depth_to_depth_solver import DepthToDepthCameraModel, _project

    model = "mei" if isinstance(camera, MeiCamera) else "double_sphere"
    reference = DepthToDepthCameraModel(
        width=320,
        height=240,
        fx=180,
        fy=180,
        cx=159.5,
        cy=119.5,
        projection=model,
        xi=camera.xi,
        alpha=getattr(camera, "alpha", 0.5),  # type: ignore[arg-type,attr-defined]
        distortion=getattr(camera, "distortion", (0.0, 0.0, 0.0, 0.0)),
    )
    points = np.array([[-2.0, 1.0, 3.0], [0.1, -0.4, 1.0], [0.0, 0.0, 5.0]])
    u, v, valid = camera.project(points)  # type: ignore[attr-defined]
    expected_u, expected_v, expected_valid = _project(points, reference)
    np.testing.assert_allclose(u, expected_u, atol=1e-10)
    np.testing.assert_allclose(v, expected_v, atol=1e-10)
    np.testing.assert_array_equal(valid, expected_valid)
