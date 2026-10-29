import numpy as np
from tools.joint_metric_pose import depth_residual

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_depth_to_depth_solver import (
    DepthToDepthCameraModel,
    DepthToDepthObservation,
)


def test_bilinear_metric_z_and_range_equivalence():
    camera = DepthToDepthCameraModel(width=16, height=12, fx=10.0, fy=10.0, cx=7.5, cy=5.5)
    v, u = np.indices((12, 16))
    depth = 8 + 0.1 * u + 0.2 * v
    pu, pv = np.array([5.25, 8.5]), np.array([4.3, 7.2])
    z = 8 + 0.1 * pu + 0.2 * pv
    points = np.stack([(pu - 7.5) * z / 10, (pv - 5.5) * z / 10, z], axis=1)
    radial = depth * np.sqrt(1 + ((u - 7.5) / 10) ** 2 + ((v - 5.5) / 10) ** 2)
    for d, kind in [(depth, "metric_z"), (radial, "metric_range")]:
        obs = DepthToDepthObservation("test", d, points, camera, kind)
        residual, valid = depth_residual(obs, SE3.identity())
        assert valid.all()
        np.testing.assert_allclose(residual, 0.0, atol=1e-12)


def test_invalid_samples_keep_fixed_penalty_and_population():
    camera = DepthToDepthCameraModel(width=16, height=12, fx=10.0, fy=10.0, cx=7.5, cy=5.5)
    depth = np.full((12, 16), 10.0)
    depth[5, 7] = np.nan
    points = np.array([[0.0, 0.0, 10.0], [100.0, 0.0, 10.0], [0.0, 0.0, -1.0]])
    residual, valid = depth_residual(
        DepthToDepthObservation("test", depth, points, camera), SE3.identity()
    )
    assert not valid.any()
    np.testing.assert_array_equal(residual, np.ones(3))
