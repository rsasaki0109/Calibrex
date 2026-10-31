import numpy as np
from tools.joint_geometric_validation import holdout_points, validate_pose
from tools.known_geometry_joint import CAMERA, observations


def test_plane_depth_and_radial_units():
    v, u = np.meshgrid(np.arange(30, 60), np.arange(80, 100), indexing="ij")
    rays = np.stack([(u - CAMERA.cx) / CAMERA.fx, (v - CAMERA.cy) / CAMERA.fy, np.ones_like(u)], -1)
    points = (rays * 10).reshape(-1, 3)
    z = np.full((192, 320), 10.0)
    assert validate_pose(z, points, np.eye(4))["accepted"]
    vv, uu = np.indices(z.shape)
    radial = z * np.sqrt(
        1 + ((uu - CAMERA.cx) / CAMERA.fx) ** 2 + ((vv - CAMERA.cy) / CAMERA.fy) ** 2
    )
    assert validate_pose(radial, points, np.eye(4), True)["accepted"]
    shifted = np.eye(4)
    shifted[2, 3] = 0.4
    wrong = validate_pose(z, points, shifted)
    assert not wrong["accepted"]
    assert wrong["behind_surface_gt_20cm"] == len(points)


def test_missing_support_is_counted_and_rejected():
    points = np.tile([0.0, 0.0, 10.0], (200, 1))
    result = validate_pose(np.full((192, 320), np.nan), points, np.eye(4))
    assert not result["accepted"] and result["invalid_depth"] == 200
    assert result["supported"] == 0 and result["median_absolute_z_m"] is None


def test_holdout_clouds_are_disjoint_from_fit_clouds():
    from scipy.spatial import cKDTree

    training = observations("exact_z_all")
    for fitted, heldout in zip(training, holdout_points(), strict=True):
        distance, _ = cKDTree(fitted.lidar_points).query(heldout)
        assert (distance > 1e-6).all()
        assert len(heldout) == 3840
