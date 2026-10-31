"""Geometry of re-rectification and the return to original optical depth."""

import numpy as np
import pytest

pytest.importorskip("cv2")

from tools.stereo_rerectification import rectification, restore_left_depth


def test_nonhorizontal_pair_has_equal_rows_after_rectification():
    k = np.array([[700.0, 0, 600], [0, 700, 180], [0, 0, 1]])
    pl = np.column_stack((k, k @ [0.06, -0.001, 0.005]))
    pr = pl.copy()
    delta = np.array([-0.54, 0.004, -0.002])
    pr[:, 3] += k @ delta
    r1, r2, p1, p2 = rectification(pl, pr, (1200, 360))
    points = np.array([[0, 0, 5], [1, -1, 10], [-2, 1, 40]])
    ql = points @ r1.T @ p1[:, :3].T
    qr = (points + delta) @ r2.T @ p2[:, :3].T
    np.testing.assert_allclose(ql[:, 1] / ql[:, 2], qr[:, 1] / qr[:, 2], atol=1e-9)
    v, u = np.indices((360, 1200))
    rays = np.stack([u, v, np.ones_like(u)], axis=-1) @ np.linalg.inv(p1[:, :3]).T
    z = 10 / (rays @ r1)[..., 2]
    restored = restore_left_depth(z, r1, k, p1[:, :3])
    interior = restored[40:-40, 40:-40]
    assert np.isfinite(interior).all()
    np.testing.assert_allclose(interior, 10, atol=0.001)


def test_identity_depth_restore_preserves_holes():
    depth = np.full((8, 8), 10.0)
    depth[3, 3] = np.nan
    restored = restore_left_depth(depth, np.eye(3), np.eye(3), np.eye(3))
    np.testing.assert_allclose(restored, depth, equal_nan=True)
