"""Opt-in scan motion grids preserve SE(3), inputs and the default arithmetic."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibrex.solvers.scan_to_scan_odometry import ScanOdometryOptions, _canonicalize_motion


def test_disabled_motion_grid_returns_original_motion() -> None:
    motion = np.eye(4)
    assert _canonicalize_motion(motion, ScanOdometryOptions()) is motion


def test_motion_grid_is_bounded_and_keeps_a_proper_rotation() -> None:
    motion = np.eye(4)
    vector = np.array([0.013000023, -0.014000031, 0.009000049])
    motion[:3, :3] = Rotation.from_rotvec(vector).as_matrix()
    motion[:3, 3] = [0.0210002, -0.0330004, 0.0040001]
    before = motion.copy()
    options = ScanOdometryOptions(motion_translation_quantum_m=1e-6,
                                  motion_rotation_quantum_rad=1e-7)
    actual = _canonicalize_motion(motion, options)
    np.testing.assert_array_equal(motion, before)
    np.testing.assert_array_equal(actual[:3, 3], np.rint(before[:3, 3] / 1e-6) * 1e-6)
    np.testing.assert_array_equal(actual[:3, :3],
                                  Rotation.from_rotvec(np.rint(vector / 1e-7) * 1e-7).as_matrix())
    assert np.max(np.abs(actual[:3, 3] - before[:3, 3])) <= 0.5e-6
    assert np.linalg.det(actual[:3, :3]) == pytest.approx(1, abs=1e-14)
    np.testing.assert_allclose(actual[:3, :3].T @ actual[:3, :3], np.eye(3), atol=1e-14)


@pytest.mark.parametrize("field", ["motion_translation_quantum_m", "motion_rotation_quantum_rad"])
@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_motion_grid_rejects_invalid_quanta(field: str, value: float) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        ScanOdometryOptions(**{field: value})
