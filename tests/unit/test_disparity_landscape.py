import numpy as np
from scipy.spatial.transform import Rotation
from tools.diagnose_disparity_landscape import interpolate

from calibrex.core.geometry import SE3


def test_pose_path_preserves_endpoints_and_half_rotation():
    start = SE3.identity()
    end = SE3((2.0, 0.0, 0.0), tuple(Rotation.from_euler("z", 90, degrees=True).as_quat()))
    for fraction in [0.0, 0.5, 1.0]:
        pose = interpolate(start, end, fraction)
        np.testing.assert_allclose(pose.translation_m, [2 * fraction, 0, 0])
        np.testing.assert_allclose(
            Rotation.from_quat(pose.rotation_quat_xyzw).as_rotvec(),
            [0, 0, fraction * np.pi / 2],
            atol=1e-15,
        )
