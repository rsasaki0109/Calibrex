import json

import numpy as np
from scipy.spatial.transform import Rotation
from tools import joint_block_bias as block

from calibrex.core.geometry import SE3


def test_block_updates_count_work_and_preserve_monotonic_cost(monkeypatch):
    def residual(batch, pose, bias):
        rotation = Rotation.from_quat(pose.rotation_quat_xyzw).as_rotvec()
        r = np.tile(
            np.r_[np.array(pose.translation_m) - [0.1, -0.05, 0.08], rotation, bias + 0.2], 20
        )
        return r, r, np.ones(len(r), dtype=bool)

    monkeypatch.setattr(block, "residuals", residual)
    monkeypatch.setattr(block, "sensitivity", lambda *args: {"robust": {"near_degenerate": False}})
    fit = block.solve([object()], SE3.identity())
    json.dumps(fit, allow_nan=False)
    assert fit["converged"] and fit["actual_residual_calls"] <= 800
    np.testing.assert_allclose(fit["bias_m"], -0.2, atol=1e-5)
    np.testing.assert_allclose(fit["transform"]["translation_m"], [0.1, -0.05, 0.08], atol=1e-5)
    for cycle in fit["cycles"]:
        assert cycle["cost_after_pose"] <= cycle["cost_after_bias"] <= cycle["cost_before"]
    assert fit["cycles"][-1]["stable_bias"] and fit["cycles"][-1]["stable_cost"]
