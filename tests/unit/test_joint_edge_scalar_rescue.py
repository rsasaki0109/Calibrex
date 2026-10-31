import json

import numpy as np
import pytest
from tools import joint_edge_scalar_rescue as rescue

from calibrex.core.geometry import SE3


@pytest.mark.parametrize(
    "target,accepted,degenerate,calls",
    [
        (-0.25, True, False, 100),
        (0.0, True, False, 100),
        (-0.25, False, False, 100),
        (-0.25, True, True, 100),
        (-0.25, True, False, 780),
    ],
)
def test_rescue_preserves_pose_and_fallback(monkeypatch, target, accepted, degenerate, calls):
    baseline = {
        "transform": SE3.identity().as_dict(),
        "bias_m": 0.0,
        "delta": [0.0] * 7,
        "cost": 1.0,
        "accepted": accepted,
        "converged": accepted,
        "actual_residual_calls": calls,
    }
    measured = []

    def residual(batch, pose, bias):
        assert pose == SE3.identity()
        measured.append(bias)
        r = np.full(100, bias - target)
        return r, r, np.ones(100, dtype=bool)

    monkeypatch.setattr(rescue, "joint_solve", lambda *args: baseline)
    monkeypatch.setattr(rescue, "residuals", residual)
    monkeypatch.setattr(
        rescue, "sensitivity", lambda *args: {"robust": {"near_degenerate": degenerate}}
    )
    result = rescue.solve([], SE3.identity())
    json.dumps(result, allow_nan=False)
    assert result["actual_residual_calls"] == calls + len(measured) <= 800
    assert result["baseline_fit"] == baseline
    assert baseline["bias_m"] == baseline["delta"][6] == 0
    assert result["transform"] == baseline["transform"]
    if target and accepted and not degenerate and calls == 100:
        assert result["rescue"]["adopted"]
        assert abs(result["bias_m"] - target) < 1e-6
    else:
        assert not result["rescue"]["adopted"]
        for key in baseline:
            if key != "actual_residual_calls":
                assert result[key] == baseline[key]
