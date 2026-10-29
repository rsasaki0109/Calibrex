from types import SimpleNamespace

import numpy as np
from tools import joint_search_continuation as control

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_depth_to_depth_solver import DepthToDepthOptions


def test_split_budget_preserves_exact_search_state(monkeypatch):
    monkeypatch.setattr(
        control,
        "evaluate_depth_to_depth_mi",
        lambda batch, pose, opts: SimpleNamespace(
            mutual_information=-sum((x - 0.3) ** 2 for x in pose.translation_m),
            evaluated_frame_count=3,
        ),
    )
    full = {"delta": [0.0] * 6}
    split = {"delta": [0.0] * 6}
    args = ([], SE3.identity(), DepthToDepthOptions())
    control.advance(*args, full, 397)
    control.advance(*args, split, 40)
    used = split["evaluations"]
    assert used <= 40
    control.advance(*args, split, 397 - used)
    assert full == split
    assert split["evaluations"] <= 397
    assert split["mi"] == max(t["mi"] for t in split["trace"])
    for t in split["trace"]:
        assert (np.abs(t["delta"][:3]) <= 2).all()
        assert (np.abs(t["delta"][3:]) <= 1).all()


def test_insufficient_sweep_budget_does_not_replay(monkeypatch):
    monkeypatch.setattr(
        control,
        "evaluate_depth_to_depth_mi",
        lambda *args: SimpleNamespace(mutual_information=1.0, evaluated_frame_count=3),
    )
    state = {"delta": [0.0] * 6}
    args = ([], SE3.identity(), DepthToDepthOptions())
    control.advance(*args, state, 1)
    assert state["evaluations"] == 1
    control.advance(*args, state, 11)
    assert state["evaluations"] == 1
    control.advance(*args, state, 12)
    assert state["evaluations"] == 13
    assert state["step"] == [0.125] * 3 + [0.05] * 3
