from types import SimpleNamespace

import numpy as np
from tools import joint_multistart_control as control

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_depth_to_depth_solver import DepthToDepthOptions


def test_budget_bounds_and_best_value(monkeypatch):
    def objective(batch, pose, options):
        value = -sum((x - 0.3) ** 2 for x in pose.translation_m)
        return SimpleNamespace(mutual_information=value, evaluated_frame_count=3)

    monkeypatch.setattr(control, "evaluate_depth_to_depth_mi", objective)
    runs = [
        control.search([], SE3.identity(), start, DepthToDepthOptions()) for start in control.STARTS
    ]
    assert sum(r["evaluations"] for r in runs) <= 800
    for run in runs:
        assert run["evaluations"] == len(run["trace"])
        assert run["mi"] == max(t["mi"] for t in run["trace"])
        for t in run["trace"]:
            assert (np.abs(t["delta"][:3]) <= 2).all()
            assert (np.abs(t["delta"][3:]) <= 1).all()


def test_flat_objective_preserves_start_and_stops(monkeypatch):
    monkeypatch.setattr(
        control,
        "evaluate_depth_to_depth_mi",
        lambda *args: SimpleNamespace(mutual_information=1.0, evaluated_frame_count=3),
    )
    run = control.search([], SE3.identity(), control.STARTS[1], DepthToDepthOptions())
    assert run["converged"]
    assert run["evaluations"] < 160
    np.testing.assert_array_equal(run["delta"], control.STARTS[1])
