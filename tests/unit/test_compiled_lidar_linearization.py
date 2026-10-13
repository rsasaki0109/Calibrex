"""Dense finite differences and robust equations preserve all float bits."""

import math
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("numba")

from calibrex.core.geometry import SE3
from calibrex.graph.lidar_point_to_plane import (
    LidarPointToPlaneObservation,
    LidarRigPointToPlaneFactor,
)
from calibrex.solvers.fixed_trajectory_se3_solver import (
    FixedTrajectorySe3ExtrinsicSolver,
    FixedTrajectorySe3SolverOptions,
    _weighted_normal_equations,
)
from calibrex.solvers.numba_continuous_time_adapter import (
    numba_rig_linearization,
    numba_rig_normal_equations,
    numba_rig_residuals,
)


def _bits(value):
    return np.asarray(value, dtype=np.float64).view(np.uint64).tolist()


def _factors(count=65):
    rng = np.random.default_rng(211)
    observations = [
        LidarPointToPlaneObservation(
            tuple(point),
            (0.1, 0.2, -0.3),
            (0.1, -0.2, 0.9),
            SE3(tuple(point), (0.1, 0.2, -0.3, 0.8)),
            weight=0.0 if i % 17 == 0 else 0.4 + i / count,
        )
        for i, point in enumerate(rng.normal(size=(count, 3)).tolist())
    ]
    kwargs = {
        "variable": "extrinsic",
        "observations": observations,
        "t_ego_lidar": SE3((0.1, -0.2, 0.3), (0.1, 0.2, -0.3, 0.8)),
    }
    return LidarRigPointToPlaneFactor(**kwargs, _array_kernel=numba_rig_residuals), (
        LidarRigPointToPlaneFactor(
            **kwargs,
            _array_kernel=numba_rig_residuals,
            _array_linearizer=numba_rig_linearization,
            _array_normal_kernel=numba_rig_normal_equations,
        )
    )


@pytest.mark.parametrize("count", [0, 1, 63, 64, 4096, 4097])
@pytest.mark.parametrize("huber", [False, True])
def test_weighted_equations_retain_row_and_product_order_and_huber_boundaries(count, huber):
    rng = np.random.default_rng(29)
    rows = rng.normal(size=(count, 6))
    choices = np.array([-0.0, 0.0, 0.05, -0.05, np.nextafter(0.05, math.inf), -0.2])
    residuals = np.resize(choices, count)
    options = FixedTrajectorySe3SolverOptions(robust_loss="huber" if huber else "none")
    expected = _weighted_normal_equations(residuals.tolist(), rows.tolist(), options)
    actual = numba_rig_normal_equations(rows, residuals, huber, 0.05)
    for a, b in zip(actual, expected, strict=True):
        assert _bits(a) == _bits(b)


@pytest.mark.parametrize("count", [0, 1, 63, 64, 4097])
def test_complete_solver_history_and_diagnostics_match(count):
    baseline, compiled = _factors(count)
    for correction in (None, (0.1, -0.2, 0.3, 0.04, -0.02, 0.01)):
        a, b = baseline.linearize(correction), compiled.linearize(correction)
        for field in ("residuals_m", "jacobian", "hessian", "gradient"):
            assert _bits(getattr(a, field)) == _bits(getattr(b, field))
        assert baseline.evaluate(correction) == compiled.evaluate(correction)
    solver = FixedTrajectorySe3ExtrinsicSolver()
    for loss in ("none", "huber"):
        options = FixedTrajectorySe3SolverOptions(max_iterations=3, robust_loss=loss)
        assert solver.solve(compiled, options) == solver.solve(baseline, options)


def test_replacement_and_custom_residuals_keep_the_expected_behavior(monkeypatch):
    baseline, compiled = _factors()
    assert compiled._compiled_system([0.0] * 6, robust_loss="huber", huber_delta_m=0.05) is not None
    baseline.observations = compiled.observations = tuple(reversed(compiled.observations))
    baseline.t_ego_lidar = compiled.t_ego_lidar = SE3((0.4, -0.3, 0.2), (0.2, -0.3, 0.1, 0.9))
    assert baseline.linearize() == compiled.linearize()
    original = compiled.residuals
    monkeypatch.setattr(compiled, "residuals", lambda correction=None: original(correction))
    assert compiled._compiled_linearization(None) is None
    assert baseline.linearize() == compiled.linearize()


def test_nonfinite_data_and_corrected_transform_use_existing_scalar_handling():
    for value in (math.nan, math.inf, -math.inf):
        baseline, compiled = _factors()
        baseline.observations = compiled.observations = tuple(
            replace(observation, point_lidar_m=(value, 0.1, 0.2))
            for observation in compiled.observations
        )
        assert compiled._compiled_linearization(None) is None
        assert all(math.isnan(r) for r in compiled.residuals())
        baseline, compiled = _factors()
        assert compiled._compiled_linearization((value, 0.0, 0.0, 0.0, 0.0, 0.0)) is None


def test_zero_difference_step_keeps_the_error():
    baseline, compiled = _factors()
    for factor in (baseline, compiled):
        with pytest.raises(ZeroDivisionError):
            factor.linearize(translation_step_m=0.0)
