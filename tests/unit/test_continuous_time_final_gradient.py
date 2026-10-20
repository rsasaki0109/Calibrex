"""Final-state gradient reporting and sparse model reuse for the native fitter."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibrex.core import continuous_time_sparse as sparse
from calibrex.core.geometry import SE3

_ANCHORS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def _problem(
    *, calibration: bool = False, initial_x: float = 1.5, rotated_world: bool = False
) -> sparse.ContinuousTimeTrajectoryFitProblem:
    rotation = Rotation.from_rotvec([0.2, -0.15, 0.1] if rotated_world else [0.0] * 3)
    matrix = rotation.as_matrix()
    offset = np.array([0.3, -0.2, 0.1]) if rotated_world else np.zeros(3)
    quaternion = tuple(rotation.as_quat())
    initial = SE3(tuple(offset + matrix @ [initial_x, 0.0, 0.0]), quaternion)
    gyro = ()
    if calibration:
        gyro = (sparse.TrajectoryImuPreintegrationMeasurement("gyro", (
            sparse.TrajectoryImuGyroSample(0.0, (0.0, 0.0, 0.2)),
            sparse.TrajectoryImuGyroSample(1.0, (0.0, 0.0, 0.2)),
        )),)
    return sparse.ContinuousTimeTrajectoryFitProblem(
        (0.0, 1.0), (initial, initial),
        pose_measurements=tuple(
            sparse.TrajectoryPoseMeasurement(
                f"{timestamp}-{target}", timestamp,
                SE3(tuple(offset + matrix @ [target, 0.0, 0.0]), quaternion), weight,
            )
            for timestamp in (0.0, 1.0) for target, weight in ((0.0, 1.0), (2.0, 3.0))
        ),
        imu_preintegration_measurements=gyro,
        initial_gyro_bias_rad_s=(0.0, 0.0, 0.2 if initial_x == 1.8 else 0.0),
        estimate_gyro_bias=calibration,
    )


def _geometric_gradient_norm(
    problem: sparse.ContinuousTimeTrajectoryFitProblem, knots: tuple[SE3, ...]
) -> float:
    """Direct right-tangent anchor geometry; no production residual/Jacobian helpers."""
    values = []
    for index, knot in enumerate(knots):
        rotation = Rotation.from_quat(knot.rotation_quat_xyzw).as_matrix()
        gradient = np.zeros(6)
        for measurement in problem.pose_measurements:
            if measurement.timestamp_sec != problem.knot_timestamps[index]:
                continue
            target = measurement.pose_world_body
            target_rotation = Rotation.from_quat(target.rotation_quat_xyzw).as_matrix()
            for anchor in _ANCHORS:
                residual = (
                    rotation @ anchor + knot.translation_m
                    - target_rotation @ anchor - target.translation_m
                )
                body_residual = rotation.T @ residual
                gradient[:3] += measurement.weight**2 * body_residual
                gradient[3:] += measurement.weight**2 * np.cross(anchor, body_residual)
        values.extend(gradient)
    return float(np.max(np.abs(values)))


@pytest.mark.parametrize("rotated_world", [False, True])
@pytest.mark.parametrize("budget,damping", [(1, 1e-3), (2, 1.0), (1, 10.0)])
def test_reported_gradient_matches_independent_final_pose_geometry(
    rotated_world: bool, budget: int, damping: float
) -> None:
    problem = _problem(rotated_world=rotated_world)
    result = sparse.fit_continuous_trajectory(
        problem, sparse.ContinuousTimeTrajectoryFitOptions(
            max_iterations=budget, initial_damping=damping,
        ),
    )
    assert result.gradient_norm == pytest.approx(
        _geometric_gradient_norm(problem, result.knot_poses), rel=1e-10, abs=1e-10
    )


@pytest.mark.parametrize("calibration", [False, True])
@pytest.mark.parametrize("mode", [
    "stationary", "gradient_stop", "iteration_limit", "accepted_small_step",
    "rejected", "accepted_then_rejected", "damping_limit",
    "singular_exception", "singular_nonfinite",
])
def test_final_model_is_refreshed_only_after_state_changes(
    calibration: bool, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem = _problem(calibration=calibration, initial_x=1.8 if mode == "stationary" else 1.5)
    budget = 30 if mode in ("stationary", "gradient_stop", "damping_limit") else 1
    if mode == "accepted_then_rejected":
        budget = 2
    damping = 1e12 if mode in ("accepted_small_step", "damping_limit") else 1e-3
    settings = sparse.ContinuousTimeTrajectoryFitOptions(
        max_iterations=budget, initial_damping=damping
    )
    original_assembly = sparse._assemble_sparse_jacobian
    original_solve = sparse.spsolve
    models: list[float] = []
    solves = 0

    def assemble(blocks: Any, *, residual_count: int, parameter_count: int) -> Any:
        matrix = original_assembly(
            blocks, residual_count=residual_count, parameter_count=parameter_count
        )
        frame = inspect.currentframe()
        assert frame is not None and frame.f_back is not None
        state = frame.f_back.f_locals
        weighted = matrix.multiply(state["weight_array"][:, None])
        gradient = weighted.T @ (state["weight_array"] * state["residual_array"])
        models.append(float(np.max(np.abs(gradient))))
        return matrix

    def solve(matrix: Any, rhs: Any) -> Any:
        nonlocal solves
        solves += 1
        if mode == "singular_exception":
            raise RuntimeError("controlled failed solve")
        if mode == "singular_nonfinite":
            return np.full(np.asarray(rhs).shape, np.nan)
        delta = original_solve(matrix, rhs)
        if mode in ("rejected", "damping_limit") or (
            mode == "accepted_then_rejected" and solves == 2
        ):
            return -delta
        return delta

    monkeypatch.setattr(sparse, "_assemble_sparse_jacobian", assemble)
    monkeypatch.setattr(sparse, "spsolve", solve)
    result = sparse.fit_continuous_trajectory(problem, settings)
    # For calibration this also checks reuse of the model needed by covariance.
    refresh = mode in ("iteration_limit", "accepted_small_step")
    assert len(models) == result.iterations + int(refresh)
    assert result.gradient_norm == pytest.approx(models[-1], rel=1e-12, abs=1e-12)
    if not calibration:
        assert result.gradient_norm == pytest.approx(
            _geometric_gradient_norm(problem, result.knot_poses), rel=1e-10, abs=1e-10
        )
    if mode in ("iteration_limit", "accepted_small_step", "accepted_then_rejected"):
        assert result.knot_poses != problem.initial_knot_poses
    if mode in (
        "stationary", "rejected", "damping_limit", "singular_exception", "singular_nonfinite"
    ):
        assert result.knot_poses == problem.initial_knot_poses
    if mode.startswith("singular"):
        assert result.status == "singular_system"
        assert result.calibration_covariance is None
    elif calibration:
        assert result.calibration_covariance is not None
        assert result.calibration_parameter_names == (
            "gyro_bias[0]", "gyro_bias[1]", "gyro_bias[2]"
        )
    if mode in ("stationary", "gradient_stop"):
        assert result.status == "converged"
        assert result.gradient_norm < settings.convergence_gradient_norm
    if mode == "accepted_small_step":
        # A small accepted damped step remains an existing stop reason; this
        # reporting correction must disclose its large final gradient.
        assert result.max_step_translation_m < settings.convergence_step_norm
        assert result.gradient_norm > 1.0
