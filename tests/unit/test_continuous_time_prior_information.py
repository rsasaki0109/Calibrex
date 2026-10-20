"""Valid information priors must preserve their declared quadratic."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from calibrex.core import continuous_time_marginalization as marginal
from calibrex.core import continuous_time_sparse as sparse
from calibrex.core.geometry import SE3
from calibrex.core.se3_manifold import se3_exp, se3_log

_UPPER = np.eye(6)
_UPPER[0, 1], _UPPER[1, 2], _UPPER[3, 4] = 0.6, -0.3, 0.2
_LOW_RANK = np.array([[1.0, 2.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, -1.0, 0.0, 1.0, 0.0]])
_INFORMATION = {
    "diagonal": np.diag([2.0, 3.0, 5.0, 7.0, 11.0, 13.0]),
    "correlated": _UPPER.T @ _UPPER,
    "rank_two": _LOW_RANK.T @ _LOW_RANK,
    "zero": np.zeros((6, 6)),
    "weak_psd": np.diag([1e12, 1e-3, 1e-3, 1e-16, 0.0, 0.0]),
}
_DELTAS = (
    np.array([0.3, -0.2, 0.1, 0.04, -0.02, 0.03]),
    np.array([-0.1, 0.4, -0.3, -0.2, 0.1, 0.05]),
    np.array([0.2, 0.1, -0.4, 0.3, 0.2, -0.1]),
)


@pytest.mark.parametrize("name", _INFORMATION)
def test_prior_at_anchor_has_declared_information(name: str) -> None:
    info = _INFORMATION[name]
    residual, jacobian = marginal.marginalization_prior_blocks(
        marginal.KnotMarginalizationPrior(0, SE3.identity(), info), SE3.identity()
    )
    np.testing.assert_array_equal(residual, np.zeros(6))
    scale = np.sqrt(np.diag(info))
    scale = scale[:, None] * scale[None, :]
    assert np.all(np.abs(jacobian.T @ jacobian - info) <= 128 * np.finfo(float).eps * scale)
    if name == "zero":
        np.testing.assert_array_equal(jacobian, np.zeros((6, 6)))
    if name == "rank_two":
        for null in (np.array([2.0, -1.0, -1.0, 0.0, 0.0, 0.0]), np.eye(6)[5]):
            np.testing.assert_allclose(jacobian @ null, np.zeros(6), atol=1e-7, rtol=0)


@pytest.mark.parametrize("name", _INFORMATION)
@pytest.mark.parametrize("delta", _DELTAS, ids=("small", "mixed", "larger"))
def test_prior_residual_norm_equals_declared_quadratic(name: str, delta: np.ndarray) -> None:
    info = _INFORMATION[name]
    anchor = se3_exp(SE3.identity(), np.array([0.1, 0.2, -0.1, 0.1, -0.05, 0.2]))
    knot = se3_exp(anchor, delta)
    prior = marginal.KnotMarginalizationPrior(0, anchor, info)
    residual, _ = marginal.marginalization_prior_blocks(prior, knot)
    actual_delta = se3_log(anchor, knot)
    np.testing.assert_allclose(
        residual @ residual, actual_delta @ info @ actual_delta, rtol=1e-12, atol=1e-18
    )


@pytest.mark.parametrize("name", _INFORMATION)
def test_prior_gradient_matches_independent_declared_objective(name: str) -> None:
    info = _INFORMATION[name]
    anchor = SE3.identity()
    knot = se3_exp(anchor, _DELTAS[0])
    prior = marginal.KnotMarginalizationPrior(0, anchor, info)
    residual, jacobian = marginal.marginalization_prior_blocks(prior, knot)
    numeric = np.zeros(6)
    epsilon = 1e-6
    for axis in range(6):
        step = epsilon * np.eye(6)[axis]
        high, low = (se3_log(anchor, se3_exp(knot, sign * step)) for sign in (1, -1))
        numeric[axis] = (high @ info @ high - low @ info @ low) / (2 * epsilon)
    np.testing.assert_allclose(2 * jacobian.T @ residual, numeric, rtol=1e-6, atol=1e-8)


def _problem(info: np.ndarray) -> sparse.ContinuousTimeTrajectoryFitProblem:
    knots = (se3_exp(SE3.identity(), _DELTAS[0]), SE3.identity())
    points = tuple(
        sparse.TrajectoryPointMeasurement(
            f"{i}-{a}", float(i), body, knots[i].transform_point(body), 1.7
        )
        for i in range(2)
        for a, body in enumerate(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        )
    )
    return sparse.ContinuousTimeTrajectoryFitProblem(
        (0.0, 1.0),
        knots,
        point_measurements=points,
        knot_marginalization_priors=(marginal.KnotMarginalizationPrior(0, SE3.identity(), info),),
    )


def _score(problem: sparse.ContinuousTimeTrajectoryFitProblem, knots: list[SE3]) -> float:
    return sparse._objective(
        problem,
        [(0, m.point_body_m) for m in problem.point_measurements],
        [],
        [],
        [],
        [],
        knots,
        np.zeros(3),
        np.zeros(3),
        0.0,
        np.zeros(3),
        np.zeros(3),
        np.ones(3),
        np.ones(3),
    )


@pytest.mark.parametrize("name", ("diagonal", "correlated", "rank_two", "zero"))
@pytest.mark.parametrize("path", ("sparse", "dense"))
def test_both_assembled_gradients_match_declared_prior_objective(
    name: str, path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem = _problem(_INFORMATION[name])
    knots = list(problem.initial_knot_poses)
    if path == "dense":
        _, half_gradient = sparse.build_dense_knot_normal_equations(problem, knots)
    else:
        captured = []
        original = sparse._assemble_sparse_jacobian

        class Captured(Exception):
            pass

        def assembly(blocks, *, residual_count, parameter_count):
            matrix = original(
                blocks, residual_count=residual_count, parameter_count=parameter_count
            )
            state = inspect.currentframe().f_back.f_locals
            weighted = matrix.multiply(state["weight_array"][:, None])
            captured.append(
                np.asarray(weighted.T @ (state["weight_array"] * state["residual_array"])).ravel()
            )
            raise Captured

        monkeypatch.setattr(sparse, "_assemble_sparse_jacobian", assembly)
        with pytest.raises(Captured):
            sparse.fit_continuous_trajectory(problem)
        assert len(captured) == 1
        half_gradient = captured[0]
    epsilon = 1e-6
    numeric = np.zeros(12)
    for index in range(2):
        for axis in range(6):
            plus, minus = list(knots), list(knots)
            step = epsilon * np.eye(6)[axis]
            plus[index], minus[index] = se3_exp(knots[index], step), se3_exp(knots[index], -step)
            numeric[6 * index + axis] = (_score(problem, plus) - _score(problem, minus)) / (
                2 * epsilon
            )
    np.testing.assert_allclose(2 * half_gradient, numeric, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("global_scale", (1e-200, 1e-20, 1.0, 1e20, 1e200))
@pytest.mark.parametrize("mixed_units", (False, True))
@pytest.mark.parametrize("name", ("correlated", "rank_two"))
def test_information_is_preserved_across_global_and_coordinate_scales(
    name: str,
    mixed_units: bool,
    global_scale: float,
) -> None:
    units = np.array([1e-6, 2e3, 1e-2, 3e5, 7e-4, 0.5]) if mixed_units else np.ones(6)
    information = global_scale * (units[:, None] * _INFORMATION[name] * units[None, :])
    prior = marginal.KnotMarginalizationPrior(0, SE3.identity(), information)
    _, jacobian = marginal.marginalization_prior_blocks(prior, SE3.identity())
    scale = np.sqrt(np.diag(information))
    bound = 128 * np.finfo(float).eps * (scale[:, None] * scale[None, :])
    assert np.all(np.abs(jacobian.T @ jacobian - information) <= bound)


def _invalid_information() -> dict[str, np.ndarray]:
    negative = np.diag([1e12, 2.0, 3.0, 4.0, 0.0, -1e-15])
    indefinite = np.diag([1e12, 1.0, 1.0, 1e-3, 1e-3, 1.0])
    indefinite[3, 4] = indefinite[4, 3] = 1.1e-3
    asymmetric = np.diag([1e12, 1.0, 1.0, 1e-3, 1e-3, 1.0])
    asymmetric[3, 4] = 1e-5
    zero_diagonal = np.eye(6)
    zero_diagonal[5, 5] = 0.0
    zero_diagonal[0, 5] = zero_diagonal[5, 0] = 1e-20
    zero_diagonal_skew = zero_diagonal.copy()
    zero_diagonal_skew[5, 0] *= -1.0
    return {
        "negative_diagonal": negative,
        "indefinite_weak_block": indefinite,
        "asymmetric_weak_block": asymmetric,
        "nonzero_zero_diagonal_row": zero_diagonal,
        "skew_zero_diagonal_row": zero_diagonal_skew,
    }


@pytest.mark.parametrize("global_scale", (1e-20, 1.0, 1e20))
@pytest.mark.parametrize("name", tuple(_invalid_information()))
def test_invalid_information_is_rejected_independently_of_scale(
    name: str,
    global_scale: float,
) -> None:
    information = global_scale * _invalid_information()[name]
    prior = marginal.KnotMarginalizationPrior(0, SE3.identity(), information)
    with pytest.raises(ValueError):
        marginal.marginalization_prior_blocks(prior, SE3.identity())


@pytest.mark.parametrize("shape", ((5, 5), (6, 5), (6, 6, 1), (7, 7), (0, 0)))
def test_prior_information_requires_six_by_six_matrix(shape: tuple[int, ...]) -> None:
    prior = marginal.KnotMarginalizationPrior(0, SE3.identity(), np.zeros(shape))
    with pytest.raises(ValueError):
        marginal.marginalization_prior_blocks(prior, SE3.identity())


@pytest.mark.parametrize("value", (float("nan"), float("inf"), -float("inf")))
def test_prior_information_requires_finite_values(value: float) -> None:
    information = np.eye(6)
    information[0, 0] = value
    prior = marginal.KnotMarginalizationPrior(0, SE3.identity(), information)
    with pytest.raises(ValueError):
        marginal.marginalization_prior_blocks(prior, SE3.identity())


def test_complex_information_is_rejected() -> None:
    information = np.eye(6, dtype=complex)
    information[0, 1], information[1, 0] = 0.1j, -0.1j
    prior = marginal.KnotMarginalizationPrior(0, SE3.identity(), information)
    with pytest.raises(ValueError):
        marginal.marginalization_prior_blocks(prior, SE3.identity())


def test_readonly_information_is_not_modified() -> None:
    information = _INFORMATION["correlated"].copy()
    original = information.copy()
    information.setflags(write=False)
    prior = marginal.KnotMarginalizationPrior(0, SE3.identity(), information)
    marginal.marginalization_prior_blocks(prior, se3_exp(SE3.identity(), _DELTAS[0]))
    np.testing.assert_array_equal(information, original)


def test_only_normalized_eigensolver_roundoff_may_be_projected() -> None:
    information = np.eye(6)
    information[0, 1] = information[1, 0] = 1.0 + 2 * np.finfo(float).eps
    prior = marginal.KnotMarginalizationPrior(0, SE3.identity(), information)
    _, jacobian = marginal.marginalization_prior_blocks(prior, SE3.identity())
    # A tiny negative eigenvalue near an exact rank deficiency is within the
    # declared dimensionless roundoff allowance. This does not add a floor.
    np.testing.assert_allclose(
        jacobian.T @ jacobian, information, atol=128 * np.finfo(float).eps, rtol=0
    )
