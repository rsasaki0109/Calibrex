import math

import pytest

from calibrex.evaluation.numerical_curvature import evaluate_numerical_curvature


def test_numerical_curvature_recovers_coupled_convex_quadratic() -> None:
    evaluation = evaluate_numerical_curvature(
        lambda x: 2.0 * x[0] ** 2 + 3.0 * x[0] * x[1] + 5.0 * x[1] ** 2,
        (0.3, -0.2),
        (1.0e-3, 2.0e-3),
    )

    assert evaluation.hessian[0] == pytest.approx((4.0, 3.0), abs=1e-9)
    assert evaluation.hessian[1] == pytest.approx((3.0, 10.0), abs=1e-9)
    assert evaluation.rank == 2
    assert evaluation.positive_eigenvalue_count == 2
    assert evaluation.negative_eigenvalue_count == 0
    assert evaluation.evaluation_count == 9


def test_numerical_curvature_detects_saddle_and_flat_direction() -> None:
    evaluation = evaluate_numerical_curvature(
        lambda x: x[0] ** 2 - 2.0 * x[1] ** 2,
        (0.0, 0.0, 0.0),
        (1.0e-3,) * 3,
    )

    assert evaluation.eigenvalues == pytest.approx((-4.0, 0.0, 2.0), abs=1e-10)
    assert evaluation.rank == 2
    assert evaluation.negative_eigenvalue_count == 1
    assert evaluation.near_zero_eigenvalue_count == 1
    assert evaluation.evaluation_count == 19


def test_numerical_curvature_rejects_nonfinite_probe() -> None:
    with pytest.raises(ValueError, match="finite"):
        evaluate_numerical_curvature(
            lambda x: math.inf if x[0] > 0.0 else 0.0,
            (0.0,),
            (0.1,),
        )
