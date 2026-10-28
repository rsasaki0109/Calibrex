"""Independent stationarity checks and regressions for false convergence."""

from __future__ import annotations

import numpy as np
import pytest

from calibrex.core.fit_diagnostics import normalized_gradient_norm


def test_stationarity_is_invariant_to_parameter_units() -> None:
    jacobian = np.array([[2.0, 0.0, 1.0], [0.0, 3.0, -1.0], [1.0, 1.0, 0.0]])
    residual = np.array([0.3, -0.4, 0.5])
    expected = max(
        abs(jacobian[:, i] @ residual) / np.linalg.norm(jacobian[:, i]) for i in range(3)
    )
    for units in ([1.0, 1.0, 1.0], [1000.0, 0.001, 1.0e6]):
        j = jacobian * units
        assert normalized_gradient_norm(
            j.T @ residual, np.sum(j * j, axis=0), np.linalg.norm(residual)
        ) == pytest.approx(expected)


def test_zero_columns_do_not_invent_information() -> None:
    assert normalized_gradient_norm(np.zeros(2), np.zeros(2), 3.0) == 0.0
    with pytest.raises(ValueError, match="zero Jacobian"):
        normalized_gradient_norm(np.ones(2), np.zeros(2), 3.0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_invalid_normal_model_is_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        normalized_gradient_norm(np.ones(1), np.array([value]), 1.0)


def test_step_norm_is_invariant_to_parameter_units() -> None:
    from calibrex.core.fit_diagnostics import normalized_step_norm

    diagonal = np.array([4.0, 9.0, 16.0])
    step = np.array([0.1, -0.2, 0.3])
    expected = 1.2 / 2.0
    for units in (np.ones(3), np.array([1000.0, 0.001, 1.0e6])):
        assert normalized_step_norm(step / units, diagonal * units**2, 2.0) == pytest.approx(
            expected
        )
