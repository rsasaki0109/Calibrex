"""Invalid solver settings must fail before numerical work begins."""
import numpy as np
import pytest

from calibrex.core.continuous_time_bspline_fit import BSplineLidarImuFitOptions
from calibrex.core.continuous_time_sparse import ContinuousTimeTrajectoryFitOptions


@pytest.mark.parametrize('options,field', [
    (ContinuousTimeTrajectoryFitOptions, 'initial_damping'),
    (ContinuousTimeTrajectoryFitOptions, 'convergence_gradient_norm'),
    (ContinuousTimeTrajectoryFitOptions, 'convergence_step_norm'),
    (BSplineLidarImuFitOptions, 'initial_damping'),
    (BSplineLidarImuFitOptions, 'gradient_tolerance'),
])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), 0., -1.])
def test_positive_settings_require_finite_positive_values(options, field, value):
    with pytest.raises(ValueError, match=field):
        options(**{field: value})


@pytest.mark.parametrize('options', [ContinuousTimeTrajectoryFitOptions, BSplineLidarImuFitOptions])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1, 1.5, 2., True, np.bool_(False)])
def test_iteration_budgets_require_actual_integers(options, value):
    with pytest.raises(ValueError, match='max_iterations'):
        options(max_iterations=value)


@pytest.mark.parametrize('field', ['relative_cost_tolerance', 'step_tolerance'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), -1.])
def test_spline_stall_tolerances_are_finite_and_nonnegative(field, value):
    with pytest.raises(ValueError, match=field):
        BSplineLidarImuFitOptions(**{field: value})


@pytest.mark.parametrize('field,value', [
    ('damping_scale_down', float('nan')), ('damping_scale_down', float('inf')),
    ('damping_scale_down', -1.), ('damping_scale_down', 0.), ('damping_scale_down', 1.1),
    ('damping_scale_up', float('nan')), ('damping_scale_up', float('inf')),
    ('damping_scale_up', -1.), ('damping_scale_up', 0.), ('damping_scale_up', .9),
])
def test_damping_scales_cannot_reverse_adaptation_or_be_nonfinite(field, value):
    with pytest.raises(ValueError, match=field):
        ContinuousTimeTrajectoryFitOptions(**{field: value})


def test_valid_boundaries_and_zero_budget_spline_evaluation_are_preserved():
    assert BSplineLidarImuFitOptions(max_iterations=0).max_iterations == 0
    spline = BSplineLidarImuFitOptions(relative_cost_tolerance=0., step_tolerance=0.)
    assert spline.step_tolerance == spline.relative_cost_tolerance == 0.
    native = ContinuousTimeTrajectoryFitOptions(
        max_iterations=np.int64(1), damping_scale_up=1., damping_scale_down=1.,
    )
    assert native.max_iterations == 1
    assert BSplineLidarImuFitOptions(max_iterations=np.int64(1)).max_iterations == 1
    with pytest.raises(ValueError, match='max_iterations'):
        ContinuousTimeTrajectoryFitOptions(max_iterations=0)
