"""Native rematching score reuse keeps scalar values and conservative fallbacks."""

import json
from types import SimpleNamespace

import pytest
from tests.unit.test_tum_diagnostic_batches import _pair_problem, _values
from tests.unit.test_tum_scoped_diagnostics import _scalar_score

from calibrex.graph import _probe_scores as scores
from calibrex.graph import joint_reassociation as r
from calibrex.graph._rgbd_batches import _rgbd_binding
from calibrex.graph.joint_optimization import JointParameterBlock
from calibrex.solvers import native_tum_joint_slac_solver as a


@pytest.mark.parametrize("axis,amount", [(0, 0.0), (0, 0.01), (1, -0.02), (2, 0.2), (17, -0.03)])
def test_every_captured_residual_and_fixed_population_float_matches_scalar(axis, amount):
    problem = _pair_problem()
    options = a.TUMJointSlacOptions()
    context = a._TUMXYZPairReassociationContext(problem, options)
    values = _values(problem.query_frames)
    field = list(values[a._XYZ_LATTICE_BLOCK])
    field[axis] += amount
    values[a._XYZ_LATTICE_BLOCK] = tuple(field)
    population = ("pair-0-7", "absent", "pair-0-1", "pair-0-7", "pair-2-10")

    @scores._with_probe_scores
    def check():
        scores._set_probe_population(population)
        state = a._xyz_pair_reassociation_state(problem, values, options, context=context)
        captured = scores._captured_scores(state, values, population)
        assert captured is not None
        assert set(captured) <= set(population)
        by_id = {factor.factor_id: factor for factor in state.factors}
        for name, value in captured.items():
            assert value.hex() == by_id[name].residuals(values)[0].hex()
        expected = _scalar_score(state, values, population, 0.15)
        actual = r._fixed_population_rmse(state, values, population, 0.15)
        assert actual[0].hex() == expected[0].hex() and actual[1] == expected[1]
        assert scores._captured_scores(state, values, ("not-captured",)) is None

    check()
    assert scores._SCORES.get() is None
    assert scores._REPORTS.get()[-1]["captures"] == 1
    assert scores._REPORTS.get()[-1]["hits"] == 2


@pytest.mark.parametrize(
    "change", ["values", "frame", "measurement", "weight", "whitening", "evaluator", "math"],
)
def test_native_score_snapshot_rejects_changes_before_consumption(change, monkeypatch):
    problem = _pair_problem()
    values = _values(problem.query_frames)
    options = a.TUMJointSlacOptions()
    population = ("pair-0-1",)

    @scores._with_probe_scores
    def check():
        scores._set_probe_population(population)
        state = a._xyz_pair_reassociation_state(problem, values, options)
        assert scores._captured_scores(state, values, population) is not None
        factor = next(f for f in state.factors if f.factor_id == population[0])
        binding = _rgbd_binding(factor)
        if change == "values":
            values[a._XYZ_LATTICE_BLOCK] = (0.01,) * 24
        elif change == "frame":
            object.__setattr__(binding.measurement.transform_world_source_initial,
                               "translation_m", (0.01, 0.02, 0.03))
        elif change == "measurement":
            object.__setattr__(binding.measurement, "target_point_sensor_m", (0.2, 0.3, 1.0))
        elif change == "weight":
            object.__setattr__(factor, "weight", 2.0)
        elif change == "whitening":
            object.__setattr__(factor, "sqrt_information", ((2.0,),))
        elif change == "evaluator":
            original = factor.evaluator
            object.__setattr__(factor, "evaluator", lambda v: original(v))
        else:
            original = a._transform_points
            monkeypatch.setattr(a, "_transform_points", lambda *args: original(*args))
        assert scores._captured_scores(state, values, population) is None
        expected = _scalar_score(state, values, population, 0.15)
        actual = r._fixed_population_rmse(state, values, population, 0.15)
        assert actual[0].hex() == expected[0].hex() and actual[1] == expected[1]

    check()
    assert scores._SCORES.get() is None


def test_complete_public_probe_result_and_order_match_the_scalar_path(monkeypatch):
    problem = _pair_problem()
    values = _values(problem.query_frames)
    options = a.TUMJointSlacOptions()
    blocks = tuple(JointParameterBlock(name, vector, known_bad_steps=(0.01,) * len(vector)
                                       if name == a._XYZ_LATTICE_BLOCK else ())
                   for name, vector in values.items())
    result = SimpleNamespace(final_result=SimpleNamespace(optimized_values=values),
                             holdout_observation_groups=("tum-xyz-pair-000-001",))
    def evaluate():
        context = a._TUMXYZPairReassociationContext(problem, options)
        def callback(v):
            return a._xyz_pair_reassociation_state(problem, v, options, context=context)
        initial = callback(values)
        return r.evaluate_joint_reassociation_probes(
            blocks, initial, result, callback, r.JointReassociationProbeOptions(0.15),
        )
    actual = evaluate()
    assert scores._REPORTS.get()[-1] == {"captures": 49, "hits": 49}
    monkeypatch.setattr(a, "_PAIR_FACTOR_IMPLEMENTATION", object())
    expected = evaluate()
    assert scores._REPORTS.get()[-1] == {"captures": 0, "hits": 0}
    assert json.dumps(actual.as_dict(), sort_keys=True) == json.dumps(
        expected.as_dict(), sort_keys=True,
    )


def test_nested_score_scopes_restore_outer_state_even_on_error():
    @scores._with_probe_scores
    def outer():
        current = scores._SCORES.get()
        @scores._with_probe_scores
        def inner():
            assert scores._SCORES.get() is not current
            raise ValueError("inner probe stopped")
        with pytest.raises(ValueError, match="inner probe stopped"):
            inner()
        assert scores._SCORES.get() is current
    outer()
    assert scores._SCORES.get() is None


def test_frame_guard_budget_exhaustion_keeps_the_scalar_path(monkeypatch):
    from calibrex.graph import _rgbd_batches

    monkeypatch.setattr(_rgbd_batches, "_MAX_GROUPS", 1)
    problem = _pair_problem()
    values = _values(problem.query_frames)
    population = ("pair-0-1", "pair-1-1")

    @scores._with_probe_scores
    def check():
        scores._set_probe_population(population)
        state = a._xyz_pair_reassociation_state(problem, values, a.TUMJointSlacOptions())
        assert scores._captured_scores(state, values, population) is None
        expected = _scalar_score(state, values, population, 0.15)
        actual = r._fixed_population_rmse(state, values, population, 0.15)
        assert actual[0].hex() == expected[0].hex() and actual[1] == expected[1]

    check()
