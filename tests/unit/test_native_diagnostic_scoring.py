"""Exact native diagnostic scores, population order and custom callback fallback."""

import math
import struct
from dataclasses import replace

import pytest
from tests.unit.test_rgbd_transform_reuse import KINDS, batch_problem, inputs

from calibrex.graph import _rgbd_execution as execution
from calibrex.graph import joint_reassociation as reassociation
from calibrex.graph._point_plane_batches import _OwnedPointFactors
from calibrex.graph._rgbd_scoring import _native_residuals
from calibrex.graph.joint_optimization import JointResidualBlock
from calibrex.solvers import native_tum_joint_slac_solver as tum


def _bytes(values):
    return struct.pack(f"<{len(values)}d", *values)


def _scalar_population(state, values, population, penalty):
    factors = {factor.factor_id: factor for factor in state.factors}
    scores, matched = [], 0
    for query_id in population:
        factor = factors.get(query_id)
        if factor is None:
            scores.append(penalty * penalty)
        else:
            residuals = factor.residuals(values)
            scores.append(sum(value * value for value in residuals) / len(residuals))
            matched += 1
    return math.sqrt(sum(scores) / len(scores)), matched


@pytest.mark.parametrize("kind", KINDS)
def test_native_score_keeps_all_residual_bits_and_python_float_accumulation(kind):
    _, factors = batch_problem(kind, count=96, regularizers=True)
    values = inputs()
    expected = [r for factor in factors for r in factor.residuals(values)]
    actual = _native_residuals(factors, values)
    assert actual is not None and all(type(value) is float for value in actual)
    assert _bytes(actual) == _bytes(expected)
    expected_rmse = math.sqrt(sum(value * value for value in expected) / len(expected))
    assert tum._factor_rmse(list(factors), values).hex() == expected_rmse.hex()
    assert _native_residuals(factors, values, scalar_factors=True) is None


@pytest.mark.parametrize("kind", KINDS)
def test_fixed_population_keeps_reordering_duplicates_unmatched_and_all_scores(kind, monkeypatch):
    _, factors = batch_problem(kind, count=96)
    state = reassociation.JointReassociationState(_OwnedPointFactors(factors), ())
    population = (
        *reversed([factor.factor_id for factor in factors]),
        "missing",
        factors[0].factor_id,
        "missing",
        factors[7].factor_id,
    )
    values = inputs()
    expected = _scalar_population(state, values, population, 0.15)
    calls, original = [], reassociation._native_residuals

    def native(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result)
        return result

    monkeypatch.setattr(reassociation, "_native_residuals", native)
    actual = reassociation._fixed_population_rmse(state, values, population, 0.15)
    assert calls and calls[0] is not None
    assert actual[0].hex() == expected[0].hex() and actual[1] == expected[1] == 98


@pytest.mark.parametrize("owned", [False, True])
def test_custom_factor_keeps_per_query_calls_and_vector_residual_normalization(owned):
    _, factors = batch_problem("xyz", count=96)
    calls = []

    def callback(values):
        calls.append(values["xyz"])
        return 0.1, -0.2, 0.3

    custom = replace(factors[0], evaluator=callback)
    selected = (custom, *factors[1:])
    state = reassociation.JointReassociationState(
        _OwnedPointFactors(selected) if owned else selected, ()
    )
    population = (*(f.factor_id for f in selected), custom.factor_id)
    expected = _scalar_population(state, inputs(), population, 0.15)
    expected_calls = calls.copy()
    calls.clear()
    actual = reassociation._fixed_population_rmse(state, inputs(), population, 0.15)
    assert actual == expected and calls == expected_calls and len(calls) == 2


def test_replaced_residual_method_keeps_every_callback_in_original_order(monkeypatch):
    _, factors = batch_problem("centered", count=96)
    state = reassociation.JointReassociationState(_OwnedPointFactors(factors), ())
    population = tuple(f.factor_id for f in reversed(factors))
    calls, original = [], JointResidualBlock.residuals

    def residuals(self, values):
        calls.append(self.factor_id)
        return original(self, values)

    monkeypatch.setattr(JointResidualBlock, "residuals", residuals)
    expected = _scalar_population(state, inputs(), population, 0.15)
    calls.clear()
    assert reassociation._fixed_population_rmse(state, inputs(), population, 0.15) == expected
    assert tuple(calls) == population


def test_custom_parameter_mapping_keeps_original_parameter_reads():
    _, factors = batch_problem("pair", count=96)
    calls = []

    class Values(dict):
        def __getitem__(self, key):
            calls.append(key)
            return super().__getitem__(key)

    values = Values(inputs())
    expected = [r for factor in factors for r in factor.residuals(values)]
    expected_calls = calls.copy()
    calls.clear()
    rmse = tum._factor_rmse(list(factors), values)
    assert rmse.hex() == math.sqrt(sum(value * value for value in expected) / len(expected)).hex()
    assert calls == expected_calls


@pytest.mark.parametrize("kind", KINDS)
def test_cpu_family_dimensions_use_only_validated_owned_native_factors(kind):
    _, factors = batch_problem(kind, count=96, regularizers=True)
    values = inputs()
    expected = tuple(len(f.residuals(values)) for f in factors)
    with execution._using_rgbd_backend("numpy"):
        assert execution._diagnostic_dimensions(tuple(factors), values) is None
        assert execution._diagnostic_dimensions(_OwnedPointFactors(factors), values) == expected
    assert execution._last_execution_report() is None


def test_missing_population_preserves_penalties_and_empty_population_error():
    _, factors = batch_problem("depth", count=96)
    state = reassociation.JointReassociationState(_OwnedPointFactors(factors), ())
    population = ("missing",) * 72
    expected = _scalar_population(state, inputs(), population, 0.15)
    assert reassociation._fixed_population_rmse(state, inputs(), population, 0.15) == expected
    with pytest.raises(ZeroDivisionError):
        reassociation._fixed_population_rmse(state, inputs(), (), 0.15)


def test_native_scoring_does_not_import_cupy():
    import subprocess
    import sys

    code = (
        "import sys; from tests.unit.test_rgbd_transform_reuse import batch_problem,inputs; "
        "from calibrex.graph._rgbd_scoring import _native_residuals; "
        "_,f=batch_problem('xyz'); assert _native_residuals(f,inputs()) is not None; "
        "assert 'cupy' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
