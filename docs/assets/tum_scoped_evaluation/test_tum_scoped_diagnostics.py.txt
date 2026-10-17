"""Diagnostic scopes share transforms without bypassing residual callbacks."""

import math
from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import pytest
from tests.unit.test_rgbd_transform_reuse import KINDS, factory, inputs, measurement

from calibrex.graph import joint_factors as f
from calibrex.graph import joint_reassociation as r
from calibrex.graph._evaluation_scope import _EVALUATION_CACHE, _with_evaluation_cache
from calibrex.graph.joint_optimization import JointResidualBlock
from calibrex.solvers import native_tum_joint_slac_solver as a


def _scalar_score(state, values, population, penalty):
    factors = {factor.factor_id: factor for factor in state.factors}
    scores, matched = [], 0
    for name in population:
        if name not in factors:
            scores.append(penalty * penalty)
        else:
            residuals = factors[name].residuals(values)
            scores.append(sum(x * x for x in residuals) / len(residuals))
            matched += 1
    return math.sqrt(sum(scores) / len(scores)), matched


@pytest.mark.parametrize("kind", KINDS)
def test_fixed_population_preserves_all_scores_and_reuses_two_transforms(kind, monkeypatch):
    factors = tuple(factory(kind, measurement(kind, i)) for i in range(12))
    state = r.JointReassociationState(factors, ())
    population = ("rgbd-7", "missing", "rgbd-1", "rgbd-7", "rgbd-0", "rgbd-11")
    values = inputs()
    expected, matched = _scalar_score(state, values, population, 0.15)
    calls, original = [], f.se3_from_tangent

    def decode(vector):
        calls.append(vector)
        return original(vector)

    monkeypatch.setattr(f, "se3_from_tangent", decode)
    actual, count = r._fixed_population_rmse(state, values, population, 0.15)
    assert actual.hex() == expected.hex() and count == matched == 5
    assert len(calls) == 2
    assert _EVALUATION_CACHE.get() is None


@pytest.mark.parametrize("kind", KINDS)
def test_cross_window_factor_rmse_has_no_plan_and_keeps_scalar_float_bits(kind, monkeypatch):
    factors = [factory(kind, measurement(kind, i)) for i in range(12)]
    values = inputs()
    residuals = [x for factor in factors for x in factor.residuals(values)]
    expected = math.sqrt(sum(x * x for x in residuals) / len(residuals))
    calls, original = [], f.se3_from_tangent

    def decode(vector):
        calls.append(vector)
        return original(vector)

    monkeypatch.setattr(f, "se3_from_tangent", decode)
    assert a._factor_rmse(factors, values).hex() == expected.hex()
    assert len(calls) == 2
    assert _EVALUATION_CACHE.get() is None


def test_custom_vector_callbacks_keep_each_query_call_and_parameter_access():
    seen = []

    def callback(values):
        seen.append(dict(values))
        return values["x"][0], -2.0

    factor = JointResidualBlock("custom", "group", ("x",), callback, weight=0.5)
    state = r.JointReassociationState((factor,), ())
    values = {"x": (0.7,)}
    population = ("custom", "absent", "custom")
    expected = _scalar_score(state, values, population, 0.15)
    seen.clear()
    actual = r._fixed_population_rmse(state, values, population, 0.15)
    assert actual[0].hex() == expected[0].hex() and actual[1] == expected[1]
    assert seen == [values, values]


def test_outer_scope_restored_after_nested_diagnostic_error():
    native = factory("pair", measurement("pair"))
    values = inputs()
    entries = []

    def failing(_values):
        assert _EVALUATION_CACHE.get() is not None
        raise ValueError("diagnostic callback stopped")

    broken = JointResidualBlock("broken", "group", (), failing)
    state = r.JointReassociationState((native, broken), ())

    @_with_evaluation_cache
    def outer():
        cache = _EVALUATION_CACHE.get()
        native.residuals(values)
        entries.extend(cache.items())
        with pytest.raises(ValueError, match="diagnostic callback stopped"):
            r._fixed_population_rmse(state, values, (native.factor_id, "broken"), 0.15)
        assert _EVALUATION_CACHE.get() is cache and list(cache.items()) == entries
        assert a._factor_rmse([], values) is None
        assert _EVALUATION_CACHE.get() is cache

    outer()
    assert _EVALUATION_CACHE.get() is None


def test_parameter_mapping_is_still_read_once_per_owned_variable_and_factor():
    class Values(Mapping):
        def __init__(self, data):
            self.data, self.reads = data, []

        def __getitem__(self, name):
            self.reads.append(name)
            return self.data[name]

        def __iter__(self):
            return iter(self.data)

        def __len__(self):
            return len(self.data)

    factors = tuple(factory("pair", measurement("pair", i)) for i in range(5))
    state = r.JointReassociationState(factors, ())
    population = tuple(factor.factor_id for factor in reversed(factors))
    values = Values(inputs())
    expected = _scalar_score(state, values, population, 0.15)
    expected_reads = values.reads[:]
    values.reads.clear()
    actual = r._fixed_population_rmse(state, values, population, 0.15)
    assert actual[0].hex() == expected[0].hex() and actual[1] == expected[1]
    assert values.reads == expected_reads


def test_mutated_native_frame_leaves_do_not_reuse_a_stale_transform():
    item = measurement("pair")
    factors = tuple(factory("pair", replace(item, measurement_id=f"rgbd-{i}")) for i in range(2))
    values = inputs()
    seen = []

    def edit(_values):
        object.__setattr__(item.transform_world_source_initial, "translation_m", (0.4, -0.2, 0.3))
        return (0.0,)

    modifier = JointResidualBlock("edit", "group", (), edit)
    state = r.JointReassociationState((factors[0], modifier, factors[1]), ())
    population = ("rgbd-0", "edit", "rgbd-1")
    expected = _scalar_score(state, values, population, 0.15)
    object.__setattr__(item.transform_world_source_initial, "translation_m", (0.1, -0.3, 0.2))
    actual = r._fixed_population_rmse(state, values, population, 0.15)
    seen.append(actual)
    assert np.asarray([actual[0]]).tobytes() == np.asarray([expected[0]]).tobytes()
    assert actual[1] == expected[1]
