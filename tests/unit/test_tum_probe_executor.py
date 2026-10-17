"""Ordered native probe mapping through real spawned CPU workers."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from functools import partial

import pytest
from tests.unit.test_joint_reassociation import _initial, _state
from tests.unit.test_tum_diagnostic_batches import _pair_problem, _spatial, _values

from calibrex.graph import joint_reassociation as r
from calibrex.solvers import _tum_probe_executor as workers
from calibrex.solvers import native_tum_joint_slac_solver as a


def _context(callback, values):
    state = callback(values)
    groups = {f.factor_id: f.observation_group for f in state.factors}
    population = tuple(sorted(groups))
    score, _ = r._fixed_population_rmse(state, values, population, 0.15)
    return r._ProbeContext(
        callback,
        groups,
        population,
        score,
        r._population_assignments(state, population),
        r.JointReassociationProbeOptions(0.15),
    )


def _tasks(values, block):
    return tuple(
        r._ProbeTask(i, block, 0, amount, r._perturbed_values(values, block, 0, amount))
        for i, amount in enumerate((-0.01, 0.01, -0.02, 0.02))
    )


def _canonical(probes):
    return json.dumps([p.as_dict() for p in probes], sort_keys=True, allow_nan=False)


def test_same_spawned_pool_preserves_complete_pair_and_spatial_probe_values():
    pair = _pair_problem()
    options = a.TUMJointSlacOptions(probe_workers=2)
    values = _values(pair.query_frames)
    callback = partial(a._xyz_pair_reassociation_state, pair, options=options)
    context = _context(callback, values)
    tasks = _tasks(values, a._XYZ_LATTICE_BLOCK)
    expected = r._map_probe_tasks(tasks, context, None)
    with workers._TUMProbePool(2) as pool:
        with pool.callback("pair", pair, options, callback) as proxy:
            actual = r._map_probe_tasks(tasks, replace(context, reassociate=proxy), pool.executor)
        assert _canonical(actual) == _canonical(expected)
        assert workers._CALLBACKS.get() is None
        assert pool.executor.submit(os.getpid).result() != os.getpid()
        spatial, planes, options = _spatial()
        values = spatial.result.optimized_values
        callback = partial(a._reassociated_spatial_state, spatial, planes, options)
        context = _context(callback, values)
        tasks = _tasks(values, a._SPATIAL_BIAS_BLOCK)
        expected = r._map_probe_tasks(tasks, context, None)
        with pool.callback("spatial", spatial, options, callback, planes) as proxy:
            actual = r._map_probe_tasks(tasks, replace(context, reassociate=proxy), pool.executor)
        assert _canonical(actual) == _canonical(expected)
        folder = pool.folder
    assert not folder.exists() and workers._CALLBACKS.get() is None


def test_public_executor_retains_probe_order_errors_and_all_result_fields():
    initial = _state("wrong", 1.0)
    blocks, policy, fitted = _initial(initial)
    refined = r.BackendNeutralJointReassociation().refine(
        blocks,
        initial,
        (),
        fitted,
        lambda v: _state("truth", 2.0),
        policy,
    )
    baseline = refined.final_result.optimized_values["x"][0]

    def callback(values):
        if values["x"][0] < baseline:
            raise ValueError("negative signed probe")
        return _state("truth", 2.0)

    options = r.JointReassociationProbeOptions(0.3)
    expected = r.evaluate_joint_reassociation_probes(blocks, initial, refined, callback, options)
    with ThreadPoolExecutor(max_workers=2) as executor:
        actual = r.evaluate_joint_reassociation_probes(
            blocks,
            initial,
            refined,
            callback,
            options,
            executor=executor,
        )
    assert json.dumps(actual.as_dict(), sort_keys=True) == json.dumps(
        expected.as_dict(), sort_keys=True
    )
    assert actual.probes[0].error == "negative signed probe"


@pytest.mark.parametrize(
    "cpus,windows,expected", [(None, 3, 1), (2, 3, 1), (6, 3, 2), (12, 3, 4), (64, 1, 4)]
)
def test_probe_pool_has_a_cpu_budget_and_four_worker_cap(cpus, windows, expected, monkeypatch):
    monkeypatch.setattr(workers.os, "cpu_count", lambda: cpus)
    options = a.TUMJointSlacOptions(window_workers=windows, probe_workers=100)
    assert workers._effective_probe_workers(options) == expected


def test_default_probe_policy_keeps_previous_option_representation():
    assert "probe_workers" not in a.TUMJointSlacOptions().as_dict()
    assert a.TUMJointSlacOptions(probe_workers=4).as_dict()["probe_workers"] == 4


def test_probe_context_is_restored_when_a_native_mapping_fails():
    pair = _pair_problem()
    options = a.TUMJointSlacOptions()
    values = _values(pair.query_frames)
    callback = partial(a._xyz_pair_reassociation_state, pair, options=options)
    with workers._TUMProbePool(2) as pool:
        with (
            pytest.raises(RuntimeError, match="mapping stopped"),
            pool.callback("pair", pair, options, callback) as proxy,
        ):
            assert proxy(values).factors
            raise RuntimeError("mapping stopped")
        assert workers._CALLBACKS.get() is None
