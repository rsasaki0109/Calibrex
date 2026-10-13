"""CPU difference cache preserves exact keys, writable ownership and invalidation."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event

import numpy as np
import pytest

pytest.importorskip("numba")
from test_compiled_lidar_linearization import _factors

from calibrex.core.geometry import SE3
from calibrex.solvers import numba_continuous_time_adapter as a


def _case(count=8192):
    _, factor = _factors(count)
    steps = [1e-4]*3+[1e-5]*3
    base = [.1, -.2, .3, .04, -.02, .01]
    transforms = [factor._corrected_transform(base)]
    for column, step in enumerate(steps):
        plus, minus = base.copy(), base.copy()
        plus[column] += step
        minus[column] -= step
        transforms += [factor._corrected_transform(plus), factor._corrected_transform(minus)]
    return factor._prepared_arrays(), transforms, steps


def _assert_bits(left, right):
    for x, y in zip(left, right, strict=True):
        assert np.array_equal(x.view(np.uint64), y.view(np.uint64))


def test_exact_reuse_keeps_independent_writable_outputs_and_readonly_cache():
    arrays, transforms, steps = _case()
    context = a.NumbaRigLinearizer()
    expected = a.numba_rig_linearization(arrays, transforms, steps)
    first = context.linearize(arrays, transforms, steps)
    _assert_bits(first, expected)
    first[0][0] = first[1][0, 0] = 123.
    second = context.linearize(arrays, transforms, steps)
    _assert_bits(second, expected)
    assert all(value.flags.writeable for value in second)
    assert all(not np.shares_memory(x, y) for x, y in zip(first, second, strict=True))
    assert all(not value.flags.writeable for value in context._result)
    assert context.calls == context.cache_hits == 1
    assert context.cache_bytes == sum(value.nbytes for value in expected)
    assert context.cache_bytes <= 64*1024**2


def test_transform_step_and_payload_replacements_invalidate_exact_cache():
    arrays, transforms, steps = _case()
    context = a.NumbaRigLinearizer()
    context.linearize(arrays, transforms, steps)
    changed = list(transforms)
    t = changed[12]
    changed[12] = SE3((np.nextafter(t.translation_m[0], np.inf), *t.translation_m[1:]),
                     t.rotation_quat_xyzw)
    result = context.linearize(arrays, changed, steps)
    _assert_bits(result, a.numba_rig_linearization(arrays, changed, steps))
    modified_steps = list(steps)
    modified_steps[5] = np.nextafter(steps[5], np.inf)
    context.linearize(arrays, changed, modified_steps)
    other = replace(arrays, points=arrays.points.copy())
    other.points.flags.writeable = False
    context.linearize(other, changed, modified_steps)
    assert context.calls == 4 and context.cache_hits == 0
    assert context._arrays is other


def test_writable_inputs_bypass_cache_and_zero_steps_fail_before_a_hit():
    arrays, transforms, steps = _case()
    mutable = replace(arrays, points=arrays.points.copy())
    context = a.NumbaRigLinearizer()
    first = context.linearize(mutable, transforms, steps)
    mutable.points[1, 0] += 1.
    second = context.linearize(mutable, transforms, steps)
    assert first[0][1] != second[0][1]
    assert context.calls == 2 and context.cache_hits == context.cache_bytes == 0
    context.linearize(arrays, transforms, steps)
    with pytest.raises(ZeroDivisionError):
        context.linearize(arrays, transforms, [0., *steps[1:]])
    with pytest.raises(ValueError, match="thirteen transforms"):
        context.linearize(arrays, transforms[:12], steps)
    assert context.calls == 3


def test_shared_cache_serializes_readers_and_each_request_owns_outputs():
    arrays, transforms, steps = _case()
    context = a.NumbaRigLinearizer()
    with ThreadPoolExecutor(max_workers=3) as workers:
        results = list(workers.map(
            lambda _: context.linearize(arrays, transforms, steps), range(6),
        ))
    for result in results[1:]:
        _assert_bits(result, results[0])
        assert not np.shares_memory(result[0], results[0][0])
    assert context.calls == 1 and context.cache_hits == 5


def test_over_budget_results_clear_previous_cache_without_retaining_payload(monkeypatch):
    arrays, transforms, steps = _case()
    context = a.NumbaRigLinearizer()
    context.linearize(arrays, transforms, steps)
    large = replace(arrays, points=np.broadcast_to(arrays.points[:1], (1200000, 3)))
    monkeypatch.setattr(a, "numba_rig_linearization", lambda *args: (
        np.broadcast_to(np.zeros(1), (1200000,)),
        np.broadcast_to(np.zeros((1, 6)), (1200000, 6)),
    ))
    context.linearize(large, transforms, steps)
    assert context.cache_bytes == 0 and context._arrays is context._key is context._result is None


@pytest.mark.parametrize("count", [32767, 32768, 65537])
@pytest.mark.parametrize("reverse", [False, True])
def test_parallel_differences_keep_residual_jacobian_and_robust_system_bits(count, reverse):
    arrays, transforms, steps = _case(count)
    if reverse:
        arrays = replace(arrays, points=arrays.points[::-1], plane_points=arrays.plane_points[::-1],
                         normals=arrays.normals[::-1], poses=arrays.poses[::-1],
                         sqrt_weights=arrays.sqrt_weights[::-1])
    expected = a.numba_rig_linearization(arrays, transforms, steps)
    with ThreadPoolExecutor(max_workers=3) as helpers:
        context = a.NumbaRigLinearizer(helpers)
        result = context.linearize(arrays, transforms, steps)
        cached = context.linearize(arrays, transforms, steps)
    _assert_bits(result, expected)
    _assert_bits(cached, expected)
    for huber in (False, True):
        assert a.numba_rig_normal_equations(result[1], result[0], huber, .1) == (
            a.numba_rig_normal_equations(expected[1], expected[0], huber, .1))
    assert context.parallel_calls == int(count >= 32768)
    assert context.calls == context.cache_hits == 1
    assert all(value.flags.writeable for value in result + cached)
    if count >= 32768:
        assert all(value.flags.owndata for value in result)
    assert not np.shares_memory(result[1], cached[1])


def test_parallel_difference_scratch_and_mutability_limits_do_not_submit(monkeypatch):
    arrays, transforms, steps = _case(32768)
    large = replace(arrays, points=np.broadcast_to(arrays.points[:1], (2097153, 3)))
    mutable = replace(arrays, points=arrays.points.copy())
    values = np.asarray([(*t.translation_m, *t.rotation_quat_xyzw) for t in transforms])
    with ThreadPoolExecutor(max_workers=3) as helpers:
        monkeypatch.setattr(helpers, "submit", lambda *args: pytest.fail("must use CPU fallback"))
        for payload in (large, mutable):
            assert a._numba_parallel_rig_linearization(
                payload, values, np.asarray(steps), helpers) is None


def test_parallel_difference_failure_joins_helper_before_releasing_owned_outputs(monkeypatch):
    arrays, transforms, steps = _case(32768)
    finished = Event()
    def failure(*args):
        if args[-2] == 0:
            raise RuntimeError("caller error")
        finished.set()
        raise ValueError("helper error")
    monkeypatch.setattr(a, "_jacobian_rows_into", failure)
    with ThreadPoolExecutor(max_workers=3) as helpers:
        context = a.NumbaRigLinearizer(helpers)
        with pytest.raises(RuntimeError, match="caller error"):
            context.linearize(arrays, transforms, steps)
    assert finished.is_set()
    assert context.calls == context.parallel_calls == context.cache_bytes == 0


def test_three_parallel_difference_callers_own_outputs_and_retain_exact_cache():
    arrays, transforms, steps = _case(65537)
    expected = a.numba_rig_linearization(arrays, transforms, steps)
    with ThreadPoolExecutor(max_workers=3) as helpers, ThreadPoolExecutor(max_workers=3) as callers:
        contexts = [a.NumbaRigLinearizer(helpers) for _ in range(3)]
        results = list(callers.map(lambda c: c.linearize(arrays, transforms, steps), contexts))
        for context in contexts:
            _assert_bits(context.linearize(arrays, transforms, steps), expected)
            assert context.calls == context.parallel_calls == context.cache_hits == 1
    for result in results:
        _assert_bits(result, expected)
    assert not np.shares_memory(results[0][1], results[1][1])
