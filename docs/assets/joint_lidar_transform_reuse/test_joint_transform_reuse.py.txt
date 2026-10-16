"""Parity and lifetime checks for scoped native LiDAR transform reuse."""

import math
import struct
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import replace
from threading import Barrier

import numpy as np
import pytest

from calibrex.core.geometry import SE3
from calibrex.graph import joint_factors as factors
from calibrex.graph import joint_optimization as graph
from calibrex.graph._evaluation_scope import (
    _EVALUATION_CACHE,
    _EVALUATION_CACHE_LIMIT,
    _with_evaluation_cache,
)


def measurement(index: int = 0) -> factors.JointPointToPlaneMeasurement:
    return factors.JointPointToPlaneMeasurement(
        f"point-{index}",
        f"group-{index // 4}",
        (1.0 + index * 0.1, math.sin(index), -0.2 + math.cos(index)),
        (0.2, -0.5, 0.7),
        (0.3 + math.sin(0.71 * index), -0.2 + math.cos(0.37 * index), 0.9),
        SE3((0.1, -0.2, 0.05), (0.01, 0.02, -0.03, 0.99)),
        SE3((-0.2, 0.4, 0.02), (-0.02, 0.03, 0.01, 0.98)),
        0.5 + (index % 3),
    )


def factory(item: factors.JointPointToPlaneMeasurement) -> graph.JointResidualBlock:
    return factors.make_joint_point_to_plane_factor(item, pose_block="pose", extrinsic_block="ext")


def values() -> dict[str, tuple[float, ...]]:
    return {"pose": (0.02, -0.01, 0.03, 0.04, -0.02, 0.01), "ext": (0.01,) * 6}


def test_only_transforms_are_reused_across_distinct_point_factors(monkeypatch) -> None:
    point_factors = [factory(measurement(i)) for i in range(12)]
    expected = [factor.residuals(values()) for factor in point_factors]
    calls = []
    original = factors.se3_from_tangent

    def decode(vector):
        calls.append(vector)
        return original(vector)

    monkeypatch.setattr(factors, "se3_from_tangent", decode)

    @_with_evaluation_cache
    def evaluate():
        return [factor.residuals(values()) for factor in point_factors]

    assert evaluate() == expected
    assert len(calls) == 2
    assert _EVALUATION_CACHE.get() is None
    assert evaluate() == expected
    assert len(calls) == 4
    point_factors[0].residuals(values())
    point_factors[0].residuals(values())
    assert len(calls) == 8


def test_key_tracks_both_tangents_initial_transforms_and_signed_zero() -> None:
    item = measurement()
    variants = [
        item,
        replace(item, transform_world_body_initial=SE3.identity()),
        replace(item, transform_body_sensor_initial=SE3.identity()),
    ]
    inputs = [values(), {**values(), "pose": (0.0,) * 6}, {**values(), "pose": (-0.0,) * 6}]
    expected = [factory(m).residuals(v) for m in variants for v in inputs]

    @_with_evaluation_cache
    def evaluate():
        result = [factory(m).residuals(v) for m in variants for v in inputs]
        cache = _EVALUATION_CACHE.get()
        assert cache is not None and len(cache) == 9
        changed = {**values(), "ext": (-0.1,) * 6}
        result.append(factory(item).residuals(changed))
        assert len(cache) == 10
        return result

    expected.append(factory(item).residuals({**values(), "ext": (-0.1,) * 6}))
    actual = evaluate()
    assert struct.pack("<10d", *(r[0] for r in actual)) == struct.pack(
        "<10d", *(r[0] for r in expected)
    )


def test_cache_is_bounded_and_eviction_recomputes_exactly() -> None:
    factor = factory(measurement())
    original = factor.residuals(values())

    @_with_evaluation_cache
    def evaluate():
        factor.residuals(values())
        cache = _EVALUATION_CACHE.get()
        assert cache is not None
        first = next(iter(cache))
        for index in range(_EVALUATION_CACHE_LIMIT + 5):
            factor.residuals({**values(), "pose": (float(index),) + (0.0,) * 5})
        assert len(cache) == _EVALUATION_CACHE_LIMIT == 128
        assert first not in cache
        assert factor.residuals(values()) == original
        assert len(cache) == _EVALUATION_CACHE_LIMIT

    evaluate()
    assert _EVALUATION_CACHE.get() is None


@pytest.mark.parametrize("bad", [(0.0,) * 5, (float("nan"),) * 6, (float("inf"),) * 6])
def test_invalid_values_keep_the_original_error_and_release_scope(bad) -> None:
    factor = factory(measurement())
    supplied = {**values(), "pose": bad}
    with pytest.raises(ValueError) as expected:
        factor.residuals(supplied)

    @_with_evaluation_cache
    def evaluate():
        factor.residuals(supplied)

    with pytest.raises(ValueError, match=str(expected.value)):
        evaluate()
    assert _EVALUATION_CACHE.get() is None


@pytest.mark.parametrize("kind", [int, np.float64])
def test_unusual_numeric_values_keep_the_scalar_path(kind) -> None:
    factor = factory(measurement())
    supplied = {"pose": tuple(kind(0) for _ in range(6)), "ext": values()["ext"]}
    expected = factor.residuals(supplied)

    @_with_evaluation_cache
    def evaluate():
        assert factor.residuals(supplied) == expected
        assert not _EVALUATION_CACHE.get()

    evaluate()


def test_custom_se3_uses_scalar_path() -> None:
    class CustomSE3(SE3):
        pass

    item = replace(measurement(), transform_world_body_initial=CustomSE3.identity())
    factor = factory(item)
    expected = factor.residuals(values())

    @_with_evaluation_cache
    def evaluate():
        assert factor.residuals(values()) == expected
        assert not _EVALUATION_CACHE.get()

    evaluate()


def test_direct_evaluator_retains_invalid_pose_error_before_extrinsic_lookup() -> None:
    factor = factory(measurement())
    with pytest.raises(ValueError, match="SE3 tangent block requires six values"):
        factor.evaluator({"pose": (0.0,) * 5})


def test_nested_and_concurrent_evaluations_have_separate_scopes() -> None:
    @_with_evaluation_cache
    def nested():
        factory(measurement()).residuals(values())
        return _EVALUATION_CACHE.get()

    @_with_evaluation_cache
    def outer():
        cache = _EVALUATION_CACHE.get()
        assert cache is not None
        assert nested() is not cache
        assert _EVALUATION_CACHE.get() is cache and not cache
        return cache

    outer()
    barrier = Barrier(2)

    @_with_evaluation_cache
    def worker(index):
        cache = _EVALUATION_CACHE.get()
        factory(measurement(index)).residuals(values())
        barrier.wait(timeout=10)
        assert cache is _EVALUATION_CACHE.get() and len(cache) == 1
        return cache

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(worker, range(2)))
    assert results[0] is not results[1]
    assert _EVALUATION_CACHE.get() is None


@pytest.mark.parametrize(
    "linear_solver,fixed_pose", [("dense", False), ("schur", False), ("dense", True)]
)
@pytest.mark.parametrize("with_outliers", [False, True])
def test_complete_lm_jacobians_probes_and_diagnostics_match_scalar(
    monkeypatch, linear_solver, fixed_pose, with_outliers
) -> None:
    pose_truth = (0.0,) * 6
    ext_truth = (0.04, -0.03, 0.02, 0.01, -0.02, 0.015)
    point_factors = []
    for index in range(48):
        item = measurement(index)
        transform = (
            factors.se3_from_tangent(pose_truth)
            .compose(item.transform_world_body_initial)
            .compose(
                factors.se3_from_tangent(ext_truth).compose(item.transform_body_sensor_initial)
            )
        )
        truth = transform.transform_point(item.point_sensor_m)
        if with_outliers and index % 13 == 0:
            truth = (truth[0] + 2.0, truth[1] - 1.0, truth[2] + 0.5)
        point_factors.append(
            replace(factory(replace(item, plane_point_world_m=truth)), sqrt_information=((1.2,),))
        )
    point_factors.append(
        factors.make_joint_prior_factor(
            factor_id="prior",
            observation_group="gauge",
            block="pose",
            target=pose_truth,
            sigma=(0.01,) * 6,
        )
    )
    blocks = [
        graph.JointParameterBlock(
            "pose",
            pose_truth,
            fixed=fixed_pose,
            known_bad_steps=(0.1,) * 6,
            schur_role="eliminated",
        ),
        graph.JointParameterBlock("ext", (0.0,) * 6, known_bad_steps=(0.1,) * 6),
    ]
    options = graph.JointOptimizerOptions(
        max_iterations=12,
        split_seed=7,
        linear_solver=linear_solver,
    )
    traces = []
    original = graph._linearize

    def linearize(*args):
        residual, jacobian = original(*args)
        traces.append((residual.tobytes(), jacobian.tobytes()))
        return residual, jacobian

    monkeypatch.setattr(graph, "_linearize", linearize)
    accelerated = graph.BackendNeutralJointOptimizer().solve(blocks, point_factors, options)
    fast_traces = traces.copy()
    traces.clear()
    disabled: ContextVar[OrderedDict[bytes, object] | None] = ContextVar("disabled", default=None)
    monkeypatch.setattr(factors, "_EVALUATION_CACHE", disabled)
    scalar = graph.BackendNeutralJointOptimizer().solve(blocks, point_factors, options)
    assert traces == fast_traces
    assert scalar.as_dict() == accelerated.as_dict()
    assert len(accelerated.history) > 1
    assert len(accelerated.probes) == (12 if fixed_pose else 24)
    if not with_outliers:
        assert accelerated.status == "converged"
        assert accelerated.optimized_values["ext"] == pytest.approx(ext_truth, abs=1e-7)
    assert _EVALUATION_CACHE.get() is None


def test_optimizer_still_invokes_every_custom_callback_and_releases_on_error() -> None:
    calls = []

    def evaluator(supplied):
        calls.append(supplied.copy())
        if len(calls) == 4:
            raise RuntimeError("custom callback failed")
        return (supplied["state"][0],)

    factor = graph.JointResidualBlock("custom", "capture", ("state",), evaluator)
    for _ in range(3):
        assert graph._factor_residuals([factor], {"state": (1.0,)}).tolist() == [1.0]
        assert _EVALUATION_CACHE.get() is None
    with pytest.raises(RuntimeError, match="custom callback failed"):
        graph._factor_residuals([factor], {"state": (1.0,)})
    assert len(calls) == 4
    assert _EVALUATION_CACHE.get() is None
