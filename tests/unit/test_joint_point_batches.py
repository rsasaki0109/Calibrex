"""Exact numerical parity and conservative eligibility for native point batches."""

import functools
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import numpy as np
import pytest

from calibrex.core.geometry import SE3
from calibrex.graph import _point_plane_batches as batches
from calibrex.graph import joint_factors as factors
from calibrex.graph import joint_optimization as graph
from calibrex.graph._evaluation_scope import _PREPARATION_CACHE, _with_preparation_cache


def problem(count=96, outliers=False, extra=False):
    point_factors = []
    truth = (0.04, -0.03, 0.02, 0.01, -0.02, 0.015)
    world = SE3((0.1, -0.2, 0.05), (0.01, 0.02, -0.03, 0.99))
    sensor = SE3((-0.2, 0.4, 0.02), (-0.02, 0.03, 0.01, 0.98))
    for index in range(count):
        initial = world if index % 2 else SE3.identity()
        point = (1.0 + index * 0.01, math.sin(index), -0.2 + math.cos(index))
        target = initial.compose(factors.se3_from_tangent(truth).compose(sensor)).transform_point(
            point
        )
        if outliers and index % 13 == 0:
            target = (target[0] + 2.0, target[1] - 1.0, target[2] + 0.5)
        measurement = factors.JointPointToPlaneMeasurement(
            f"point-{index}",
            f"group-{index // 4}",
            point,
            target,
            (0.3 + math.sin(0.71 * index), -0.2 + math.cos(0.37 * index), 0.9),
            initial,
            sensor,
            0.5 + index % 3,
        )
        point_factors.append(
            factors.make_joint_point_to_plane_factor(
                measurement,
                pose_block="pose",
                extrinsic_block="ext",
            )
        )
    point_factors.append(
        factors.make_joint_prior_factor(
            factor_id="pose-prior",
            observation_group="gauge",
            block="pose",
            target=(0.0,) * 6,
            sigma=(0.01,) * 6,
        )
    )
    blocks = [
        graph.JointParameterBlock(
            "pose", (0.0,) * 6, known_bad_steps=(0.1,) * 6, schur_role="eliminated"
        ),
        graph.JointParameterBlock("ext", (0.0,) * 6, known_bad_steps=(0.1,) * 6),
    ]
    if extra:
        blocks.append(graph.JointParameterBlock("bias", (0.1, -0.2)))
        point_factors.insert(
            5,
            factors.make_joint_prior_factor(
                factor_id="bias-prior",
                observation_group="gauge-bias",
                block="bias",
                target=(0.0, 0.0),
                sigma=(0.2, 0.3),
            ),
        )
    return blocks, point_factors, truth


@pytest.mark.parametrize("rotation", [(0.0, -0.0, 0.0, 1.0), (-0.3, 0.6, 0.2, -0.4)])
@pytest.mark.parametrize("translation", [(0.01, -0.02, 0.03), (-0.0, 0.0, -0.0)])
def test_ordered_rotation_matches_scalar_point_bytes(rotation, translation) -> None:
    rng = np.random.default_rng(17)
    points = np.concatenate(
        (
            rng.normal(size=(128, 3)),
            np.asarray(
                [
                    (-0.0, 0.0, -0.0),
                    (1e-300, -1e-300, 0.0),
                    (1e100, -1e100, 1e99),
                ]
            ),
        )
    )
    transform = SE3(translation, rotation)
    expected = np.asarray([transform.transform_point(point) for point in points])
    # The public scalar geometry does not expose NumPy underflow handling.
    with np.errstate(all="raise"), np.errstate(under="ignore"):
        actual = batches._transform_points(points, transform)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("count", [63, 64, 8205])
def test_every_residual_and_jacobian_byte_matches_scalar_across_chunks(monkeypatch, count) -> None:
    blocks, point_factors, _ = problem(count, extra=True)
    values = {block.name: block.initial_values for block in blocks}
    values["pose"] = (0.02, -0.01, 0.03, 0.04, -0.02, 0.01)
    options = graph.JointOptimizerOptions()
    plan = batches._prepare_point_plane_plan(point_factors, values)
    assert (plan is not None) == (count >= 64)
    if plan is not None:
        assert all(not group.points.flags.writeable for group in plan.groups)
    layout = graph._variable_layout(blocks)
    accelerated = graph._linearize(point_factors, values, blocks, layout, options)
    fast_residuals = graph._factor_residuals(point_factors, values)
    monkeypatch.setattr(batches, "_prepare_point_plane_plan", lambda *_: None)
    reference = graph._linearize(point_factors, values, blocks, layout, options)
    assert all(a.tobytes() == b.tobytes() for a, b in zip(accelerated, reference, strict=True))
    assert fast_residuals.tobytes() == graph._factor_residuals(point_factors, values).tobytes()


@pytest.mark.parametrize("solver,fixed", [("dense", False), ("schur", False), ("dense", True)])
@pytest.mark.parametrize("outliers", [False, True])
def test_full_solve_diagnostics_probes_and_history_are_exact(monkeypatch, solver, fixed, outliers):
    blocks, point_factors, truth = problem(outliers=outliers)
    blocks[0] = replace(blocks[0], fixed=fixed)
    settings = graph.JointOptimizerOptions(max_iterations=12, split_seed=7, linear_solver=solver)
    calls = []
    original = batches._PointPlanePlan.linearize

    def measured(self, *args):
        answer = original(self, *args)
        calls.append(tuple(v.tobytes() for v in answer))
        return answer

    monkeypatch.setattr(batches._PointPlanePlan, "linearize", measured)
    accelerated = graph.BackendNeutralJointOptimizer().solve(blocks, point_factors, settings)
    assert len(calls) > 1
    fast_traces = calls.copy()
    scalar_traces = []
    old_linearize = graph._linearize

    def capture(*args):
        answer = old_linearize(*args)
        scalar_traces.append(tuple(v.tobytes() for v in answer))
        return answer

    monkeypatch.setattr(batches, "_prepare_point_plane_plan", lambda *_: None)
    monkeypatch.setattr(graph, "_linearize", capture)
    reference = graph.BackendNeutralJointOptimizer().solve(blocks, point_factors, settings)
    assert scalar_traces == fast_traces
    assert json.dumps(reference.as_dict()) == json.dumps(accelerated.as_dict())
    if not outliers:
        assert accelerated.status == "converged"
        assert accelerated.optimized_values["ext"] == pytest.approx(truth, abs=1e-7)


def test_generic_callbacks_and_wrapped_builtins_retain_original_call_order(monkeypatch):
    blocks, point_factors, _ = problem(64)
    values = {block.name: block.initial_values for block in blocks}
    original = point_factors[10].evaluator
    calls = []

    @functools.wraps(original)
    def wrapped(supplied):
        calls.append("wrapped")
        return original(supplied)

    point_factors[10] = replace(point_factors[10], evaluator=wrapped)
    point_factors.insert(
        2,
        graph.JointResidualBlock(
            "custom",
            "custom",
            ("pose",),
            lambda _: calls.append("custom") or (0.1,),
        ),
    )
    assert batches._prepare_point_plane_plan(point_factors, values) is None
    graph._factor_residuals(point_factors, values)
    assert calls == ["custom", "wrapped"]


@pytest.mark.parametrize("change", ["values", "point", "whitening", "subclass", "method", "limit"])
def test_unusual_inputs_and_modified_methods_keep_scalar_fallback(monkeypatch, change):
    blocks, point_factors, _ = problem(65)
    values = {block.name: block.initial_values for block in blocks}
    if change == "values":
        values["pose"] = tuple(np.float64(v) for v in values["pose"])
    elif change == "point":
        original = batches._binding(point_factors[0])
        item = replace(original.measurement, point_sensor_m=(1e200, 0.0, 0.0))
        point_factors[0] = factors.make_joint_point_to_plane_factor(
            item, pose_block="pose", extrinsic_block="ext"
        )
    elif change == "whitening":
        point_factors[0] = replace(point_factors[0], sqrt_information=((1.2,),))
    elif change == "subclass":

        class CustomFactor(graph.JointResidualBlock):
            pass

        point_factors[0] = CustomFactor(**point_factors[0].__dict__)
    elif change == "method":
        original = graph.JointResidualBlock.residuals
        monkeypatch.setattr(
            graph.JointResidualBlock, "residuals", lambda self, supplied: original(self, supplied)
        )
    else:
        monkeypatch.setattr(batches, "_MAX_POINTS", 64)
    assert batches._prepare_point_plane_plan(point_factors, values) is None


def test_changed_values_and_weights_are_observed_on_the_next_pass(monkeypatch):
    blocks, point_factors, _ = problem(64)
    values = {block.name: block.initial_values for block in blocks}
    first = graph._factor_residuals(point_factors, values)
    values["ext"] = (0.1,) * 6
    point_factors[4] = replace(point_factors[4], weight=3.2)
    changed = graph._factor_residuals(point_factors, values)
    assert first.tobytes() != changed.tobytes()
    monkeypatch.setattr(batches, "_prepare_point_plane_plan", lambda *_: None)
    assert changed.tobytes() == graph._factor_residuals(point_factors, values).tobytes()


def test_concurrent_passes_use_separate_prepared_arrays(monkeypatch):
    blocks, point_factors, _ = problem(128)
    initial = {block.name: block.initial_values for block in blocks}
    inputs = [initial, {**initial, "ext": (0.1,) * 6}]
    with ThreadPoolExecutor(max_workers=2) as executor:
        actual = list(executor.map(lambda v: graph._factor_residuals(point_factors, v), inputs))
    assert actual[0].tobytes() != actual[1].tobytes()
    monkeypatch.setattr(batches, "_prepare_point_plane_plan", lambda *_: None)
    expected = [graph._factor_residuals(point_factors, v) for v in inputs]
    assert [v.tobytes() for v in actual] == [v.tobytes() for v in expected]


def test_complete_solver_builds_only_train_and_holdout_plans(monkeypatch):
    blocks, point_factors, _ = problem(384)
    original = batches._build_point_plane_plan
    builds = []

    def build(*args):
        plan = original(*args)
        builds.append((len(args[0]), plan is not None))
        return plan

    monkeypatch.setattr(batches, "_build_point_plane_plan", build)
    options = graph.JointOptimizerOptions(max_iterations=6, split_seed=7)
    result = graph.BackendNeutralJointOptimizer().solve(blocks, point_factors, options)
    assert len(result.history) > 1
    assert len(result.probes) == 24
    assert builds == [(len(result.train_factor_ids), True), (len(result.holdout_factor_ids), True)]
    assert _PREPARATION_CACHE.get() is None
    values = {block.name: block.initial_values for block in blocks}
    first = batches._prepare_point_plane_plan(point_factors, values)
    assert first is not batches._prepare_point_plane_plan(point_factors, values)
    assert len(builds) == 4


@pytest.mark.parametrize(
    "change",
    [
        "replacement",
        "weight",
        "point",
        "plane",
        "world",
        "sensor",
        "rotation",
        "cell",
        "code",
        "descriptor",
        "owner",
        "prior",
        "names",
        "whitening",
        "subclass",
    ],
)
def test_reused_plans_observe_changes_between_evaluations(monkeypatch, change):
    from contextvars import ContextVar

    blocks, point_factors, _ = problem(96)
    values = {block.name: block.initial_values for block in blocks}

    @_with_preparation_cache
    def evaluate():
        first = batches._prepare_point_plane_plan(point_factors, values)
        assert first is not None
        binding = batches._binding(point_factors[0])
        if change == "replacement":
            point_factors[0] = replace(point_factors[0], weight=3.25)
        elif change == "weight":
            object.__setattr__(point_factors[0], "weight", 3.25)
        elif change == "point":
            object.__setattr__(binding.measurement, "point_sensor_m", (-0.0, 0.0, -0.0))
        elif change == "plane":
            object.__setattr__(binding.measurement, "plane_point_world_m", (0.2, 0.1, -0.1))
        elif change in ("world", "sensor"):
            transform = (
                binding.measurement.transform_world_body_initial
                if change == "world"
                else binding.measurement.transform_body_sensor_initial
            )
            object.__setattr__(transform, "translation_m", (-0.0, 0.0, -0.0))
        elif change == "rotation":
            object.__setattr__(
                binding.measurement.transform_world_body_initial,
                "rotation_quat_xyzw",
                (0.01, -0.02, 0.03, 0.99),
            )
        elif change == "cell":
            evaluator = point_factors[0].evaluator
            index = evaluator.__code__.co_freevars.index("normal")
            evaluator.__closure__[index].cell_contents = (-0.2, 0.4, 0.8)
        elif change == "code":
            evaluator = point_factors[0].evaluator
            evaluator.__code__ = evaluator.__code__.replace(co_filename="changed_native.py")
        elif change == "descriptor":
            setattr(point_factors[0].evaluator, batches._ATTRIBUTE, replace(binding))
        elif change == "owner":
            object.__setattr__(binding.owner, "code", binding.owner.code.replace(co_name="changed"))
        elif change == "prior":
            prior = batches._binding(point_factors[-1])
            object.__setattr__(prior, "dimension", 7)
        elif change == "names":
            point_factors[0] = replace(point_factors[0], variable_names=("ext", "pose"))
        elif change == "whitening":
            point_factors[0] = replace(point_factors[0], sqrt_information=((1.2,),))
        else:

            class Custom(graph.JointResidualBlock):
                pass

            point_factors[0] = Custom(**point_factors[0].__dict__)
        assert batches._prepare_point_plane_plan(point_factors, values) is not first
        actual = graph._factor_residuals(point_factors, values)
        with monkeypatch.context() as fresh:
            fresh.setattr(batches, "_PREPARATION_CACHE", ContextVar("disabled", default=None))
            expected = graph._factor_residuals(point_factors, values)
        assert actual.tobytes() == expected.tobytes()

    evaluate()
    assert _PREPARATION_CACHE.get() is None


def test_reuse_keeps_dynamic_values_and_rebuilds_for_factor_order():
    blocks, point_factors, _ = problem(96)
    values = {block.name: block.initial_values for block in blocks}

    @_with_preparation_cache
    def evaluate():
        first = batches._prepare_point_plane_plan(point_factors, values)
        changed = {**values, "ext": (0.05,) * 6}
        assert batches._prepare_point_plane_plan(point_factors, changed) is first
        assert first.evaluate(values).tobytes() != first.evaluate(changed).tobytes()
        point_factors.reverse()
        reordered = batches._prepare_point_plane_plan(point_factors, changed)
        assert reordered is not None and reordered is not first
        reference = batches._build_point_plane_plan(point_factors, changed)
        assert reordered.evaluate(changed).tobytes() == reference.evaluate(changed).tobytes()
        assert batches._prepare_point_plane_plan(point_factors, {"pose": (0.0,) * 6}) is None
        assert (
            batches._prepare_point_plane_plan(point_factors, {**values, "pose": (0.0,) * 5}) is None
        )

    evaluate()


def test_cache_bounds_evict_whole_plans_and_never_cache_ineligible_graphs(monkeypatch):
    blocks, first, _ = problem(64)
    _, second, _ = problem(64)
    _, third, _ = problem(64)
    values = {block.name: block.initial_values for block in blocks}
    monkeypatch.setattr(batches, "_PLAN_FACTOR_LIMIT", 150)

    @_with_preparation_cache
    def evaluate():
        original = batches._prepare_point_plane_plan(first, values)
        batches._prepare_point_plane_plan(second, values)
        cache = _PREPARATION_CACHE.get()
        assert len(cache) == 2
        batches._prepare_point_plane_plan(third, values)
        assert len(cache) == 2 and id(first) not in cache
        rebuilt = batches._prepare_point_plane_plan(first, values)
        assert rebuilt is not original
        assert original.evaluate(values).tobytes() == rebuilt.evaluate(values).tobytes()
        monkeypatch.setattr(batches, "_PLAN_FACTOR_LIMIT", 64)
        cache.clear()
        assert batches._prepare_point_plane_plan(first, values) is not None
        assert not cache
        first[0] = replace(first[0], evaluator=lambda _: (0.0,))
        assert batches._prepare_point_plane_plan(first, values) is None and not cache

    evaluate()


def test_nested_concurrent_and_failed_solver_scopes_are_released():
    blocks, point_factors, _ = problem(96)
    values = {block.name: block.initial_values for block in blocks}

    @_with_preparation_cache
    def nested():
        return batches._prepare_point_plane_plan(point_factors, values)

    @_with_preparation_cache
    def outer():
        plan = batches._prepare_point_plane_plan(point_factors, values)
        assert nested() is not plan
        assert batches._prepare_point_plane_plan(point_factors, values) is plan
        raise RuntimeError("release prepared data")

    with pytest.raises(RuntimeError, match="release prepared data"):
        outer()
    assert _PREPARATION_CACHE.get() is None
    barrier = Barrier(2)

    @_with_preparation_cache
    def worker(_):
        plan = batches._prepare_point_plane_plan(point_factors, values)
        barrier.wait(timeout=10)
        assert batches._prepare_point_plane_plan(point_factors, values) is plan
        return plan

    with ThreadPoolExecutor(max_workers=2) as executor:
        plans = list(executor.map(worker, range(2)))
    assert plans[0] is not plans[1]
    assert _PREPARATION_CACHE.get() is None


def test_modified_global_methods_and_eligibility_limits_disable_cached_plans(monkeypatch):
    blocks, point_factors, _ = problem(96)
    values = {block.name: block.initial_values for block in blocks}

    @_with_preparation_cache
    def evaluate():
        assert batches._prepare_point_plane_plan(point_factors, values) is not None
        with monkeypatch.context() as changed:
            changed.setattr(batches, "_MAX_POINTS", 64)
            assert batches._prepare_point_plane_plan(point_factors, values) is None
        assert batches._prepare_point_plane_plan(point_factors, values) is not None
        with monkeypatch.context() as changed:
            original = graph.JointResidualBlock.residuals
            changed.setattr(
                graph.JointResidualBlock, "residuals", lambda self, v: original(self, v)
            )
            assert batches._prepare_point_plane_plan(point_factors, values) is None
        with monkeypatch.context() as changed:
            original = factors.se3_from_tangent
            changed.setattr(factors, "se3_from_tangent", lambda v: original(v))
            assert batches._prepare_point_plane_plan(point_factors, values) is None

    evaluate()
