"""Exact numerical parity and conservative eligibility for native point batches."""

import functools
import json
import math
from dataclasses import replace

import numpy as np
import pytest

from calibrex.core.geometry import SE3
from calibrex.graph import _point_plane_batches as batches
from calibrex.graph import joint_factors as factors
from calibrex.graph import joint_optimization as graph


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
def test_ordered_rotation_matches_scalar_point_bytes(rotation) -> None:
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
    transform = SE3((0.01, -0.02, 0.03), rotation)
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
