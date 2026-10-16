"""Exact residual, derivative and optimizer parity for scoped RGB-D transforms."""

import math
import struct
from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import replace

import numpy as np
import pytest

from calibrex.core.geometry import SE3
from calibrex.graph import joint_factors as factors
from calibrex.graph import joint_optimization as graph
from calibrex.graph._evaluation_scope import _EVALUATION_CACHE, _with_evaluation_cache

KINDS = ("depth", "trilinear", "centered", "xyz", "pair")


def inputs():
    return {
        "pose": (0.01, -0.02, 0.03, 0.04, -0.01, 0.02),
        "target": (-0.03, 0.02, 0.01, -0.01, 0.04, 0.02),
        "ext": (0.02, 0.01, -0.01, 0.03, -0.02, 0.04),
        "depth": (0.01, -0.02),
        "lattice": tuple(0.002 * math.sin(i) for i in range(8)),
        "bias": (0.01,),
        "xyz": tuple(0.002 * math.cos(i) for i in range(24)),
    }


def measurement(kind, index=0):
    world = SE3((0.1, -0.3, 0.2), (0.03, -0.02, 0.01, 0.99))
    sensor = SE3((-0.2, 0.1, 0.4), (-0.02, 0.01, 0.04, 0.98))
    normal = (0.2 + math.sin(index), -0.3 + math.cos(0.7 * index), 0.8)
    ray = (0.3 * math.sin(index), 0.3 * math.cos(index), 1.0)
    depth = 1.0 + index * 0.01
    point = tuple(value * depth for value in ray)
    weights = (0.125,) * 8
    common = {
        "measurement_id": f"rgbd-{index}",
        "observation_group": f"frame-{index % 8}",
        "plane_normal_world": normal,
        "weight": 0.5 + index % 3,
    }
    if kind == "pair":
        return factors.JointTrilinearXYZPairMeasurement(
            **common,
            source_point_sensor_m=point,
            target_point_sensor_m=(0.2, -0.1, 1.1),
            source_lattice_weights=weights,
            target_lattice_weights=tuple(reversed(weights)),
            transform_world_source_initial=world,
            transform_world_target_initial=sensor,
        )
    common.update(
        plane_point_world_m=(0.2, -0.1, 0.3),
        transform_world_body_initial=world,
        transform_body_sensor_initial=sensor,
    )
    if kind == "depth":
        return factors.JointDepthPointToPlaneMeasurement(
            **common,
            normalized_ray_sensor=ray,
            nominal_depth_m=depth,
        )
    if kind == "xyz":
        return factors.JointTrilinearXYZPointToPlaneMeasurement(
            **common,
            point_sensor_m=point,
            lattice_weights=weights,
        )
    return factors.JointTrilinearDepthPointToPlaneMeasurement(
        **common,
        normalized_ray_sensor=ray,
        nominal_depth_m=depth,
        lattice_weights=weights,
    )


def factory(kind, item):
    if kind == "pair":
        return factors.make_joint_trilinear_xyz_pair_factor(
            item, source_pose_block="pose", target_pose_block="target", xyz_lattice_block="xyz"
        )
    names = {"pose_block": "pose", "extrinsic_block": "ext"}
    if kind == "depth":
        return factors.make_joint_depth_point_to_plane_factor(
            item, **names, depth_calibration_block="depth"
        )
    if kind == "trilinear":
        return factors.make_joint_trilinear_depth_point_to_plane_factor(
            item, **names, depth_lattice_block="lattice"
        )
    if kind == "centered":
        return factors.make_joint_centered_trilinear_depth_point_to_plane_factor(
            item, **names, depth_bias_block="bias", centered_lattice_block="lattice"
        )
    return factors.make_joint_trilinear_xyz_point_to_plane_factor(
        item, **names, xyz_lattice_block="xyz"
    )


def byte_values(values):
    return struct.pack(f"<{len(values)}d", *values)


@pytest.mark.parametrize("kind", KINDS)
def test_distinct_points_and_changing_depth_keep_exact_residuals(kind, monkeypatch):
    items = [factory(kind, measurement(kind, index)) for index in range(12)]
    supplied = [
        inputs(),
        {
            **inputs(),
            "depth": (-0.02, 0.03),
            "bias": (-0.03,),
            "lattice": (0.01,) * 8,
            "xyz": (-0.02,) * 24,
        },
    ]
    expected = [byte_values(factor.residuals(value)) for value in supplied for factor in items]
    calls = []
    original = factors.se3_from_tangent

    def decode(vector):
        calls.append(vector)
        return original(vector)

    monkeypatch.setattr(factors, "se3_from_tangent", decode)

    @_with_evaluation_cache
    def evaluate():
        return [byte_values(factor.residuals(value)) for value in supplied for factor in items]

    assert evaluate() == expected
    assert len(calls) == 2
    assert _EVALUATION_CACHE.get() is None
    assert evaluate() == expected
    assert len(calls) == 4
    items[0].residuals(inputs())
    assert len(calls) == 6


def test_different_rgbd_families_share_only_the_common_transform(monkeypatch):
    items = [factory(kind, measurement(kind, index)) for index, kind in enumerate(KINDS[:-1])]
    expected = [factor.residuals(inputs()) for factor in items]
    calls = []
    original = factors.se3_from_tangent

    def decode(vector):
        calls.append(vector)
        return original(vector)

    monkeypatch.setattr(factors, "se3_from_tangent", decode)

    @_with_evaluation_cache
    def evaluate():
        return [factor.residuals(inputs()) for factor in items]

    assert evaluate() == expected
    assert len(calls) == 2


@pytest.mark.parametrize("kind", KINDS)
def test_key_distinguishes_both_tangents_frames_and_signed_zero(kind):
    item = measurement(kind)
    names = (
        ("transform_world_source_initial", "transform_world_target_initial")
        if kind == "pair"
        else ("transform_world_body_initial", "transform_body_sensor_initial")
    )
    variants = [item, *(replace(item, **{name: SE3.identity()}) for name in names)]
    supplied = [
        inputs(),
        {**inputs(), "pose": (0.0,) * 6},
        {**inputs(), "pose": (-0.0,) * 6},
        {**inputs(), "target" if kind == "pair" else "ext": (0.1,) * 6},
    ]
    expected = [byte_values(factory(kind, m).residuals(v)) for m in variants for v in supplied]

    @_with_evaluation_cache
    def evaluate():
        return [byte_values(factory(kind, m).residuals(v)) for m in variants for v in supplied]

    assert evaluate() == expected


@pytest.mark.parametrize("kind", KINDS)
def test_replaced_initial_frames_are_recomputed(kind):
    item = measurement(kind)
    factor = factory(kind, item)
    name = "transform_world_source_initial" if kind == "pair" else "transform_world_body_initial"

    @_with_evaluation_cache
    def evaluate():
        before = factor.residuals(inputs())
        object.__setattr__(getattr(item, name), "translation_m", (0.4, -0.2, 0.3))
        changed = factor.residuals(inputs())
        assert changed != before
        assert changed == factory(kind, item).residuals(inputs())
        object.__setattr__(item, name, SE3.identity())
        changed_again = factor.residuals(inputs())
        assert changed_again != changed
        return changed_again

    assert evaluate() == factor.residuals(inputs())


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("numeric", [int, np.float64])
def test_custom_numeric_inputs_retain_scalar_path(kind, numeric):
    factor = factory(kind, measurement(kind))
    supplied = {**inputs(), "pose": tuple(numeric(0) for _ in range(6))}
    expected = factor.residuals(supplied)

    @_with_evaluation_cache
    def evaluate():
        assert factor.residuals(supplied) == expected
        # Pair's ordinary target pose may still use its independent key.
        assert len(_EVALUATION_CACHE.get()) == (1 if kind == "pair" else 0)

    evaluate()


@pytest.mark.parametrize("kind", KINDS)
def test_mutable_or_custom_initial_frames_retain_scalar_path(kind):
    class CustomSE3(SE3):
        pass

    item = measurement(kind)
    name = "transform_world_source_initial" if kind == "pair" else "transform_world_body_initial"
    frame = getattr(item, name)
    mutable = [*frame.translation_m]
    object.__setattr__(frame, "translation_m", mutable)
    factor = factory(kind, item)

    @_with_evaluation_cache
    def evaluate():
        before = factor.residuals(inputs())
        mutable[0] += 0.2
        changed = factor.residuals(inputs())
        assert changed != before
        assert changed == factory(kind, item).residuals(inputs())
        object.__setattr__(item, name, CustomSE3.identity())
        return factor.residuals(inputs())

    assert evaluate() == factor.residuals(inputs())


@pytest.mark.parametrize("kind", KINDS)
def test_invalid_pose_preserves_error_order_and_cleans_up(kind):
    factor = factory(kind, measurement(kind))
    supplied = {**inputs(), "pose": (0.0,) * 5}
    supplied.pop("ext", None)
    supplied.pop("target", None)

    @_with_evaluation_cache
    def evaluate():
        return factor.evaluator(supplied)

    with pytest.raises(ValueError, match="SE3 tangent block requires six values"):
        evaluate()
    assert _EVALUATION_CACHE.get() is None


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("linear_solver", ["dense", "schur"])
def test_complete_lm_derivatives_probes_whitening_and_diagnostics_match_scalar(
    kind, linear_solver, monkeypatch
):
    items = [factory(kind, measurement(kind, index)) for index in range(48)]
    items[0] = replace(items[0], sqrt_information=((1.2,),))
    names = tuple(dict.fromkeys(name for factor in items for name in factor.variable_names))
    blocks = [
        graph.JointParameterBlock(
            name,
            inputs()[name],
            known_bad_steps=(0.1,) * len(inputs()[name]),
            schur_role="eliminated" if name == "pose" else "retained",
        )
        for name in names
    ]
    for name in names:
        items.append(
            factors.make_joint_prior_factor(
                factor_id=f"prior-{name}",
                observation_group=f"gauge-{name}",
                block=name,
                target=(0.0,) * len(inputs()[name]),
                sigma=(0.5,) * len(inputs()[name]),
            )
        )
    options = graph.JointOptimizerOptions(
        max_iterations=3, split_seed=17, linear_solver=linear_solver, huber_delta=0.1
    )
    traces = []
    original = graph._linearize

    def linearize(*args):
        residual, jacobian = original(*args)
        traces.append((residual.tobytes(), jacobian.tobytes()))
        return residual, jacobian

    monkeypatch.setattr(graph, "_linearize", linearize)
    accelerated = graph.BackendNeutralJointOptimizer().solve(blocks, items, options)
    fast_traces = traces.copy()
    traces.clear()
    disabled: ContextVar[OrderedDict[bytes, object] | None] = ContextVar(
        "disabled_rgbd", default=None
    )
    monkeypatch.setattr(factors, "_EVALUATION_CACHE", disabled)
    scalar = graph.BackendNeutralJointOptimizer().solve(blocks, items, options)
    assert traces == fast_traces
    assert scalar.as_dict() == accelerated.as_dict()
    assert len(fast_traces) >= 2
    assert len(accelerated.probes) == 2 * sum(block.dimension for block in blocks)
    assert accelerated.train_factor_family_diagnostics
    assert _EVALUATION_CACHE.get() is None
