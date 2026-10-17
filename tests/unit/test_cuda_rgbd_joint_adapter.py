"""Real-device byte parity and bounds for optional native RGB-D execution."""

import json
from dataclasses import replace

import pytest
from tests.unit.test_rgbd_transform_reuse import KINDS, batch_problem, factory, inputs, measurement

from calibrex.graph import _rgbd_batches as batches
from calibrex.graph import _rgbd_execution as runtime
from calibrex.graph import joint_optimization as graph
from calibrex.graph._point_plane_batches import _OwnedPointFactors

cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device unavailable", allow_module_level=True)
except Exception:
    pytest.skip("CUDA runtime unavailable", allow_module_level=True)

from calibrex.solvers import cuda_rgbd_joint_adapter as cuda  # noqa: E402


@pytest.fixture(autouse=True)
def exercise_real_device_for_small_test_graphs(monkeypatch):
    monkeypatch.setattr(runtime, "_MIN_CUDA_POINTS", 0)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("count", [96, 4097])
def test_complete_residual_and_jacobian_bytes_match_numpy(kind, count):
    blocks, factors = batch_problem(kind, count=count, regularizers=True)
    values = inputs()
    layout = graph._variable_layout(blocks)
    options = graph.JointOptimizerOptions()
    expected = graph._linearize(factors, values, blocks, layout, options)
    with runtime._using_rgbd_backend("cuda"):
        actual = graph._linearize(factors, values, blocks, layout, options)
        residual = graph._factor_residuals(factors, values)
    assert actual[0].tobytes() == expected[0].tobytes()
    assert actual[1].tobytes() == expected[1].tobytes()
    assert residual.tobytes() == expected[0].tobytes()
    report = runtime._last_execution_report()
    assert report["linearizations"] == 1 and report["kernel_launches"] > 0
    assert report["peak_owned_bytes_bound"] <= report["cache_budget_bytes"]


@pytest.mark.parametrize("kind", KINDS)
def test_complete_optimizer_history_and_family_diagnostics_match_numpy(kind):
    blocks, factors = batch_problem(kind, count=384, regularizers=True)
    options = graph.JointOptimizerOptions(max_iterations=3, split_seed=17, linear_solver="schur")
    factors = _OwnedPointFactors(factors)
    expected = graph.BackendNeutralJointOptimizer().solve(blocks, factors, options)
    with runtime._using_rgbd_backend("cuda"):
        actual = graph.BackendNeutralJointOptimizer().solve(blocks, factors, options)
    assert json.dumps(actual.as_dict(), sort_keys=True) == json.dumps(
        expected.as_dict(), sort_keys=True
    )
    assert runtime._last_execution_report()["linearizations"] == len(actual.history) + 1


@pytest.mark.parametrize("kind", ["trilinear", "centered", "xyz", "pair"])
def test_two_calibration_states_keep_signed_zero_and_release_device_data(kind):
    _, factors = batch_problem(kind, count=96, regularizers=True)
    values = inputs()
    plan = batches._prepare_rgbd_plan(factors, values)
    backend = cuda._CudaRGBDBackend()
    assert plan is not None and backend.accepts(plan)
    try:
        name = "xyz" if kind in ("xyz", "pair") else "lattice"
        for sign in (0.0, -0.0, 0.01, -0.01):
            current = {**values, name: (sign,) * len(values[name])}
            assert backend.evaluate(plan, current).tobytes() == plan.evaluate(current).tobytes()
            assert all(len(group.points) <= 2 for group in backend.plans[id(plan)].groups)
        assert len(backend.plans) == 1
    finally:
        backend.close()
    assert not backend.plans


def test_column_scratch_budget_and_device_rejection_keep_all_values(monkeypatch):
    blocks, factors = batch_problem("pair", count=97, regularizers=True)
    values = inputs()
    layout = graph._variable_layout(blocks)
    options = graph.JointOptimizerOptions()
    expected = graph._linearize(factors, values, blocks, layout, options)
    monkeypatch.setattr(cuda, "_COLUMN_ELEMENTS", 512)
    with runtime._using_rgbd_backend("cuda"):
        actual = graph._linearize(factors, values, blocks, layout, options)
    assert actual[1].tobytes() == expected[1].tobytes()
    assert runtime._last_execution_report()["max_column_elements"] <= 512
    plan = batches._prepare_rgbd_plan(factors, values)
    backend = cuda._CudaRGBDBackend(cache_bytes=0)
    assert plan is not None and not backend.accepts(plan) and not backend.plans


def test_custom_callback_preserves_scalar_calls_without_loading_device_data():
    blocks, factors = batch_problem("xyz", count=96, regularizers=True)
    called = []
    original = factors[0]

    def custom(values):
        called.append(1)
        return original.evaluator(values)

    modified = (replace(original, evaluator=custom), *factors[1:])
    values = inputs()
    options = graph.JointOptimizerOptions()
    layout = graph._variable_layout(blocks)
    expected = graph._linearize(modified, values, blocks, layout, options)
    count = len(called)
    called.clear()
    with runtime._using_rgbd_backend("cuda"):
        actual = graph._linearize(modified, values, blocks, layout, options)
    assert actual[0].tobytes() == expected[0].tobytes()
    assert actual[1].tobytes() == expected[1].tobytes()
    assert len(called) == count
    assert runtime._last_execution_report() is None


@pytest.mark.parametrize("kind", KINDS[1:])
@pytest.mark.parametrize("controls", [1, 8, 64])
def test_device_calibration_keeps_ordered_cancellation_and_subnormal_bytes(kind, controls):
    weights = (1.0 / controls,) * controls
    factors = []
    for index in range(96):
        item = measurement(kind, index)
        changes = (
            {"source_lattice_weights": weights, "target_lattice_weights": weights}
            if kind == "pair"
            else {"lattice_weights": weights}
        )
        factors.append(factory(kind, replace(item, **changes)))
    size = controls * (3 if kind in ("xyz", "pair") else 1)
    backend = cuda._CudaRGBDBackend()
    try:
        for offsets in (
            (0.0,) * size,
            (-0.0,) * size,
            tuple((1e12, 1e-12, -1e12, -0.0)[i % 4] for i in range(size)),
            tuple((1e-300, -1e-300, 0.0)[i % 3] for i in range(size)),
        ):
            values = {**inputs(), "xyz" if kind in ("xyz", "pair") else "lattice": offsets}
            plan = batches._prepare_rgbd_plan(factors, values)
            assert plan is not None and backend.accepts(plan)
            for packed in backend.plans[id(plan)].groups:
                actual = packed.calibrated(backend, values)
                expected = packed.group.calibrated_points(values)
                assert actual[0].get().tobytes() == expected[0].tobytes()
                if actual[1] is not None:
                    assert actual[1].get().tobytes() == expected[1].tobytes()
    finally:
        backend.close()


def test_three_native_graphs_evict_the_oldest_device_plan():
    backend = cuda._CudaRGBDBackend()
    plans = []
    try:
        for count in (96, 97, 98):
            _, factors = batch_problem("xyz", count=count, regularizers=True)
            plan = batches._prepare_rgbd_plan(factors, inputs())
            plans.append(plan)
            assert plan is not None and backend.accepts(plan)
            assert len(backend.plans) <= 2
            assert backend.report()["peak_owned_bytes_bound"] <= backend.budget
        assert id(plans[0]) not in backend.plans
        assert id(plans[1]) in backend.plans and id(plans[2]) in backend.plans
    finally:
        backend.close()
