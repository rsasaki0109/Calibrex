"""CPU import isolation, scoped CUDA ownership and explicit TUM options."""

import json
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from calibrex.graph import _rgbd_execution as runtime
from calibrex.graph import joint_optimization as graph
from calibrex.graph._point_plane_batches import _OwnedPointFactors
from calibrex.solvers import native_tum_joint_slac_solver as tum


def test_default_tum_import_and_options_do_not_load_cuda():
    code = (
        "import json,sys; from calibrex.solvers.native_tum_joint_slac_solver "
        "import TUMJointSlacOptions; o=TUMJointSlacOptions(); "
        "assert o.joint_evaluation_backend=='numpy'; "
        "assert 'joint_evaluation_backend' not in o.as_dict(); "
        "print(json.dumps([n for n in sys.modules if n=='cupy' or n.startswith('cupy.')]))"
    )
    done = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert json.loads(done.stdout) == []


@pytest.mark.parametrize("value", ["auto", "numba_cpu", None, True, 1, [], {}])
def test_unknown_backend_is_rejected(value):
    with pytest.raises(ValueError, match="joint_evaluation_backend"):
        tum._joint_backend({"joint_evaluation_backend": value})


def test_cuda_option_is_explicit_and_cpu_provenance_stays_unchanged():
    assert tum._joint_backend({}) == "numpy"
    assert tum._joint_backend({"joint_evaluation_backend": "cuda"}) == "cuda"
    assert (
        tum.TUMJointSlacOptions(joint_evaluation_backend="cuda").as_dict()[
            "joint_evaluation_backend"
        ]
        == "cuda"
    )


class Plan:
    groups = (SimpleNamespace(rows=range(8192)),)

    def evaluate(self, values):
        return np.asarray([1.0, -0.0])


class Adapter:
    def __init__(self):
        self.closed = False

    def accepts(self, plan):
        return True

    def evaluate(self, plan, values):
        return np.asarray([2.0, -0.0])

    def report(self):
        return {"closed_before_report": self.closed}

    def close(self):
        self.closed = True


def test_nested_cpu_scope_restores_cuda_and_releases_it_on_error(monkeypatch):
    created = []

    def new():
        result = Adapter()
        created.append(result)
        return result

    monkeypatch.setattr(runtime, "_new_cuda_backend", new)
    plan = Plan()
    assert runtime._evaluate_rgbd(plan, {}).tobytes() == plan.evaluate({}).tobytes()
    with pytest.raises(RuntimeError, match="stop"), runtime._using_rgbd_backend("cuda"):
        assert created == []
        assert runtime._evaluate_rgbd(plan, {})[0] == 2.0
        with runtime._using_rgbd_backend("numpy"):
            assert runtime._evaluate_rgbd(plan, {})[0] == 1.0
        assert runtime._evaluate_rgbd(plan, {})[0] == 2.0
        assert len(created) == 1
        raise RuntimeError("stop")
    assert created[0].closed
    assert runtime._EXECUTION.get() is None
    report = runtime._last_execution_report()
    assert report == {"closed_before_report": False}
    report["changed"] = True
    assert "changed" not in runtime._last_execution_report()


def test_rejected_device_plan_uses_the_complete_cpu_plan(monkeypatch):
    adapter = Adapter()
    monkeypatch.setattr(adapter, "accepts", lambda plan: False)
    monkeypatch.setattr(runtime, "_new_cuda_backend", lambda: adapter)
    with runtime._using_rgbd_backend("cuda"):
        plan = Plan()
        assert runtime._evaluate_rgbd(plan, {}).tobytes() == plan.evaluate({}).tobytes()
    assert adapter.closed


def test_missing_cuda_dependency_reports_the_requested_extra_and_resets_scope(monkeypatch):
    monkeypatch.setitem(sys.modules, "calibrex.solvers.cuda_rgbd_joint_adapter", None)
    with pytest.raises(ValueError, match=r"calibrex\[cuda\]"), runtime._using_rgbd_backend("cuda"):
        runtime._evaluate_rgbd(Plan(), {})
    assert runtime._EXECUTION.get() is None


def test_small_native_graph_does_not_initialize_cuda(monkeypatch):
    def unexpected():
        raise AssertionError("small graph loaded CUDA")

    monkeypatch.setattr(runtime, "_new_cuda_backend", unexpected)
    plan = Plan()
    plan.groups = (SimpleNamespace(rows=range(runtime._MIN_CUDA_POINTS - 1)),)
    with runtime._using_rgbd_backend("cuda"):
        assert runtime._evaluate_rgbd(plan, {}).tobytes() == plan.evaluate({}).tobytes()
    assert runtime._last_execution_report() is None


def test_cuda_policy_provenance_does_not_import_cupy():
    code = (
        "import sys; from calibrex.graph._rgbd_execution import _cuda_identity; "
        "value=_cuda_identity(); assert 'initial_min_points=4096' in value; "
        "assert 'kernel_sha256=' in value; assert 'cupy' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("kind", ["depth", "trilinear", "centered", "xyz", "pair"])
def test_owned_cpu_diagnostics_use_dimensions_without_evaluating_callbacks(kind, monkeypatch):
    from test_rgbd_transform_reuse import batch_problem, inputs

    blocks, factors = batch_problem(kind, regularizers=True)
    factors = _OwnedPointFactors(factors)
    values = inputs()
    expected_dimensions = tuple(len(factor.residuals(values)) for factor in factors)
    options = graph.JointOptimizerOptions()
    residual, jacobian = graph._linearize(
        factors, values, blocks, graph._variable_layout(blocks), options
    )
    metadata = runtime._diagnostic_dimensions
    monkeypatch.setattr(runtime, "_diagnostic_dimensions", lambda *_: None)
    expected = graph._factor_family_diagnostics(factors, values, residual, jacobian, options)
    monkeypatch.setattr(runtime, "_diagnostic_dimensions", metadata)
    codes = {factor.evaluator.__code__ for factor in factors}
    calls = []

    def profile(frame, event, arg):
        if event == "call" and frame.f_code in codes:
            calls.append(frame.f_code)

    previous = sys.getprofile()
    try:
        sys.setprofile(profile)
        assert metadata(factors, values) == expected_dimensions
        actual = graph._factor_family_diagnostics(factors, values, residual, jacobian, options)
    finally:
        sys.setprofile(previous)
    assert actual == expected
    assert calls == []
    assert metadata(tuple(factors), values) is None


@pytest.mark.parametrize("mode", ["unowned", "custom", "changed_code"])
def test_cpu_diagnostics_preserve_callback_path_without_verified_dimensions(mode):
    from test_rgbd_transform_reuse import batch_problem, inputs

    _blocks, items = batch_problem("depth", regularizers=True)
    if mode == "custom":
        items[0] = replace(items[0], evaluator=lambda _: (0.125, -0.25))
    elif mode == "changed_code":
        items[0].evaluator.__code__ = items[0].evaluator.__code__.replace(co_name="custom")
    factors = tuple(items) if mode == "unowned" else _OwnedPointFactors(items)
    values = inputs()
    assert runtime._diagnostic_dimensions(factors, values) is None
    residuals = np.asarray([value for factor in factors for value in factor.residuals(values)])
    codes = {factor.evaluator.__code__ for factor in factors}
    calls = []

    def profile(frame, event, arg):
        if event == "call" and frame.f_code in codes:
            calls.append(frame.f_code)

    previous = sys.getprofile()
    try:
        sys.setprofile(profile)
        diagnostics = graph._factor_family_diagnostics(
            factors, values, residuals, None, graph.JointOptimizerOptions()
        )
    finally:
        sys.setprofile(previous)
    assert len(calls) == len(factors)
    assert sum(item.residual_dimension for item in diagnostics) == len(residuals)
