"""Explicit, synchronous native RGB-D execution; CUDA imports stay optional."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.graph.joint_optimization import (
    JointOptimizerOptions,
    JointParameterBlock,
    JointResidualBlock,
    ParameterValues,
)

if TYPE_CHECKING:
    from calibrex.graph._rgbd_batches import _RGBDPlan
    from calibrex.solvers.cuda_rgbd_joint_adapter import _CudaRGBDBackend

_RGBDBackend: TypeAlias = Literal["numpy", "cuda"]
_FloatArray: TypeAlias = NDArray[np.float64]
_MIN_CUDA_POINTS = 4096


@dataclass
class _Execution:
    adapter: _CudaRGBDBackend | None = None


_EXECUTION: ContextVar[_Execution | None] = ContextVar("rgbd_execution", default=None)
_LAST_REPORT: ContextVar[dict[str, object] | None] = ContextVar("rgbd_report", default=None)


@contextmanager
def _using_rgbd_backend(backend: _RGBDBackend) -> Iterator[None]:
    if backend not in ("numpy", "cuda"):
        raise ValueError("joint_evaluation_backend must be numpy or cuda")
    execution = _Execution() if backend == "cuda" else None
    token = _EXECUTION.set(execution)
    try:
        yield
    finally:
        try:
            adapter = execution.adapter if execution is not None else None
            _LAST_REPORT.set(adapter.report() if adapter is not None else None)
            if adapter is not None:
                adapter.close()
        finally:
            _EXECUTION.reset(token)


def _new_cuda_backend() -> _CudaRGBDBackend:
    try:
        from calibrex.solvers.cuda_rgbd_joint_adapter import _CudaRGBDBackend
    except ImportError as error:
        raise ValueError("CUDA RGB-D evaluation requires calibrex[cuda]") from error
    return _CudaRGBDBackend()


def _adapter(plan: _RGBDPlan) -> _CudaRGBDBackend | None:
    execution = _EXECUTION.get()
    if execution is None:
        return None
    if execution.adapter is None:
        if sum(len(group.rows) for group in plan.groups) < _MIN_CUDA_POINTS:
            return None
        execution.adapter = _new_cuda_backend()
    return execution.adapter if execution.adapter.accepts(plan) else None


def _linearize_rgbd(
    plan: _RGBDPlan,
    values: dict[str, tuple[float, ...]],
    blocks: Sequence[JointParameterBlock],
    layout: dict[str, slice],
    options: JointOptimizerOptions,
) -> tuple[_FloatArray, _FloatArray]:
    adapter = _adapter(plan)
    return (
        plan.linearize(values, blocks, layout, options)
        if adapter is None
        else adapter.linearize(plan, values, blocks, layout, options)
    )


def _evaluate_rgbd(plan: _RGBDPlan, values: ParameterValues) -> _FloatArray:
    adapter = _adapter(plan)
    return plan.evaluate(values) if adapter is None else adapter.evaluate(plan, values)


def _diagnostic_dimensions(
    factors: Sequence[JointResidualBlock], values: ParameterValues
) -> tuple[int, ...] | None:
    if _EXECUTION.get() is None:
        return None
    from calibrex.graph._point_plane_batches import _OwnedPointFactors
    from calibrex.graph._rgbd_batches import _prepare_rgbd_plan

    if type(factors) is not _OwnedPointFactors:
        return None
    plan = _prepare_rgbd_plan(factors, values)
    if plan is None:
        return None
    scalar = {row: dimension for row, dimension, _ in plan.scalars}
    cursor = 0
    dimensions = []
    for _ in factors:
        dimension = scalar.get(cursor, 1)
        dimensions.append(dimension)
        cursor += dimension
    assert cursor == plan.size
    return tuple(dimensions)


def _last_execution_report() -> dict[str, object] | None:
    value = _LAST_REPORT.get()
    return None if value is None else dict(value)


def _cuda_identity() -> str:
    import hashlib

    from calibrex.solvers._cuda_rgbd_joint_kernels import SOURCE

    return (
        f"calibrex.cuda_rgbd_joint/v0.1;initial_min_points={_MIN_CUDA_POINTS};"
        f"fp64;fmad=false;kernel_sha256={hashlib.sha256(SOURCE.encode()).hexdigest()}"
    )
