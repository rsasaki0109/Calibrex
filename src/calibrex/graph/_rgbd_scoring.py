"""Native diagnostic residuals with the factories' exact scalar float bits."""

from __future__ import annotations

from collections.abc import Sequence

from calibrex.graph.joint_optimization import JointResidualBlock, ParameterValues


def _native_residuals(
    factors: Sequence[JointResidualBlock], values: ParameterValues, *, scalar_factors: bool = False,
) -> list[float] | None:
    from calibrex.graph._rgbd_batches import _prepare_rgbd_plan

    plan = _prepare_rgbd_plan(factors, values)
    if plan is None or (scalar_factors and plan.size != len(factors)):
        return None
    from calibrex.graph._rgbd_execution import _evaluate_rgbd

    # Builtin floats retain CPython's ordered/compensated sum behavior in the
    # caller. Iterating NumPy scalars would use a different accumulation path.
    result: list[float] = _evaluate_rgbd(plan, values).tolist()
    return result
