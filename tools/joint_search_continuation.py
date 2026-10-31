"""Resumable pattern search with baseline-preserving global budget allocation."""

from __future__ import annotations

from typing import Any

import numpy as np

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_depth_to_depth_solver import (
    DepthToDepthObservation,
    DepthToDepthOptions,
    evaluate_depth_to_depth_mi,
)
from calibrex.solvers.borer_six_dof_solver import apply_local_se3_delta


def advance(
    batch: list[DepthToDepthObservation],
    initial: SE3,
    options: DepthToDepthOptions,
    state: dict[str, Any],
    budget: int,
) -> None:
    """Spend at most budget NEW evaluations; stop only between complete sweeps."""
    step = np.array(state.get("step", [0.25] * 3 + [0.1] * 3))
    best = np.array(state["delta"], dtype=float)
    trace = state.setdefault("trace", [])
    before = len(trace)
    minimum = np.array([0.01] * 3 + [0.005] * 3)
    bounds = np.array([2.0] * 3 + [1.0] * 3)

    def evaluate(delta: np.ndarray) -> float:
        value = evaluate_depth_to_depth_mi(
            batch, apply_local_se3_delta(initial, delta[:3], delta[3:]), options
        )
        trace.append(
            {
                "delta": delta.tolist(),
                "mi": value.mutual_information,
                "frames": value.evaluated_frame_count,
            }
        )
        return value.mutual_information

    if "mi" not in state:
        if budget < 1:
            raise ValueError("initial evaluation requires budget")
        state["mi"] = evaluate(best)
    score = state["mi"]
    while np.any(step >= minimum):
        deltas = []
        for axis in range(6):
            if step[axis] < minimum[axis]:
                continue
            for direction in [-1.0, 1.0]:
                delta = best.copy()
                delta[axis] = np.clip(
                    delta[axis] + direction * step[axis], -bounds[axis], bounds[axis]
                )
                if not np.array_equal(delta, best):
                    deltas.append(delta)
        if not deltas or len(trace) - before + len(deltas) > budget:
            break
        candidates = [(evaluate(d), tuple(-d), d) for d in deltas]
        value, _, delta = max(candidates, key=lambda c: (c[0], c[1]))
        if value > score + 1e-9:
            best, score = delta, value
        else:
            step *= 0.5
    state.update(
        delta=best.tolist(),
        mi=score,
        step=step.tolist(),
        converged=bool(np.all(step < minimum)),
        evaluations=len(trace),
    )
