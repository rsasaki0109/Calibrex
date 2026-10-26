"""Optional local updates restricted to the measured numerical eigenspace."""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from calibrex.core.result import StrictModel


class ProjectedStepEvidence(StrictModel):
    """Rank selection at one attempted linear update, before candidate acceptance."""

    parameter_count: int = Field(ge=1)
    numerical_rank: int = Field(ge=0)
    zero_information_count: int = Field(ge=0)
    eigenvalue_threshold: float = Field(ge=0, allow_inf_nan=False)
    relative_tolerance: float = Field(gt=0, allow_inf_nan=False)
    relative_null_gradient: float = Field(ge=0, allow_inf_nan=False)


def observable_subspace_step(
    normal: NDArray[np.float64], gradient: NDArray[np.float64],
    damped_normal: NDArray[np.float64],
) -> tuple[NDArray[np.float64], ProjectedStepEvidence]:
    """Solve the damped quadratic only in the diagonally scaled observed subspace.

    Damping selects a step within that subspace; it adds no residual or prior.
    A material gradient in a numerical null direction is rejected, not discarded.
    """
    if any(np.iscomplexobj(x) for x in (normal, gradient, damped_normal)):
        raise ValueError("step model must be real")
    h, g, damped = (np.asarray(x, dtype=float) for x in (normal, gradient, damped_normal))
    count = g.size
    if not count or g.shape != (count,) or h.shape != (count, count) or damped.shape != h.shape:
        raise ValueError("step model dimensions must agree")
    if not all(np.all(np.isfinite(x)) for x in (h, g, damped)):
        raise ValueError("step model must be finite")
    tolerance = 60. * float(np.finfo(float).eps) * count
    diagonal = np.diag(h)
    if np.any(diagonal < 0):
        raise ValueError("information diagonal must be nonnegative")
    active = diagonal > 0
    if np.any(h[~active, :] != 0) or np.any(h[:, ~active] != 0) or np.any(g[~active] != 0):
        raise ValueError("zero-information coordinates must have zero coupling and gradient")
    step = np.zeros(count, dtype=float)
    rank, threshold, null_gradient = 0, 0., 0.
    if np.any(active):
        scale = np.sqrt(diagonal[active])
        scaled = h[np.ix_(active, active)] / scale[:, None] / scale[None, :]
        scaled_damped = damped[np.ix_(active, active)] / scale[:, None] / scale[None, :]
        scaled_gradient = g[active] / scale
        if not all(np.all(np.isfinite(x)) for x in (scaled, scaled_damped, scaled_gradient)):
            raise ValueError("scaled step model must be finite")
        if any(not np.allclose(x, x.T, rtol=0, atol=tolerance)
               for x in (scaled, scaled_damped)):
            raise ValueError("scaled step model must be symmetric")
        values, vectors = np.linalg.eigh(.5 * (scaled + scaled.T))
        threshold = tolerance * max(float(np.max(np.abs(values))), 1.)
        if values[0] < -threshold:
            raise ValueError("information must be positive semidefinite")
        observed = values > threshold
        rank = int(np.count_nonzero(observed))
        null_gradient = float(np.max(np.abs(vectors[:, ~observed].T @ scaled_gradient), initial=0.))
        null_gradient /= max(float(np.max(np.abs(scaled_gradient))), 1e-300)
        if not math.isfinite(null_gradient) or null_gradient > tolerance:
            raise ValueError("gradient is outside the numerical information range")
        if np.all(observed):
            np.linalg.cholesky(scaled_damped)
            step[active] = np.linalg.solve(damped[np.ix_(active, active)], -g[active])
        else:
            basis = vectors[:, observed]
            reduced = basis.T @ scaled_damped @ basis
            np.linalg.cholesky(reduced)
            coordinates = np.linalg.solve(reduced, -(basis.T @ scaled_gradient))
            step[active] = (basis @ coordinates) / scale
    if not np.all(np.isfinite(step)):
        raise ValueError("projected step must be finite")
    return step, ProjectedStepEvidence(
        parameter_count=count, numerical_rank=rank,
        zero_information_count=int(np.count_nonzero(~active)), eigenvalue_threshold=threshold,
        relative_tolerance=tolerance, relative_null_gradient=null_gradient,
    )
