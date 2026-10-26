"""Local, diagonally scaled information diagnostics for trajectory knots."""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from calibrex.core.continuous_time_marginalization import SchurEliminationEvidence
from calibrex.core.result import StrictModel


class KnotInformationEvidence(StrictModel):
    """Numerical rank at a fixed linearization, not a global observability proof."""

    parameter_count: int = Field(ge=1)
    numerical_rank: int = Field(ge=0)
    numerical_nullity: int = Field(ge=0)
    relative_eigenvalue_tolerance: float = Field(gt=0, lt=1)
    eigenvalue_threshold: float = Field(ge=0, allow_inf_nan=False)
    scaled_eigenvalues: list[float]
    zero_information_parameters: list[str]
    nullspace_participation: dict[str, float]


class WindowObservabilityEvidence(StrictModel):
    """Separate measurement information from information inherited through priors."""

    elimination: SchurEliminationEvidence | None = None
    frozen_parameters: list[str] = Field(default_factory=list)
    window_index: int = Field(ge=0)
    global_knot_indices: list[int]
    measurements: KnotInformationEvidence
    with_marginal_prior: KnotInformationEvidence


def analyze_knot_information(
    information: NDArray[np.float64],
    knot_indices: tuple[int, ...],
    *, relative_tolerance: float = 1.e-10,
) -> KnotInformationEvidence:
    """Assess rank after diagonal scaling; report nullspace projector diagonals.

    Translation and rotation columns use their local tangent coordinates. Scaling
    removes arbitrary independent units; participation refers to these scaled
    coordinates and includes coupled null modes, not only zero columns.
    """
    if np.iscomplexobj(information):
        raise ValueError("information must be real")
    matrix = np.asarray(information, dtype=float)
    dimension = 6 * len(knot_indices)
    if len(set(knot_indices)) != len(knot_indices) or any(index < 0 for index in knot_indices):
        raise ValueError("knot indices must be distinct and nonnegative")
    if not dimension or matrix.shape != (dimension, dimension):
        raise ValueError("information shape must match nonempty knot indices")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("information must be finite")
    if not math.isfinite(relative_tolerance) or not 0 < relative_tolerance < 1:
        raise ValueError("relative tolerance must be between zero and one")
    diagonal = np.diag(matrix)
    if np.any(diagonal < 0):
        raise ValueError("information diagonal must be nonnegative")
    active = diagonal > 0
    if np.any(matrix[~active, :] != 0) or np.any(matrix[:, ~active] != 0):
        raise ValueError("zero information diagonal must have zero row and column")
    scale = np.sqrt(np.where(active, diagonal, 1.))
    scaled = matrix / scale[:, None] / scale[None, :]
    if not np.all(np.isfinite(scaled)) or not np.allclose(scaled, scaled.T, rtol=0, atol=1e-12):
        raise ValueError("scaled information must be finite and symmetric")
    values, vectors = np.linalg.eigh(.5 * (scaled + scaled.T))
    threshold = relative_tolerance * max(float(values[-1]), 0.)
    if float(values[0]) < -max(threshold, 1.e-14):
        raise ValueError("information must be positive semidefinite")
    observed = values > threshold
    participation = np.sum(vectors[:, ~observed] ** 2, axis=1)
    names = [f"knot[{index}].{axis}" for index in knot_indices
             for axis in ("tx", "ty", "tz", "rx", "ry", "rz")]
    rank = int(np.count_nonzero(observed))
    return KnotInformationEvidence(
        parameter_count=dimension, numerical_rank=rank, numerical_nullity=dimension - rank,
        relative_eigenvalue_tolerance=relative_tolerance, eigenvalue_threshold=threshold,
        scaled_eigenvalues=[float(v) for v in np.maximum(values, 0.)],
        zero_information_parameters=[name for name, flag in zip(names, active, strict=True)
                                     if not flag],
        nullspace_participation=dict(zip(names, (float(v) for v in participation), strict=True)),
    )
