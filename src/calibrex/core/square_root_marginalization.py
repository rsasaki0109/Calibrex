"""Dense residual-space marginalization without subtracting normal matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from calibrex.core.result import StrictModel


class SquareRootEliminationEvidence(StrictModel):
    """Rank decision for residual-space elimination, distinct from normal rank."""

    eliminated_dimension: int = Field(ge=0)
    numerical_rank: int = Field(ge=0)
    zero_information_dimension: int = Field(ge=0)
    singular_value_threshold: float = Field(ge=0, allow_inf_nan=False)


@dataclass(frozen=True)
class SquareRootMarginalizationResult:
    """Retained factor for ``min_e ||J_e e + J_r x + r||²``.

    Rows include the constant residual cost. Rank evidence refers to eliminated
    columns after independent column normalization, at this linearization only.
    """

    jacobian: NDArray[np.float64]
    residual: NDArray[np.float64]
    eliminated_dimension: int
    numerical_rank: int
    zero_information_dimension: int
    singular_value_threshold: float

    def evidence(self) -> SquareRootEliminationEvidence:
        """Return schema-valid numerical rank evidence without factor arrays."""
        return SquareRootEliminationEvidence(
            eliminated_dimension=self.eliminated_dimension, numerical_rank=self.numerical_rank,
            zero_information_dimension=self.zero_information_dimension,
            singular_value_threshold=self.singular_value_threshold,
        )


def marginalize_residual_prefix(
    jacobian: NDArray[np.float64],
    residual: NDArray[np.float64],
    eliminated_dim: int,
) -> SquareRootMarginalizationResult:
    """Eliminate a column prefix of an already weighted least-squares factor.

    Apply the orthogonal complement of the eliminated column space directly to
    retained columns and residuals. No normal matrix, damping, eigenvalue floor,
    or subtraction of two information matrices is used. Independent nonzero
    eliminated columns are normalized before the numerical rank decision.

    This dense reference API constructs a full residual-space orthogonal basis
    (quadratic memory in the row count); it is intended for small systems.
    """
    if np.iscomplexobj(jacobian) or np.iscomplexobj(residual):
        raise ValueError("residual equations must be real")
    matrix = np.asarray(jacobian, dtype=float)
    rhs = np.asarray(residual, dtype=float)
    if matrix.ndim != 2 or rhs.shape != (matrix.shape[0],):
        raise ValueError("residual equation dimensions must agree")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(rhs)):
        raise ValueError("residual equations must be finite")
    if (not isinstance(eliminated_dim, (int, np.integer))
            or not 0 <= eliminated_dim < matrix.shape[1]):
        raise ValueError("eliminated dimension must be a nonnegative integer below column count")
    rows = matrix.shape[0]
    eliminated = matrix[:, :eliminated_dim]
    retained = matrix[:, eliminated_dim:]
    rank = 0
    threshold = 0.
    active = np.zeros(eliminated_dim, dtype=bool)
    if rows and eliminated_dim:
        # Normalize in two stages to avoid overflow in squaring large entries
        # and underflow in squaring subnormal, but independent, columns.
        maxima = np.max(np.abs(eliminated), axis=0)
        active = maxima > 0.
        if np.any(active):
            normalized = eliminated[:, active] / maxima[active]
            normalized /= np.linalg.norm(normalized, axis=0)
            left, singular_values, _ = np.linalg.svd(normalized, full_matrices=True)
            threshold = (
                60. * float(np.finfo(float).eps) * max(rows, eliminated_dim)
                * float(singular_values[0])
            )
            rank = int(np.count_nonzero(singular_values > threshold))
            complement = left[:, rank:]
            with np.errstate(over="ignore", invalid="ignore"):
                retained = complement.T @ retained
                rhs = complement.T @ rhs
    if not np.all(np.isfinite(retained)) or not np.all(np.isfinite(rhs)):
        raise ValueError("marginalized residual equations must be finite")
    return SquareRootMarginalizationResult(
        jacobian=np.array(retained, copy=True), residual=np.array(rhs, copy=True),
        eliminated_dimension=int(eliminated_dim), numerical_rank=rank,
        zero_information_dimension=int(np.count_nonzero(~active)),
        singular_value_threshold=threshold,
    )
