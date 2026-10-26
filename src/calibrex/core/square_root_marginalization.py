"""Dense residual-space marginalization without subtracting normal matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from pydantic import Field
from scipy.linalg.lapack import dormqr

from calibrex.core.result import StrictModel


class SquareRootEliminationEvidence(StrictModel):
    """Rank decision for residual-space elimination, distinct from normal rank."""

    eliminated_dimension: int = Field(ge=0)
    numerical_rank: int = Field(ge=0)
    zero_information_dimension: int = Field(ge=0)
    singular_value_threshold: float = Field(ge=0, allow_inf_nan=False)
    retained_rows_before_compression: int | None = Field(default=None, ge=0)
    retained_rows_after_compression: int | None = Field(default=None, ge=0)


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

    A thin SVD and implicit Householder transformations avoid constructing a
    square residual-space basis. Storage is linear in rows for fixed columns.
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
            left, singular_values, _ = np.linalg.svd(normalized, full_matrices=False)
            threshold = (
                60. * float(np.finfo(float).eps) * max(rows, eliminated_dim)
                * float(singular_values[0])
            )
            rank = int(np.count_nonzero(singular_values > threshold))
            if rank:
                transformed = _complement_rows(
                    left[:, :rank], np.column_stack((retained, rhs)),
                )
                retained, rhs = transformed[:, :-1], transformed[:, -1]
    if not np.all(np.isfinite(retained)) or not np.all(np.isfinite(rhs)):
        raise ValueError("marginalized residual equations must be finite")
    return SquareRootMarginalizationResult(
        jacobian=np.array(retained, copy=True), residual=np.array(rhs, copy=True),
        eliminated_dimension=int(eliminated_dim), numerical_rank=rank,
        zero_information_dimension=int(np.count_nonzero(~active)),
        singular_value_threshold=threshold,
    )


def _complement_rows(
    basis: NDArray[np.float64], target: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply Q.T implicitly and keep rows orthogonal to the supplied basis."""
    rank = basis.shape[1]
    if rank == basis.shape[0]:
        return np.zeros((0, target.shape[1]), dtype=float)
    reflectors, tau = np.linalg.qr(basis, mode="raw")
    reflectors = np.array(reflectors.T, order="F", copy=True)
    target = np.array(target, order="F", copy=True)
    _unused, work, info = dormqr("L", "T", reflectors, tau, target, lwork=-1)
    if info != 0:
        raise ValueError("residual projection workspace query failed")
    transformed, _work, info = dormqr(
        "L", "T", reflectors, tau, target, lwork=max(1, int(work[0])), overwrite_c=1,
    )
    if info != 0:
        raise ValueError("residual projection failed")
    return np.asarray(transformed[rank:], dtype=float)


def compress_residual_factor(
    jacobian: NDArray[np.float64], residual: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Bound factor rows by column count plus one, preserving constant cost.

    Unpivoted QR of the augmented matrix ``[J, r]`` preserves
    ``||J x + r||²`` without forming normal equations or discarding small positive
    directions. Only R is constructed, avoiding a square residual-space Q.
    This is a floating-point orthogonal transformation, not exact arithmetic.
    """
    if np.iscomplexobj(jacobian) or np.iscomplexobj(residual):
        raise ValueError("residual equations must be real")
    matrix = np.asarray(jacobian, dtype=float)
    rhs = np.asarray(residual, dtype=float)
    if matrix.ndim != 2 or rhs.shape != (matrix.shape[0],):
        raise ValueError("residual equation dimensions must agree")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(rhs)):
        raise ValueError("residual equations must be finite")
    if matrix.shape[0] <= matrix.shape[1] + 1:
        return np.array(matrix, copy=True), np.array(rhs, copy=True)
    with np.errstate(over="ignore", invalid="ignore"):
        triangular = np.linalg.qr(np.column_stack((matrix, rhs)), mode="r")
    if not np.all(np.isfinite(triangular)):
        raise ValueError("compressed residual equations must be finite")
    return triangular[:, :-1].copy(), triangular[:, -1].copy()
