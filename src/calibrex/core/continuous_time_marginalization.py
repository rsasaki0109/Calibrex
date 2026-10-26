"""Schur marginalization and priors for continuous-time trajectory windows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from calibrex.core.geometry import SE3
from calibrex.core.result import StrictModel
from calibrex.core.se3_manifold import se3_exp, se3_log

FloatArray: TypeAlias = NDArray[np.float64]
_INFORMATION_ROUNDOFF: float = 60.0 * float(np.finfo(np.float64).eps)


@dataclass(frozen=True)
class KnotMarginalizationPrior:
    """Information prior on one window-local knot pose, in its anchor tangent.

    Weighting requires a real, finite, symmetric positive-semidefinite 6x6
    information matrix. A zero diagonal must have an exactly zero row/column.
    """

    knot_index: int
    anchor_pose: SE3
    information: FloatArray
    gradient: FloatArray = field(default_factory=lambda: np.zeros(6, dtype=float))


@dataclass(frozen=True)
class JointKnotMarginalizationPrior:
    """Retained joint quadratic in fixed anchor tangents, including cross blocks."""

    knot_indices: tuple[int, ...]
    anchor_poses: tuple[SE3, ...]
    information: FloatArray
    gradient: FloatArray

    def __post_init__(self) -> None:
        count = len(self.knot_indices)
        if not count or len(self.anchor_poses) != count:
            raise ValueError("joint prior needs matching nonempty knot indices and anchors")
        if len(set(self.knot_indices)) != count or min(self.knot_indices) < 0:
            raise ValueError("joint prior knot indices must be distinct and nonnegative")
        if np.shape(self.information) != (6 * count, 6 * count):
            raise ValueError("joint prior information size must match its knot count")
        weight = _square_root_information(self.information, dimension=6 * count)
        _prior_residual_offset(self, weight)


@dataclass(frozen=True)
class SquareRootKnotMarginalizationPrior:
    """Joint anchor-tangent factor retained without conversion to information."""

    knot_indices: tuple[int, ...]
    anchor_poses: tuple[SE3, ...]
    jacobian: FloatArray
    residual: FloatArray

    def __post_init__(self) -> None:
        count = len(self.knot_indices)
        if (not count or len(self.anchor_poses) != count
                or len(set(self.knot_indices)) != count or min(self.knot_indices) < 0):
            raise ValueError(
                "square-root prior needs matching distinct nonnegative knots and anchors"
            )
        j, r = np.asarray(self.jacobian), np.asarray(self.residual)
        if (np.iscomplexobj(j) or np.iscomplexobj(r) or j.ndim != 2
                or j.shape[1] != 6 * count or r.shape != (j.shape[0],)
                or not np.all(np.isfinite(j)) or not np.all(np.isfinite(r))):
            raise ValueError("square-root prior needs finite real matching factors")


JointKnotPrior: TypeAlias = JointKnotMarginalizationPrior | SquareRootKnotMarginalizationPrior


def _joint_factor(prior: JointKnotPrior) -> tuple[FloatArray, FloatArray]:
    if isinstance(prior, SquareRootKnotMarginalizationPrior):
        return prior.jacobian, prior.residual
    weight = _square_root_information(prior.information, dimension=6 * len(prior.knot_indices))
    return weight, _prior_residual_offset(prior, weight)


def joint_marginalization_prior_residual(
    prior: JointKnotPrior, knots: list[SE3],
) -> FloatArray:
    """Evaluate the full retained quadratic up to a constant independent of pose."""
    if max(prior.knot_indices) >= len(knots):
        raise ValueError("joint prior knot index is out of range")
    delta = np.concatenate([
        se3_log(anchor, knots[index])
        for index, anchor in zip(prior.knot_indices, prior.anchor_poses, strict=True)
    ])
    weight, offset = _joint_factor(prior)
    return weight @ delta + offset


def joint_marginalization_prior_blocks(
    prior: JointKnotPrior, knots: list[SE3],
) -> tuple[FloatArray, list[tuple[int, FloatArray]]]:
    """Whiten all knot columns together so cross-knot curvature is preserved."""
    residual = joint_marginalization_prior_residual(prior, knots)
    weight, _offset = _joint_factor(prior)
    blocks: list[tuple[int, FloatArray]] = []
    epsilon = 1.e-7
    for local, (index, anchor) in enumerate(zip(
        prior.knot_indices, prior.anchor_poses, strict=True,
    )):
        delta = se3_log(anchor, knots[index])
        jacobian = np.zeros((6, 6), dtype=float)
        for axis in range(6):
            step = np.zeros(6, dtype=float)
            step[axis] = epsilon
            jacobian[:, axis] = (
                se3_log(anchor, se3_exp(knots[index], step)) - delta
            ) / epsilon
        blocks.append((index, weight[:, 6 * local:6 * local + 6] @ jacobian))
    return residual, blocks


class SchurEliminationEvidence(StrictModel):
    """Numerical elimination rank and range checks in diagonally scaled coordinates."""

    eliminated_dimension: int = Field(ge=0)
    numerical_rank: int = Field(ge=0)
    zero_information_dimension: int = Field(ge=0)
    coupled_nullity: int = Field(ge=0)
    eigenvalue_threshold: float = Field(ge=0, allow_inf_nan=False)
    relative_range_tolerance: float = Field(gt=0, allow_inf_nan=False)
    max_relative_range_error: float = Field(ge=0, allow_inf_nan=False)


def schur_complement_normal(
    hessian: FloatArray,
    gradient: FloatArray,
    eliminated_dim: int,
) -> tuple[FloatArray, FloatArray]:
    """Eliminate a prefix while checking numerical nullspace consistency."""
    normal, rhs, _evidence = schur_complement_normal_with_evidence(
        hessian, gradient, eliminated_dim,
    )
    return normal, rhs


def schur_complement_normal_with_evidence(
    hessian: FloatArray, gradient: FloatArray, eliminated_dim: int,
) -> tuple[FloatArray, FloatArray, SchurEliminationEvidence]:
    """Generalized Schur elimination with explicit scaled rank and range evidence.

    Null directions may be removed only when both the retained coupling and the
    eliminated gradient lie in the numerical range. No damping or prior is added.
    """
    range_tolerance = _INFORMATION_ROUNDOFF * max(1, eliminated_dim)
    if np.iscomplexobj(hessian) or np.iscomplexobj(gradient):
        raise ValueError("normal equations must be real")
    hessian = np.asarray(hessian, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    if (hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]
            or gradient.shape != (hessian.shape[0],)):
        raise ValueError("normal equation dimensions must agree")
    if not np.all(np.isfinite(hessian)) or not np.all(np.isfinite(gradient)):
        raise ValueError("normal equations must be finite")
    if eliminated_dim < 0:
        raise ValueError("eliminated dimension must be nonnegative")
    if eliminated_dim == 0:
        evidence = SchurEliminationEvidence(
            eliminated_dimension=0, numerical_rank=0, zero_information_dimension=0,
            coupled_nullity=0, eigenvalue_threshold=0.,
            relative_range_tolerance=range_tolerance, max_relative_range_error=0.,
        )
        return np.array(hessian, copy=True), np.array(gradient, copy=True), evidence
    if eliminated_dim >= hessian.shape[0]:
        raise ValueError("eliminated dimension must be smaller than the system size")
    if not np.allclose(hessian, hessian.T, rtol=range_tolerance, atol=0.):
        raise ValueError("normal information must be symmetric")
    retained = slice(eliminated_dim, None)
    eliminated = slice(None, eliminated_dim)
    hee = hessian[eliminated, eliminated]
    her = hessian[eliminated, retained]
    hre = hessian[retained, eliminated]
    hrr = hessian[retained, retained]
    ge = gradient[eliminated]
    gr = gradient[retained]
    diagonal = np.diag(hee)
    if np.any(diagonal < 0.):
        raise ValueError("eliminated information diagonal must be nonnegative")
    active = diagonal > 0.
    missing = np.flatnonzero(~active)
    if (np.any(hessian[missing, :] != 0.) or np.any(hessian[:, missing] != 0.)
            or np.any(gradient[missing] != 0.)):
        raise ValueError(
            "zero-information eliminated coordinates require zero coupling and gradient"
        )
    correction = np.zeros((hrr.shape[0], her.shape[1] + 1), dtype=float)
    rank = 0
    threshold = 0.
    range_error = 0.
    if np.any(active):
        block = hee[np.ix_(active, active)]
        scale = np.sqrt(diagonal[active])
        normalized = block / scale[:, None] / scale[None, :]
        if not np.all(np.isfinite(normalized)) or not np.allclose(
            normalized, normalized.T, rtol=0., atol=range_tolerance,
        ):
            raise ValueError("eliminated information must be symmetric after scaling")
        values, vectors = np.linalg.eigh(.5 * (normalized + normalized.T))
        threshold = range_tolerance * max(float(np.max(np.abs(values))), 1.)
        if values[0] < -threshold:
            raise ValueError("eliminated information must be positive semidefinite")
        observed = values > threshold
        rank = int(np.count_nonzero(observed))
        right = np.column_stack((her[active], ge[active]))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            scaled_right = right / scale[:, None]
            scaled_left = hre[:, active] / scale[None, :]
        if not np.all(np.isfinite(scaled_right)) or not np.all(np.isfinite(scaled_left)):
            raise ValueError("scaled eliminated right-hand side and coupling must be finite")
        if np.all(observed):
            solved = np.linalg.solve(normalized, scaled_right)
        else:
            projections = vectors[:, ~observed].T @ scaled_right
            denominators = np.maximum(np.max(np.abs(scaled_right), axis=0), 1.e-300)
            range_error = float(np.max(np.abs(projections) / denominators[None, :]))
            if range_error > range_tolerance:
                raise ValueError("eliminated coupling or gradient is outside the numerical range")
            basis = vectors[:, observed]
            solved = (
                basis @ ((basis.T @ scaled_right) / values[observed, None])
            )
        # Contract in scaled coordinates: unscaling the eliminated solution
        # first can overflow even when the retained quadratic is representable.
        with np.errstate(over="ignore", invalid="ignore"):
            correction = scaled_left @ solved
    with np.errstate(over="ignore", invalid="ignore"):
        marginalized = hrr - correction[:, :-1]
        marginal_gradient = gr - correction[:, -1]
    if not np.all(np.isfinite(marginalized)) or not np.all(np.isfinite(marginal_gradient)):
        raise ValueError("marginalized normal equations must be finite")
    return marginalized, marginal_gradient, SchurEliminationEvidence(
        eliminated_dimension=eliminated_dim, numerical_rank=rank,
        zero_information_dimension=int(np.count_nonzero(~active)),
        coupled_nullity=int(np.count_nonzero(active)) - rank,
        eigenvalue_threshold=threshold, relative_range_tolerance=range_tolerance,
        max_relative_range_error=range_error,
    )


def marginalize_knot_prefix(
    hessian: FloatArray,
    gradient: FloatArray,
    *,
    eliminated_knot_count: int,
) -> tuple[FloatArray, FloatArray]:
    """Marginalize out the first ``eliminated_knot_count`` knot poses (6 DoF each)."""

    return schur_complement_normal(
        hessian,
        gradient,
        6 * eliminated_knot_count,
    )


def information_from_normal(
    hessian: FloatArray,
    gradient: FloatArray,
    *,
    anchor_pose: SE3,
    knot_index: int,
) -> KnotMarginalizationPrior:
    """Pack one retained information block at the supplied anchor.

    Preserve the linear term as well as the retained curvature.
    """

    if hessian.shape != (6, 6):
        raise ValueError("single-knot marginalization requires a 6x6 information block")
    symmetrized = 0.5 * (hessian + hessian.T)
    return KnotMarginalizationPrior(
        knot_index=knot_index,
        anchor_pose=anchor_pose,
        information=symmetrized,
        gradient=np.array(gradient, dtype=float, copy=True),
    )


def split_retained_information(
    hessian: FloatArray,
    gradient: FloatArray,
    *,
    retained_knot_count: int,
    anchor_poses: tuple[SE3, ...],
    start_knot_index: int,
) -> tuple[KnotMarginalizationPrior, ...]:
    """Pack per-knot diagonal information blocks.

    This block-diagonal approximation drops cross-knot blocks, but keeps gradients.
    """

    if hessian.shape[0] != 6 * retained_knot_count:
        raise ValueError("retained Hessian size does not match retained knot count")
    if len(anchor_poses) != retained_knot_count:
        raise ValueError("anchor pose count must match retained knot count")
    priors: list[KnotMarginalizationPrior] = []
    for local_index in range(retained_knot_count):
        offset = 6 * local_index
        block = hessian[offset : offset + 6, offset : offset + 6]
        priors.append(
            information_from_normal(
                block,
                gradient[offset : offset + 6],
                anchor_pose=anchor_poses[local_index],
                knot_index=start_knot_index + local_index,
            )
        )
    return tuple(priors)


def _square_root_information(information: FloatArray, *, dimension: int = 6) -> FloatArray:
    """Return W with W.T @ W equal to the declared information up to roundoff.

    Equilibrate by the positive diagonal before testing symmetry and the
    spectrum, so large weights cannot hide an invalid weak direction. The
    dimensionless allowance is 60 machine epsilons times the corresponding
    matrix/spectral norm. Only negative eigensolver roundoff is clipped; no
    positive floor is added and positive weak information is retained.
    """

    array = np.asarray(information)
    if array.shape != (dimension, dimension) or np.iscomplexobj(array):
        raise ValueError(f"prior information must be a real {dimension}x{dimension} matrix")
    try:
        matrix: FloatArray = np.asarray(array, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"prior information must be a real finite {dimension}x{dimension} matrix"
        ) from error
    if not np.all(np.isfinite(matrix)):
        raise ValueError("prior information must be finite")
    diagonal = np.diag(matrix)
    if np.any(diagonal < 0.0):
        raise ValueError("prior information must have a nonnegative diagonal")
    active = diagonal > 0.0
    if np.any(matrix[~active, :] != 0.0) or np.any(matrix[:, ~active] != 0.0):
        raise ValueError("zero prior information diagonal requires a zero row and column")
    weight: FloatArray = np.zeros((dimension, dimension), dtype=float)
    if not np.any(active):
        return weight
    scale = np.sqrt(diagonal[active])
    # Sequential division avoids overflowing/underflowing an outer product of
    # diagonal scales when the original finite information has extreme units.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized = matrix[np.ix_(active, active)] / scale[:, None] / scale[None, :]
    if not np.all(np.isfinite(normalized)) or np.any(
        np.abs(normalized) > 1.0 + _INFORMATION_ROUNDOFF
    ):
        raise ValueError("prior information must be positive semidefinite")
    tolerance = _INFORMATION_ROUNDOFF * float(np.linalg.norm(normalized, ord=np.inf))
    if np.linalg.norm(normalized - normalized.T, ord=np.inf) > tolerance:
        raise ValueError("prior information must be symmetric after diagonal scaling")
    symmetric = 0.5 * (normalized + normalized.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    spectral_tolerance = _INFORMATION_ROUNDOFF * float(np.max(np.abs(eigenvalues)))
    if np.min(eigenvalues) < -spectral_tolerance:
        raise ValueError("prior information must be positive semidefinite")
    # Rows are eigen-directions: diag(sqrt(lambda)) @ Q.T, then undo scaling.
    weight[np.ix_(active, active)] = (
        np.sqrt(np.maximum(eigenvalues, 0.0))[:, None] * eigenvectors.T
    ) * scale[None, :]
    return weight


def marginalization_prior_blocks(
    prior: KnotMarginalizationPrior,
    knot: SE3,
) -> tuple[FloatArray, FloatArray]:
    """Whiten the prior's Log residual and local Jacobian consistently.

    Invalid information raises ValueError. The finite-difference Log Jacobian
    is evaluated in the fixed anchor tangent.
    """

    weight = _square_root_information(prior.information)
    delta = se3_log(prior.anchor_pose, knot)
    jacobian: NDArray[np.float64] = np.zeros((6, 6), dtype=float)
    epsilon = 1.0e-7
    for axis in range(6):
        step: NDArray[np.float64] = np.zeros(6, dtype=float)
        step[axis] = epsilon
        perturbed = se3_log(prior.anchor_pose, se3_exp(knot, step))
        jacobian[:, axis] = (perturbed - delta) / epsilon
    weighted_jacobian = weight @ jacobian
    weighted_residual = weight @ delta + _prior_residual_offset(prior, weight)
    return weighted_residual, weighted_jacobian


def _prior_residual_offset(
    prior: KnotMarginalizationPrior | JointKnotMarginalizationPrior, weight: FloatArray,
) -> FloatArray:
    """Find b with W.T b = g without suppressing weak coordinate information."""
    dimension = weight.shape[0]
    gradient = np.asarray(prior.gradient)
    if gradient.shape != (dimension,) or np.iscomplexobj(gradient):
        raise ValueError("prior gradient must be real, finite, and match the information size")
    gradient = np.asarray(gradient, dtype=float)
    if not np.all(np.isfinite(gradient)):
        raise ValueError("prior gradient must be finite")
    if not np.any(gradient):
        return np.zeros(dimension, dtype=float)
    scale = np.sqrt(np.diag(prior.information))
    active = scale > 0.
    if np.any(gradient[~active] != 0.):
        raise ValueError("prior gradient is outside the information range")
    if not np.any(active):
        return np.zeros(dimension, dtype=float)
    matrix = weight[:, active].T / scale[active, None]
    target = gradient[active] / scale[active]
    if not np.all(np.isfinite(target)):
        raise ValueError("scaled prior gradient must be finite")
    offset = np.linalg.lstsq(matrix, target, rcond=_INFORMATION_ROUNDOFF)[0]
    error = np.linalg.norm(matrix @ offset - target)
    if error > _INFORMATION_ROUNDOFF * max(float(np.linalg.norm(target)), 1.e-300):
        raise ValueError("prior gradient is outside the information range")
    return offset


def marginalization_prior_residual(prior: KnotMarginalizationPrior, knot: SE3) -> FloatArray:
    """Evaluate the prior including its linear term and a fixed additive constant."""
    weight = _square_root_information(prior.information)
    return weight @ se3_log(prior.anchor_pose, knot) + _prior_residual_offset(prior, weight)
