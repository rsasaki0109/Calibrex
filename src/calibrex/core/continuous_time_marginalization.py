"""Schur marginalization and priors for continuous-time trajectory windows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3
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


def joint_marginalization_prior_residual(
    prior: JointKnotMarginalizationPrior, knots: list[SE3],
) -> FloatArray:
    """Evaluate the full retained quadratic up to a constant independent of pose."""
    if max(prior.knot_indices) >= len(knots):
        raise ValueError("joint prior knot index is out of range")
    delta = np.concatenate([
        se3_log(anchor, knots[index])
        for index, anchor in zip(prior.knot_indices, prior.anchor_poses, strict=True)
    ])
    weight = _square_root_information(prior.information, dimension=len(delta))
    return weight @ delta + _prior_residual_offset(prior, weight)


def joint_marginalization_prior_blocks(
    prior: JointKnotMarginalizationPrior, knots: list[SE3],
) -> tuple[FloatArray, list[tuple[int, FloatArray]]]:
    """Whiten all knot columns together so cross-knot curvature is preserved."""
    residual = joint_marginalization_prior_residual(prior, knots)
    dimension = len(residual)
    weight = _square_root_information(prior.information, dimension=dimension)
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


def schur_complement_normal(
    hessian: FloatArray,
    gradient: FloatArray,
    eliminated_dim: int,
) -> tuple[FloatArray, FloatArray]:
    """Return the Schur complement on retained variables after eliminating a prefix."""

    if eliminated_dim <= 0:
        return np.array(hessian, copy=True), np.array(gradient, copy=True)
    if eliminated_dim >= hessian.shape[0]:
        raise ValueError("eliminated dimension must be smaller than the system size")
    retained = slice(eliminated_dim, None)
    eliminated = slice(None, eliminated_dim)
    hee = hessian[eliminated, eliminated]
    her = hessian[eliminated, retained]
    hre = hessian[retained, eliminated]
    hrr = hessian[retained, retained]
    ge = gradient[eliminated]
    gr = gradient[retained]
    solved = np.linalg.solve(hee, np.column_stack((her, ge)))
    marginalized = hrr - hre @ solved[:, :-1]
    marginal_gradient = gr - hre @ solved[:, -1]
    return marginalized, marginal_gradient


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
