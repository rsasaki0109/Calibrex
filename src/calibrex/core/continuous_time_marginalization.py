"""Schur marginalization and priors for continuous-time trajectory windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex.core.geometry import SE3
from calibrex.core.se3_manifold import se3_exp, se3_log

FloatArray: TypeAlias = NDArray[np.float64]
_INFORMATION_ROUNDOFF: float = 60.0 * float(np.finfo(np.float64).eps)


@dataclass(frozen=True)
class KnotMarginalizationPrior:
    """Zero-centered information prior on one window-local knot pose.

    Weighting requires a real, finite, symmetric positive-semidefinite 6x6
    information matrix. A zero diagonal must have an exactly zero row/column.
    """

    knot_index: int
    anchor_pose: SE3
    information: FloatArray


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
    hee_inv = np.linalg.inv(hee)
    marginalized = hrr - hre @ hee_inv @ her
    marginal_gradient = gr - hre @ hee_inv @ ge
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

    The current prior representation does not encode the input gradient.
    """

    if hessian.shape != (6, 6):
        raise ValueError("single-knot marginalization requires a 6x6 information block")
    symmetrized = 0.5 * (hessian + hessian.T)
    return KnotMarginalizationPrior(
        knot_index=knot_index,
        anchor_pose=anchor_pose,
        information=symmetrized,
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

    The current representation drops cross-knot blocks and the input gradient.
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


def _square_root_information(information: FloatArray) -> FloatArray:
    """Return W with W.T @ W equal to the declared information up to roundoff.

    Equilibrate by the positive diagonal before testing symmetry and the
    spectrum, so large weights cannot hide an invalid weak direction. The
    dimensionless allowance is 60 machine epsilons times the corresponding
    matrix/spectral norm. Only negative eigensolver roundoff is clipped; no
    positive floor is added and positive weak information is retained.
    """

    array = np.asarray(information)
    if array.shape != (6, 6) or np.iscomplexobj(array):
        raise ValueError("prior information must be a real 6x6 matrix")
    try:
        matrix: FloatArray = np.asarray(array, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("prior information must be a real finite 6x6 matrix") from error
    if not np.all(np.isfinite(matrix)):
        raise ValueError("prior information must be finite")
    diagonal = np.diag(matrix)
    if np.any(diagonal < 0.0):
        raise ValueError("prior information must have a nonnegative diagonal")
    active = diagonal > 0.0
    if np.any(matrix[~active, :] != 0.0) or np.any(matrix[:, ~active] != 0.0):
        raise ValueError("zero prior information diagonal requires a zero row and column")
    weight: FloatArray = np.zeros((6, 6), dtype=float)
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
    and the zero-centered prior model are unchanged.
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
    weighted_residual = weight @ delta
    return weighted_residual, weighted_jacobian
