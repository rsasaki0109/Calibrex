"""Schur marginalization and priors for continuous-time trajectory windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cholesky

from calibrex.core.geometry import SE3
from calibrex.core.se3_manifold import se3_exp, se3_log

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class KnotMarginalizationPrior:
    """Information-form prior on one knot pose in a window-local index."""

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
    """Pack a dense marginalized normal system for one retained knot."""

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
    """Split a block-diagonal retained Hessian into per-knot priors."""

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


def marginalization_prior_blocks(
    prior: KnotMarginalizationPrior,
    knot: SE3,
) -> tuple[FloatArray, FloatArray]:
    """Return weighted residual and Jacobian rows for one information prior."""

    delta = se3_log(prior.anchor_pose, knot)
    try:
        weight = cholesky(prior.information, lower=True)
    except np.linalg.LinAlgError:
        symmetrized = 0.5 * (prior.information + prior.information.T)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetrized)
        clipped = np.clip(eigenvalues, 1.0e-9, None)
        weight = eigenvectors @ np.diag(np.sqrt(clipped))
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
