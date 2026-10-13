"""Which sensor's candidate rotation the pair errors of a check run point to.

Each pair of a run reads the candidate's rotation error between two frames,
``e = rotvec(R_candidate R_estimate^T)`` about the parent frame's axes. When several
pairs share frames, the errors carry more than each verdict: they say *which* frame's
candidate is off.

Model
-----
* If only frame ``f``'s candidate is off by a small rotation ``w`` about ``f``'s own
  axes (the true pose is the candidate turned by ``-w``), a pair with ``f`` as child
  reads ``R_parent_child w`` and a pair with ``f`` as parent reads ``-w``, to first
  order; pairs without ``f`` read zero.
* For every frame of a connected group of pairs, ``w`` is fitted by weighted least
  squares on the observed axes, and the fit's chi-square is compared with the
  hypothesis that nothing is off (every error zero).
* A direction of ``w`` the pairs constrain worse than the rotation floor (its std
  would exceed it) is not fitted, so a far-fetched turn cannot absorb the errors; an
  own axis that lies mostly along such directions is reported undetermined.
* An axis's weight is ``1 / max(std, floor)**2`` with ``floor`` the verdicts' rotation
  floor over ``sigma_k``. The check judges an error against
  ``max(sigma_k std, floor)``, so attribution works at the same resolution, and the
  floor absorbs what the std leaves out (on Hilti, other parts of one recording move a
  camera pair's reading by about 1.5 times its std).
* A group is ``consistent`` when nothing-off fits. Otherwise a frame *explains* the
  errors when its fit is acceptable, its move is significant, and its chi-square is
  within the 3-dof critical value of the best acceptable fit. The group is
  ``attributed`` when exactly one frame explains them, ``ambiguous`` when several do
  (a frame seen by a single pair cannot be told from its partner), and
  ``unexplained`` when none does.

The report is informational and changes no verdict.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from scipy.stats import chi2

from calibrex.check.closure import DERIVED_PAIRS, ClosureEdge
from calibrex.check.verdict import VerdictOptions
from calibrex.core.calibration_check import (
    CheckAttributionGroup,
    CheckAttributionHypothesis,
    CheckPairRecord,
    CheckRotationAttribution,
)

FloatArray: TypeAlias = NDArray[np.float64]
ALPHA = 0.01
UNDETERMINED_MIX = 0.05
"""An own axis is undetermined when this much of it lies in the fit's null space."""
ASSUMPTIONS: tuple[str, ...] = (
    "first order: each pair reads R_parent_child w for a child frame turned by w, -w for a "
    "parent frame turned by w",
    "the pairs' errors are independent; pairs that share an input make the tests optimistic",
    "a bias shared by every pair through one frame (one estimator type) reads as that frame turned",
    "an axis std is used at least at the verdicts' rotation floor over sigma_k",
    "a direction of a frame's turn that the pairs constrain worse than the rotation floor "
    "(std above it) is not fitted; an own axis mostly along such directions is undetermined",
    "informational: no verdict changes",
)


@dataclass(frozen=True)
class AttributionObservation:
    """One pair's rotation error ``rotvec(R_candidate R_estimate^T)`` about its parent's axes."""

    label: str
    parent: str
    child: str
    candidate_rotation: FloatArray
    """``R_parent_child`` of the candidate."""
    error_deg: FloatArray
    std_deg: FloatArray
    observed: tuple[bool, bool, bool]


def observations_from_run(
    records: Sequence[CheckPairRecord],
    edges: Mapping[tuple[str, tuple[str, ...]], ClosureEdge],
) -> list[AttributionObservation]:
    """The rotation error of every pair that produced an independent estimate.

    The same estimates as the closure report: derived pairs, ``camera-focal`` and
    inconclusive pairs are left out.
    """

    observations: list[AttributionObservation] = []
    for record in records:
        if (
            record.status not in {"pass", "warn", "fail"}
            or record.pair in DERIVED_PAIRS
            or record.pair == "camera-focal"
            or record.compared_transform is None
        ):
            continue
        edge = edges.get((record.pair, tuple(record.frames)))
        if edge is None or edge.parent == edge.child or not any(edge.rot_observed):
            continue
        candidate = Rotation.from_quat(record.compared_transform.rotation_quat_xyzw).as_matrix()
        error = np.degrees(Rotation.from_matrix(candidate @ edge.rotation.T).as_rotvec())
        observations.append(
            AttributionObservation(
                label=f"{record.pair}:{edge.parent}>{edge.child}",
                parent=edge.parent,
                child=edge.child,
                candidate_rotation=candidate,
                error_deg=error,
                std_deg=np.asarray(edge.rot_std_deg, dtype=np.float64),
                observed=edge.rot_observed,
            )
        )
    return observations


def build_rotation_attribution(
    observations: Sequence[AttributionObservation],
    options: VerdictOptions,
    *,
    alpha: float = ALPHA,
) -> CheckRotationAttribution | None:
    """Single-frame hypotheses for every connected group of pairs; ``None`` without pairs."""

    if not observations:
        return None
    floor = options.rotation_floor_deg / options.sigma_k
    groups = [
        _attribute(group, floor, options.rotation_floor_deg, alpha)
        for group in _groups(observations)
    ]
    return CheckRotationAttribution(
        assumptions=list(ASSUMPTIONS), sigma_floor_deg=floor, alpha=alpha, groups=groups
    )


def _groups(observations: Sequence[AttributionObservation]) -> list[list[AttributionObservation]]:
    parent: dict[str, str] = {}

    def find(frame: str) -> str:
        parent.setdefault(frame, frame)
        while parent[frame] != frame:
            parent[frame] = parent[parent[frame]]
            frame = parent[frame]
        return frame

    for item in observations:
        parent[find(item.parent)] = find(item.child)
    grouped: dict[str, list[AttributionObservation]] = {}
    for item in observations:
        grouped.setdefault(find(item.parent), []).append(item)
    return sorted(grouped.values(), key=lambda group: sorted(_frames(group)))


def _frames(group: Sequence[AttributionObservation]) -> list[str]:
    return sorted({frame for item in group for frame in (item.parent, item.child)})


@dataclass(frozen=True)
class _Fit:
    hypothesis: CheckAttributionHypothesis
    move_p_value: float


def _attribute(
    group: Sequence[AttributionObservation], floor: float, max_std: float, alpha: float
) -> CheckAttributionGroup:
    rows = [(item, axis) for item in group for axis in range(3) if item.observed[axis]]
    errors = np.array([item.error_deg[axis] for item, axis in rows])
    weights = np.array([1.0 / max(float(item.std_deg[axis]), floor) ** 2 for item, axis in rows])
    null_chi2 = float(np.sum(weights * errors**2))
    null_p = float(chi2.sf(null_chi2, len(rows)))
    frames = _frames(group)
    fits = [_fit(frame, group, rows, errors, weights, null_chi2, max_std) for frame in frames]
    fits.sort(key=lambda fit: fit.hypothesis.chi2)
    status: Literal["consistent", "attributed", "ambiguous", "unexplained"]
    acceptable = [
        fit for fit in fits if fit.hypothesis.p_value >= alpha and fit.move_p_value < alpha
    ]
    # a frame competes with the best one when its chi-square is within the critical value
    # of a 3-dof rotation: a far better fit is preferred, not called ambiguous
    margin = float(chi2.isf(alpha, 3))
    explaining = [
        fit.hypothesis.frame
        for fit in acceptable
        if fit.hypothesis.chi2 - acceptable[0].hypothesis.chi2 < margin
    ]
    frame: str | None = None
    if null_p >= alpha:
        status = "consistent"
        summary = (
            f"the pair errors fit the candidate (chi2 {null_chi2:.1f} on {len(rows)} axes, "
            f"p {null_p:.2f}): no frame's rotation is off beyond the verdicts' resolution"
        )
    elif len(explaining) == 1:
        status = "attributed"
        frame = explaining[0]
        best = next(fit.hypothesis for fit in fits if fit.hypothesis.frame == frame)
        summary = (
            f"{frame} alone explains the pair errors: its candidate is turned by "
            f"{_rotation_text(best)} (chi2 {best.chi2:.1f} on {best.dof} dof, "
            f"p {best.p_value:.2f}; nothing off: chi2 {null_chi2:.1f} on {len(rows)} axes)"
        )
    elif explaining:
        status = "ambiguous"
        summary = (
            f"{', '.join(explaining)} each explain the pair errors alone; "
            "these pairs cannot tell which candidate is off"
        )
    else:
        status = "unexplained"
        best = fits[0].hypothesis
        summary = (
            f"no single frame explains the pair errors (best: {best.frame}, chi2 "
            f"{best.chi2:.1f} on {best.dof} dof, p {best.p_value:.2g}): more than one "
            "candidate is off, or the estimates disagree"
        )
    return CheckAttributionGroup(
        frames=frames,
        pairs=[item.label for item in group],
        observations=len(rows),
        null_chi2=null_chi2,
        null_p_value=null_p,
        hypotheses=[fit.hypothesis for fit in fits],
        status=status,
        frame=frame,
        candidates=explaining if status == "ambiguous" else [],
        summary=summary,
    )


def _fit(
    frame: str,
    group: Sequence[AttributionObservation],
    rows: Sequence[tuple[AttributionObservation, int]],
    errors: FloatArray,
    weights: FloatArray,
    null_chi2: float,
    max_std: float,
) -> _Fit:
    """Least-squares ``w`` (about ``frame``'s axes) when only ``frame`` is off."""

    design = np.zeros((len(rows), 3))
    for index, (item, axis) in enumerate(rows):
        if item.child == frame:
            design[index] = item.candidate_rotation[axis]
        elif item.parent == frame:
            design[index, axis] = -1.0
    information = design.T @ (weights[:, None] * design)
    values, vectors = np.linalg.eigh(information)
    # a direction the pairs pin down worse than ``max_std`` is not fitted: it would let a
    # far-fetched turn absorb the errors
    keep = values >= 1.0 / max_std**2
    rank = int(keep.sum())
    covariance = (vectors[:, keep] / values[keep]) @ vectors[:, keep].T
    rotation = covariance @ (design.T @ (weights * errors))
    residual = errors - design @ rotation
    fit_chi2 = float(np.sum(weights * residual**2))
    dof = len(rows) - rank
    null_space = vectors[:, ~keep]
    determined = [float(np.linalg.norm(null_space[axis])) < UNDETERMINED_MIX for axis in range(3)]
    std = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return _Fit(
        hypothesis=CheckAttributionHypothesis(
            frame=frame,
            rotation_deg=[float(rotation[k]) if determined[k] else None for k in range(3)],
            std_deg=[float(std[k]) if determined[k] else None for k in range(3)],
            chi2=fit_chi2,
            dof=dof,
            p_value=float(chi2.sf(fit_chi2, dof)) if dof > 0 else 1.0,
            pairs=[item.label for item in group if frame in (item.parent, item.child)],
        ),
        move_p_value=float(chi2.sf(max(null_chi2 - fit_chi2, 0.0), rank)) if rank else 1.0,
    )


def _rotation_text(hypothesis: CheckAttributionHypothesis) -> str:
    parts = [
        f"{name} {value:+.2f} ± {std:.2f}"
        for name, value, std in zip(
            ("roll", "pitch", "yaw"), hypothesis.rotation_deg, hypothesis.std_deg, strict=True
        )
        if value is not None and std is not None
    ]
    return (", ".join(parts) or "an undetermined rotation") + " deg about its own axes"


__all__ = [
    "AttributionObservation",
    "build_rotation_attribution",
    "observations_from_run",
]
