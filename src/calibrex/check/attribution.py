"""Which sensor's candidate the pair errors of a check run point to.

Each pair of a run reads the candidate's rotation error between two frames,
``e = rotvec(R_candidate R_estimate^T)`` about the parent frame's axes, and pairs that
estimate translation also read ``t_candidate - t_estimate`` along the parent's axes.
When several pairs share frames, the errors carry more than each verdict: they say
*which* frame's candidate is off.

Model
-----
* If only frame ``f``'s candidate is off by a small rotation ``w`` and a shift ``u``
  about and along ``f``'s own axes (the true pose is the candidate turned by ``-w`` and
  moved by ``-u``), a pair with ``f`` as child reads rotation ``R_parent_child w`` and
  translation ``R_parent_child u``, and a pair with ``f`` as parent reads rotation
  ``-w`` and translation ``t_parent_child x w - u``, to first order (the parent's turn
  swings the child's origin on its lever arm ``t_parent_child``, the candidate's
  translation); pairs without ``f`` read zero.
* For every frame of a connected group of pairs, ``w`` and ``u`` are fitted by
  weighted least squares on the observed axes, and the fit's chi-square is compared
  with the hypothesis that nothing is off (every error zero). Without translation in
  the group only ``w`` is fitted.
* A direction of ``(w, u)`` the pairs constrain worse than the floors (its std would
  exceed the rotation floor, the translation floor, measured in floors) is not fitted,
  so a far-fetched turn or shift cannot absorb the errors; an own axis that lies mostly
  along such directions is reported undetermined.
* An axis's weight is ``1 / max(std, floor)**2`` with ``floor`` the verdicts' rotation
  or translation floor over ``sigma_k``. The check judges an error against
  ``max(sigma_k std, floor)``, so attribution works at the same resolution, and the
  floor absorbs what the std leaves out (on Hilti, other parts of one recording move a
  camera pair's reading by about 1.5 times its std).
* A group is ``consistent`` when nothing-off fits. Otherwise a frame *explains* the
  errors when its fit is acceptable, its move is significant, and its chi-square is
  within the critical value of the best acceptable fit (3 dof for a turn, 6 when the
  group observes translation). The group is ``attributed`` when exactly one frame
  explains them, ``ambiguous`` when several do (a frame seen by a single pair cannot be
  told from its partner), and ``unexplained`` when none does.

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
NO_AXES = (False, False, False)
ASSUMPTIONS: tuple[str, ...] = (
    "first order: each pair reads R_parent_child w (rotation) and R_parent_child u "
    "(translation) for a child frame turned by w and moved by u, -w and t_parent_child x w "
    "- u for a parent frame",
    "the pairs' errors are independent; pairs that share an input make the tests optimistic",
    "a bias shared by every pair through one frame (one estimator type) reads as that frame "
    "turned or moved",
    "an axis std is used at least at the verdicts' rotation or translation floor over sigma_k",
    "a direction of a frame's turn and shift that the pairs constrain worse than the floors "
    "(std above them) is not fitted; an own axis mostly along such directions is undetermined",
    "informational: no verdict changes",
)


@dataclass(frozen=True)
class AttributionObservation:
    """One pair's errors about and along its parent's axes.

    Rotation ``rotvec(R_candidate R_estimate^T)`` in degrees; translation
    ``t_candidate - t_estimate`` in metres when the pair estimates it.
    """

    label: str
    parent: str
    child: str
    candidate_rotation: FloatArray
    """``R_parent_child`` of the candidate."""
    error_deg: FloatArray
    std_deg: FloatArray
    observed: tuple[bool, bool, bool]
    candidate_translation: FloatArray | None = None
    """``t_parent_child`` of the candidate: the lever arm of the parent's turn."""
    translation_error_m: FloatArray | None = None
    translation_std_m: FloatArray | None = None
    translation_observed: tuple[bool, bool, bool] = NO_AXES


def observations_from_run(
    records: Sequence[CheckPairRecord],
    edges: Mapping[tuple[str, tuple[str, ...]], ClosureEdge],
) -> list[AttributionObservation]:
    """The rotation and translation errors of every pair with an independent estimate.

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
        if edge is None or edge.parent == edge.child:
            continue
        translated = edge.translation is not None and any(edge.trans_observed)
        if not any(edge.rot_observed) and not translated:
            continue
        candidate = Rotation.from_quat(record.compared_transform.rotation_quat_xyzw).as_matrix()
        error = np.degrees(Rotation.from_matrix(candidate @ edge.rotation.T).as_rotvec())
        lever = np.asarray(record.compared_transform.translation_m, dtype=np.float64)
        observations.append(
            AttributionObservation(
                label=f"{record.pair}:{edge.parent}>{edge.child}",
                parent=edge.parent,
                child=edge.child,
                candidate_rotation=candidate,
                error_deg=error,
                std_deg=np.asarray(edge.rot_std_deg, dtype=np.float64),
                observed=edge.rot_observed,
                candidate_translation=lever,
                translation_error_m=(
                    lever - edge.translation
                    if translated and edge.translation is not None
                    else None
                ),
                translation_std_m=(
                    np.asarray(edge.trans_std_m, dtype=np.float64) if translated else None
                ),
                translation_observed=edge.trans_observed if translated else NO_AXES,
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
    floors = _Floors(
        rotation_deg=options.rotation_floor_deg / options.sigma_k,
        translation_m=options.translation_floor_m / options.sigma_k,
        max_rotation_std_deg=options.rotation_floor_deg,
        max_translation_std_m=options.translation_floor_m,
    )
    groups = [_attribute(group, floors, alpha) for group in _groups(observations)]
    return CheckRotationAttribution(
        assumptions=list(ASSUMPTIONS),
        sigma_floor_deg=floors.rotation_deg,
        sigma_floor_m=floors.translation_m,
        alpha=alpha,
        groups=groups,
    )


@dataclass(frozen=True)
class _Floors:
    rotation_deg: float
    translation_m: float
    max_rotation_std_deg: float
    max_translation_std_m: float


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
class _Row:
    item: AttributionObservation
    kind: Literal["rotation", "translation"]
    axis: int


@dataclass(frozen=True)
class _Fit:
    hypothesis: CheckAttributionHypothesis
    move_p_value: float


def _rows(group: Sequence[AttributionObservation]) -> list[_Row]:
    rows = [
        _Row(item, "rotation", axis) for item in group for axis in range(3) if item.observed[axis]
    ]
    rows += [
        _Row(item, "translation", axis)
        for item in group
        for axis in range(3)
        if item.translation_observed[axis] and item.translation_error_m is not None
    ]
    return rows


def _attribute(
    group: Sequence[AttributionObservation], floors: _Floors, alpha: float
) -> CheckAttributionGroup:
    rows = _rows(group)
    errors = np.array([_error(row) for row in rows])
    weights = np.array([1.0 / max(_std(row), _floor(row, floors)) ** 2 for row in rows])
    null_chi2 = float(np.sum(weights * errors**2))
    null_p = float(chi2.sf(null_chi2, len(rows)))
    translated = any(row.kind == "translation" for row in rows)
    frames = _frames(group)
    fits = [
        _fit(frame, group, rows, errors, weights, null_chi2, floors, translated) for frame in frames
    ]
    fits.sort(key=lambda fit: fit.hypothesis.chi2)
    status: Literal["consistent", "attributed", "ambiguous", "unexplained"]
    acceptable = [
        fit for fit in fits if fit.hypothesis.p_value >= alpha and fit.move_p_value < alpha
    ]
    # a frame competes with the best one when its chi-square is within the critical value
    # of one frame's move (3 dof for a turn, 6 for a turn and a shift): a far better fit is
    # preferred, not called ambiguous
    margin = float(chi2.isf(alpha, 6 if translated else 3))
    explaining = [
        fit.hypothesis.frame
        for fit in acceptable
        if fit.hypothesis.chi2 - acceptable[0].hypothesis.chi2 < margin
    ]
    rotation_rows = sum(row.kind == "rotation" for row in rows)
    frame: str | None = None
    if null_p >= alpha:
        status = "consistent"
        what = "rotation or translation" if translated else "rotation"
        summary = (
            f"the pair errors fit the candidate (chi2 {null_chi2:.1f} on {len(rows)} axes, "
            f"p {null_p:.2f}): no frame's {what} is off beyond the verdicts' resolution"
        )
    elif len(explaining) == 1:
        status = "attributed"
        frame = explaining[0]
        best = next(fit.hypothesis for fit in fits if fit.hypothesis.frame == frame)
        summary = (
            f"{frame} alone explains the pair errors: its candidate is {_move_text(best)} "
            f"(chi2 {best.chi2:.1f} on {best.dof} dof, p {best.p_value:.2f}; nothing off: "
            f"chi2 {null_chi2:.1f} on {len(rows)} axes)"
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
        observations=rotation_rows,
        translation_observations=len(rows) - rotation_rows,
        null_chi2=null_chi2,
        null_p_value=null_p,
        hypotheses=[fit.hypothesis for fit in fits],
        status=status,
        frame=frame,
        candidates=explaining if status == "ambiguous" else [],
        summary=summary,
    )


def _error(row: _Row) -> float:
    if row.kind == "rotation":
        return float(row.item.error_deg[row.axis])
    assert row.item.translation_error_m is not None
    return float(row.item.translation_error_m[row.axis])


def _std(row: _Row) -> float:
    if row.kind == "rotation":
        return float(row.item.std_deg[row.axis])
    assert row.item.translation_std_m is not None
    return float(row.item.translation_std_m[row.axis])


def _floor(row: _Row, floors: _Floors) -> float:
    return floors.rotation_deg if row.kind == "rotation" else floors.translation_m


def _skew(vector: FloatArray) -> FloatArray:
    x, y, z = (float(item) for item in vector)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _fit(
    frame: str,
    group: Sequence[AttributionObservation],
    rows: Sequence[_Row],
    errors: FloatArray,
    weights: FloatArray,
    null_chi2: float,
    floors: _Floors,
    translated: bool,
) -> _Fit:
    """Least-squares ``w`` (deg) and ``u`` (m) about ``frame``'s axes when only it is off."""

    design = np.zeros((len(rows), 6))
    for index, row in enumerate(rows):
        item, axis = row.item, row.axis
        if row.kind == "rotation":
            if item.child == frame:
                design[index, :3] = item.candidate_rotation[axis]
            elif item.parent == frame:
                design[index, axis] = -1.0
        elif item.child == frame:
            design[index, 3:] = item.candidate_rotation[axis]
        elif item.parent == frame:
            lever = item.candidate_translation
            if lever is not None:  # t x w, w in degrees
                design[index, :3] = np.radians(_skew(lever)[axis])
            design[index, 3 + axis] = -1.0
    # in units of the floors, a direction the pairs pin down worse than one floor is not
    # fitted: it would let a far-fetched turn or shift absorb the errors
    scale = np.array([floors.max_rotation_std_deg] * 3 + [floors.max_translation_std_m] * 3)
    scaled = design * scale
    information = scaled.T @ (weights[:, None] * scaled)
    values, vectors = np.linalg.eigh(information)
    keep = values >= 1.0
    rank = int(keep.sum())
    covariance = (vectors[:, keep] / values[keep]) @ vectors[:, keep].T
    move = scale * (covariance @ (scaled.T @ (weights * errors)))
    residual = errors - design @ move
    fit_chi2 = float(np.sum(weights * residual**2))
    dof = len(rows) - rank
    null_space = vectors[:, ~keep]
    determined = [float(np.linalg.norm(null_space[axis])) < UNDETERMINED_MIX for axis in range(6)]
    std = scale * np.sqrt(np.maximum(np.diag(covariance), 0.0))

    def axes(offset: int) -> tuple[list[float | None], list[float | None]]:
        picked = range(offset, offset + 3)
        return (
            [float(move[k]) if determined[k] else None for k in picked],
            [float(std[k]) if determined[k] else None for k in picked],
        )

    rotation, rotation_std = axes(0)
    translation, translation_std = axes(3) if translated else (None, None)
    return _Fit(
        hypothesis=CheckAttributionHypothesis(
            frame=frame,
            rotation_deg=rotation,
            std_deg=rotation_std,
            translation_m=translation,
            translation_std_m=translation_std,
            chi2=fit_chi2,
            dof=dof,
            p_value=float(chi2.sf(fit_chi2, dof)) if dof > 0 else 1.0,
            pairs=[item.label for item in group if frame in (item.parent, item.child)],
        ),
        move_p_value=float(chi2.sf(max(null_chi2 - fit_chi2, 0.0), rank)) if rank else 1.0,
    )


def _move_text(hypothesis: CheckAttributionHypothesis) -> str:
    turn = [
        f"{name} {value:+.2f} ± {std:.2f}"
        for name, value, std in zip(
            ("roll", "pitch", "yaw"), hypothesis.rotation_deg, hypothesis.std_deg, strict=True
        )
        if value is not None and std is not None
    ]
    text = "turned by " + (", ".join(turn) or "an undetermined rotation") + " deg"
    if hypothesis.translation_m is None or hypothesis.translation_std_m is None:
        return text + " about its own axes"
    shift = [
        f"{name} {value:+.3f} ± {std:.3f}"
        for name, value, std in zip(
            ("x", "y", "z"), hypothesis.translation_m, hypothesis.translation_std_m, strict=True
        )
        if value is not None and std is not None
    ]
    return (
        text
        + " and moved by "
        + (", ".join(shift) or "an undetermined shift")
        + " m, about and along its own axes"
    )


__all__ = [
    "AttributionObservation",
    "build_rotation_attribution",
    "observations_from_run",
]
