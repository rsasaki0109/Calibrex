"""Rig closure evidence: the estimates of one check run judged against each other.

Each pair of a run is judged against the *candidate* on its own. Closure adds a
second, candidate-free test: the estimates form a graph of frames, and every
loop of it (two estimates of the same frame pair, or three pairs around a
triangle) must multiply to the identity. A loop that does not close says the
estimators disagree with each other, so the verdicts they give on the candidate
are less trustworthy.

Model
-----
* Edges are the run's per-pair estimates ``T_parent_child`` (rotation, and
  translation where the estimator reports it) with the per-axis standard
  deviations of the estimator. Estimates that are composed from other pairs'
  results (``gnss-imu``) are *derived*: they are not independent evidence and
  are excluded.
* Loops are a fundamental cycle basis (a spanning forest, then one loop per
  remaining edge), so parallel estimates and cycles are covered.
* The closure error is ``rotvec(R_loop)`` in degrees and the translation of the
  loop product in metres, both in the axes of the loop's first frame.
* The standard deviation is propagated to first order assuming the members'
  errors are independent, which is optimistic when members share an input (the
  vehicle pairs share the LiDAR odometry). Loops of a basis that share a member
  are correlated with each other as well.
* An axis is judged only if every member observes everything that enters it:
  a closure axis on which an unobserved member axis has a first-order weight
  above :data:`MIX_THRESHOLD` is unchecked. Translation is judged only if every
  member estimates it and every member rotation is fully observed.
* The verdict rule is the one of the pairs (:mod:`calibrex.check.verdict`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from calibrex.check.verdict import AxisEstimate, VerdictOptions, judge_pair, worst_verdict
from calibrex.core.calibration_check import (
    CHECK_PAIR_NAMES,
    CheckAxisName,
    CheckClosure,
    CheckClosureEdge,
    CheckClosureMember,
    CheckClosureReport,
    CheckPairRecord,
)

FloatArray: TypeAlias = NDArray[np.float64]
MIX_THRESHOLD = 0.05
"""Weight above which an unobserved member axis makes a closure axis unchecked."""
DERIVED_PAIRS: Mapping[str, str] = {
    "gnss-imu": "composed from this run's gnss-lidar and imu-lidar results; not independent",
}
ASSUMPTIONS: tuple[str, ...] = (
    "edges are this run's per-pair estimates, not the candidate; derived pairs are excluded",
    "loops are a fundamental cycle basis; loops that share a member are correlated",
    "the std of a closure axis is propagated to first order assuming independent members; "
    "members that share an input (the LiDAR odometry of the vehicle pairs) make it optimistic",
    "an axis is judged only if every member observes everything entering it; "
    "translation needs every member to estimate it",
)
ROTATION_NAMES: tuple[CheckAxisName, ...] = ("roll", "pitch", "yaw")
TRANSLATION_NAMES: tuple[CheckAxisName, ...] = ("x", "y", "z")


@dataclass(frozen=True)
class ClosureEdge:
    """One estimate ``T_parent_child`` with its per-axis standard deviations."""

    pair: str
    parent: str
    child: str
    rotation: FloatArray
    rot_std_deg: tuple[float, float, float]
    rot_observed: tuple[bool, bool, bool]
    translation: FloatArray | None = None
    trans_std_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    trans_observed: tuple[bool, bool, bool] = (False, False, False)
    estimator: str | None = None
    shared_inputs: tuple[str, ...] = ()
    """Inputs this estimate shares with other pairs (see :func:`shared_inputs_of`)."""


def shared_inputs_of(pair: str, frames: Sequence[str]) -> tuple[str, ...]:
    """The measurement stream a pair's estimate is computed from, when others use it too.

    The three vehicle pairs that need LiDAR motion read the same LiDAR odometry
    of the LiDAR frame, so their errors are correlated.
    """

    if pair in {"lidar-vehicle", "lidar-wheel_odometry"} and frames:
        return (f"lidar odometry of {frames[0]}",)
    if pair == "ins-lidar" and len(frames) > 1:
        return (f"lidar odometry of {frames[1]}",)
    return ()


def edge_from_estimates(
    pair: str,
    parent: str,
    child: str,
    compared: FloatArray,
    estimates: Sequence[AxisEstimate],
    estimator: str | None = None,
    shared_inputs: tuple[str, ...] = (),
) -> ClosureEdge:
    """Recover the estimate from the candidate (``compared``) and its per-axis errors.

    The error of an axis is candidate minus estimate; the rotation error is
    ``E = R_candidate R_estimate^T`` so ``R_estimate = E^T R_candidate``.
    """

    by_name = {item.name: item for item in estimates}
    rot_error = np.array(
        [by_name[name].candidate_error if name in by_name else 0.0 for name in ROTATION_NAMES]
    )
    rotation = (
        Rotation.from_rotvec(np.radians(rot_error)).as_matrix().T @ compared[:3, :3]
    ).astype(np.float64)
    translation: FloatArray | None = None
    if any(name in by_name for name in TRANSLATION_NAMES):
        error = np.array(
            [by_name[n].candidate_error if n in by_name else 0.0 for n in TRANSLATION_NAMES]
        )
        translation = compared[:3, 3] - error

    def std(name: CheckAxisName) -> float:
        return by_name[name].std if name in by_name else 0.0

    def observed(name: CheckAxisName) -> bool:
        return name in by_name and by_name[name].estimated

    return ClosureEdge(
        pair=pair,
        parent=parent,
        child=child,
        rotation=rotation,
        rot_std_deg=(std("roll"), std("pitch"), std("yaw")),
        rot_observed=(observed("roll"), observed("pitch"), observed("yaw")),
        translation=translation,
        trans_std_m=(std("x"), std("y"), std("z")),
        trans_observed=(observed("x"), observed("y"), observed("z")),
        estimator=estimator,
        shared_inputs=shared_inputs,
    )


def _skew(vector: FloatArray) -> FloatArray:
    x, y, z = (float(item) for item in vector)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


@dataclass(frozen=True)
class _Step:
    edge: ClosureEdge
    forward: bool


def _loop_steps(
    edges: Sequence[ClosureEdge], extra: int, forest: Sequence[bool]
) -> tuple[list[_Step], list[str]] | None:
    """Fundamental loop of ``edges[extra]`` through the spanning forest of the others.

    The loop starts at the extra edge's parent, goes down the extra edge and
    returns through the forest.
    """

    target = edges[extra]
    adjacency: dict[str, list[tuple[str, _Step]]] = {}
    for index, edge in enumerate(edges):
        if index == extra or not forest[index]:
            continue
        adjacency.setdefault(edge.parent, []).append((edge.child, _Step(edge, True)))
        adjacency.setdefault(edge.child, []).append((edge.parent, _Step(edge, False)))
    # breadth first from the extra edge's child back to its parent
    previous: dict[str, tuple[str, _Step] | None] = {target.child: None}
    queue = [target.child]
    for node in queue:
        for neighbour, step in adjacency.get(node, []):
            if neighbour not in previous:
                previous[neighbour] = (node, step)
                queue.append(neighbour)
    if target.parent not in previous:
        return None
    path: list[_Step] = []
    cursor = target.parent
    while (link := previous[cursor]) is not None:
        cursor, step = link
        path.append(step)
    path.reverse()  # collected from the parent back to the child; the loop walks child to parent
    steps = [_Step(target, True), *path]
    destinations = [step.edge.child if step.forward else step.edge.parent for step in steps]
    return steps, [target.parent, *destinations[:-1]]


def _spanning_edges(edges: Sequence[ClosureEdge]) -> list[bool]:
    """``True`` for edges of the spanning forest (in the given order)."""

    root: dict[str, str] = {}

    def find(node: str) -> str:
        root.setdefault(node, node)
        while root[node] != node:
            root[node] = root[root[node]]
            node = root[node]
        return node

    flags: list[bool] = []
    for edge in edges:
        first, second = find(edge.parent), find(edge.child)
        if first == second:
            flags.append(False)
        else:
            root[first] = second
            flags.append(True)
    return flags


def evaluate_loop(
    steps: Sequence[_Step], frames: Sequence[str], options: VerdictOptions
) -> CheckClosure:
    """Closure error, first-order std and verdict of one loop."""

    count = len(steps)
    prefix = [np.eye(3)]
    contributions: list[FloatArray] = []
    for step in steps:
        edge = step.edge
        rotation = edge.rotation if step.forward else edge.rotation.T
        if edge.translation is None:
            contribution = np.zeros(3)
        elif step.forward:
            contribution = prefix[-1] @ edge.translation
        else:
            contribution = -prefix[-1] @ edge.rotation.T @ edge.translation
        contributions.append(contribution)
        prefix.append(prefix[-1] @ rotation)
    loop_rotation = prefix[-1]
    loop_translation = np.sum(contributions, axis=0)
    rotation_error = np.degrees(Rotation.from_matrix(loop_rotation).as_rotvec())

    has_translation = all(step.edge.translation is not None for step in steps)
    rot_cov = np.zeros((3, 3))
    trans_cov = np.zeros((3, 3))
    rot_masked = [False, False, False]
    trans_masked = [False, False, False]
    reasons: dict[str, str] = {}
    full_rotation = all(all(step.edge.rot_observed) for step in steps)
    for k, step in enumerate(steps):
        edge = step.edge
        base = prefix[k]
        j_rot = base if step.forward else -prefix[k + 1]
        later = np.sum(contributions[k + 1 :], axis=0) if k + 1 < count else np.zeros(3)
        rot_sigma = np.radians(np.array(edge.rot_std_deg))
        for axis in range(3):
            if edge.rot_observed[axis]:
                column = j_rot[:, axis] * rot_sigma[axis]
                rot_cov += np.outer(column, column)
            else:
                for target_axis in range(3):
                    if (
                        abs(j_rot[target_axis, axis]) > MIX_THRESHOLD
                        and not rot_masked[target_axis]
                    ):
                        rot_masked[target_axis] = True
                        reasons[ROTATION_NAMES[target_axis]] = (
                            f"{edge.pair} does not observe its {ROTATION_NAMES[axis]}, "
                            "which enters this axis with weight "
                            f"{abs(j_rot[target_axis, axis]):.2f}"
                        )
        if has_translation and edge.translation is not None:
            j_trans_eps = base if step.forward else -prefix[k + 1]
            j_trans_rot = -_skew(later) @ j_rot
            if not step.forward:
                j_trans_rot = j_trans_rot - base @ edge.rotation.T @ _skew(edge.translation)
            trans_sigma = np.array(edge.trans_std_m)
            for axis in range(3):
                if edge.trans_observed[axis]:
                    column = j_trans_eps[:, axis] * trans_sigma[axis]
                    trans_cov += np.outer(column, column)
                else:
                    for target_axis in range(3):
                        if (
                            abs(j_trans_eps[target_axis, axis]) > MIX_THRESHOLD
                            and not trans_masked[target_axis]
                        ):
                            trans_masked[target_axis] = True
                            reasons[TRANSLATION_NAMES[target_axis]] = (
                                f"{edge.pair} does not observe its {TRANSLATION_NAMES[axis]}, "
                                "which enters this axis with weight "
                                f"{abs(j_trans_eps[target_axis, axis]):.2f}"
                            )
            if full_rotation:
                for axis in range(3):
                    column = j_trans_rot[:, axis] * rot_sigma[axis]
                    trans_cov += np.outer(column, column)

    notes: list[str] = []
    estimates: list[AxisEstimate] = []
    rot_std = np.degrees(np.sqrt(np.diag(rot_cov)))
    for axis, name in enumerate(ROTATION_NAMES):
        estimates.append(
            AxisEstimate(
                name,
                "deg",
                float(rotation_error[axis]),
                float(rot_std[axis]),
                not rot_masked[axis],
                reasons.get(name),
                None if not rot_masked[axis] else "unobservable",
            )
        )
    if has_translation:
        trans_std = np.sqrt(np.diag(trans_cov))
        for axis, name in enumerate(TRANSLATION_NAMES):
            masked = trans_masked[axis] or not full_rotation
            reason = reasons.get(name)
            if reason is None and not full_rotation:
                reason = (
                    "a member's rotation is not fully observed, which enters the translation "
                    "closure through the lever arms"
                )
            estimates.append(
                AxisEstimate(
                    name,
                    "m",
                    float(loop_translation[axis]),
                    float(trans_std[axis]),
                    not masked,
                    reason,
                    "unobservable" if masked else None,
                )
            )
    else:
        lacking = sorted({s.edge.pair for s in steps if s.edge.translation is None})
        notes.append(f"translation not closed: {', '.join(lacking)} do(es) not estimate it")
    counts: dict[str, list[str]] = {}
    for step in steps:
        for shared in step.edge.shared_inputs:
            counts.setdefault(shared, []).append(step.edge.pair)
    for shared, users in sorted(counts.items()):
        if len(users) > 1:
            notes.append(
                f"{' and '.join(users)} share the {shared}, so their errors are correlated: "
                "this loop tests the solvers on top of it, not the odometry, and its std is "
                "optimistic"
            )
    judgement = judge_pair(estimates, options)
    members = [
        CheckClosureMember(
            pair=step.edge.pair,  # type: ignore[arg-type]
            parent_frame=step.edge.parent,
            child_frame=step.edge.child,
            direction="forward" if step.forward else "reverse",
            estimator=step.edge.estimator,
        )
        for step in steps
    ]
    return CheckClosure(
        kind="parallel" if count == 2 else "cycle",
        frames=list(frames),
        members=members,
        axes=list(judgement.axes),
        unchecked_axes=list(judgement.unchecked),
        verdict=judgement.verdict,
        notes=notes,
    )


def _pair_rank(pair: str) -> int:
    return CHECK_PAIR_NAMES.index(pair) if pair in CHECK_PAIR_NAMES else len(CHECK_PAIR_NAMES)


def build_closure_report(
    records: Sequence[CheckPairRecord],
    edges: Mapping[tuple[str, tuple[str, ...]], ClosureEdge],
    options: VerdictOptions,
) -> CheckClosureReport | None:
    """Closure loops over the run's estimates.

    ``edges`` maps ``(pair, frames)`` of a record to its estimate when the
    estimator produced one. Returns ``None`` when no pair ran at all.
    """

    ran = [
        record
        for record in records
        if record.status in {"pass", "warn", "fail", "inconclusive"}
        and record.pair != "camera-focal"  # intrinsics, not an extrinsic: no edge in the graph
    ]
    if not ran:
        return None
    listing: list[CheckClosureEdge] = []
    usable: list[ClosureEdge] = []
    for record in sorted(ran, key=lambda item: (_pair_rank(item.pair), item.frames)):
        edge = edges.get((record.pair, tuple(record.frames)))
        transform = record.compared_transform
        parent = transform.parent_frame if transform else None
        child = transform.child_frame if transform else None
        if record.pair in DERIVED_PAIRS:
            listing.append(
                CheckClosureEdge(
                    pair=record.pair,
                    parent_frame=parent,
                    child_frame=child,
                    role="derived",
                    reason=DERIVED_PAIRS[record.pair],
                )
            )
            continue
        reason: str | None = None
        if edge is None or record.status == "inconclusive":
            reason = record.reason or "no usable estimate"
        elif edge.parent == edge.child:
            reason = "its frames coincide"
        elif not any(edge.rot_observed) and not any(edge.trans_observed):
            reason = "no axis was observed"
        if reason is not None or edge is None:
            listing.append(
                CheckClosureEdge(
                    pair=record.pair,
                    parent_frame=parent,
                    child_frame=child,
                    role="unusable",
                    reason=reason,
                )
            )
            continue
        usable.append(edge)
        listing.append(
            CheckClosureEdge(
                pair=record.pair, parent_frame=parent, child_frame=child, role="independent"
            )
        )
    loops: list[CheckClosure] = []
    spanning = _spanning_edges(usable)
    for index, in_forest in enumerate(spanning):
        if in_forest:
            continue
        found = _loop_steps(usable, index, spanning)
        if found is None:
            continue
        steps, frames = found
        loops.append(evaluate_loop(steps, frames, options))
    return CheckClosureReport(
        assumptions=list(ASSUMPTIONS),
        edges=listing,
        loops=loops,
        verdict=worst_verdict([loop.verdict for loop in loops]),
    )


def closure_floor_verdict(report: CheckClosureReport | None) -> str | None:
    """The verdict a closure contributes to the overall verdict: ``warn`` if a loop fails."""

    if report is not None and any(loop.verdict == "fail" for loop in report.loops):
        return "warn"
    return None


__all__ = [
    "ClosureEdge",
    "build_closure_report",
    "closure_floor_verdict",
    "edge_from_estimates",
    "evaluate_loop",
]
