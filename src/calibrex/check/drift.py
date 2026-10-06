"""``calibrex drift``: did a rig's calibration change between recordings of it?

Every bag is estimated with :func:`calibrex.check.estimate.build_bag_estimate`
(its per-bag ``slac.bag_estimate`` artifact is written to ``DIR/<bag-name>/``;
the estimator cache makes a re-run cheap) and the axes that the data observed in
at least two bags are tested for consistency. The statistics are pure functions
(:func:`assess_axis`, :func:`compare_pair`) over ``(bag, value, std)``.

Rule, per axis with observations ``x_i`` and standard deviations ``s_i`` in
``n >= 2`` bags (``k = sigma_k``, ``floor`` the minimum detectable change of the
axis type: the rotation / rigid-scan rotation / translation floors of
``calibrex check``):

* pairwise: ``d_ij = x_i - x_j``, ``sd_ij = sqrt(s_i^2 + s_j^2)``,
  ``tol_ij = max(k * sd_ij, floor)``; the pair *exceeds* when ``|d_ij| > tol_ij``;
* homogeneity: ``Q = sum((x_i - m)^2 / s_i^2)`` with ``m`` the inverse-variance
  weighted mean, compared with chi-square on ``n - 1`` degrees of freedom;
* the axis is ``drift`` when a pair exceeds its tolerance **and** ``p(Q) <
  alpha``; ``stable`` otherwise; ``inconclusive`` with fewer than two bags that
  observed it;
* with ``n >= 3`` the deviating bags are found by removing, one at a time, the bag
  whose removal leaves the most homogeneous rest, until the rest passes the
  criterion above; fewer than half the bags may be removed, otherwise the bags
  split and nobody is blamed. Each bag's ``delta`` is against the weighted mean
  of the bags that were not removed. With ``n = 2`` a difference cannot be
  attributed to a bag.

Pair verdict: ``drift`` if any axis drifts, else ``stable`` if any axis was
tested, else ``inconclusive``; overall: the worst of the pairs
(``stable < inconclusive < drift``, as ``check`` ranks ``pass < inconclusive``).
Axes are compared as the estimators report them: rotation vector components in
degrees about the parent frame's axes, so the small-angle difference is used.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import chi2 as chi2_distribution

from calibrex import __version__
from calibrex.check.estimate import ARTIFACT_FILENAME, build_bag_estimate
from calibrex.check.progress import CheckProgress, as_progress
from calibrex.check.runner import CheckRunOptions
from calibrex.check.verdict import RIGID_SCAN_ROTATION_FLOOR_DEG
from calibrex.core.bag_estimate import BagEstimateArtifact, EstimateAxis, EstimatePairRecord
from calibrex.core.calibration_check import CheckAxisName, CheckProvenance, OdometryKind
from calibrex.core.calibration_drift import (
    CalibrationDriftArtifact,
    DriftAxisRecord,
    DriftBagChange,
    DriftBagRef,
    DriftLeaveOneOut,
    DriftObservation,
    DriftPairRecord,
    DriftPairwise,
    DriftSummary,
    DriftThresholds,
    DriftVerdict,
    FloorSource,
)
from calibrex.core.io import write_mapping
from calibrex.core.provenance import git_commit, sha256_path

DRIFT_VERDICT_ORDER: tuple[str, ...] = ("stable", "inconclusive", "drift")
ARTIFACT_NAME = "calibration_drift.json"
ROTATION_AXES: tuple[str, ...] = ("roll", "pitch", "yaw")
_MIN_STD = 1e-9


@dataclass(frozen=True)
class DriftOptions:
    """Thresholds of the consistency rule."""

    sigma_k: float = 3.0
    rotation_floor_deg: float = 0.5
    rigid_scan_rotation_floor_deg: float = RIGID_SCAN_ROTATION_FLOOR_DEG
    translation_floor_m: float = 0.02
    chi2_alpha: float = 0.01

    def __post_init__(self) -> None:
        positive = (
            self.sigma_k,
            self.rotation_floor_deg,
            self.rigid_scan_rotation_floor_deg,
            self.translation_floor_m,
        )
        if min(positive) <= 0.0 or not 0.0 < self.chi2_alpha < 1.0:
            raise ValueError(
                "sigma_k and the floors must be positive and chi2_alpha must be in (0, 1)"
            )

    def thresholds(self) -> DriftThresholds:
        return DriftThresholds(
            sigma_k=self.sigma_k,
            rotation_floor_deg=self.rotation_floor_deg,
            rigid_scan_rotation_floor_deg=self.rigid_scan_rotation_floor_deg,
            translation_floor_m=self.translation_floor_m,
            chi2_alpha=self.chi2_alpha,
        )


@dataclass(frozen=True)
class Observation:
    """One bag's estimate of one axis (input of :func:`assess_axis`)."""

    bag: str
    value: float | None
    std: float | None
    status: str
    """The estimate's axis status, or ``pair_not_estimated``."""

    @property
    def used(self) -> bool:
        return (
            self.status == "observed"
            and self.value is not None
            and self.std is not None
            and math.isfinite(self.value)
            and math.isfinite(self.std)
        )


def _weighted_mean(values: Sequence[float], stds: Sequence[float]) -> tuple[float, float]:
    weights = [1.0 / max(s, _MIN_STD) ** 2 for s in stds]
    total = sum(weights)
    mean = sum(w * v for w, v in zip(weights, values, strict=True)) / total
    return mean, math.sqrt(1.0 / total)


def _chi_square(values: Sequence[float], stds: Sequence[float]) -> float:
    mean, _ = _weighted_mean(values, stds)
    return sum(((v - mean) / s) ** 2 for v, s in zip(values, stds, strict=True))


def _inconsistent(
    values: Sequence[float], stds: Sequence[float], options: DriftOptions, floor: float
) -> bool:
    """The drift criterion: a pair beyond its tolerance and chi-square rejecting homogeneity."""

    if len(values) < 2:
        return False
    exceeding = any(
        abs(values[i] - values[j]) > max(options.sigma_k * math.hypot(stds[i], stds[j]), floor)
        for i, j in itertools.combinations(range(len(values)), 2)
    )
    p_value = float(chi2_distribution.sf(_chi_square(values, stds), len(values) - 1))
    return exceeding and p_value < options.chi2_alpha


def assess_axis(
    name: CheckAxisName,
    unit: Literal["deg", "m"],
    observations: Sequence[Observation],
    options: DriftOptions,
    *,
    rigid_scan: bool = False,
    deviating: Sequence[str] | None = None,
) -> DriftAxisRecord:
    """Test one axis for consistency across the bags that observed it.

    ``deviating`` names the bags to treat as the odd ones out (the pair-level attribution of
    :func:`compare_pair`); by default they are found on this axis alone.
    """

    source: FloorSource
    if unit == "m":
        source, floor = "translation", options.translation_floor_m
    elif rigid_scan:
        source, floor = "rigid_scan_rotation", options.rigid_scan_rotation_floor_deg
    else:
        source, floor = "rotation", options.rotation_floor_deg
    records = [
        DriftObservation.model_validate(
            {
                "bag": o.bag,
                "value": o.value,
                "std": o.std,
                "status": o.status,
                "used": o.used,
            }
        )
        for o in observations
    ]
    used = [o for o in observations if o.used]
    if len(used) < 2:
        return DriftAxisRecord(
            name=name,
            unit=unit,
            observations=records,
            bags_observed=len(used),
            floor=floor,
            floor_source=source,
            status="inconclusive",
            reason=f"observed in {len(used)} bag(s); at least 2 are needed to compare",
        )
    values = [float(o.value) for o in used if o.value is not None]
    stds = [max(float(o.std), _MIN_STD) for o in used if o.std is not None]
    k = options.sigma_k
    mean, mean_std = _weighted_mean(values, stds)
    q = sum(((v - mean) / s) ** 2 for v, s in zip(values, stds, strict=True))
    dof = len(used) - 1
    p_value = float(chi2_distribution.sf(q, dof))
    pairwise: list[DriftPairwise] = []
    for (ia, a), (ib, b) in itertools.combinations(enumerate(used), 2):
        sd = math.hypot(stds[ia], stds[ib])
        tolerance = max(k * sd, floor)
        difference = values[ia] - values[ib]
        pairwise.append(
            DriftPairwise(
                bag_a=a.bag,
                bag_b=b.bag,
                difference=difference,
                std=sd,
                z=difference / sd,
                tolerance=tolerance,
                ratio=abs(difference) / tolerance,
                exceeds=abs(difference) > tolerance,
            )
        )
    # Which bags disagree with the rest: remove, one at a time, the bag whose removal leaves
    # the most homogeneous set, while the rest is still inconsistent and fewer than half the
    # bags are removed. If that does not restore consistency the bags split and nobody is blamed.
    leave_one_out: list[DriftLeaveOneOut] = []
    removed: list[int] = []
    if deviating is not None:
        removed = [i for i, o in enumerate(used) if o.bag in deviating]
    elif len(used) >= 3 and _inconsistent(values, stds, options, floor):
        keep = list(range(len(used)))
        while len(removed) < (len(used) - 1) // 2 and _inconsistent(
            [values[j] for j in keep], [stds[j] for j in keep], options, floor
        ):
            best = min(
                keep,
                key=lambda i: _chi_square(
                    [values[j] for j in keep if j != i], [stds[j] for j in keep if j != i]
                ),
            )
            keep.remove(best)
            removed.append(best)
        if _inconsistent([values[j] for j in keep], [stds[j] for j in keep], options, floor):
            removed = []
    if len(used) >= 3:
        consensus = [j for j in range(len(used)) if j not in removed]
        for i, o in enumerate(used):
            rest = [j for j in consensus if j != i]
            m_rest, sd_rest = _weighted_mean([values[j] for j in rest], [stds[j] for j in rest])
            sd = math.hypot(stds[i], sd_rest)
            delta = values[i] - m_rest
            leave_one_out.append(
                DriftLeaveOneOut(
                    bag=o.bag,
                    delta=delta,
                    std=sd,
                    z=delta / sd,
                    tolerance=max(k * sd, floor),
                    deviates=i in removed,
                )
            )
    exceeding = any(p.exceeds for p in pairwise)
    drifting = exceeding and p_value < options.chi2_alpha
    reason = None
    if exceeding and not drifting:
        reason = (
            "a pair exceeds its tolerance but the chi-square test does not reject homogeneity "
            f"(p = {p_value:.3g} >= {options.chi2_alpha:g})"
        )
    return DriftAxisRecord(
        name=name,
        unit=unit,
        observations=records,
        bags_observed=len(used),
        floor=floor,
        floor_source=source,
        status="drift" if drifting else "stable",
        reason=reason,
        weighted_mean=mean,
        weighted_mean_std=mean_std,
        chi2=q,
        dof=dof,
        p_value=p_value,
        minimum_detectable_change=min(p.tolerance for p in pairwise),
        max_abs_difference=max(abs(p.difference) for p in pairwise),
        pairwise=pairwise,
        leave_one_out=leave_one_out,
    )


def _rotation_delta_deg(bag_deg: Sequence[float], reference_deg: Sequence[float]) -> float:
    """Angle (deg) of the rotation taking the reference rotation to the bag's."""

    rotation = Rotation.from_rotvec(np.deg2rad(np.asarray(bag_deg, dtype=float)))
    reference = Rotation.from_rotvec(np.deg2rad(np.asarray(reference_deg, dtype=float)))
    return float(np.rad2deg((rotation * reference.inv()).magnitude()))


def _changes(
    axes: Sequence[DriftAxisRecord], deviating: Sequence[str], bags: Sequence[str]
) -> list[DriftBagChange]:
    """Per deviating bag: its offset from the rest, and the angle on the rotation when complete."""

    ambiguous = not deviating
    targets = list(deviating)
    if ambiguous:
        # two bags differ and neither can be blamed: report the later one against the earlier
        seen = [b for b in bags if any(o.bag == b and o.used for a in axes for o in a.observations)]
        targets = seen[1:2]
    result: list[DriftBagChange] = []
    for bag in targets:
        axes_delta: dict[str, float] = {}
        absolute: dict[str, tuple[float, float]] = {}
        for axis in axes:
            used = [o for o in axis.observations if o.used and o.value is not None]
            if axis.bags_observed < 2 or bag not in {o.bag for o in used}:
                continue
            value = next(float(o.value or 0.0) for o in used if o.bag == bag)
            others = [o for o in used if o.bag != bag and o.bag not in deviating] or [
                o for o in used if o.bag != bag
            ]
            if ambiguous:
                others = others[:1]
            mean, _ = _weighted_mean(
                [float(o.value or 0.0) for o in others],
                [float(o.std or _MIN_STD) for o in others],
            )
            axes_delta[axis.name] = value - mean
            absolute[axis.name] = (value, mean)
        rotation = None
        if all(n in absolute for n in ROTATION_AXES):
            rotation = _rotation_delta_deg(
                [absolute[n][0] for n in ROTATION_AXES], [absolute[n][1] for n in ROTATION_AXES]
            )
        result.append(
            DriftBagChange(
                bag=bag,
                reference="other_bag" if ambiguous else "mean_of_others",
                rotation_delta_deg=rotation,
                axes_delta=axes_delta,
            )
        )
    return result


def _pair_deviating(records: Sequence[DriftAxisRecord], options: DriftOptions) -> list[str]:
    """The bags whose removal makes every tested axis of the pair consistent, or ``[]``.

    Greedy: remove the bag whose removal leaves the smallest total chi-square over the axes,
    while some axis is still inconsistent and fewer than half the bags are removed. If the
    rest is still inconsistent the bags split and nobody is blamed.
    """

    tested = [a for a in records if a.status != "inconclusive"]
    bags = list(dict.fromkeys(o.bag for a in tested for o in a.observations if o.used))

    def series(axis: DriftAxisRecord, keep: Sequence[str]) -> tuple[list[float], list[float]]:
        used = [o for o in axis.observations if o.used and o.bag in keep]
        return (
            [float(o.value or 0.0) for o in used],
            [max(float(o.std or 0.0), _MIN_STD) for o in used],
        )

    def inconsistent(keep: Sequence[str]) -> bool:
        return any(_inconsistent(*series(a, keep), options, a.floor) for a in tested)

    def score(keep: Sequence[str]) -> float:
        total = 0.0
        for axis in tested:
            values, stds = series(axis, keep)
            if len(values) >= 2:
                total += _chi_square(values, stds)
        return total

    if len(bags) < 3 or not inconsistent(bags):
        return []
    keep = list(bags)
    removed: list[str] = []
    while len(removed) < (len(bags) - 1) // 2 and inconsistent(keep):
        best = min(keep, key=lambda b: score([x for x in keep if x != b]))
        keep.remove(best)
        removed.append(best)
    return [] if inconsistent(keep) else removed


def compare_pair(
    pair: str,
    parent_frame: str | None,
    child_frame: str | None,
    axes: Sequence[tuple[CheckAxisName, Literal["deg", "m"], Sequence[Observation]]],
    bags: Sequence[str],
    options: DriftOptions,
    *,
    rigid_scan: bool = False,
) -> DriftPairRecord:
    """Test every axis of one pair and derive the pair verdict and the deviating bags."""

    records = [
        assess_axis(name, unit, observations, options, rigid_scan=rigid_scan)
        for name, unit, observations in axes
    ]
    drifting = [a for a in records if a.status == "drift"]
    tested = [a for a in records if a.status != "inconclusive"]
    verdict: DriftVerdict = "drift" if drifting else "stable" if tested else "inconclusive"
    deviating: list[str] = []
    if drifting:
        # attribute on all the pair's axes together: one bag moved, not one axis' worth of bags
        deviating = _pair_deviating(records, options)
        records = [
            assess_axis(
                name, unit, observations, options, rigid_scan=rigid_scan, deviating=deviating
            )
            for name, unit, observations in axes
        ]
    changes: list[DriftBagChange] = []
    attribution: Literal["bag", "ambiguous"] | None = None
    if verdict == "drift":
        attribution = "bag" if deviating else "ambiguous"
        changes = _changes(records, deviating, bags)
    return DriftPairRecord.model_validate(
        {
            "pair": pair,
            "parent_frame": parent_frame,
            "child_frame": child_frame,
            "verdict": verdict,
            "reason": "no axis was observed in at least two bags"
            if verdict == "inconclusive"
            else None,
            "bags_compared": [
                b
                for b in bags
                if any(o.bag == b and o.used for a in records for o in a.observations)
            ],
            "axes": records,
            "deviating_bags": deviating,
            "attribution": attribution,
            "changes": changes,
        }
    )


def worst_drift_verdict(verdicts: Sequence[str]) -> DriftVerdict:
    """The worst of ``drift > inconclusive > stable``; ``inconclusive`` for an empty list."""

    ranked = [v for v in verdicts if v in DRIFT_VERDICT_ORDER]
    if not ranked:
        return "inconclusive"
    worst = max(ranked, key=DRIFT_VERDICT_ORDER.index)
    return "drift" if worst == "drift" else "inconclusive" if worst == "inconclusive" else "stable"


# ------------------------------------------------------------------- pair grouping


def _group_key(record: EstimatePairRecord) -> tuple[str, tuple[str, ...]]:
    """A pair is the same pair across bags when its name and its two frames agree."""

    return (record.pair, tuple(record.frames))


_AXIS_ORDER = {n: i for i, n in enumerate(("roll", "pitch", "yaw", "x", "y", "z"))}


def compare_estimates(
    estimates: Sequence[tuple[str, BagEstimateArtifact]], options: DriftOptions
) -> list[DriftPairRecord]:
    """Compare the pairs of per-bag estimates (``(bag name, estimate)``, in recording order)."""

    names = [name for name, _ in estimates]
    groups: dict[tuple[str, tuple[str, ...]], dict[str, EstimatePairRecord]] = {}
    for name, estimate in estimates:
        for record in estimate.pairs:
            if record.axes:
                groups.setdefault(_group_key(record), {}).setdefault(name, record)
    pairs: list[DriftPairRecord] = []
    for (pair, _frames), per_bag in sorted(groups.items(), key=lambda item: item[0]):
        transforms = [r.transform for r in per_bag.values() if r.transform is not None]
        parent = transforms[0].parent_frame if transforms else None
        child = transforms[0].child_frame if transforms else None
        axis_names: list[tuple[CheckAxisName, Literal["deg", "m"]]] = []
        for record in per_bag.values():
            for axis in record.axes:
                if axis.status != "not_estimated" and (axis.name, axis.unit) not in axis_names:
                    axis_names.append((axis.name, axis.unit))
        axis_names.sort(key=lambda item: _AXIS_ORDER.get(item[0], 99))
        axes: list[tuple[CheckAxisName, Literal["deg", "m"], list[Observation]]] = []
        for axis_name, unit in axis_names:
            observations: list[Observation] = []
            for name in names:
                bag_record = per_bag.get(name)
                found: EstimateAxis | None = None
                if bag_record is not None:
                    found = next((a for a in bag_record.axes if a.name == axis_name), None)
                if found is None:
                    observations.append(Observation(name, None, None, "pair_not_estimated"))
                else:
                    observations.append(Observation(name, found.value, found.std, found.status))
            axes.append((axis_name, unit, observations))
        rigid = any(r.deskew == "none" for r in per_bag.values())
        pairs.append(compare_pair(pair, parent, child, axes, names, options, rigid_scan=rigid))
    return pairs


# ------------------------------------------------------------------------- next steps


def next_steps(pairs: Sequence[DriftPairRecord]) -> list[str]:
    """Actionable text for the drifting and the inconclusive pairs."""

    steps: list[str] = []
    for pair in pairs:
        if pair.verdict == "drift" and pair.attribution == "bag":
            who = ", ".join(pair.deviating_bags)
            steps.append(
                f"{pair.pair}: {who} disagrees with the other recording(s); recalibrate it "
                f"(calibrex estimate {who} --output DIR) or check the mounting, and confirm "
                f"with calibrex check {who} --tf <the calibration in use>"
            )
        elif pair.verdict == "drift":
            steps.append(
                f"{pair.pair}: the recordings disagree but with two bags the one that moved "
                "cannot be told; add a third recording of the rig (calibrex drift A B C ...) "
                "or run calibrex check on each with the calibration in use"
            )
        elif pair.verdict == "inconclusive":
            steps.append(
                f"{pair.pair}: no axis was observed in two bags; use longer or more exciting "
                "recordings (raise --max-duration-s, rotate and translate the rig) or leave it "
                "out with --pairs"
            )
    if not steps and pairs:
        floors = [
            a.minimum_detectable_change
            for p in pairs
            for a in p.axes
            if a.minimum_detectable_change is not None and a.unit == "deg"
        ]
        extra = (
            f" (rotation changes below {min(floors):.2g} deg cannot be flagged)" if floors else ""
        )
        steps.append(
            "stable means no change larger than each axis' minimum detectable change was "
            f"found{extra}; axes the data did not observe in two bags are not covered"
        )
    return steps


# ------------------------------------------------------------------------------ main


def _unique_names(paths: Sequence[Path]) -> list[str]:
    names: list[str] = []
    for path in paths:
        base = path.name or path.resolve().name
        name, n = base, 2
        while name in names:
            name, n = f"{base}_{n}", n + 1
        names.append(name)
    return names


def build_calibration_drift(
    bags: Sequence[str | Path],
    *,
    output_dir: Path,
    run: CheckRunOptions,
    options: DriftOptions | None = None,
    tf_files: Sequence[str | Path] = (),
    vehicle_frame: str | None = None,
    frame_overrides: Mapping[str, str] | None = None,
    topic_kinds: Mapping[str, OdometryKind] | None = None,
    command: Sequence[str] | None = None,
    progress: Callable[[str], None] | CheckProgress | None = None,
) -> CalibrationDriftArtifact:
    """Estimate every bag (cached) and test the pairs' axes for consistency across them."""

    if len(bags) < 2:
        raise ValueError("calibrex drift needs at least two bags")
    options = options or DriftOptions()
    report = as_progress(progress)
    paths = [Path(b) for b in bags]
    names = _unique_names(paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    estimates: list[tuple[str, BagEstimateArtifact]] = []
    refs: list[DriftBagRef] = []
    for index, (path, name) in enumerate(zip(paths, names, strict=True), start=1):
        report(f"drift: bag {index}/{len(paths)}: {name}")
        bag_dir = output_dir / name
        started = time.monotonic()
        bag_run = dataclasses.replace(run, evidence_dir=bag_dir / "evidence", base_dir=bag_dir)
        estimate = build_bag_estimate(
            path,
            output_dir=bag_dir,
            run=bag_run,
            tf_files=tf_files,
            vehicle_frame=vehicle_frame,
            frame_overrides=frame_overrides,
            topic_kinds=topic_kinds,
            command=command,
            progress=report,
        )
        runtime = time.monotonic() - started
        estimate_path = bag_dir / ARTIFACT_FILENAME
        write_mapping(estimate_path, estimate.model_dump(mode="json", exclude_none=True))
        sha = sha256_path(estimate_path)
        if sha is None:
            raise OSError(f"could not read back {estimate_path}")
        estimates.append((name, estimate))
        ran = [p for p in estimate.pairs if p.evidence_from_cache is not None]
        refs.append(
            DriftBagRef(
                name=name,
                path=str(path),
                input_sha256=estimate.bag.input_sha256,
                input_digest_scope=estimate.bag.input_digest_scope,
                estimate_path=estimate_path.relative_to(output_dir).as_posix(),
                estimate_sha256=sha,
                pair_statuses={p.pair: p.status for p in estimate.pairs if p.status != "skipped"},
                runtime_s=runtime,
                estimator_cache_hits=sum(1 for p in ran if p.evidence_from_cache),
                estimator_runs=sum(1 for p in ran if not p.evidence_from_cache),
            )
        )
    pairs = compare_estimates(estimates, options)
    counts: dict[str, int] = {}
    for pair in pairs:
        counts[pair.verdict] = counts.get(pair.verdict, 0) + 1
    tested = [a for p in pairs for a in p.axes if a.status != "inconclusive"]
    digest = hashlib.sha256(
        "\n".join(sorted(ref.input_sha256 for ref in refs)).encode("utf-8")
    ).hexdigest()
    notes = [
        "each bag is estimated independently (calibrex estimate); the per-bag artifacts are "
        "under the bag-named directories and are referenced by digest",
        "bag order is the order given; with two bags a difference cannot be attributed to one",
        "the standard deviations are the estimators' own and are not guaranteed calibrated; "
        "the floors set the smallest change that can be flagged",
    ]
    if run.max_duration_s is not None:
        notes.append(f"only the first {run.max_duration_s:g} s of each bag's sensor streams")
    return CalibrationDriftArtifact(
        bags=refs,
        vehicle_frame=estimates[0][1].vehicle_frame,
        pairs=pairs,
        overall_verdict=worst_drift_verdict([p.verdict for p in pairs]),
        summary=DriftSummary(
            bag_count=len(refs),
            pair_count=len(pairs),
            verdict_counts=dict(sorted(counts.items())),
            axes_tested=len(tested),
            axes_drifting=sum(1 for a in tested if a.status == "drift"),
        ),
        thresholds=options.thresholds(),
        options=estimates[0][1].options,
        next_steps=next_steps(pairs),
        provenance=CheckProvenance(
            generator="calibrex drift",
            generator_version=__version__,
            git_commit=git_commit(),
            command=list(command or []),
            input_sha256=digest,
            input_digest_scope="sha256 of the sorted input digests of every bag "
            "(each bag's digest scope is in bags[].input_digest_scope)",
            created_at=datetime.now(timezone.utc).isoformat(),
            notes=notes,
        ),
    )


# ------------------------------------------------------------------------------ text


def _cell(value: float, std: float, unit: str) -> str:
    if unit == "deg":
        return f"{value:+.3f} +- {std:.3f}"
    return f"{value * 100:+.2f} +- {std * 100:.2f}"


def format_drift_text(artifact: CalibrationDriftArtifact, output_dir: Path) -> str:
    """Per pair a table of the axes across bags, the verdict and the next steps."""

    bag_names = [b.name for b in artifact.bags]
    width = max(16, *(len(n) for n in bag_names))
    lines = [
        "calibrex drift",
        f"bags ({len(artifact.bags)}): "
        + ", ".join(f"{b.name} [{b.runtime_s:.0f}s]" for b in artifact.bags),
        f"overall: {artifact.overall_verdict}  (a change is flagged beyond "
        f"max({artifact.thresholds.sigma_k:g} sigma, floor) with chi-square "
        f"p < {artifact.thresholds.chi2_alpha:g}; rotation in deg, translation in cm)",
    ]
    for pair in artifact.pairs:
        frames = f"  T_{pair.parent_frame}_{pair.child_frame}" if pair.parent_frame else ""
        lines += ["", f"{pair.pair}: {pair.verdict}{frames}"]
        if pair.reason:
            lines.append(f"  {pair.reason}")
        header = "  axis  " + "  ".join(n.rjust(width) for n in bag_names)
        lines.append(header + "   max|diff|  min detectable  status")
        for axis in pair.axes:
            scale = 1.0 if axis.unit == "deg" else 100.0
            cells = []
            for name in bag_names:
                obs = next((o for o in axis.observations if o.bag == name), None)
                if obs is not None and obs.used and obs.value is not None and obs.std is not None:
                    cells.append(_cell(obs.value, obs.std, axis.unit).rjust(width))
                else:
                    cells.append("-".rjust(width))
            diff = (
                "-" if axis.max_abs_difference is None else f"{axis.max_abs_difference * scale:.3f}"
            )
            floor = (
                "-"
                if axis.minimum_detectable_change is None
                else f"{axis.minimum_detectable_change * scale:.3f}"
            )
            lines.append(
                f"  {axis.name:<5} {'  '.join(cells)}   {diff:>8}  {floor:>14}  {axis.status}"
            )
        if pair.verdict == "drift":
            for change in pair.changes:
                angle = (
                    f"; rotation {change.rotation_delta_deg:.2f} deg"
                    if change.rotation_delta_deg is not None
                    else ""
                )
                ref = "the other bag" if change.reference == "other_bag" else "the others"
                deltas = ", ".join(
                    f"{k} {v:+.3f} deg" if k in ROTATION_AXES else f"{k} {v * 100:+.2f} cm"
                    for k, v in change.axes_delta.items()
                )
                lines.append(f"  {change.bag} vs {ref}: {deltas}{angle}")
            if pair.attribution == "ambiguous":
                lines.append("  deviating bag: cannot be told (two bags, or the bags split)")
            else:
                lines.append(f"  deviating bag(s): {', '.join(pair.deviating_bags)}")
    if not artifact.pairs:
        lines += ["", "no pair could be estimated in any bag"]
    if artifact.next_steps:
        lines += ["", "next steps:", *(f"  - {s}" for s in artifact.next_steps)]
    lines.append(f"per-bag estimates: {output_dir}/<bag>/{ARTIFACT_FILENAME}")
    return "\n".join(lines)
