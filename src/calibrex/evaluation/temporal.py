"""Temporal (time-offset) evidence for motion-compensated online calibration.

Sign convention
---------------
Positive ``delta_t_s`` means target capture timestamps are evaluated as if they
**lag** the odometry clock: poses use ``T_world_source(t_capture + delta_t_s)``.
Injecting a positive stamp bias on target messages therefore requires a
negative estimated offset to correct it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from calibrex.core.geometry import SE3, Vector3
from calibrex.core.time import apply_time_offset_ns
from calibrex.data.livox import LivoxPointRecord
from calibrex.data.odometry_track import OdometryTrack
from calibrex.evaluation.holdout import split_indices
from calibrex.evaluation.lidar import build_rig_point_to_plane_observations
from calibrex.graph.lidar_point_to_plane import LidarRigPointToPlaneFactor

TimeOffsetAnchor = Literal["final", "initial"]
TemporalSeparabilityVerdict = Literal["separable", "degenerate", "consistent"]

TemporalGateStatus = Literal["pass", "fail", "inconclusive"]
_SEARCH_RESOLUTION_S = 0.001
_GOLDEN_RATIO = 0.6180339887498949


@dataclass(frozen=True)
class TemporalEvidenceOptions:
    """Factor options controlling post-run temporal evidence."""

    probe_offsets_s: tuple[float, ...] = ()
    probe_min_rmse_increase: float = 0.10
    probe_min_detection_ratio: float = 1.0
    estimate_time_offset: bool = False
    search_bound_s: float = 0.20
    max_abs_time_offset_s: float = 0.05
    min_significant_relative_improvement: float = 0.10
    time_offset_anchor: TimeOffsetAnchor = "final"
    inject_time_offset_s: float = 0.0


@dataclass(frozen=True)
class TimeOffsetProbeResult:
    """One perturbation-probe row."""

    offset_s: float
    baseline_rmse_m: float
    shifted_rmse_m: float
    relative_increase: float
    detected: bool


@dataclass(frozen=True)
class TimeOffsetEstimateResult:
    """1D holdout-RMSE search outcome."""

    estimated_offset_s: float
    rmse_at_estimate_m: float
    rmse_at_zero_s: float
    relative_improvement: float
    curve_sample_count: int
    curve_flatness: float


@dataclass(frozen=True)
class TemporalEvidenceResult:
    """Aggregated temporal evidence block for provenance and metrics."""

    sign_convention: str
    options: TemporalEvidenceOptions
    baseline_holdout_rmse_m: float
    holdout_point_count: int
    probes: tuple[TimeOffsetProbeResult, ...]
    probe_detection_ratio: float | None
    probe_gate_status: TemporalGateStatus | None
    probe_gate_reason: str | None
    estimate: TimeOffsetEstimateResult | None
    estimate_gate_status: TemporalGateStatus | None
    estimate_gate_reason: str | None
    verdict: TemporalGateStatus | None
    verdict_reason: str | None


@dataclass(frozen=True)
class TemporalSeparabilityResult:
    """Joint observability verdict comparing anchored vs adapted holdout-RMSE curves."""

    verdict: TemporalSeparabilityVerdict
    reason: str
    anchored_curve_flatness: float
    adapted_curve_flatness: float
    flatness_margin: float


@dataclass(frozen=True)
class DualTemporalEvidenceResult:
    """Temporal evidence evaluated at both the adapted and anchored extrinsics."""

    time_offset_anchor: TimeOffsetAnchor
    anchored_transform: SE3
    time_offset_injected_s: float
    selected: TemporalEvidenceResult
    adapted: TemporalEvidenceResult
    anchored: TemporalEvidenceResult
    separability: TemporalSeparabilityResult | None


def temporal_evidence_options_from_factor(options: dict[str, Any]) -> TemporalEvidenceOptions:
    """Parse temporal-evidence options from ``lidar_rig_point_to_plane`` factor options."""

    raw_probes = options.get("time_offset_probe_s", [])
    probe_offsets: tuple[float, ...] = ()
    if isinstance(raw_probes, list):
        probe_offsets = tuple(float(value) for value in raw_probes)

    anchor_raw = options.get("time_offset_anchor", "final")
    anchor: TimeOffsetAnchor = "initial" if anchor_raw == "initial" else "final"

    return TemporalEvidenceOptions(
        probe_offsets_s=probe_offsets,
        probe_min_rmse_increase=_float_option(
            options, "time_offset_probe_min_rmse_increase", default=0.10, minimum=0.0
        ),
        probe_min_detection_ratio=_float_option(
            options,
            "online_gate_min_time_offset_probe_detection_ratio",
            default=1.0,
            minimum=0.0,
        ),
        estimate_time_offset=bool(options.get("estimate_time_offset", False)),
        search_bound_s=_float_option(
            options, "time_offset_search_bound_s", default=0.20, minimum=0.01
        ),
        max_abs_time_offset_s=_float_option(
            options, "online_gate_max_abs_time_offset_s", default=0.05, minimum=0.0
        ),
        min_significant_relative_improvement=_float_option(
            options, "time_offset_probe_min_rmse_increase", default=0.10, minimum=0.0
        ),
        time_offset_anchor=anchor,
        inject_time_offset_s=_float_option(
            options, "inject_time_offset_s", default=0.0, minimum=0.0
        ),
    )


def temporal_evidence_requested(options: TemporalEvidenceOptions) -> bool:
    """Return whether any temporal-evidence machinery should run."""

    return bool(options.probe_offsets_s) or options.estimate_time_offset


def evaluate_temporal_evidence(
    *,
    source_records: list[LivoxPointRecord],
    target_points: list[Vector3],
    capture_timestamps_ns: list[int],
    final_t_source_target: SE3,
    odometry_track: OdometryTrack,
    t_base_source: SE3,
    variable: str,
    sensor: str,
    voxel_size_m: float,
    correspondence_gate_m: float,
    holdout_ratio: float,
    seed: int,
    options: TemporalEvidenceOptions,
    max_odometry_extrapolation_s: float | None = None,
) -> TemporalEvidenceResult:
    """Run probe and/or estimator temporal evidence on a motion-compensated replay."""

    if len(target_points) != len(capture_timestamps_ns):
        msg = "capture_timestamps_ns length must match target_points"
        raise ValueError(msg)

    _train_indices, holdout_indices = split_indices(len(target_points), holdout_ratio, seed=seed)
    holdout_points = [target_points[index] for index in holdout_indices]
    holdout_capture_ns = [capture_timestamps_ns[index] for index in holdout_indices]

    baseline_rmse = _holdout_rmse_at_offset(
        source_records=source_records,
        holdout_points=holdout_points,
        holdout_capture_ns=holdout_capture_ns,
        t_source_target=final_t_source_target,
        odometry_track=odometry_track,
        t_base_source=t_base_source,
        variable=variable,
        sensor=sensor,
        voxel_size_m=voxel_size_m,
        correspondence_gate_m=correspondence_gate_m,
        delta_t_s=0.0,
        max_odometry_extrapolation_s=max_odometry_extrapolation_s,
    )
    if baseline_rmse is None:
        return _empty_inconclusive(
            options=options,
            reason="too few holdout correspondences for temporal evidence",
            holdout_point_count=len(holdout_points),
        )

    probes = _run_probes(
        options=options,
        baseline_rmse_m=baseline_rmse,
        source_records=source_records,
        holdout_points=holdout_points,
        holdout_capture_ns=holdout_capture_ns,
        t_source_target=final_t_source_target,
        odometry_track=odometry_track,
        t_base_source=t_base_source,
        variable=variable,
        sensor=sensor,
        voxel_size_m=voxel_size_m,
        correspondence_gate_m=correspondence_gate_m,
        max_odometry_extrapolation_s=max_odometry_extrapolation_s,
    )
    probe_gate_status, probe_gate_reason, probe_detection_ratio = _evaluate_probe_gate(
        probes, options
    )

    estimate_result: TimeOffsetEstimateResult | None = None
    estimate_gate_status: TemporalGateStatus | None = None
    estimate_gate_reason: str | None = None
    if options.estimate_time_offset:
        estimate_result = _estimate_time_offset(
            source_records=source_records,
            holdout_points=holdout_points,
            holdout_capture_ns=holdout_capture_ns,
            t_source_target=final_t_source_target,
            odometry_track=odometry_track,
            t_base_source=t_base_source,
            variable=variable,
            sensor=sensor,
            voxel_size_m=voxel_size_m,
            correspondence_gate_m=correspondence_gate_m,
            search_bound_s=options.search_bound_s,
            rmse_at_zero_s=baseline_rmse,
            max_odometry_extrapolation_s=max_odometry_extrapolation_s,
        )
        estimate_gate_status, estimate_gate_reason = _evaluate_estimate_gate(
            estimate_result,
            options,
        )

    verdict, verdict_reason = _combine_verdicts(
        probe_gate_status=probe_gate_status if options.probe_offsets_s else None,
        probe_gate_reason=probe_gate_reason,
        estimate_gate_status=estimate_gate_status if options.estimate_time_offset else None,
        estimate_gate_reason=estimate_gate_reason,
    )

    return TemporalEvidenceResult(
        sign_convention=(
            "positive delta_t_s evaluates T_world_source(t_capture + delta_t_s); "
            "target timestamps lag the odometry clock when delta_t_s > 0"
        ),
        options=options,
        baseline_holdout_rmse_m=baseline_rmse,
        holdout_point_count=len(holdout_points),
        probes=probes,
        probe_detection_ratio=probe_detection_ratio,
        probe_gate_status=probe_gate_status if options.probe_offsets_s else None,
        probe_gate_reason=probe_gate_reason,
        estimate=estimate_result,
        estimate_gate_status=estimate_gate_status,
        estimate_gate_reason=estimate_gate_reason,
        verdict=verdict,
        verdict_reason=verdict_reason,
    )


def evaluate_dual_temporal_evidence(
    *,
    source_records: list[LivoxPointRecord],
    target_points: list[Vector3],
    capture_timestamps_ns: list[int],
    adapted_t_source_target: SE3,
    anchored_t_source_target: SE3,
    odometry_track: OdometryTrack,
    t_base_source: SE3,
    variable: str,
    sensor: str,
    voxel_size_m: float,
    correspondence_gate_m: float,
    holdout_ratio: float,
    seed: int,
    options: TemporalEvidenceOptions,
    max_odometry_extrapolation_s: float | None = None,
) -> DualTemporalEvidenceResult:
    """Evaluate temporal evidence at adapted and anchored extrinsics."""

    adapted = evaluate_temporal_evidence(
        source_records=source_records,
        target_points=target_points,
        capture_timestamps_ns=capture_timestamps_ns,
        final_t_source_target=adapted_t_source_target,
        odometry_track=odometry_track,
        t_base_source=t_base_source,
        variable=variable,
        sensor=sensor,
        voxel_size_m=voxel_size_m,
        correspondence_gate_m=correspondence_gate_m,
        holdout_ratio=holdout_ratio,
        seed=seed,
        options=options,
        max_odometry_extrapolation_s=max_odometry_extrapolation_s,
    )
    anchored = evaluate_temporal_evidence(
        source_records=source_records,
        target_points=target_points,
        capture_timestamps_ns=capture_timestamps_ns,
        final_t_source_target=anchored_t_source_target,
        odometry_track=odometry_track,
        t_base_source=t_base_source,
        variable=variable,
        sensor=sensor,
        voxel_size_m=voxel_size_m,
        correspondence_gate_m=correspondence_gate_m,
        holdout_ratio=holdout_ratio,
        seed=seed,
        options=options,
        max_odometry_extrapolation_s=max_odometry_extrapolation_s,
    )
    selected = adapted if options.time_offset_anchor == "final" else anchored
    separability: TemporalSeparabilityResult | None = None
    if options.estimate_time_offset:
        separability = derive_temporal_separability(
            anchored=anchored,
            adapted=adapted,
            margin=options.min_significant_relative_improvement,
        )
    return DualTemporalEvidenceResult(
        time_offset_anchor=options.time_offset_anchor,
        anchored_transform=anchored_t_source_target,
        time_offset_injected_s=options.inject_time_offset_s,
        selected=selected,
        adapted=adapted,
        anchored=anchored,
        separability=separability,
    )


def derive_temporal_separability(
    *,
    anchored: TemporalEvidenceResult,
    adapted: TemporalEvidenceResult,
    margin: float,
) -> TemporalSeparabilityResult:
    """Classify extrinsic/temporal joint observability from both holdout-RMSE curves."""

    anchored_flatness = _curve_flatness(anchored)
    adapted_flatness = _curve_flatness(adapted)
    anchored_localizes = _estimate_localizes(anchored, margin)
    adapted_localizes = _estimate_localizes(adapted, margin)
    adapted_absorbed = _adapted_extrinsic_absorbed(anchored, adapted, margin)

    if anchored_localizes and (not adapted_localizes or adapted_absorbed):
        verdict: TemporalSeparabilityVerdict = "separable"
        if adapted_absorbed:
            reason = (
                f"anchored curve localizes δt̂={_estimate_offset_s(anchored):.4f} s while "
                f"adapted δt̂≈{_estimate_offset_s(adapted):.4f} s after extrinsic absorption "
                f"(anchored flatness {anchored_flatness:.4f}, adapted {adapted_flatness:.4f})"
            )
        else:
            reason = (
                f"anchored curve localizes (flatness {anchored_flatness:.4f} >= {margin:.4f}) "
                f"while adapted curve is flat ({adapted_flatness:.4f} < {margin:.4f})"
            )
    elif not anchored_localizes and not adapted_localizes:
        verdict = "degenerate"
        reason = (
            f"both holdout-RMSE curves are flat (anchored {anchored_flatness:.4f}, "
            f"adapted {adapted_flatness:.4f}; margin {margin:.4f})"
        )
    else:
        verdict = "consistent"
        reason = (
            f"both curves localize a minimum (anchored flatness {anchored_flatness:.4f}, "
            f"adapted {adapted_flatness:.4f}; margin {margin:.4f})"
        )

    return TemporalSeparabilityResult(
        verdict=verdict,
        reason=reason,
        anchored_curve_flatness=anchored_flatness,
        adapted_curve_flatness=adapted_flatness,
        flatness_margin=margin,
    )


def temporal_evidence_to_provenance(result: TemporalEvidenceResult) -> dict[str, Any]:
    """Serialize a :class:`TemporalEvidenceResult` for run provenance."""

    block: dict[str, Any] = {
        "sign_convention": result.sign_convention,
        "baseline_holdout_rmse_m": result.baseline_holdout_rmse_m,
        "holdout_point_count": result.holdout_point_count,
        "time_offset_probe_s": list(result.options.probe_offsets_s),
        "time_offset_probe_min_rmse_increase": result.options.probe_min_rmse_increase,
        "online_gate_min_time_offset_probe_detection_ratio": (
            result.options.probe_min_detection_ratio
        ),
        "estimate_time_offset": result.options.estimate_time_offset,
        "time_offset_search_bound_s": result.options.search_bound_s,
        "online_gate_max_abs_time_offset_s": result.options.max_abs_time_offset_s,
    }
    if result.probes:
        block["probes"] = [
            {
                "offset_s": probe.offset_s,
                "baseline_rmse_m": probe.baseline_rmse_m,
                "shifted_rmse_m": probe.shifted_rmse_m,
                "relative_increase": probe.relative_increase,
                "detected": probe.detected,
            }
            for probe in result.probes
        ]
        block["probe_detection_ratio"] = result.probe_detection_ratio
        block["probe_gate_status"] = result.probe_gate_status
        block["probe_gate_reason"] = result.probe_gate_reason
    if result.estimate is not None:
        block["estimate"] = {
            "estimated_offset_s": result.estimate.estimated_offset_s,
            "rmse_at_estimate_m": result.estimate.rmse_at_estimate_m,
            "rmse_at_zero_s": result.estimate.rmse_at_zero_s,
            "relative_improvement": result.estimate.relative_improvement,
            "curve_sample_count": result.estimate.curve_sample_count,
            "curve_flatness": result.estimate.curve_flatness,
        }
        block["estimate_gate_status"] = result.estimate_gate_status
        block["estimate_gate_reason"] = result.estimate_gate_reason
    block["verdict"] = result.verdict
    block["verdict_reason"] = result.verdict_reason
    return block


def dual_temporal_evidence_to_provenance(result: DualTemporalEvidenceResult) -> dict[str, Any]:
    """Serialize dual-mode temporal evidence for run provenance."""

    block = temporal_evidence_to_provenance(result.selected)
    block["time_offset_anchor"] = result.time_offset_anchor
    block["anchored_transform"] = _se3_to_provenance(result.anchored_transform)
    block["adapted"] = temporal_evidence_to_provenance(result.adapted)
    block["anchored"] = temporal_evidence_to_provenance(result.anchored)
    if result.separability is not None:
        block["separability"] = {
            "verdict": result.separability.verdict,
            "reason": result.separability.reason,
            "anchored_curve_flatness": result.separability.anchored_curve_flatness,
            "adapted_curve_flatness": result.separability.adapted_curve_flatness,
            "flatness_margin": result.separability.flatness_margin,
        }
    injected_s = result.time_offset_injected_s
    block["time_offset_injected_s"] = injected_s
    if abs(injected_s) > 1.0e-12:
        block["time_offset_injection_warning"] = (
            "VALIDATION-ONLY: target capture timestamps were shifted at load time; "
            "this run is not clean-field data"
        )
    return block


def _holdout_rmse_at_offset(
    *,
    source_records: list[LivoxPointRecord],
    holdout_points: list[Vector3],
    holdout_capture_ns: list[int],
    t_source_target: SE3,
    odometry_track: OdometryTrack,
    t_base_source: SE3,
    variable: str,
    sensor: str,
    voxel_size_m: float,
    correspondence_gate_m: float,
    delta_t_s: float,
    max_odometry_extrapolation_s: float | None = None,
) -> float | None:
    poses, max_extrapolation_s = _poses_at_offset(
        holdout_capture_ns,
        odometry_track=odometry_track,
        t_base_source=t_base_source,
        delta_t_s=delta_t_s,
    )
    if (
        max_odometry_extrapolation_s is not None
        and max_extrapolation_s > max_odometry_extrapolation_s
    ):
        return None
    observations = build_rig_point_to_plane_observations(
        source_records=source_records,
        target_points=holdout_points,
        initial_t_source_target=t_source_target,
        target_t_world_source=poses,
        voxel_size_m=voxel_size_m,
        correspondence_gate_m=correspondence_gate_m,
    )
    if len(observations) < 3:
        return None
    factor = LidarRigPointToPlaneFactor(
        variable=variable,
        t_ego_lidar=t_source_target,
        observations=observations,
        sensor=sensor,
    )
    evaluation = factor.evaluate()
    return evaluation.rmse_m


def _poses_at_offset(
    capture_timestamps_ns: list[int],
    *,
    odometry_track: OdometryTrack,
    t_base_source: SE3,
    delta_t_s: float,
) -> tuple[list[SE3], float]:
    poses: list[SE3] = []
    max_extrapolation_s = 0.0
    for capture_ns in capture_timestamps_ns:
        shifted_ns = apply_time_offset_ns(capture_ns, delta_t_s)
        t_world_base, _clamped, extrapolation_s = odometry_track.interpolate(shifted_ns)
        max_extrapolation_s = max(max_extrapolation_s, extrapolation_s)
        poses.append(t_world_base.compose(t_base_source))
    return poses, max_extrapolation_s


def _run_probes(
    *,
    options: TemporalEvidenceOptions,
    baseline_rmse_m: float,
    source_records: list[LivoxPointRecord],
    holdout_points: list[Vector3],
    holdout_capture_ns: list[int],
    t_source_target: SE3,
    odometry_track: OdometryTrack,
    t_base_source: SE3,
    variable: str,
    sensor: str,
    voxel_size_m: float,
    correspondence_gate_m: float,
    max_odometry_extrapolation_s: float | None = None,
) -> tuple[TimeOffsetProbeResult, ...]:
    probes: list[TimeOffsetProbeResult] = []
    for offset_s in options.probe_offsets_s:
        shifted_rmse = _holdout_rmse_at_offset(
            source_records=source_records,
            holdout_points=holdout_points,
            holdout_capture_ns=holdout_capture_ns,
            t_source_target=t_source_target,
            odometry_track=odometry_track,
            t_base_source=t_base_source,
            variable=variable,
            sensor=sensor,
            voxel_size_m=voxel_size_m,
            correspondence_gate_m=correspondence_gate_m,
            delta_t_s=offset_s,
            max_odometry_extrapolation_s=max_odometry_extrapolation_s,
        )
        if shifted_rmse is None:
            relative_increase = 0.0
            detected = False
            shifted_value = baseline_rmse_m
        else:
            shifted_value = shifted_rmse
            relative_increase = _relative_increase(baseline_rmse_m, shifted_value)
            detected = relative_increase >= options.probe_min_rmse_increase
        probes.append(
            TimeOffsetProbeResult(
                offset_s=offset_s,
                baseline_rmse_m=baseline_rmse_m,
                shifted_rmse_m=shifted_value,
                relative_increase=relative_increase,
                detected=detected,
            )
        )
    return tuple(probes)


def _evaluate_probe_gate(
    probes: tuple[TimeOffsetProbeResult, ...],
    options: TemporalEvidenceOptions,
) -> tuple[TemporalGateStatus, str, float]:
    if not probes:
        return "inconclusive", "no time-offset probes configured", 0.0
    detected_count = sum(1 for probe in probes if probe.detected)
    ratio = detected_count / len(probes)
    if ratio >= options.probe_min_detection_ratio:
        return (
            "pass",
            f"{detected_count}/{len(probes)} probes detected "
            f"(ratio {ratio:.3f} >= {options.probe_min_detection_ratio:.3f})",
            ratio,
        )
    return (
        "fail",
        f"only {detected_count}/{len(probes)} probes detected "
        f"(ratio {ratio:.3f} < {options.probe_min_detection_ratio:.3f})",
        ratio,
    )


def _estimate_time_offset(
    *,
    source_records: list[LivoxPointRecord],
    holdout_points: list[Vector3],
    holdout_capture_ns: list[int],
    t_source_target: SE3,
    odometry_track: OdometryTrack,
    t_base_source: SE3,
    variable: str,
    sensor: str,
    voxel_size_m: float,
    correspondence_gate_m: float,
    search_bound_s: float,
    rmse_at_zero_s: float,
    max_odometry_extrapolation_s: float | None = None,
) -> TimeOffsetEstimateResult:
    sample_count = 0

    def objective(delta_t_s: float) -> float:
        nonlocal sample_count
        sample_count += 1
        rmse = _holdout_rmse_at_offset(
            source_records=source_records,
            holdout_points=holdout_points,
            holdout_capture_ns=holdout_capture_ns,
            t_source_target=t_source_target,
            odometry_track=odometry_track,
            t_base_source=t_base_source,
            variable=variable,
            sensor=sensor,
            voxel_size_m=voxel_size_m,
            correspondence_gate_m=correspondence_gate_m,
            delta_t_s=delta_t_s,
            max_odometry_extrapolation_s=max_odometry_extrapolation_s,
        )
        if rmse is None:
            return float("inf")
        return rmse

    coarse_step = max(0.01, search_bound_s / 10.0)
    coarse_best = 0.0
    coarse_best_rmse = objective(0.0)
    coarse_rmses: list[float] = [coarse_best_rmse]
    offset = -search_bound_s
    while offset <= search_bound_s + 1.0e-12:
        rmse = objective(offset)
        coarse_rmses.append(rmse)
        if rmse < coarse_best_rmse:
            coarse_best_rmse = rmse
            coarse_best = offset
        offset += coarse_step

    finite_coarse = [value for value in coarse_rmses if math.isfinite(value)]
    if finite_coarse:
        min_coarse = min(finite_coarse)
        max_coarse = max(finite_coarse)
        curve_flatness = (
            0.0 if min_coarse <= 0.0 else (max_coarse - min_coarse) / min_coarse
        )
    else:
        curve_flatness = 0.0

    left = max(-search_bound_s, coarse_best - coarse_step)
    right = min(search_bound_s, coarse_best + coarse_step)
    estimated_offset, rmse_at_estimate = _golden_section_minimize(
        objective,
        left=left,
        right=right,
        resolution_s=_SEARCH_RESOLUTION_S,
    )
    relative_improvement = _relative_improvement(rmse_at_zero_s, rmse_at_estimate)
    return TimeOffsetEstimateResult(
        estimated_offset_s=estimated_offset,
        rmse_at_estimate_m=rmse_at_estimate,
        rmse_at_zero_s=rmse_at_zero_s,
        relative_improvement=relative_improvement,
        curve_sample_count=sample_count,
        curve_flatness=curve_flatness,
    )


def _golden_section_minimize(
    objective: Any,
    *,
    left: float,
    right: float,
    resolution_s: float,
) -> tuple[float, float]:
    a = left
    b = right
    c = b - _GOLDEN_RATIO * (b - a)
    d = a + _GOLDEN_RATIO * (b - a)
    fc = objective(c)
    fd = objective(d)
    while (b - a) > resolution_s:
        if fc < fd:
            b = d
            d = c
            fd = fc
            c = b - _GOLDEN_RATIO * (b - a)
            fc = objective(c)
        else:
            a = c
            c = d
            fc = fd
            d = a + _GOLDEN_RATIO * (b - a)
            fd = objective(d)
    if fc < fd:
        best_offset = c
        best_rmse = fc
    else:
        best_offset = d
        best_rmse = fd
    return best_offset, best_rmse


def _evaluate_estimate_gate(
    estimate: TimeOffsetEstimateResult,
    options: TemporalEvidenceOptions,
) -> tuple[TemporalGateStatus, str]:
    abs_offset = abs(estimate.estimated_offset_s)
    improvement = estimate.relative_improvement
    if estimate.curve_flatness < options.min_significant_relative_improvement:
        return (
            "inconclusive",
            f"holdout RMSE curve is too flat to localize a minimum "
            f"(curve flatness {estimate.curve_flatness:.4f} < "
            f"{options.min_significant_relative_improvement:.4f})",
        )
    if abs_offset <= options.max_abs_time_offset_s:
        return (
            "pass",
            f"|estimated offset| {abs_offset:.4f} s <= "
            f"{options.max_abs_time_offset_s:.4f} s gate",
        )
    if improvement < options.min_significant_relative_improvement:
        return (
            "inconclusive",
            f"holdout RMSE improvement is too small to confirm offset "
            f"(relative improvement {improvement:.4f} < "
            f"{options.min_significant_relative_improvement:.4f})",
        )
    return (
        "fail",
        f"|estimated offset| {abs_offset:.4f} s exceeds "
        f"{options.max_abs_time_offset_s:.4f} s gate with significant RMSE "
        f"improvement ({improvement:.4f})",
    )


def _combine_verdicts(
    *,
    probe_gate_status: TemporalGateStatus | None,
    probe_gate_reason: str | None,
    estimate_gate_status: TemporalGateStatus | None,
    estimate_gate_reason: str | None,
) -> tuple[TemporalGateStatus | None, str | None]:
    statuses: list[TemporalGateStatus] = []
    reasons: list[str] = []
    if probe_gate_status is not None:
        statuses.append(probe_gate_status)
        if probe_gate_reason:
            reasons.append(f"probe: {probe_gate_reason}")
    if estimate_gate_status is not None:
        statuses.append(estimate_gate_status)
        if estimate_gate_reason:
            reasons.append(f"estimate: {estimate_gate_reason}")
    if not statuses:
        return None, None
    verdict = _worst_status(statuses)
    return verdict, "; ".join(reasons)


def _worst_status(statuses: list[TemporalGateStatus]) -> TemporalGateStatus:
    order = {"pass": 0, "inconclusive": 1, "fail": 2}
    return max(statuses, key=lambda status: order[status])


def _relative_increase(baseline: float, shifted: float) -> float:
    if baseline <= 0.0:
        return 0.0 if shifted <= baseline else 1.0
    return max(0.0, (shifted - baseline) / baseline)


def _relative_improvement(baseline: float, improved: float) -> float:
    if baseline <= 0.0:
        return 0.0
    return max(0.0, (baseline - improved) / baseline)


def _empty_inconclusive(
    *,
    options: TemporalEvidenceOptions,
    reason: str,
    holdout_point_count: int,
) -> TemporalEvidenceResult:
    return TemporalEvidenceResult(
        sign_convention=(
            "positive delta_t_s evaluates T_world_source(t_capture + delta_t_s); "
            "target timestamps lag the odometry clock when delta_t_s > 0"
        ),
        options=options,
        baseline_holdout_rmse_m=0.0,
        holdout_point_count=holdout_point_count,
        probes=(),
        probe_detection_ratio=None,
        probe_gate_status="inconclusive" if options.probe_offsets_s else None,
        probe_gate_reason=reason if options.probe_offsets_s else None,
        estimate=None,
        estimate_gate_status="inconclusive" if options.estimate_time_offset else None,
        estimate_gate_reason=reason if options.estimate_time_offset else None,
        verdict="inconclusive",
        verdict_reason=reason,
    )


def _float_option(
    options: dict[str, Any], key: str, *, default: float, minimum: float
) -> float:
    try:
        value = float(options[key])
    except (KeyError, TypeError, ValueError):
        return default
    return max(value, minimum)


def _curve_flatness(result: TemporalEvidenceResult) -> float:
    if result.estimate is not None:
        return result.estimate.curve_flatness
    return 0.0


def _estimate_offset_s(result: TemporalEvidenceResult) -> float:
    if result.estimate is not None:
        return result.estimate.estimated_offset_s
    return 0.0


def _estimate_localizes(result: TemporalEvidenceResult, margin: float) -> bool:
    if result.estimate is None:
        return False
    if result.estimate_gate_status == "inconclusive":
        reason = (result.estimate_gate_reason or "").lower()
        if "flat" in reason:
            return False
    return _curve_flatness(result) >= margin


def _adapted_extrinsic_absorbed(
    anchored: TemporalEvidenceResult,
    adapted: TemporalEvidenceResult,
    margin: float,
) -> bool:
    """True when anchored finds a δt offset but adapted δt̂≈0 (v0.2 absorption)."""

    if anchored.estimate is None or adapted.estimate is None:
        return False
    anchored_magnitude = abs(anchored.estimate.estimated_offset_s)
    adapted_magnitude = abs(adapted.estimate.estimated_offset_s)
    absorption_margin_s = max(0.01, margin * 0.1)
    return anchored_magnitude >= absorption_margin_s and adapted_magnitude < absorption_margin_s


def _se3_to_provenance(transform: SE3) -> dict[str, Any]:
    return {
        "translation_m": list(transform.translation_m),
        "rotation_quat_xyzw": list(transform.rotation_quat_xyzw),
    }
