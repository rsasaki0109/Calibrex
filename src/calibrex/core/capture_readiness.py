"""Schema-versioned capture readiness and recapture guidance for LiDAR calibration."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from itertools import pairwise
from typing import Any, Literal

from pydantic import Field, field_validator

from calibrex.core.geometry import SE3
from calibrex.core.result import Grade, StrictModel
from calibrex.data.odometry_track import OdometryPoseSample

CAPTURE_READINESS_SCHEMA_VERSION: Literal["slac.capture_readiness/v0.1"] = (
    "slac.capture_readiness/v0.1"
)

CaptureReadinessDecision = Literal["proceed", "recapture", "review"]
CaptureReadinessAction = Literal["proceed", "recapture", "review"]

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class CaptureReadinessThresholds(StrictModel):
    """Configurable gates for deciding whether a capture window is usable."""

    min_pose_count: int = Field(default=10, ge=1)
    min_time_span_s: float = Field(default=5.0, ge=0.0)
    min_path_length_m: float = Field(default=1.0, ge=0.0)
    min_endpoint_rotation_deg: float = Field(default=10.0, ge=0.0)
    warn_endpoint_rotation_deg: float = Field(default=25.0, ge=0.0)
    min_plane_count: int = Field(default=10, ge=1)
    min_normal_rank: int = Field(default=2, ge=1, le=3)
    max_normal_condition_number: float = Field(default=100.0, gt=0.0)
    min_observable_dof: int = Field(default=6, ge=1, le=6)
    max_angular_speed_dps: float = Field(default=120.0, ge=0.0)
    max_linear_speed_mps: float = Field(default=3.0, ge=0.0)


class CaptureReadinessParameters(StrictModel):
    """Replay and geometry-proxy settings needed to reproduce the artifact."""

    voxel_size_m: float = Field(gt=0.0)
    correspondence_gate_m: float = Field(gt=0.0)
    max_scans_per_window: int = Field(ge=1)
    max_points_per_window: int = Field(ge=1)
    source_point_time_field: str | None = None
    use_point_time_offsets: bool = True
    odometry_burst_policy: Literal["preserve", "keep_first", "keep_last"] = "preserve"
    odometry_burst_min_interval_s: float = Field(gt=0.0)
    capture_window_record_prefilter_margin_s: float = Field(ge=0.0)
    capture_window_sampling_policy: str
    geometry_proxy: Literal["source_world_plane_map_virtual_target"] = (
        "source_world_plane_map_virtual_target"
    )


class CaptureReadinessMotion(StrictModel):
    """Coverage and kinematic excitation measured in one capture window."""

    pose_count: int = Field(ge=0)
    first_timestamp_ns: int | None = Field(default=None, ge=0)
    last_timestamp_ns: int | None = Field(default=None, ge=0)
    time_span_s: float = Field(ge=0.0)
    path_length_m: float = Field(ge=0.0)
    endpoint_displacement_m: float = Field(ge=0.0)
    endpoint_rotation_deg: float = Field(ge=0.0)
    linear_speed_p95_mps: float | None = Field(default=None, ge=0.0)
    linear_speed_max_mps: float | None = Field(default=None, ge=0.0)
    angular_speed_p95_dps: float | None = Field(default=None, ge=0.0)
    angular_speed_max_dps: float | None = Field(default=None, ge=0.0)


class CaptureReadinessGeometry(StrictModel):
    """Surface-normal and local point-to-plane observability evidence."""

    source_scan_count: int = Field(ge=0)
    source_point_count: int = Field(ge=0)
    plane_count: int = Field(ge=0)
    normal_rank: int = Field(ge=0, le=3)
    normal_condition_number: float | None = Field(default=None, ge=0.0)
    normal_diversity_score: float | None = Field(default=None, ge=0.0, le=1.0)
    normal_eigenvalues: list[float] = Field(default_factory=list, max_length=3)
    estimated_observable_dof: int = Field(ge=0, le=6)
    weak_directions: list[str] = Field(default_factory=list)
    correspondence_count: int = Field(ge=0)
    normal_source: Literal["voxel_plane_map", "unavailable"] = "voxel_plane_map"


class CaptureReadinessWindow(StrictModel):
    """Actionable readiness result for one declared capture window."""

    window_index: int = Field(ge=0)
    requested_start_timestamp_ns: int | None = Field(default=None, ge=0)
    requested_end_timestamp_ns: int | None = Field(default=None, ge=0)
    motion: CaptureReadinessMotion
    geometry: CaptureReadinessGeometry
    grade: Grade
    decision: CaptureReadinessDecision
    reasons: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)


class CaptureReadinessRecommendation(StrictModel):
    """Human-facing summary of the next safe action."""

    action: CaptureReadinessAction
    message: str
    instructions: list[str] = Field(default_factory=list)
    window_indices: list[int] = Field(default_factory=list, min_length=1)


class CaptureReadinessProvenance(StrictModel):
    """Input digests and generation details for a readiness artifact."""

    source_paths: list[str] = Field(min_length=2)
    source_sha256: dict[str, str] = Field(min_length=2)
    tool_name: str = "calibrex"
    tool_version: str | None = None
    git_commit: str | None = None
    notes: list[str] = Field(default_factory=list)

    @field_validator("source_sha256")
    @classmethod
    def validate_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        invalid = [key for key, digest in value.items() if not _SHA256_RE.fullmatch(digest)]
        if invalid:
            raise ValueError(f"source_sha256 contains invalid SHA-256 values: {invalid}")
        return value


class CaptureReadinessArtifact(StrictModel):
    """Ground-truth-free preflight artifact for solid-state LiDAR calibration."""

    schema_version: Literal["slac.capture_readiness/v0.1"] = (
        CAPTURE_READINESS_SCHEMA_VERSION
    )
    config_path: str
    dataset_path: str
    source_sensor: str
    target_sensor: str
    odometry_topic: str
    parameters: CaptureReadinessParameters
    thresholds: CaptureReadinessThresholds
    windows: list[CaptureReadinessWindow] = Field(min_length=1)
    grade: Grade
    decision: CaptureReadinessDecision
    summary: str
    recommendations: list[CaptureReadinessRecommendation] = Field(min_length=1)
    provenance: CaptureReadinessProvenance


def summarize_capture_window_motion(
    samples: Sequence[OdometryPoseSample],
    window: tuple[int | None, int | None],
) -> CaptureReadinessMotion:
    """Summarize odometry coverage, displacement, and speed for one window."""

    if not samples:
        return CaptureReadinessMotion(
            pose_count=0,
            time_span_s=0.0,
            path_length_m=0.0,
            endpoint_displacement_m=0.0,
            endpoint_rotation_deg=0.0,
        )
    first_ns = samples[0].timestamp_ns
    last_ns = samples[-1].timestamp_ns
    start_ns = max(first_ns, window[0] if window[0] is not None else first_ns)
    end_ns = min(last_ns, window[1] if window[1] is not None else last_ns)
    selected = [sample for sample in samples if start_ns <= sample.timestamp_ns <= end_ns]
    if not selected:
        return CaptureReadinessMotion(
            pose_count=0,
            time_span_s=0.0,
            path_length_m=0.0,
            endpoint_displacement_m=0.0,
            endpoint_rotation_deg=0.0,
        )
    linear_speeds: list[float] = []
    angular_speeds: list[float] = []
    path_length_m = 0.0
    for left, right in pairwise(selected):
        dt_s = max((right.timestamp_ns - left.timestamp_ns) / 1_000_000_000.0, 1.0e-9)
        translation_m = math.dist(left.pose.translation_m, right.pose.translation_m)
        path_length_m += translation_m
        linear_speeds.append(translation_m / dt_s)
        angular_speeds.append(_rotation_speed_dps(left.pose, right.pose, dt_s))
    relative = selected[0].pose.inverse().compose(selected[-1].pose)
    quaternion = relative.rotation_quat_xyzw
    scalar = min(1.0, max(-1.0, abs(quaternion[3])))
    endpoint_rotation_deg = math.degrees(2.0 * math.acos(scalar))
    return CaptureReadinessMotion(
        pose_count=len(selected),
        first_timestamp_ns=selected[0].timestamp_ns,
        last_timestamp_ns=selected[-1].timestamp_ns,
        time_span_s=(selected[-1].timestamp_ns - selected[0].timestamp_ns) / 1_000_000_000.0,
        path_length_m=path_length_m,
        endpoint_displacement_m=math.dist(
            selected[0].pose.translation_m, selected[-1].pose.translation_m
        ),
        endpoint_rotation_deg=endpoint_rotation_deg,
        linear_speed_p95_mps=_percentile(linear_speeds, 95.0),
        linear_speed_max_mps=max(linear_speeds) if linear_speeds else None,
        angular_speed_p95_dps=_percentile(angular_speeds, 95.0),
        angular_speed_max_dps=max(angular_speeds) if angular_speeds else None,
    )


def classify_capture_readiness_window(
    *,
    window_index: int,
    requested_window: tuple[int | None, int | None],
    motion: CaptureReadinessMotion,
    geometry: CaptureReadinessGeometry,
    thresholds: CaptureReadinessThresholds,
) -> CaptureReadinessWindow:
    """Apply explicit motion and geometry gates to one readiness window."""

    failures: list[str] = []
    warnings: list[str] = []
    actions: list[str] = []
    if motion.pose_count < thresholds.min_pose_count:
        failures.append(
            f"only {motion.pose_count} odometry poses; need >= {thresholds.min_pose_count}"
        )
        actions.append("record a longer window with a continuous odometry track")
    if motion.time_span_s < thresholds.min_time_span_s:
        failures.append(
            f"time span {motion.time_span_s:.2f} s is below "
            f"{thresholds.min_time_span_s:.2f} s"
        )
        actions.append("extend the capture window")
    if motion.path_length_m < thresholds.min_path_length_m:
        failures.append(
            f"path length {motion.path_length_m:.2f} m is below "
            f"{thresholds.min_path_length_m:.2f} m"
        )
        actions.append("include controlled translation as well as rotation")
    if motion.endpoint_rotation_deg < thresholds.min_endpoint_rotation_deg:
        failures.append(
            f"endpoint rotation {motion.endpoint_rotation_deg:.1f} deg is below "
            f"{thresholds.min_endpoint_rotation_deg:.1f} deg"
        )
        actions.append("add deliberate roll, pitch, and yaw excitation")
    elif motion.endpoint_rotation_deg < thresholds.warn_endpoint_rotation_deg:
        warnings.append(
            f"endpoint rotation {motion.endpoint_rotation_deg:.1f} deg is marginal; "
            f"target >= {thresholds.warn_endpoint_rotation_deg:.1f} deg"
        )
        actions.append("prefer a wider set of roll, pitch, and yaw views")
    if (
        motion.linear_speed_max_mps is not None
        and motion.linear_speed_max_mps > thresholds.max_linear_speed_mps
    ):
        failures.append(
            f"maximum linear speed {motion.linear_speed_max_mps:.2f} m/s exceeds "
            f"{thresholds.max_linear_speed_mps:.2f} m/s"
        )
        actions.append("repeat the motion more slowly to reduce deskew and odometry error")
    if (
        motion.angular_speed_max_dps is not None
        and motion.angular_speed_max_dps > thresholds.max_angular_speed_dps
    ):
        failures.append(
            f"maximum angular speed {motion.angular_speed_max_dps:.1f} deg/s exceeds "
            f"{thresholds.max_angular_speed_dps:.1f} deg/s"
        )
        actions.append("repeat the rotation more slowly")
    if geometry.plane_count < thresholds.min_plane_count:
        failures.append(
            f"only {geometry.plane_count} usable voxel planes; need >= "
            f"{thresholds.min_plane_count}"
        )
        actions.append("include several stable surfaces at different orientations")
    if geometry.normal_rank < thresholds.min_normal_rank:
        failures.append(
            f"plane-normal rank {geometry.normal_rank} is below "
            f"{thresholds.min_normal_rank}"
        )
        actions.append("sweep across floor, walls, and other non-parallel surfaces")
    if (
        geometry.normal_condition_number is not None
        and geometry.normal_condition_number > thresholds.max_normal_condition_number
    ):
        warnings.append(
            f"plane-normal condition {geometry.normal_condition_number:.1f} exceeds "
            f"{thresholds.max_normal_condition_number:.1f}"
        )
        actions.append("add a view containing a third, clearly different surface normal")
    if geometry.estimated_observable_dof < thresholds.min_observable_dof:
        failures.append(
            f"estimated observable DoF {geometry.estimated_observable_dof} is below "
            f"{thresholds.min_observable_dof}"
        )
        if geometry.weak_directions:
            actions.append("add views that excite: " + ", ".join(geometry.weak_directions))
        else:
            actions.append("add complementary motion and geometry before calibration")
    if not geometry.correspondence_count:
        failures.append("no point-to-plane correspondences were available for the geometry proxy")
        actions.append("check LiDAR overlap, frame transforms, voxel size, and gate distance")

    if failures:
        grade: Grade = "fail"
        decision: CaptureReadinessDecision = "recapture"
        reasons = failures + warnings
    elif warnings:
        grade = "warn"
        decision = "review"
        reasons = warnings
    else:
        grade = "pass"
        decision = "proceed"
        reasons = ["motion excitation and geometry observability passed the declared gates"]
    return CaptureReadinessWindow(
        window_index=window_index,
        requested_start_timestamp_ns=requested_window[0],
        requested_end_timestamp_ns=requested_window[1],
        motion=motion,
        geometry=geometry,
        grade=grade,
        decision=decision,
        reasons=_unique(reasons),
        actions=_unique(actions),
    )


def build_capture_readiness_recommendations(
    windows: Sequence[CaptureReadinessWindow],
) -> tuple[Grade, CaptureReadinessDecision, str, list[CaptureReadinessRecommendation]]:
    """Build one concise, user-facing next action from per-window verdicts."""

    failed = [window.window_index for window in windows if window.grade == "fail"]
    warned = [window.window_index for window in windows if window.grade == "warn"]
    if failed:
        instructions = _unique(
            action for window in windows if window.grade == "fail" for action in window.actions
        )
        recommendations = [
            CaptureReadinessRecommendation(
                action="recapture",
                message=(
                    "再収録してください。失敗した窓ではソルバを回さず、"
                    "運動励起と面の多様性を増やします。"
                ),
                instructions=instructions,
                window_indices=failed,
            )
        ]
        if warned:
            recommendations.append(
                CaptureReadinessRecommendation(
                    action="review",
                    message="警告窓は再収録後にもう一度確認してください。",
                    instructions=["警告は致命的ではありませんが、SOTA比較では記録しておく"],
                    window_indices=warned,
                )
            )
        return "fail", "recapture", "one or more capture windows are not ready", recommendations
    if warned:
        return (
            "warn",
            "review",
            "capture windows are usable but at least one has marginal excitation",
            [
                CaptureReadinessRecommendation(
                    action="review",
                    message="続行できますが、警告窓は品質レビュー対象です。",
                    instructions=_unique(
                        action
                        for window in windows
                        if window.grade == "warn"
                        for action in window.actions
                    ),
                    window_indices=warned,
                )
            ],
        )
    return (
        "pass",
        "proceed",
        "all capture windows passed motion and geometry readiness gates",
        [
            CaptureReadinessRecommendation(
                action="proceed",
                message=(
                    "収録はキャリブレーション開始可能です。"
                    "point-time deskewを有効にして実行してください。"
                ),
                instructions=[
                    "solver実行後も窓別holdout RMSEと外参の窓間差を確認する",
                    "Readiness artifactを結果と一緒に保存する",
                ],
                window_indices=[window.window_index for window in windows],
            )
        ],
    )


def capture_readiness_json_schema() -> dict[str, Any]:
    """Return the standalone JSON schema for capture readiness artifacts."""

    return CaptureReadinessArtifact.model_json_schema()


def _rotation_speed_dps(left: SE3, right: SE3, dt_s: float) -> float:
    relative = left.inverse().compose(right)
    scalar = min(1.0, max(-1.0, abs(relative.rotation_quat_xyzw[3])))
    return math.degrees(2.0 * math.acos(scalar)) / dt_s


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    alpha = position - lower
    return ordered[lower] + alpha * (ordered[upper] - ordered[lower])


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
