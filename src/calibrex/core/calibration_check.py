"""Schema-versioned audit of the calibration deployed on a robot (``calibrex check``).

``slac.calibration_check/v0.1`` records the candidate extrinsics read from a
bag's ``/tf_static`` or a calibration file, the sensors found in the bag, and one
record per sensor pair that could be audited.  Transforms follow the
``T_parent_child`` convention (``p_parent = T p_child``); a pair named ``A-B``
carries ``T_A_B`` composed through the static tree.

A plan-only run (``plan_only: true``) leaves every runnable pair ``planned``.
A full run executes the native estimator of each wired pair and judges the
candidate transform against it: the pair status becomes ``pass``, ``warn``,
``fail`` or ``inconclusive`` and the pair record carries per-axis judgements,
the estimator's evidence artifact (path and SHA-256) and the options used.
These fields are optional additions, so plan-only artifacts stay valid.

``slac.check_frames/v0.1`` is the small input format accepted by
``calibrex check --tf`` for hand-written static frame trees.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field, field_validator, model_validator

from calibrex.core.result import StrictModel

CALIBRATION_CHECK_SCHEMA_VERSION: Literal["slac.calibration_check/v0.1"] = (
    "slac.calibration_check/v0.1"
)
CHECK_FRAMES_SCHEMA_VERSION: Literal["slac.check_frames/v0.1"] = "slac.check_frames/v0.1"

SensorRole = Literal["lidar", "imu", "camera", "gnss", "odometry", "twist", "tf_static"]
OdometryKind = Literal["wheel", "ins", "unknown"]
CheckPairName = Literal[
    "imu-lidar",
    "lidar-lidar",
    "camera-imu",
    "camera-focal",
    "gnss-lidar",
    "gnss-imu",
    "lidar-vehicle",
    "imu-vehicle",
    "ins-lidar",
    "lidar-wheel_odometry",
]
CHECK_PAIR_NAMES: tuple[CheckPairName, ...] = (
    "imu-lidar",
    "lidar-lidar",
    "camera-imu",
    "camera-focal",
    "gnss-lidar",
    "gnss-imu",
    "lidar-vehicle",
    "imu-vehicle",
    "ins-lidar",
    "lidar-wheel_odometry",
)
CheckPairStatus = Literal["planned", "skipped", "pass", "warn", "fail", "inconclusive"]
CheckReasonCode = Literal[
    "missing_topic",
    "frame_not_in_tree",
    "frames_not_connected",
    "no_candidate_calibration",
    "no_vehicle_frame",
    "method_not_wired",
    "not_selected",
    "unsupported_sensor",
    "missing_intrinsics",
    "missing_dependency",
    "estimator_error",
    "estimator_failed",
    "no_judgeable_axes",
]
CandidateSourceKind = Literal[
    "bag_tf_static",
    "urdf",
    "frames_yaml",
    "kalibr_camchain",
    "rtk_slam_calib",
    "hilti_sensors",
]
FrameSource = Literal["override", "header", "source_topic_hint", "role_default"]


class CheckTransform(StrictModel):
    """``T_parent_child``: maps ``child_frame`` points into ``parent_frame``."""

    parent_frame: str
    child_frame: str
    translation_m: list[float] = Field(min_length=3, max_length=3)
    rotation_quat_xyzw: list[float] = Field(min_length=4, max_length=4)


class CheckBagInput(StrictModel):
    """The bag that was audited.

    The digest follows the repository precedent for tens-of-gigabyte bags:
    ``metadata.yaml`` in full, and each storage file by name, size and first
    64 MiB (see ``input_digest_scope``).
    """

    path: str
    storage_identifier: str | None = None
    topic_count: int = Field(ge=0)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_digest_scope: str


class CheckCandidateSource(StrictModel):
    """One origin of candidate extrinsics."""

    kind: CandidateSourceKind
    path: str | None = None
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    frame_count: int = Field(ge=0)
    notes: list[str] = Field(default_factory=list)


class CheckTopicRecord(StrictModel):
    """One bag topic, its role and the tree frame it maps to."""

    topic: str
    message_type: str
    message_count: int | None = Field(default=None, ge=0)
    role: SensorRole | None = None
    odometry_kind: OdometryKind | None = None
    header_frame_id: str | None = None
    mapped_frame: str | None = None
    frame_source: FrameSource | None = None
    notes: list[str] = Field(default_factory=list)


class CheckFrameEdge(StrictModel):
    """One static edge of the candidate tree."""

    transform: CheckTransform
    source: CandidateSourceKind


class CheckFrameTree(StrictModel):
    """The candidate static tree (a forest when several roots exist)."""

    roots: list[str] = Field(default_factory=list)
    frames: list[str] = Field(default_factory=list)
    edges: list[CheckFrameEdge] = Field(default_factory=list)
    overrides: list[str] = Field(default_factory=list)


CheckAxisName = Literal["roll", "pitch", "yaw", "x", "y", "z"]
CheckAxisStatus = Literal["pass", "warn", "fail"]


class CheckAxisJudgement(StrictModel):
    """One judged axis: candidate error against the estimator's value and uncertainty.

    ``candidate_error`` is candidate minus estimate: for a rotation axis the
    component of the small rotation ``R_candidate R_estimate^T`` about the axes
    of the compared transform's parent frame (degrees, the convention the
    estimator reports its standard deviation in); for a translation axis the
    difference of the translation components (metres).

    ``tolerance = max(sigma_k * std, floor)``. ``status`` is ``pass`` when
    ``|candidate_error| <= tolerance``, ``fail`` when it exceeds twice the
    tolerance, else ``warn``.

    ``detectable_error`` is ``tolerance + |candidate_error|``: a deliberate
    error larger than this on this axis, of either sign, would have pushed
    ``|candidate_error|`` beyond ``tolerance`` and been flagged. It is derived
    from the candidate error and uncertainty at hand; no solve is repeated.
    ``detects_perturbation`` says whether the probe of
    ``detection_probe`` (degrees, rotation axes only) is below that bound.
    """

    name: CheckAxisName
    unit: Literal["deg", "m"]
    candidate_error: float
    estimate_std: float = Field(ge=0.0)
    tolerance: float = Field(gt=0.0)
    tolerance_source: Literal["sigma", "floor"]
    ratio: float = Field(ge=0.0, description="abs(candidate_error) / tolerance")
    status: CheckAxisStatus
    detectable_error: float = Field(ge=0.0)
    detection_probe: float | None = None
    detects_perturbation: bool | None = None


class CheckUncheckedAxis(StrictModel):
    """An axis the estimator did not constrain, so the candidate is not judged on it."""

    name: CheckAxisName
    unit: Literal["deg", "m"]
    std: float | None = Field(default=None, ge=0.0)
    reason: str
    reason_code: Literal["unobservable", "control_not_detected", "no_estimate"] | None = None


class CheckTimeOffset(StrictModel):
    """The estimator's clock offset; reported, not judged."""

    estimate_s: float
    std_s: float | None = Field(default=None, ge=0.0)
    status: Literal["estimated", "unobservable"]
    note: str = "reported only; not judged"


class CheckEvidenceRef(StrictModel):
    """One estimator artifact written next to the check artifact."""

    path: str = Field(description="relative to the check artifact's directory")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    schema_version: str
    role: str
    policy_status: str | None = None


class CheckOptions(StrictModel):
    """Verdict thresholds and runtime controls used for a run."""

    sigma_k: float = Field(gt=0.0)
    rotation_floor_deg: float = Field(gt=0.0)
    translation_floor_m: float = Field(gt=0.0)
    detection_probe_deg: float = Field(gt=0.0)
    max_duration_s: float | None = None
    pairs: list[str] | None = None
    camera: str | None = None
    imu_lidar_translation: bool = True
    acceleration_unit: Literal["mps2", "g"] = "mps2"


class CheckPairRecord(StrictModel):
    """One candidate sensor pair and what was decided for it."""

    pair: CheckPairName
    sensors: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    frames: list[str] = Field(default_factory=list)
    candidate_transform: CheckTransform | None = None
    status: CheckPairStatus
    reason_code: CheckReasonCode | None = None
    reason: str | None = None
    evidence_artifact: str | None = None
    verdict: str | None = None
    compared_transform: CheckTransform | None = Field(
        default=None,
        description=(
            "the candidate in the estimator's convention (for example T_lidar_imu for "
            "imu-lidar); axes are those of its parent frame"
        ),
    )
    estimator: str | None = None
    estimator_policy_status: str | None = None
    estimator_policy_reasons: list[str] = Field(default_factory=list)
    axes: list[CheckAxisJudgement] = Field(default_factory=list)
    unchecked_axes: list[CheckUncheckedAxis] = Field(default_factory=list)
    time_offset: CheckTimeOffset | None = None
    evidence: list[CheckEvidenceRef] = Field(default_factory=list)
    coverage: Literal["full", "partial"] | None = Field(
        default=None,
        description="partial when any rotation axis, or any translation axis the estimator "
        "attempted, is unchecked",
    )
    runtime_s: float | None = Field(default=None, ge=0.0)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _status_fields_are_consistent(self) -> CheckPairRecord:
        if self.status == "skipped":
            if self.reason_code is None or not self.reason:
                raise ValueError("a skipped pair needs reason_code and reason")
        elif self.status == "planned":
            if self.candidate_transform is None:
                raise ValueError("a planned pair needs candidate_transform")
            if self.reason_code is not None:
                raise ValueError("a planned pair must not carry reason_code")
        return self


class CheckSummary(StrictModel):
    """Counts derived from the pair records."""

    pair_count: int = Field(ge=0)
    runnable_count: int = Field(ge=0)
    status_counts: dict[str, int] = Field(default_factory=dict)
    skipped_by_reason: dict[str, int] = Field(default_factory=dict)
    partial_pairs: int = Field(
        default=0,
        ge=0,
        description="run pairs that left at least one attempted axis unchecked",
    )


class CheckProvenance(StrictModel):
    """Generator lineage and input digests."""

    generator: str
    generator_version: str
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_digest_scope: str
    created_at: str
    notes: list[str] = Field(default_factory=list)


class CalibrationCheckArtifact(StrictModel):
    """Audit plan (Phase A) and, in later phases, verdicts for a deployed calibration."""

    schema_version: Literal["slac.calibration_check/v0.1"] = CALIBRATION_CHECK_SCHEMA_VERSION
    plan_only: bool = True
    bag: CheckBagInput
    vehicle_frame: str | None = None
    candidate_sources: list[CheckCandidateSource] = Field(default_factory=list)
    topics: list[CheckTopicRecord] = Field(default_factory=list)
    frame_tree: CheckFrameTree
    pairs: list[CheckPairRecord] = Field(default_factory=list)
    summary: CheckSummary
    overall_verdict: str | None = Field(
        default=None,
        description=(
            "worst verdict over the pairs that ran (fail > warn > inconclusive > pass); "
            "absent in a plan-only run; inconclusive when no pair ran"
        ),
    )
    options: CheckOptions | None = None
    evidence_dir: str | None = Field(
        default=None, description="evidence directory, relative to the artifact"
    )
    provenance: CheckProvenance

    @model_validator(mode="after")
    def _summary_matches_pairs(self) -> CalibrationCheckArtifact:
        expected = summarize_pairs(self.pairs)
        if self.summary != expected:
            raise ValueError("summary does not match the pair records")
        return self


def summarize_pairs(pairs: list[CheckPairRecord]) -> CheckSummary:
    """Derive the summary counts from pair records."""

    status_counts: dict[str, int] = {}
    skipped_by_reason: dict[str, int] = {}
    for pair in pairs:
        status_counts[pair.status] = status_counts.get(pair.status, 0) + 1
        if pair.status == "skipped" and pair.reason_code is not None:
            skipped_by_reason[pair.reason_code] = skipped_by_reason.get(pair.reason_code, 0) + 1
    partial = sum(1 for pair in pairs if pair.coverage == "partial")
    return CheckSummary(
        partial_pairs=partial,
        pair_count=len(pairs),
        runnable_count=len(pairs) - status_counts.get("skipped", 0),
        status_counts=dict(sorted(status_counts.items())),
        skipped_by_reason=dict(sorted(skipped_by_reason.items())),
    )


class CheckFramesEntry(StrictModel):
    """One static frame: ``T_parent_frame``."""

    name: str = Field(min_length=1)
    parent: str = Field(min_length=1)
    translation_m: list[float] = Field(min_length=3, max_length=3)
    rotation_quat_xyzw: list[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0, 1.0], min_length=4, max_length=4
    )

    @field_validator("translation_m", "rotation_quat_xyzw")
    @classmethod
    def _finite(cls, value: list[float]) -> list[float]:
        if not all(math.isfinite(item) for item in value):
            raise ValueError("values must be finite")
        return value

    @field_validator("rotation_quat_xyzw")
    @classmethod
    def _nonzero_quaternion(cls, value: list[float]) -> list[float]:
        if math.sqrt(sum(item * item for item in value)) < 1e-9:
            raise ValueError("quaternion must be non-zero")
        return value


class CheckFramesFile(StrictModel):
    """Hand-written static frame tree for ``calibrex check --tf``.

    ``topic_frames`` maps bag topics to frames and ``role_frames`` names the
    default frame of a sensor role; both are optional hints used when a
    topic's header frame id is not in the tree.
    """

    schema_version: Literal["slac.check_frames/v0.1"] = CHECK_FRAMES_SCHEMA_VERSION
    frames: list[CheckFramesEntry] = Field(min_length=1)
    topic_frames: dict[str, str] = Field(default_factory=dict)
    role_frames: dict[SensorRole, str] = Field(default_factory=dict)


def calibration_check_json_schema() -> dict[str, object]:
    """Return the standalone JSON schema for calibration check artifacts."""

    return CalibrationCheckArtifact.model_json_schema()


def check_frames_json_schema() -> dict[str, object]:
    """Return the standalone JSON schema for ``calibrex check`` frame files."""

    return CheckFramesFile.model_json_schema()
