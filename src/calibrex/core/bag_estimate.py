"""Schema-versioned calibration estimated from a bag (``calibrex estimate``).

``slac.bag_estimate/v0.1`` is the counterpart of ``slac.calibration_check/v0.1``
for a user who has a bag but no calibration yet: instead of judging a candidate
it records what the bag's own data say. Per sensor pair it carries the estimated
transform, each axis with its standard deviation and whether the data observed
it, the estimator settings, the evidence artifacts, the input topics and the bag
digest. It also records which sensor frames could be written into the exported
``slac.check_frames`` tree and which were omitted, and why.

Transforms follow the ``T_parent_child`` convention (``p_parent = T p_child``)
of the pair's estimator (the same ``parent_frame`` and ``child_frame`` that
``slac.calibration_check`` reports as ``compared_transform``). Rotation axes are
the components of the rotation vector of the estimated rotation about the axes
of the parent frame, in degrees, which is the convention the estimator reports
its standard deviation in.

An axis is ``observed`` only when the data constrained it and the estimator's
own known-bad control detected it. Every other status means the transform value
on that axis is the placeholder or prior it was started from, not a measurement.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from calibrex.core.calibration_check import (
    CheckAxisName,
    CheckBagInput,
    CheckCandidateSource,
    CheckEvidenceRef,
    CheckPairName,
    CheckProvenance,
    CheckReasonCode,
    CheckTimeOffset,
    CheckTopicRecord,
    CheckTransform,
)
from calibrex.core.result import StrictModel

BAG_ESTIMATE_SCHEMA_VERSION: Literal["slac.bag_estimate/v0.1"] = "slac.bag_estimate/v0.1"

EstimateAxisStatus = Literal[
    "observed",
    "unobservable",
    "control_not_detected",
    "no_estimate",
    "not_estimated",
]
EstimatePairStatus = Literal["estimated", "partial", "failed", "skipped"]
EstimateExportKind = Literal[
    "frames_yaml", "static_transform_publisher", "launch_yaml", "urdf_joints", "kalibr_camchain"
]
EstimateFill = Literal["prior", "placeholder_identity"]


class EstimateAxis(StrictModel):
    """One axis of a pair's estimate.

    ``value`` is the estimated value (degrees about the parent frame's axes for
    a rotation axis, metres for a translation axis); it is absent unless the
    estimator produced one. ``status`` says whether the data observed it:
    ``observed``; ``unobservable`` (the data did not constrain it);
    ``control_not_detected`` (the estimator's known-bad control on this axis was
    not detected on held-out data, so the value is not trusted);
    ``no_estimate`` (the estimator produced no value); ``not_estimated`` (this
    estimator does not attempt the axis, for example the translation of a
    camera-IMU rotation estimate).
    """

    name: CheckAxisName
    unit: Literal["deg", "m"]
    value: float | None = None
    std: float | None = Field(default=None, ge=0.0)
    status: EstimateAxisStatus
    reason: str | None = None


class EstimatePairRecord(StrictModel):
    """One sensor pair and what the bag's data say about its extrinsic."""

    pair: CheckPairName
    sensors: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    frames: list[str] = Field(default_factory=list)
    status: EstimatePairStatus
    reason_code: CheckReasonCode | None = None
    reason: str | None = None
    estimator: str | None = None
    estimator_policy_status: str | None = None
    estimator_policy_reasons: list[str] = Field(default_factory=list)
    transform: CheckTransform | None = Field(
        default=None,
        description=(
            "the estimate in the estimator's convention; axes that are not 'observed' hold "
            "the value named by 'fill' and are not measurements"
        ),
    )
    fill: EstimateFill | None = Field(
        default=None,
        description="where the non-observed axes of 'transform' come from: the --tf prior "
        "(rough, user supplied) or the identity placeholder",
    )
    axes: list[EstimateAxis] = Field(default_factory=list)
    time_offset: CheckTimeOffset | None = None
    deskew: Literal["gyro", "constant_velocity", "none"] | None = None
    evidence: list[CheckEvidenceRef] = Field(
        default_factory=list,
        description="estimator artifacts; their reference fields compare against the "
        "placeholder or prior, not against a deployed calibration, and are not meaningful",
    )
    evidence_from_cache: bool | None = None
    runtime_s: float | None = Field(default=None, ge=0.0)
    notes: list[str] = Field(default_factory=list)


class EstimateFrameEntry(StrictModel):
    """One frame written to the exported tree (``T_parent_frame``)."""

    frame: str
    parent: str
    pair: CheckPairName
    axes_from_prior: list[CheckAxisName] = Field(
        default_factory=list,
        description="axes the data did not observe, taken from the --tf prior",
    )
    transform: CheckTransform


class EstimateFrameOmission(StrictModel):
    """A frame left out of the exported tree because no complete, measured edge exists."""

    frame: str
    pair: CheckPairName | None = None
    reason: str


class EstimateFrames(StrictModel):
    """The exported static frame tree."""

    root: str | None = None
    entries: list[EstimateFrameEntry] = Field(default_factory=list)
    omitted: list[EstimateFrameOmission] = Field(default_factory=list)


class EstimateExportFile(StrictModel):
    """One file derived from the estimate."""

    kind: EstimateExportKind
    path: str = Field(description="relative to the estimate artifact's directory")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EstimateOptions(StrictModel):
    """Estimator settings used for the run."""

    max_duration_s: float | None = None
    gnss_max_duration_s: float | None = None
    pairs: list[str] | None = None
    camera: str | None = None
    imu_lidar_translation: bool = True
    acceleration_unit: Literal["mps2", "g"] = "mps2"
    imu_lidar_deskew: Literal["auto", "gyro", "none"] = "auto"
    lidar_lidar_deskew: Literal["auto", "constant_velocity", "none"] = "auto"
    gnss_lidar_deskew: Literal["auto", "constant_velocity", "none"] = "auto"
    topic_kinds: dict[str, str] = Field(default_factory=dict)
    frame_map: dict[str, str] = Field(default_factory=dict)


class EstimateSummary(StrictModel):
    """Counts over the pair records."""

    pair_count: int = Field(ge=0)
    status_counts: dict[str, int] = Field(default_factory=dict)
    skipped_by_reason: dict[str, int] = Field(default_factory=dict)
    frames_written: int = Field(ge=0)
    frames_omitted: int = Field(ge=0)


class BagEstimateArtifact(StrictModel):
    """Calibration estimated from a bag, with its observability and provenance."""

    schema_version: Literal["slac.bag_estimate/v0.1"] = BAG_ESTIMATE_SCHEMA_VERSION
    bag: CheckBagInput
    vehicle_frame: str | None = None
    prior_sources: list[CheckCandidateSource] = Field(
        default_factory=list,
        description="rough priors (--tf files and the bag's /tf_static) used only to start "
        "estimators that need an initial guess and to fill axes the data did not observe",
    )
    topics: list[CheckTopicRecord] = Field(default_factory=list)
    pairs: list[EstimatePairRecord] = Field(default_factory=list)
    frames: EstimateFrames = Field(default_factory=EstimateFrames)
    exports: list[EstimateExportFile] = Field(default_factory=list)
    summary: EstimateSummary
    options: EstimateOptions
    evidence_dir: str | None = None
    next_steps: list[str] = Field(default_factory=list)
    provenance: CheckProvenance


def bag_estimate_json_schema() -> dict[str, object]:
    """Return the standalone JSON schema for bag estimate artifacts."""

    return BagEstimateArtifact.model_json_schema()
