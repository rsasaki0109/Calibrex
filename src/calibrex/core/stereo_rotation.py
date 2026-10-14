"""Schema-valid evidence for a stereo pair's relative rotation, from its images alone.

A rectified pair puts every point on the same row in both images. When the candidate
rotation between the two cameras is wrong, the right image's rows shift by a pattern
that names the error: a constant for a rotation about the baseline, a term in the
column for one about the optical axis, and a term in column times row for one about
the vertical axis. :mod:`calibrex.evaluation.stereo_rotation` fits that pattern on
matched patches; this artifact records the fit, its block-jackknife uncertainty and a
control per axis.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.result import StrictModel

STEREO_ROTATION_SCHEMA_VERSION: Literal["slac.stereo_rotation/v0.1"] = "slac.stereo_rotation/v0.1"


class StereoRotationProvenance(StrictModel):
    """Inputs and generator lineage."""

    generator: str
    generator_version: str
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    left_topic: str
    right_topic: str
    left_frame: str
    right_frame: str
    intrinsics_source: str
    bag_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class StereoRotationAxis(StrictModel):
    """The candidate's rotation error about one axis of the left camera."""

    name: Literal["roll", "pitch", "yaw"]
    candidate_error_deg: float = Field(
        description="rotvec(R_candidate R_estimate^T) about the left camera's axis"
    )
    std_deg: float = Field(ge=0.0, description="block-jackknife standard deviation")
    control_response: float | None = Field(
        default=None,
        description="change of the error when the candidate turns by control_deg about this "
        "axis, over control_deg (1 when the axis is fully observed)",
    )
    observable: bool
    reason: str


class StereoRotationArtifact(StrictModel):
    """The relative rotation error of a stereo pair's candidate, with evidence."""

    schema_version: Literal["slac.stereo_rotation/v0.1"] = STEREO_ROTATION_SCHEMA_VERSION
    method: Literal["rectified_row_offsets/v0.1"] = "rectified_row_offsets/v0.1"
    policy_status: Literal["pass", "warn", "fail", "inconclusive"]
    policy_reasons: list[str] = Field(min_length=1)
    frames: int = Field(ge=0)
    patches: int = Field(ge=0)
    patches_constraining_rows: int = Field(ge=0)
    baseline_m: float = Field(ge=0.0)
    rectified_focal_px: float = Field(gt=0.0)
    rectified_from_left: list[list[float]] = Field(min_length=3, max_length=3)
    axes: list[StereoRotationAxis] = Field(default_factory=list)
    vertical_baseline_ratio: float | None = Field(
        default=None,
        description="row offset per pixel of disparity (a nuisance term: the baseline's "
        "vertical component over its length, in the rectified frame)",
    )
    residual_std_px: float | None = Field(default=None, ge=0.0)
    row_scale_ratio: float | None = Field(
        default=None,
        description="row offset per pixel of row (a nuisance term: the right rectified "
        "image's vertical scale over the left's, minus one; a focal length or distortion "
        "slightly off in either camera)",
    )
    row_scale_ratio_std: float | None = Field(default=None, ge=0.0)
    time_offset_s: float | None = Field(
        default=None,
        description="the right camera's clock minus the left one's: a pair stamped g apart "
        "was taken g + time_offset_s apart (only with the rig's rotation from a gyro)",
    )
    time_offset_std_s: float | None = Field(default=None, ge=0.0)
    time_offset_status: Literal["estimated", "unobservable"] | None = None
    timing_source: str | None = None
    max_stamp_gap_s: float | None = Field(default=None, ge=0.0)
    recentred_by_deg: list[float] | None = Field(
        default=None,
        min_length=3,
        max_length=3,
        description="the turn (about the left camera's axes, the candidate_error convention) by "
        "which the rectification was re-centred before the final measurement, when the "
        "candidate was too far off for the row search; candidate_error_deg includes it",
    )
    options: dict[str, Any]
    limitations: list[str] = Field(default_factory=list)
    provenance: StereoRotationProvenance

    def save(self, path: str | Path) -> None:
        """Save the artifact as YAML or JSON."""

        write_mapping(Path(path), self.model_dump(mode="json", exclude_none=True))


def stereo_rotation_json_schema() -> dict[str, Any]:
    """Return the stereo rotation artifact JSON schema."""

    return StereoRotationArtifact.model_json_schema()


def load_stereo_rotation(path: str | Path) -> StereoRotationArtifact:
    """Load and validate a stereo rotation artifact."""

    return StereoRotationArtifact.model_validate(read_mapping(Path(path)))
