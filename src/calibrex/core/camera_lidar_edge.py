"""Schema-valid targetless camera-LiDAR extrinsic evidence from edge alignment.

The extrinsic is ``T_camera_lidar`` (``T_parent_child``): it maps LiDAR points
into the camera's optical frame (``x`` right, ``y`` down, ``z`` forward).
Rotation DoFs are reported as the components of the rotation vector of its
rotation about the *camera* axes; their standard deviations and differences to a
reference are small rotations about those axes.  Translation DoFs are the camera
frame components of the translation.

Every DoF carries a held-out known-bad control: the objective at the estimate
must beat the objective at the estimate shifted by a known amount, on held-out
blocks of frames.  A DoF whose control is not detected is not trusted.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.result import StrictModel

CAMERA_LIDAR_EDGE_SCHEMA_VERSION: Literal["slac.camera_lidar_edge/v0.1"] = (
    "slac.camera_lidar_edge/v0.1"
)
CameraLidarEdgeDofName = Literal["roll", "pitch", "yaw", "x", "y", "z"]
CameraLidarEdgePolicyStatus = Literal["pass", "warn", "fail", "inconclusive"]


class CameraLidarEdgeProvenance(StrictModel):
    """Inputs and generator lineage."""

    generator: str
    generator_version: str
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    dataset_family: str
    sequence_ids: list[str] = Field(min_length=1)
    image_topic: str
    lidar_topic: str
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_digest_scope: str
    dataset_license: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class CameraLidarEdgeControl(StrictModel):
    """Held-out objective drop after shifting one DoF of the fit in both directions."""

    amount: float
    unit: Literal["deg", "m"]
    held_out_blocks: int = Field(ge=0)
    detected_blocks: int = Field(ge=0)
    mean_objective_drop: float
    min_t_statistic: float | None = Field(
        default=None,
        description="smaller of the two directions' one-sided t statistics of the per-block "
        "objective drops",
    )
    detected: bool


class CameraLidarEdgeDofRecord(StrictModel):
    """One DoF of ``T_camera_lidar`` with its uncertainty and evidence."""

    name: CameraLidarEdgeDofName
    unit: Literal["deg", "m"]
    value: float
    delta_from_start: float = Field(
        description="estimate minus the start (candidate) in the DoF's unit; rotation DoFs are "
        "the rotation vector of the left correction in the camera frame"
    )
    std_jackknife: float = Field(ge=0.0)
    std_reported: float = Field(ge=0.0)
    status: Literal["estimated", "unobservable"]
    at_search_bound: bool = False
    reference_value: float | None = None
    error_to_reference: float | None = None
    known_bad_control: CameraLidarEdgeControl | None = None

    @model_validator(mode="after")
    def check_reported_std(self) -> CameraLidarEdgeDofRecord:
        """Never report less uncertainty than the jackknife provides."""

        if self.std_reported + 1e-15 < self.std_jackknife:
            raise ValueError("std_reported must not be below the jackknife std")
        return self


class CameraLidarEdgeTransform(StrictModel):
    """``T_camera_lidar`` as translation and quaternion."""

    parent_frame: str
    child_frame: str
    translation_m: list[float] = Field(min_length=3, max_length=3)
    rotation_quat_xyzw: list[float] = Field(min_length=4, max_length=4)


class CameraLidarEdgeSamples(StrictModel):
    """The synchronized frames the estimate rests on."""

    frames: int = Field(ge=0)
    edge_points: int = Field(ge=0)
    blocks: int = Field(ge=0)
    median_abs_time_gap_s: float | None = Field(default=None, ge=0.0)
    max_abs_time_gap_s: float | None = Field(default=None, ge=0.0)
    first_frame_time_s: float | None = None
    last_frame_time_s: float | None = None
    median_rings: int | None = Field(default=None, ge=0)


class CameraLidarEdgeObjective(StrictModel):
    """The edge-alignment objective at the start and at the estimate."""

    start: float
    estimate: float
    gain: float


class CameraLidarEdgeArtifact(StrictModel):
    """Evidence for one targetless camera-LiDAR extrinsic estimate."""

    schema_version: Literal["slac.camera_lidar_edge/v0.1"] = CAMERA_LIDAR_EDGE_SCHEMA_VERSION
    method: Literal["edge_alignment/v0.1"] = "edge_alignment/v0.1"
    solver_status: Literal["converged", "max_iterations", "insufficient_samples"]
    policy_status: CameraLidarEdgePolicyStatus
    policy_reasons: list[str] = Field(min_length=1)
    calibrated_dofs: list[CameraLidarEdgeDofName]
    transform: CameraLidarEdgeTransform | None = None
    start_transform: CameraLidarEdgeTransform | None = None
    dofs: list[CameraLidarEdgeDofRecord]
    objective: CameraLidarEdgeObjective | None = None
    samples: CameraLidarEdgeSamples
    jackknife_fits: int = Field(ge=0)
    camera: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any]
    reference: str | None = None
    limitations: list[str] = Field(default_factory=list)
    provenance: CameraLidarEdgeProvenance

    @model_validator(mode="after")
    def check_policy(self) -> CameraLidarEdgeArtifact:
        """Keep calibrated DoFs and the pass verdict honest."""

        estimated = sorted(item.name for item in self.dofs if item.status == "estimated")
        if sorted(self.calibrated_dofs) != estimated:
            raise ValueError("calibrated_dofs must list exactly the estimated DoFs")
        rotation = {"roll", "pitch", "yaw"}
        if self.policy_status == "pass" and not rotation <= set(estimated):
            raise ValueError("a pass verdict requires all three rotation DoFs to be estimated")
        if self.solver_status == "insufficient_samples" and self.policy_status in {"pass", "warn"}:
            raise ValueError("an unsolved run cannot pass or warn")
        return self

    def save(self, path: str | Path) -> None:
        """Save the artifact as YAML or JSON."""

        write_mapping(Path(path), self.model_dump(mode="json", exclude_none=True))


def camera_lidar_edge_json_schema() -> dict[str, Any]:
    """Return the camera-LiDAR edge artifact JSON schema."""

    return CameraLidarEdgeArtifact.model_json_schema()


def load_camera_lidar_edge(path: str | Path) -> CameraLidarEdgeArtifact:
    """Load and validate a camera-LiDAR edge artifact."""

    return CameraLidarEdgeArtifact.model_validate(read_mapping(Path(path)))
