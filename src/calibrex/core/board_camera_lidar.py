"""Schema-valid evidence for a camera-LiDAR extrinsic measured with a checkerboard.

:mod:`calibrex.evaluation.board_camera_lidar` solves ``T_camera_lidar`` from the board's
plane as the camera and the LiDAR see it at several still poses
(:mod:`calibrex.evaluation.board_camera_lidar_bag` finds the poses in a bag). This artifact
records the board, the poses found and used, the transform, and each axis with its
leave-one-pose-out std and whether the poses observed it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.result import StrictModel

BOARD_CAMERA_LIDAR_SCHEMA_VERSION: Literal["slac.board_camera_lidar/v0.1"] = (
    "slac.board_camera_lidar/v0.1"
)
BoardAxisName = Literal["roll", "pitch", "yaw", "x", "y", "z"]


class BoardSpec(StrictModel):
    """The checkerboard: inner corners and square size."""

    columns: int = Field(ge=2)
    rows: int = Field(ge=2)
    square_m: float = Field(gt=0.0)


class BoardCameraLidarTransform(StrictModel):
    """``T_camera_lidar`` as translation and quaternion."""

    parent_frame: str
    child_frame: str
    translation_m: list[float] = Field(min_length=3, max_length=3)
    rotation_quat_xyzw: list[float] = Field(min_length=4, max_length=4)


class BoardCameraLidarAxis(StrictModel):
    """One axis of ``T_camera_lidar``."""

    name: BoardAxisName
    unit: Literal["deg", "m"]
    value: float = Field(
        description="rotations: the rotation vector of T_camera_lidar about the camera's axes; "
        "translations: the camera-frame translation"
    )
    std: float | None = Field(
        default=None, ge=0.0, description="leave-one-pose-out (jackknife) standard deviation"
    )
    observed: bool
    reason: str


class BoardCameraLidarPose(StrictModel):
    """One still board pose found in the recording."""

    pose_id: str
    start_s: float | None = None
    end_s: float | None = None
    detections: int | None = Field(default=None, ge=0)
    scans: int | None = Field(default=None, ge=0)
    lidar_points: int = Field(ge=0)
    distance_m: float = Field(ge=0.0, description="the board's distance from the camera")
    reprojection_px: float = Field(ge=0.0)
    used: bool
    reason: str | None = None
    normal_residual_deg: float | None = Field(
        default=None,
        ge=0.0,
        description="angle between the camera's board normal and the LiDAR's, mapped with the "
        "estimate",
    )
    offset_residual_m: float | None = Field(
        default=None,
        description="the LiDAR plane's offset minus the camera's, in the camera frame, mapped "
        "with the estimate",
    )


class BoardCameraLidarProvenance(StrictModel):
    """Inputs and generator lineage."""

    generator: str
    generator_version: str
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    image_topic: str | None = None
    lidar_topic: str | None = None
    intrinsics_source: str | None = None
    input_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class BoardCameraLidarArtifact(StrictModel):
    """``T_camera_lidar`` from checkerboard planes, with per-axis observability."""

    schema_version: Literal["slac.board_camera_lidar/v0.1"] = BOARD_CAMERA_LIDAR_SCHEMA_VERSION
    method: Literal["planar_board/v0.1"] = "planar_board/v0.1"
    solver_status: str = Field(
        description="converged, insufficient_poses, or the plane solver's own status"
    )
    policy_status: Literal["pass", "warn", "inconclusive"]
    policy_reasons: list[str] = Field(min_length=1)
    board: BoardSpec
    transform: BoardCameraLidarTransform | None = None
    start_transform: BoardCameraLidarTransform
    axes: list[BoardCameraLidarAxis] = Field(default_factory=list)
    poses: list[BoardCameraLidarPose] = Field(default_factory=list)
    images: int | None = Field(default=None, ge=0)
    images_searched: int | None = Field(default=None, ge=0)
    images_with_board: int | None = Field(default=None, ge=0)
    normal_singular_values: list[float] | None = Field(
        default=None,
        min_length=3,
        max_length=3,
        description="singular values of the weighted camera-frame board normals",
    )
    unobserved_directions: list[list[float]] = Field(
        default_factory=list,
        description="camera-frame translation directions the board normals did not fix; the "
        "translation along them is the start's",
    )
    plane_normal_rmse_deg: float | None = Field(default=None, ge=0.0)
    plane_offset_rmse_m: float | None = Field(default=None, ge=0.0)
    camera: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    provenance: BoardCameraLidarProvenance

    @model_validator(mode="after")
    def check_policy(self) -> BoardCameraLidarArtifact:
        """Keep the verdict consistent with the axes."""

        observed = [axis for axis in self.axes if axis.observed]
        if self.transform is None and (observed or self.policy_status != "inconclusive"):
            raise ValueError("an unsolved run observes no axis and is inconclusive")
        if self.policy_status == "pass" and len(observed) != 6:
            raise ValueError("a pass verdict requires all six axes to be observed")
        return self

    def save(self, path: str | Path) -> None:
        """Save the artifact as YAML or JSON."""

        write_mapping(Path(path), self.model_dump(mode="json", exclude_none=True))


def board_camera_lidar_json_schema() -> dict[str, Any]:
    """Return the board camera-LiDAR artifact JSON schema."""

    return BoardCameraLidarArtifact.model_json_schema()


def load_board_camera_lidar(path: str | Path) -> BoardCameraLidarArtifact:
    """Load and validate a board camera-LiDAR artifact."""

    return BoardCameraLidarArtifact.model_validate(read_mapping(Path(path)))
