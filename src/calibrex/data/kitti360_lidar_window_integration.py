"""Provenance-locked KITTI-360 LiDAR window integration."""

from __future__ import annotations

import hashlib
import math
from bisect import bisect_left
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, field_validator, model_validator

from calibrex.core.geometry import (
    SE3,
    interpolate_se3,
    quaternion_xyzw_from_rotation_matrix,
)
from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import StrictModel
from calibrex.data.kitti360_camera_lidar_problem import (
    motion_compensate_kitti360_scan,
    read_kitti360_poses,
    read_kitti360_velodyne_to_pose_matrix,
)
from calibrex.data.remote_archive_selection import load_remote_archive_selection

FloatArray: TypeAlias = NDArray[np.float64]
Float32Array: TypeAlias = NDArray[np.float32]

KITTI360_LIDAR_WINDOW_INTEGRATION_SCHEMA_VERSION_V0_1: Literal[
    "slac.kitti360_lidar_window_integration/v0.1"
] = "slac.kitti360_lidar_window_integration/v0.1"
KITTI360_LIDAR_WINDOW_INTEGRATION_SCHEMA_VERSION: Literal[
    "slac.kitti360_lidar_window_integration/v0.2"
] = "slac.kitti360_lidar_window_integration/v0.2"
KITTI360_LIDAR_WINDOW_INTEGRATOR_VERSION = "calibrex.kitti360_lidar_window_integration/v0.2"


class Kitti360IntegrationFileReference(StrictModel):
    """Content-addressed regular file consumed by the integration."""

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class Kitti360LidarScanReference(Kitti360IntegrationFileReference):
    """One float32 XYZI scan bound to a numeric KITTI-360 frame."""

    frame_id: str = Field(pattern=r"^[0-9]{10}$")
    point_count: int = Field(gt=0)
    encoding: Literal["little_endian_float32_xyzi"] = "little_endian_float32_xyzi"

    @model_validator(mode="after")
    def check_binary_size(self) -> Kitti360LidarScanReference:
        """Require exactly four float32 values for every point."""

        if self.size_bytes != self.point_count * 4 * np.dtype("<f4").itemsize:
            raise ValueError("LiDAR scan size does not match float32 XYZI point count")
        return self


class Kitti360ResolvedPose(StrictModel):
    """Direct, interpolated, or bounded-extrapolated pose for one frame."""

    frame_id: str = Field(pattern=r"^[0-9]{10}$")
    method: Literal[
        "official",
        "bracketed_se3_interpolation",
        "bounded_endpoint_se3_extrapolation",
    ]
    left_official_frame_id: str | None = Field(default=None, pattern=r"^[0-9]{10}$")
    right_official_frame_id: str | None = Field(default=None, pattern=r"^[0-9]{10}$")
    interpolation_alpha: float | None = None
    translation_m: list[float] = Field(min_length=3, max_length=3)
    rotation_quat_xyzw: list[float] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def check_resolution(self) -> Kitti360ResolvedPose:
        """Bind interpolation metadata to the declared resolution method."""

        bracket = (
            self.left_official_frame_id,
            self.right_official_frame_id,
            self.interpolation_alpha,
        )
        if self.method == "official" and any(value is not None for value in bracket):
            raise ValueError("official poses must not declare interpolation metadata")
        if self.method == "bracketed_se3_interpolation":
            if any(value is None for value in bracket):
                raise ValueError("interpolated poses require both brackets and alpha")
            assert self.left_official_frame_id is not None
            assert self.right_official_frame_id is not None
            if not (
                int(self.left_official_frame_id)
                < int(self.frame_id)
                < int(self.right_official_frame_id)
            ):
                raise ValueError("pose interpolation brackets must contain the frame")
            assert self.interpolation_alpha is not None
            if not 0.0 < self.interpolation_alpha < 1.0:
                raise ValueError("pose interpolation alpha must be strictly internal")
        if self.method == "bounded_endpoint_se3_extrapolation":
            if any(value is None for value in bracket):
                raise ValueError("extrapolated poses require both endpoints and alpha")
            assert self.left_official_frame_id is not None
            assert self.right_official_frame_id is not None
            assert self.interpolation_alpha is not None
            left = int(self.left_official_frame_id)
            right = int(self.right_official_frame_id)
            frame = int(self.frame_id)
            if left >= right or not (frame < left or frame > right):
                raise ValueError("pose extrapolation frame must be outside its endpoints")
            if 0.0 <= self.interpolation_alpha <= 1.0:
                raise ValueError("pose extrapolation alpha must be outside [0, 1]")
        return self


class Kitti360LidarWindowIntegrationParameters(StrictModel):
    """Frozen geometric and filtering policy for window integration."""

    neighbor_radius_frames: int = Field(ge=0)
    maximum_pose_extrapolation_frames: int = Field(default=0, ge=0)
    output_coordinate_frame: Literal["center_velodyne"] = "center_velodyne"
    transform_convention: Literal[
        "T_center_velodyne_source_velodyne="
        "inverse(T_world_pose_center*T_pose_velodyne)*"
        "(T_world_pose_source*T_pose_velodyne)/v0.1"
    ] = (
        "T_center_velodyne_source_velodyne="
        "inverse(T_world_pose_center*T_pose_velodyne)*"
        "(T_world_pose_source*T_pose_velodyne)/v0.1"
    )
    motion_compensation: Literal["kitti360_azimuth_weighted_pose_delta/v0.1"] = (
        "kitti360_azimuth_weighted_pose_delta/v0.1"
    )
    pose_resolution: Literal[
        "official_or_bracketed_translation_lerp_quaternion_slerp/v0.1",
        "official_bracketed_or_bounded_endpoint_translation_lerp_quaternion_slerp/v0.2",
    ] = (
        "official_bracketed_or_bounded_endpoint_translation_lerp_quaternion_slerp/v0.2"
    )
    minimum_range_m: float = Field(ge=0.0)
    range_filter_frame: Literal["source_velodyne_after_motion_compensation"] = (
        "source_velodyne_after_motion_compensation"
    )
    voxel_resolution_m: float = Field(ge=0.0)
    voxel_reducer: Literal[
        "none",
        "sorted_voxel_centroid_xyz_mean_intensity/v0.1",
    ]
    output_encoding: Literal["little_endian_float32_xyzi"] = "little_endian_float32_xyzi"

    @model_validator(mode="after")
    def check_voxel_policy(self) -> Kitti360LidarWindowIntegrationParameters:
        """Keep the reducer declaration consistent with its resolution."""

        expected = (
            "none"
            if self.voxel_resolution_m == 0.0
            else "sorted_voxel_centroid_xyz_mean_intensity/v0.1"
        )
        if self.voxel_reducer != expected:
            raise ValueError("voxel reducer does not match voxel_resolution_m")
        return self


class Kitti360LidarIntegratedWindow(StrictModel):
    """Input coverage, point accounting, and output for one center frame."""

    center_frame_id: str = Field(pattern=r"^[0-9]{10}$")
    source_frame_ids: list[str] = Field(min_length=1)
    source_point_count: int = Field(gt=0)
    range_discarded_point_count: int = Field(ge=0)
    range_retained_point_count: int = Field(gt=0)
    voxel_discarded_point_count: int = Field(ge=0)
    output: Kitti360LidarScanReference

    @field_validator("source_frame_ids")
    @classmethod
    def check_source_ids(cls, value: list[str]) -> list[str]:
        normalized = [f"{int(frame_id):010d}" for frame_id in value]
        if value != normalized or value != sorted(set(value), key=int):
            raise ValueError("source frame IDs must be unique, normalized, and ordered")
        if any(int(right) != int(left) + 1 for left, right in pairwise(value)):
            raise ValueError("source frame IDs within a window must be contiguous")
        return value

    @model_validator(mode="after")
    def check_counts(self) -> Kitti360LidarIntegratedWindow:
        """Require loss accounting and center/output identity to close exactly."""

        if self.center_frame_id not in self.source_frame_ids:
            raise ValueError("each LiDAR window must contain its center frame")
        if self.output.frame_id != self.center_frame_id:
            raise ValueError("integrated output frame must equal the window center")
        if (
            self.range_retained_point_count + self.range_discarded_point_count
            != self.source_point_count
        ):
            raise ValueError("range-filter point accounting does not close")
        if (
            self.output.point_count + self.voxel_discarded_point_count
            != self.range_retained_point_count
        ):
            raise ValueError("voxel-filter point accounting does not close")
        return self


class Kitti360LidarWindowIntegrationProvenance(StrictModel):
    """Generator identity and immutable input digests."""

    generator: str
    generator_version: str
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    source_sha256: dict[str, str] = Field(min_length=4)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @field_validator("source_sha256")
    @classmethod
    def check_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        invalid = [key for key, digest in value.items() if not _is_sha256(digest)]
        if invalid:
            raise ValueError(f"source_sha256 contains invalid SHA-256 values: {invalid}")
        return value


class Kitti360LidarWindowIntegrationArtifact(StrictModel):
    """Schema-valid lineage for center-frame KITTI-360 LiDAR maps."""

    schema_version: Literal[
        "slac.kitti360_lidar_window_integration/v0.1",
        "slac.kitti360_lidar_window_integration/v0.2",
    ] = (
        KITTI360_LIDAR_WINDOW_INTEGRATION_SCHEMA_VERSION
    )
    artifact_id: str = Field(min_length=1)
    dataset_family: Literal["KITTI-360"] = "KITTI-360"
    dataset_id: str = Field(min_length=1)
    sequence_id: str = Field(min_length=1)
    dataset_license_spdx: str = Field(min_length=1)
    selection_manifest: Kitti360IntegrationFileReference
    poses_file: Kitti360IntegrationFileReference
    calibration_files: list[Kitti360IntegrationFileReference] = Field(min_length=2, max_length=2)
    center_frame_ids: list[str] = Field(min_length=1)
    parameters: Kitti360LidarWindowIntegrationParameters
    source_scans: list[Kitti360LidarScanReference] = Field(min_length=1)
    resolved_poses: list[Kitti360ResolvedPose] = Field(min_length=1)
    windows: list[Kitti360LidarIntegratedWindow] = Field(min_length=1)
    output_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance: Kitti360LidarWindowIntegrationProvenance

    @model_validator(mode="after")
    def check_coverage(self) -> Kitti360LidarWindowIntegrationArtifact:
        """Require exact, ordered source, pose, center, and output coverage."""

        if self.center_frame_ids != sorted(set(self.center_frame_ids), key=int):
            raise ValueError("center frame IDs must be unique and ordered")
        window_centers = [window.center_frame_id for window in self.windows]
        if window_centers != self.center_frame_ids:
            raise ValueError("window centers must exactly match center_frame_ids")
        source_ids = [scan.frame_id for scan in self.source_scans]
        if source_ids != sorted(set(source_ids), key=int):
            raise ValueError("source scans must be unique and ordered by frame ID")
        source_by_id = {scan.frame_id: scan for scan in self.source_scans}
        used_source_ids = {
            frame_id for window in self.windows for frame_id in window.source_frame_ids
        }
        if used_source_ids != set(source_ids):
            raise ValueError("window source coverage must exactly match source_scans")
        for window in self.windows:
            expected_ids = [
                frame_id
                for frame_id in source_ids
                if abs(int(frame_id) - int(window.center_frame_id))
                <= self.parameters.neighbor_radius_frames
            ]
            if window.source_frame_ids != expected_ids:
                raise ValueError("window source IDs do not match the declared radius")
            expected_points = sum(
                source_by_id[frame_id].point_count for frame_id in window.source_frame_ids
            )
            if window.source_point_count != expected_points:
                raise ValueError("window source point count differs from source scans")
        pose_ids = [pose.frame_id for pose in self.resolved_poses]
        if pose_ids != sorted(set(pose_ids), key=int):
            raise ValueError("resolved poses must be unique and ordered by frame ID")
        required_pose_ids = {
            adjacent
            for frame_id in source_ids
            for adjacent in (
                frame_id,
                _motion_compensation_adjacent_frame_id(frame_id),
            )
        }
        if not required_pose_ids.issubset(set(pose_ids)):
            raise ValueError("resolved poses do not cover source and adjacent frames")
        extrapolated = [
            pose
            for pose in self.resolved_poses
            if pose.method == "bounded_endpoint_se3_extrapolation"
        ]
        if self.schema_version.endswith("/v0.1"):
            if extrapolated or self.parameters.maximum_pose_extrapolation_frames != 0:
                raise ValueError("integration v0.1 cannot declare pose extrapolation")
            if self.parameters.pose_resolution != (
                "official_or_bracketed_translation_lerp_quaternion_slerp/v0.1"
            ):
                raise ValueError("integration v0.1 requires its original pose policy")
        else:
            if self.parameters.pose_resolution != (
                "official_bracketed_or_bounded_endpoint_translation_lerp_"
                "quaternion_slerp/v0.2"
            ):
                raise ValueError("integration v0.2 requires the bounded pose policy")
            for pose in extrapolated:
                assert pose.left_official_frame_id is not None
                assert pose.right_official_frame_id is not None
                endpoint_distance = min(
                    abs(int(pose.frame_id) - int(pose.left_official_frame_id)),
                    abs(int(pose.frame_id) - int(pose.right_official_frame_id)),
                )
                if endpoint_distance > self.parameters.maximum_pose_extrapolation_frames:
                    raise ValueError("resolved pose exceeds the extrapolation bound")
        outputs = [window.output for window in self.windows]
        if self.output_set_sha256 != _scan_reference_set_sha256(outputs):
            raise ValueError("output_set_sha256 does not match integrated outputs")
        calibration_names = {Path(item.path).name for item in self.calibration_files}
        if calibration_names != {"calib_cam_to_velo.txt", "calib_cam_to_pose.txt"}:
            raise ValueError("integration requires exact KITTI-360 pose-chain files")
        return self

    def save(self, path: str | Path) -> None:
        """Save this integration artifact as schema-valid YAML or JSON."""

        write_mapping(Path(path), self.model_dump(mode="json", exclude_none=True))


def integrate_kitti360_lidar_windows(
    selection_manifest_path: str | Path,
    poses_path: str | Path,
    calibration_root: str | Path,
    center_frame_ids: Sequence[str | int],
    *,
    neighbor_radius_frames: int,
    output_directory: str | Path,
    artifact_id: str | None = None,
    minimum_range_m: float = 1.0,
    voxel_resolution_m: float = 0.002,
    maximum_pose_extrapolation_frames: int = 0,
    generator: str = "calibrex.kitti360_lidar_window_integration",
    generator_version: str = KITTI360_LIDAR_WINDOW_INTEGRATOR_VERSION,
    command: Sequence[str] = (),
) -> Kitti360LidarWindowIntegrationArtifact:
    """Integrate centered, motion-compensated KITTI-360 LiDAR windows."""

    if neighbor_radius_frames < 0:
        raise ValueError("neighbor_radius_frames must be non-negative")
    if maximum_pose_extrapolation_frames < 0:
        raise ValueError("maximum_pose_extrapolation_frames must be non-negative")
    if minimum_range_m < 0.0 or voxel_resolution_m < 0.0:
        raise ValueError("range and voxel resolutions must be non-negative")
    centers = _normalize_frame_ids(center_frame_ids, label="center")
    selection_path = Path(selection_manifest_path).resolve()
    pose_path = Path(poses_path).resolve()
    calibration = Path(calibration_root).resolve()
    output_root = Path(output_directory).resolve()
    selection = load_remote_archive_selection(selection_path)
    if selection.dataset_family != "KITTI-360":
        raise ValueError("LiDAR integration requires a KITTI-360 selection")
    if selection.selection_policy != "centered_same_numeric_frame_windows/v0.2":
        raise ValueError("LiDAR integration requires a centered-window selection")
    selected_ids = selection.frame_ids
    selected_set = set(selected_ids)
    missing_centers = sorted(set(centers) - selected_set, key=int)
    if missing_centers:
        raise ValueError(f"center frames are absent from selection: {missing_centers}")
    source_ids = sorted(
        {
            frame_id
            for frame_id in selected_ids
            if any(abs(int(frame_id) - int(center)) <= neighbor_radius_frames for center in centers)
        },
        key=int,
    )
    windows_by_center = {
        center: [
            frame_id
            for frame_id in source_ids
            if abs(int(frame_id) - int(center)) <= neighbor_radius_frames
        ]
        for center in centers
    }
    _validate_window_contiguity(windows_by_center)

    pointcloud_members = {
        member.frame_id: member
        for member in selection.members
        if member.role == "pointcloud" and member.frame_id in source_ids
    }
    if set(pointcloud_members) != set(source_ids):
        raise ValueError("selection does not bind every source LiDAR scan")
    source_paths: dict[str, Path] = {}
    source_scans: list[Kitti360LidarScanReference] = []
    for frame_id in source_ids:
        member = pointcloud_members[frame_id]
        scan_path = _resolve_reference_path(member.local_path, selection_path.parent)
        digest = _required_file_sha256(scan_path)
        if digest != member.local_sha256 or scan_path.stat().st_size != member.local_size_bytes:
            raise ValueError(f"selected LiDAR scan differs from manifest: {frame_id}")
        point_count = _xyzi_point_count(scan_path)
        source_paths[frame_id] = scan_path
        source_scans.append(
            Kitti360LidarScanReference(
                frame_id=frame_id,
                path=str(scan_path),
                sha256=digest,
                size_bytes=scan_path.stat().st_size,
                point_count=point_count,
            )
        )

    calibration_paths = [
        calibration / "calib_cam_to_velo.txt",
        calibration / "calib_cam_to_pose.txt",
    ]
    input_references = {
        "selection_manifest": _file_reference(selection_path),
        "poses_file": _file_reference(pose_path),
        **{f"calibration:{path.name}": _file_reference(path) for path in calibration_paths},
    }
    official_poses = read_kitti360_poses(pose_path)
    needed_pose_ids = {int(frame_id) for frame_id in source_ids} | {
        int(_motion_compensation_adjacent_frame_id(frame_id)) for frame_id in source_ids
    }
    poses, resolved_poses = _resolve_pose_frames(
        official_poses,
        needed_pose_ids,
        maximum_extrapolation_frames=maximum_pose_extrapolation_frames,
    )
    velodyne_to_pose = read_kitti360_velodyne_to_pose_matrix(calibration)

    output_root.mkdir(parents=True, exist_ok=True)
    windows: list[Kitti360LidarIntegratedWindow] = []
    for center in centers:
        center_id = int(center)
        world_from_center = poses[center_id] @ velodyne_to_pose
        center_from_world = np.linalg.inv(world_from_center)
        integrated_parts: list[Float32Array] = []
        range_discarded = 0
        source_point_count = 0
        for source in windows_by_center[center]:
            source_id = int(source)
            points = _read_xyzi(source_paths[source])
            source_point_count += len(points)
            compensated = motion_compensate_kitti360_scan(
                points,
                frame_id=source_id,
                poses=poses,
                velodyne_to_pose=velodyne_to_pose,
            )
            ranges = np.linalg.norm(compensated[:, :3], axis=1)
            retained = compensated[ranges >= minimum_range_m]
            range_discarded += len(compensated) - len(retained)
            if len(retained) == 0:
                continue
            world_from_source = poses[source_id] @ velodyne_to_pose
            center_from_source = center_from_world @ world_from_source
            integrated_parts.append(_transform_xyzi(retained, center_from_source))
        if not integrated_parts:
            raise ValueError(f"range filtering removed every point for center {center}")
        integrated = np.concatenate(integrated_parts, axis=0)
        range_retained = len(integrated)
        output_points = _voxel_centroids(integrated, voxel_resolution_m)
        output_path = output_root / f"{center}.bin"
        output_reference = _write_xyzi_without_overwrite(output_path, output_points, center)
        windows.append(
            Kitti360LidarIntegratedWindow(
                center_frame_id=center,
                source_frame_ids=windows_by_center[center],
                source_point_count=source_point_count,
                range_discarded_point_count=range_discarded,
                range_retained_point_count=range_retained,
                voxel_discarded_point_count=range_retained - len(output_points),
                output=output_reference,
            )
        )

    parameters = Kitti360LidarWindowIntegrationParameters(
        neighbor_radius_frames=neighbor_radius_frames,
        maximum_pose_extrapolation_frames=maximum_pose_extrapolation_frames,
        minimum_range_m=minimum_range_m,
        voxel_resolution_m=voxel_resolution_m,
        voxel_reducer=(
            "none" if voxel_resolution_m == 0.0 else "sorted_voxel_centroid_xyz_mean_intensity/v0.1"
        ),
    )
    calibration_references = [
        input_references[f"calibration:{path.name}"] for path in calibration_paths
    ]
    outputs = [window.output for window in windows]
    return Kitti360LidarWindowIntegrationArtifact(
        artifact_id=artifact_id
        or f"kitti360-{selection.sequence_id}-lidar-window-r{neighbor_radius_frames}",
        dataset_id=selection.dataset_id,
        sequence_id=selection.sequence_id,
        dataset_license_spdx=selection.dataset_license_spdx,
        selection_manifest=input_references["selection_manifest"],
        poses_file=input_references["poses_file"],
        calibration_files=calibration_references,
        center_frame_ids=centers,
        parameters=parameters,
        source_scans=source_scans,
        resolved_poses=resolved_poses,
        windows=windows,
        output_set_sha256=_scan_reference_set_sha256(outputs),
        provenance=Kitti360LidarWindowIntegrationProvenance(
            generator=generator,
            generator_version=generator_version,
            git_commit=git_commit(),
            command=list(command),
            source_sha256={key: reference.sha256 for key, reference in input_references.items()},
        ),
    )


def load_kitti360_lidar_window_integration(
    path: str | Path,
) -> Kitti360LidarWindowIntegrationArtifact:
    """Load and validate a KITTI-360 LiDAR window integration artifact."""

    return Kitti360LidarWindowIntegrationArtifact.model_validate(read_mapping(Path(path)))


def verify_kitti360_lidar_window_integration_files(
    artifact: Kitti360LidarWindowIntegrationArtifact,
    *,
    base_path: str | Path | None = None,
) -> list[str]:
    """Return deterministic missing, size, or digest issues for bound files."""

    root = Path(base_path) if base_path is not None else None
    references: list[Kitti360IntegrationFileReference] = [
        artifact.selection_manifest,
        artifact.poses_file,
        *artifact.calibration_files,
        *artifact.source_scans,
        *(window.output for window in artifact.windows),
    ]
    issues: list[str] = []
    for reference in references:
        path = Path(reference.path)
        if root is not None and not path.is_absolute():
            path = root / path
        digest = sha256_path(path)
        if digest is None:
            issues.append(f"missing:{reference.path}")
            continue
        if not path.is_file():
            issues.append(f"not_file:{reference.path}")
            continue
        if path.stat().st_size != reference.size_bytes:
            issues.append(f"size_mismatch:{reference.path}")
        if digest != reference.sha256:
            issues.append(f"digest_mismatch:{reference.path}")
    return issues


def kitti360_lidar_window_integration_json_schema() -> dict[str, Any]:
    """Return the standalone JSON Schema for LiDAR window integrations."""

    return Kitti360LidarWindowIntegrationArtifact.model_json_schema()


def _normalize_frame_ids(
    frame_ids: Sequence[str | int],
    *,
    label: str,
) -> list[str]:
    if not frame_ids:
        raise ValueError(f"at least one {label} frame ID is required")
    normalized: list[str] = []
    for frame_id in frame_ids:
        raw = str(frame_id).strip()
        if not raw.isdigit() or int(raw) < 0:
            raise ValueError(f"invalid {label} frame ID: {frame_id!r}")
        normalized.append(f"{int(raw):010d}")
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{label} frame IDs must be unique")
    return sorted(normalized, key=int)


def _validate_window_contiguity(windows: dict[str, list[str]]) -> None:
    for center, frame_ids in windows.items():
        if center not in frame_ids:
            raise ValueError(f"window does not contain center {center}")
        if any(int(right) != int(left) + 1 for left, right in pairwise(frame_ids)):
            raise ValueError(f"selection has an internal gap around center {center}")


def _motion_compensation_adjacent_frame_id(frame_id: str) -> str:
    value = int(frame_id)
    adjacent = value + 1 if value in (0, 1) else value - 1
    if adjacent < 0:
        raise ValueError(f"frame {frame_id} has no valid motion-compensation neighbor")
    return f"{adjacent:010d}"


def _resolve_pose_frames(
    official_poses: dict[int, FloatArray],
    requested_frame_ids: Iterable[int],
    *,
    maximum_extrapolation_frames: int,
) -> tuple[dict[int, FloatArray], list[Kitti360ResolvedPose]]:
    if not official_poses:
        raise ValueError("KITTI-360 pose file is empty")
    official_ids = sorted(official_poses)
    resolved: dict[int, FloatArray] = {}
    records: list[Kitti360ResolvedPose] = []
    for frame_id in sorted(set(requested_frame_ids)):
        if frame_id in official_poses:
            matrix = np.asarray(official_poses[frame_id], dtype=np.float64)
            method: Literal[
                "official",
                "bracketed_se3_interpolation",
                "bounded_endpoint_se3_extrapolation",
            ] = "official"
            left_id = None
            right_id = None
            alpha = None
        else:
            position = bisect_left(official_ids, frame_id)
            outside = position == 0 or position == len(official_ids)
            if outside:
                if len(official_ids) < 2:
                    raise ValueError("at least two official poses are needed for extrapolation")
                endpoint = official_ids[0] if position == 0 else official_ids[-1]
                distance = abs(frame_id - endpoint)
                if distance > maximum_extrapolation_frames:
                    raise ValueError(
                        f"pose frame {frame_id} is {distance} frames outside official "
                        f"coverage; allowed {maximum_extrapolation_frames}"
                    )
                left, right = (
                    (official_ids[0], official_ids[1])
                    if position == 0
                    else (official_ids[-2], official_ids[-1])
                )
            else:
                left = official_ids[position - 1]
                right = official_ids[position]
            alpha_value = (frame_id - left) / (right - left)
            transform = (
                _extrapolate_se3(
                    _matrix_to_se3(official_poses[left]),
                    _matrix_to_se3(official_poses[right]),
                    alpha_value,
                )
                if outside
                else interpolate_se3(
                    _matrix_to_se3(official_poses[left]),
                    _matrix_to_se3(official_poses[right]),
                    alpha_value,
                )
            )
            matrix = _se3_to_matrix(transform)
            method = (
                "bounded_endpoint_se3_extrapolation"
                if outside
                else "bracketed_se3_interpolation"
            )
            left_id = f"{left:010d}"
            right_id = f"{right:010d}"
            alpha = float(alpha_value)
        transform = _matrix_to_se3(matrix)
        resolved[frame_id] = matrix
        records.append(
            Kitti360ResolvedPose(
                frame_id=f"{frame_id:010d}",
                method=method,
                left_official_frame_id=left_id,
                right_official_frame_id=right_id,
                interpolation_alpha=alpha,
                translation_m=list(transform.translation_m),
                rotation_quat_xyzw=list(transform.rotation_quat_xyzw),
            )
        )
    return resolved, records


def _extrapolate_se3(left: SE3, right: SE3, alpha: float) -> SE3:
    """Extrapolate translation and shortest-arc quaternion without clamping."""

    if 0.0 <= alpha <= 1.0:
        raise ValueError("endpoint extrapolation requires alpha outside [0, 1]")
    left_q = left.rotation_quat_xyzw
    right_q = right.rotation_quat_xyzw
    dot = sum(a * b for a, b in zip(left_q, right_q, strict=True))
    if dot < 0.0:
        right_q = (
            -right_q[0],
            -right_q[1],
            -right_q[2],
            -right_q[3],
        )
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 1.0 - 1.0e-8:
        quaternion = (
            left_q[0] + alpha * (right_q[0] - left_q[0]),
            left_q[1] + alpha * (right_q[1] - left_q[1]),
            left_q[2] + alpha * (right_q[2] - left_q[2]),
            left_q[3] + alpha * (right_q[3] - left_q[3]),
        )
    else:
        theta = math.acos(dot)
        denominator = math.sin(theta)
        left_weight = math.sin((1.0 - alpha) * theta) / denominator
        right_weight = math.sin(alpha * theta) / denominator
        quaternion = (
            left_weight * left_q[0] + right_weight * right_q[0],
            left_weight * left_q[1] + right_weight * right_q[1],
            left_weight * left_q[2] + right_weight * right_q[2],
            left_weight * left_q[3] + right_weight * right_q[3],
        )
    translation = (
        left.translation_m[0]
        + alpha * (right.translation_m[0] - left.translation_m[0]),
        left.translation_m[1]
        + alpha * (right.translation_m[1] - left.translation_m[1]),
        left.translation_m[2]
        + alpha * (right.translation_m[2] - left.translation_m[2]),
    )
    return SE3(translation_m=translation, rotation_quat_xyzw=quaternion)


def _matrix_to_se3(matrix: FloatArray) -> SE3:
    values = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    return SE3(
        translation_m=(
            float(values[0, 3]),
            float(values[1, 3]),
            float(values[2, 3]),
        ),
        rotation_quat_xyzw=quaternion_xyzw_from_rotation_matrix(values[:3, :3].reshape(-1)),
    )


def _se3_to_matrix(transform: SE3) -> FloatArray:
    x, y, z, w = transform.rotation_quat_xyzw
    rotation = np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    matrix: FloatArray = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(transform.translation_m, dtype=np.float64)
    return matrix


def _read_xyzi(path: Path) -> Float32Array:
    point_count = _xyzi_point_count(path)
    points = np.fromfile(path, dtype="<f4").reshape(point_count, 4)
    if not np.all(np.isfinite(points)):
        raise ValueError(f"KITTI-360 scan contains non-finite values: {path}")
    return np.asarray(points, dtype=np.float32)


def _xyzi_point_count(path: Path) -> int:
    if not path.is_file():
        raise ValueError(f"KITTI-360 scan is not a readable file: {path}")
    item_size = 4 * np.dtype("<f4").itemsize
    size = path.stat().st_size
    if size == 0 or size % item_size != 0:
        raise ValueError(f"KITTI-360 scan is not packed float32 XYZI: {path}")
    return size // item_size


def _transform_xyzi(points: Float32Array, transform: FloatArray) -> Float32Array:
    values = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    output = values.copy()
    output[:, :3] = values[:, :3] @ matrix[:3, :3].T + matrix[:3, 3]
    return output.astype(np.float32)


def _voxel_centroids(points: Float32Array, resolution_m: float) -> Float32Array:
    if resolution_m == 0.0:
        return np.asarray(points, dtype=np.float32)
    keys = np.floor(np.asarray(points[:, :3], dtype=np.float64) / resolution_m).astype(np.int64)
    _unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    sums: FloatArray = np.zeros((len(counts), 4), dtype=np.float64)
    np.add.at(sums, inverse, np.asarray(points, dtype=np.float64))
    return (sums / counts[:, None]).astype(np.float32)


def _write_xyzi_without_overwrite(
    path: Path,
    points: Float32Array,
    frame_id: str,
) -> Kitti360LidarScanReference:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.calibrex-partial")
    partial.unlink(missing_ok=True)
    np.asarray(points, dtype="<f4").tofile(partial)
    partial_digest = _required_file_sha256(partial)
    partial_size = partial.stat().st_size
    if path.exists():
        if path.stat().st_size != partial_size or _required_file_sha256(path) != partial_digest:
            partial.unlink(missing_ok=True)
            raise ValueError(f"existing integrated output differs; refusing overwrite: {path}")
        partial.unlink(missing_ok=True)
    else:
        partial.replace(path)
    return Kitti360LidarScanReference(
        frame_id=frame_id,
        path=str(path),
        sha256=partial_digest,
        size_bytes=partial_size,
        point_count=len(points),
    )


def _resolve_reference_path(raw_path: str, base_path: Path) -> Path:
    path = Path(raw_path)
    return (path if path.is_absolute() else base_path / path).resolve()


def _file_reference(path: Path) -> Kitti360IntegrationFileReference:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"required integration input is not a file: {resolved}")
    return Kitti360IntegrationFileReference(
        path=str(resolved),
        sha256=_required_file_sha256(resolved),
        size_bytes=resolved.stat().st_size,
    )


def _required_file_sha256(path: Path) -> str:
    digest = sha256_path(path)
    if digest is None or not path.is_file():
        raise ValueError(f"required file is not readable: {path}")
    return digest


def _scan_reference_set_sha256(
    references: Sequence[Kitti360LidarScanReference],
) -> str:
    digest = hashlib.sha256()
    for reference in sorted(references, key=lambda item: int(item.frame_id)):
        digest.update(reference.frame_id.encode("ascii"))
        digest.update(b"\0")
        digest.update(reference.sha256.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(reference.size_bytes).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(reference.point_count).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)
