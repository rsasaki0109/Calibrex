"""Provenance-locked KITTI raw LiDAR/OXTS window integration."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, field_validator, model_validator

from calibrex.core.geometry import SE3
from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import StrictModel
from calibrex.data.kitti import (
    KITTIOXTSPacket,
    kitti_oxts_pose_world_imu,
    read_calibration_file,
    read_oxts_packet,
)
from calibrex.data.remote_archive_selection import load_remote_archive_selection

FloatArray: TypeAlias = NDArray[np.float64]
Float32Array: TypeAlias = NDArray[np.float32]

KITTI_RAW_LIDAR_WINDOW_INTEGRATION_SCHEMA_VERSION: Literal[
    "slac.kitti_raw_lidar_window_integration/v0.1"
] = "slac.kitti_raw_lidar_window_integration/v0.1"
KITTI_RAW_LIDAR_WINDOW_INTEGRATOR_VERSION = (
    "calibrex.kitti_raw_lidar_window_integration/v0.1"
)


class KittiRawIntegrationFileReference(StrictModel):
    """Content-addressed regular file consumed by raw integration."""

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class KittiRawLidarScanReference(KittiRawIntegrationFileReference):
    """One packed little-endian float32 XYZI scan."""

    frame_id: str = Field(pattern=r"^[0-9]{10}$")
    point_count: int = Field(gt=0)
    encoding: Literal["little_endian_float32_xyzi"] = (
        "little_endian_float32_xyzi"
    )

    @model_validator(mode="after")
    def check_binary_size(self) -> KittiRawLidarScanReference:
        if self.size_bytes != self.point_count * 4 * np.dtype("<f4").itemsize:
            raise ValueError("raw LiDAR scan size differs from XYZI point count")
        return self


class KittiRawResolvedOxtsPose(KittiRawIntegrationFileReference):
    """Official OXTS packet and derived ``T_world_imu`` for one frame."""

    frame_id: str = Field(pattern=r"^[0-9]{10}$")
    method: Literal["official_oxts_mercator_rpy/v0.1"] = (
        "official_oxts_mercator_rpy/v0.1"
    )
    translation_m: list[float] = Field(min_length=3, max_length=3)
    rotation_quat_xyzw: list[float] = Field(min_length=4, max_length=4)


class KittiRawLidarWindowIntegrationParameters(StrictModel):
    """Frozen raw pose-chain, filtering, and voxel policy."""

    neighbor_radius_frames: int = Field(ge=0)
    pose_origin_frame_id: str = Field(pattern=r"^[0-9]{10}$")
    output_coordinate_frame: Literal["center_velodyne"] = "center_velodyne"
    transform_convention: Literal[
        "T_center_velodyne_source_velodyne="
        "inverse(T_world_imu_center*T_imu_velodyne)*"
        "(T_world_imu_source*T_imu_velodyne)/v0.1"
    ] = (
        "T_center_velodyne_source_velodyne="
        "inverse(T_world_imu_center*T_imu_velodyne)*"
        "(T_world_imu_source*T_imu_velodyne)/v0.1"
    )
    motion_compensation: Literal[
        "azimuth_weighted_oxts_pose_delta/v0.1"
    ] = "azimuth_weighted_oxts_pose_delta/v0.1"
    minimum_range_m: float = Field(ge=0.0)
    range_filter_frame: Literal[
        "source_velodyne_after_motion_compensation"
    ] = "source_velodyne_after_motion_compensation"
    voxel_resolution_m: float = Field(ge=0.0)
    voxel_reducer: Literal[
        "none", "sorted_voxel_centroid_xyz_mean_intensity/v0.1"
    ]
    output_encoding: Literal["little_endian_float32_xyzi"] = (
        "little_endian_float32_xyzi"
    )

    @model_validator(mode="after")
    def check_voxel_policy(self) -> KittiRawLidarWindowIntegrationParameters:
        expected = (
            "none"
            if self.voxel_resolution_m == 0.0
            else "sorted_voxel_centroid_xyz_mean_intensity/v0.1"
        )
        if self.voxel_reducer != expected:
            raise ValueError("raw voxel reducer differs from resolution")
        return self


class KittiRawLidarIntegratedWindow(StrictModel):
    """Point accounting and output for one raw center frame."""

    center_frame_id: str = Field(pattern=r"^[0-9]{10}$")
    source_frame_ids: list[str] = Field(min_length=1)
    source_point_count: int = Field(gt=0)
    range_discarded_point_count: int = Field(ge=0)
    range_retained_point_count: int = Field(gt=0)
    voxel_discarded_point_count: int = Field(ge=0)
    output: KittiRawLidarScanReference

    @field_validator("source_frame_ids")
    @classmethod
    def check_source_ids(cls, value: list[str]) -> list[str]:
        normalized = [f"{int(frame_id):010d}" for frame_id in value]
        if value != normalized or value != sorted(set(value), key=int):
            raise ValueError("raw window source IDs must be unique and ordered")
        if any(int(right) != int(left) + 1 for left, right in pairwise(value)):
            raise ValueError("raw window source IDs must be contiguous")
        return value

    @model_validator(mode="after")
    def check_counts(self) -> KittiRawLidarIntegratedWindow:
        if self.center_frame_id not in self.source_frame_ids:
            raise ValueError("raw window does not contain its center")
        if self.output.frame_id != self.center_frame_id:
            raise ValueError("raw window output frame differs from center")
        if (
            self.range_retained_point_count + self.range_discarded_point_count
            != self.source_point_count
        ):
            raise ValueError("raw range-filter point accounting does not close")
        if (
            self.output.point_count + self.voxel_discarded_point_count
            != self.range_retained_point_count
        ):
            raise ValueError("raw voxel point accounting does not close")
        return self


class KittiRawLidarWindowIntegrationProvenance(StrictModel):
    """Generator and immutable source digests for raw integration."""

    generator: str
    generator_version: str
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    source_sha256: dict[str, str] = Field(min_length=3)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @field_validator("source_sha256")
    @classmethod
    def check_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        invalid = [key for key, digest in value.items() if not _is_sha256(digest)]
        if invalid:
            raise ValueError(f"invalid raw integration source hashes: {invalid}")
        return value


class KittiRawLidarWindowIntegrationArtifact(StrictModel):
    """Schema-valid lineage for center-frame KITTI raw LiDAR maps."""

    schema_version: Literal[
        "slac.kitti_raw_lidar_window_integration/v0.1"
    ] = KITTI_RAW_LIDAR_WINDOW_INTEGRATION_SCHEMA_VERSION
    artifact_id: str
    dataset_family: Literal["KITTI raw"] = "KITTI raw"
    dataset_id: str
    sequence_id: str
    dataset_license_spdx: str
    selection_manifest: KittiRawIntegrationFileReference
    imu_to_velodyne_calibration: KittiRawIntegrationFileReference
    timestamp_files: list[KittiRawIntegrationFileReference] = Field(
        min_length=3, max_length=3
    )
    center_frame_ids: list[str] = Field(min_length=1)
    parameters: KittiRawLidarWindowIntegrationParameters
    source_scans: list[KittiRawLidarScanReference] = Field(min_length=1)
    resolved_poses: list[KittiRawResolvedOxtsPose] = Field(min_length=1)
    windows: list[KittiRawLidarIntegratedWindow] = Field(min_length=1)
    output_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    provenance: KittiRawLidarWindowIntegrationProvenance

    @model_validator(mode="after")
    def check_coverage(self) -> KittiRawLidarWindowIntegrationArtifact:
        if self.center_frame_ids != sorted(set(self.center_frame_ids), key=int):
            raise ValueError("raw center frame IDs must be unique and ordered")
        if [item.center_frame_id for item in self.windows] != self.center_frame_ids:
            raise ValueError("raw windows do not match center frame IDs")
        source_ids = [item.frame_id for item in self.source_scans]
        if source_ids != sorted(set(source_ids), key=int):
            raise ValueError("raw source scans must be unique and ordered")
        pose_ids = [item.frame_id for item in self.resolved_poses]
        required_pose_ids = sorted(
            {
                frame_id
                for source_id in source_ids
                for frame_id in (
                    source_id,
                    _motion_adjacent_frame_id(source_id),
                )
            },
            key=int,
        )
        if pose_ids != required_pose_ids:
            raise ValueError("raw OXTS pose coverage differs from motion policy")
        source_by_id = {item.frame_id: item for item in self.source_scans}
        used = {
            frame_id for window in self.windows for frame_id in window.source_frame_ids
        }
        if used != set(source_ids):
            raise ValueError("raw window coverage differs from source scans")
        for window in self.windows:
            expected = [
                frame_id
                for frame_id in source_ids
                if abs(int(frame_id) - int(window.center_frame_id))
                <= self.parameters.neighbor_radius_frames
            ]
            if window.source_frame_ids != expected:
                raise ValueError("raw window source IDs differ from radius policy")
            if window.source_point_count != sum(
                source_by_id[item].point_count for item in expected
            ):
                raise ValueError("raw window source point count is inconsistent")
        if self.parameters.pose_origin_frame_id != pose_ids[0]:
            raise ValueError("raw pose origin must be the first resolved pose")
        if self.output_set_sha256 != _scan_reference_set_sha256(
            [item.output for item in self.windows]
        ):
            raise ValueError("raw output set SHA-256 differs from outputs")
        if Path(self.imu_to_velodyne_calibration.path).name != (
            "calib_imu_to_velo.txt"
        ):
            raise ValueError("raw integration requires calib_imu_to_velo.txt")
        timestamp_names = {Path(item.path).name for item in self.timestamp_files}
        if timestamp_names != {"timestamps.txt"}:
            raise ValueError("raw integration timestamp file names are invalid")
        return self

    def save(self, path: str | Path) -> None:
        """Save this integration artifact as schema-valid YAML or JSON."""

        write_mapping(Path(path), self.model_dump(mode="json", exclude_none=True))


def integrate_kitti_raw_lidar_windows(
    selection_manifest_path: str | Path,
    calibration_path: str | Path,
    center_frame_ids: Sequence[str | int],
    *,
    neighbor_radius_frames: int,
    output_directory: str | Path,
    artifact_id: str | None = None,
    minimum_range_m: float = 1.0,
    voxel_resolution_m: float = 0.002,
    generator: str = "calibrex.kitti_raw_lidar_window_integration",
    generator_version: str = KITTI_RAW_LIDAR_WINDOW_INTEGRATOR_VERSION,
    command: Sequence[str] = (),
) -> KittiRawLidarWindowIntegrationArtifact:
    """Integrate centered KITTI raw windows using official OXTS pose packets."""

    if neighbor_radius_frames < 0:
        raise ValueError("neighbor radius must be non-negative")
    if minimum_range_m < 0.0 or voxel_resolution_m < 0.0:
        raise ValueError("range and voxel settings must be non-negative")
    centers = _normalize_frame_ids(center_frame_ids)
    selection_path = Path(selection_manifest_path).resolve()
    calibration_file = Path(calibration_path).resolve()
    output_root = Path(output_directory).resolve()
    selection = load_remote_archive_selection(selection_path)
    if selection.schema_version != "slac.remote_archive_selection/v0.3":
        raise ValueError("raw integration requires remote selection v0.3")
    if selection.dataset_family != "KITTI raw" or selection.selection_policy != (
        "center_images_with_centered_pointcloud_pose_windows/v0.3"
    ):
        raise ValueError("raw integration requires a raw multistream selection")
    if centers != selection.center_frame_ids:
        raise ValueError("raw integration centers differ from selection centers")
    assert selection.stream_frame_ids is not None
    source_ids = selection.stream_frame_ids["pointcloud"]
    pose_ids = selection.stream_frame_ids["pose"]
    windows_by_center = {
        center: [
            frame_id
            for frame_id in source_ids
            if abs(int(frame_id) - int(center)) <= neighbor_radius_frames
        ]
        for center in centers
    }
    _validate_windows(windows_by_center)
    lidar_members = {
        item.frame_id: item for item in selection.members if item.role == "pointcloud"
    }
    pose_members = {
        item.frame_id: item for item in selection.members if item.role == "pose"
    }
    if set(lidar_members) != set(source_ids) or set(pose_members) != set(pose_ids):
        raise ValueError("raw selection does not bind its LiDAR/OXTS coverage")
    source_paths: dict[str, Path] = {}
    source_scans: list[KittiRawLidarScanReference] = []
    pose_paths: dict[str, Path] = {}
    packets: dict[str, KITTIOXTSPacket] = {}
    for frame_id in source_ids:
        lidar_path = _verified_member_path(
            lidar_members[frame_id].local_path,
            lidar_members[frame_id].local_sha256,
            lidar_members[frame_id].local_size_bytes,
            selection_path.parent,
        )
        point_count = _xyzi_point_count(lidar_path)
        source_paths[frame_id] = lidar_path
        source_scans.append(
            KittiRawLidarScanReference(
                frame_id=frame_id,
                path=str(lidar_path),
                sha256=_required_sha256(lidar_path),
                size_bytes=lidar_path.stat().st_size,
                point_count=point_count,
            )
        )
    for frame_id in pose_ids:
        pose_path = _verified_member_path(
            pose_members[frame_id].local_path,
            pose_members[frame_id].local_sha256,
            pose_members[frame_id].local_size_bytes,
            selection_path.parent,
        )
        pose_paths[frame_id] = pose_path
        packets[frame_id] = read_oxts_packet(pose_path)
    origin = packets[pose_ids[0]]
    pose_transforms = {
        frame_id: kitti_oxts_pose_world_imu(packet, origin)
        for frame_id, packet in packets.items()
    }
    pose_matrices = {
        int(frame_id): _se3_to_matrix(transform)
        for frame_id, transform in pose_transforms.items()
    }
    resolved_poses = [
        KittiRawResolvedOxtsPose(
            frame_id=frame_id,
            path=str(pose_paths[frame_id]),
            sha256=_required_sha256(pose_paths[frame_id]),
            size_bytes=pose_paths[frame_id].stat().st_size,
            translation_m=list(pose_transforms[frame_id].translation_m),
            rotation_quat_xyzw=list(
                pose_transforms[frame_id].rotation_quat_xyzw
            ),
        )
        for frame_id in pose_ids
    ]
    imu_to_velodyne = _read_imu_to_velodyne(calibration_file)
    output_root.mkdir(parents=True, exist_ok=True)
    windows: list[KittiRawLidarIntegratedWindow] = []
    for center in centers:
        world_from_center = pose_matrices[int(center)] @ imu_to_velodyne
        center_from_world = np.linalg.inv(world_from_center)
        integrated_parts: list[Float32Array] = []
        range_discarded = 0
        source_point_count = 0
        for source in windows_by_center[center]:
            points = _read_xyzi(source_paths[source])
            source_point_count += len(points)
            compensated = _motion_compensate_azimuth_weighted_scan(
                points,
                frame_id=int(source),
                poses=pose_matrices,
                velodyne_to_pose=imu_to_velodyne,
            )
            ranges = np.linalg.norm(compensated[:, :3], axis=1)
            retained = compensated[ranges >= minimum_range_m]
            range_discarded += len(compensated) - len(retained)
            if len(retained) == 0:
                continue
            world_from_source = pose_matrices[int(source)] @ imu_to_velodyne
            center_from_source = center_from_world @ world_from_source
            integrated_parts.append(_transform_xyzi(retained, center_from_source))
        if not integrated_parts:
            raise ValueError(f"raw range filter removed window {center}")
        integrated = np.concatenate(integrated_parts, axis=0)
        output_points = _voxel_centroids(integrated, voxel_resolution_m)
        output = _write_xyzi_without_overwrite(
            output_root / f"{center}.bin", output_points, center
        )
        windows.append(
            KittiRawLidarIntegratedWindow(
                center_frame_id=center,
                source_frame_ids=windows_by_center[center],
                source_point_count=source_point_count,
                range_discarded_point_count=range_discarded,
                range_retained_point_count=len(integrated),
                voxel_discarded_point_count=(
                    len(integrated) - len(output_points)
                ),
                output=output,
            )
        )
    selection_reference = _file_reference(selection_path)
    calibration_reference = _file_reference(calibration_file)
    timestamp_references = [
        _file_reference(
            _verified_member_path(
                item.local_path,
                item.local_sha256,
                item.local_size_bytes,
                selection_path.parent,
            )
        )
        for item in selection.metadata_members
    ]
    source_references: dict[str, KittiRawIntegrationFileReference] = {
        "selection_manifest": selection_reference,
        "calibration:calib_imu_to_velo.txt": calibration_reference,
        **{
            f"timestamp:{index}": item
            for index, item in enumerate(timestamp_references)
        },
    }
    parameters = KittiRawLidarWindowIntegrationParameters(
        neighbor_radius_frames=neighbor_radius_frames,
        pose_origin_frame_id=pose_ids[0],
        minimum_range_m=minimum_range_m,
        voxel_resolution_m=voxel_resolution_m,
        voxel_reducer=(
            "none"
            if voxel_resolution_m == 0.0
            else "sorted_voxel_centroid_xyz_mean_intensity/v0.1"
        ),
    )
    return KittiRawLidarWindowIntegrationArtifact(
        artifact_id=(
            artifact_id
            or f"kitti-raw-{selection.sequence_id}-lidar-window-r"
            f"{neighbor_radius_frames}"
        ),
        dataset_id=selection.dataset_id,
        sequence_id=selection.sequence_id,
        dataset_license_spdx=selection.dataset_license_spdx,
        selection_manifest=selection_reference,
        imu_to_velodyne_calibration=calibration_reference,
        timestamp_files=timestamp_references,
        center_frame_ids=centers,
        parameters=parameters,
        source_scans=source_scans,
        resolved_poses=resolved_poses,
        windows=windows,
        output_set_sha256=_scan_reference_set_sha256(
            [item.output for item in windows]
        ),
        provenance=KittiRawLidarWindowIntegrationProvenance(
            generator=generator,
            generator_version=generator_version,
            git_commit=git_commit(),
            command=list(command),
            source_sha256={
                key: value.sha256 for key, value in source_references.items()
            },
        ),
    )


def load_kitti_raw_lidar_window_integration(
    path: str | Path,
) -> KittiRawLidarWindowIntegrationArtifact:
    """Load and validate a KITTI raw LiDAR window integration artifact."""

    return KittiRawLidarWindowIntegrationArtifact.model_validate(
        read_mapping(Path(path))
    )


def verify_kitti_raw_lidar_window_integration_files(
    artifact: KittiRawLidarWindowIntegrationArtifact,
    *,
    base_path: str | Path | None = None,
) -> list[str]:
    """Return deterministic file integrity issues for raw integration."""

    root = Path(base_path) if base_path is not None else None
    references: list[KittiRawIntegrationFileReference] = [
        artifact.selection_manifest,
        artifact.imu_to_velodyne_calibration,
        *artifact.timestamp_files,
        *artifact.source_scans,
        *artifact.resolved_poses,
        *(item.output for item in artifact.windows),
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
        if path.stat().st_size != reference.size_bytes:
            issues.append(f"size_mismatch:{reference.path}")
        if digest != reference.sha256:
            issues.append(f"digest_mismatch:{reference.path}")
    return issues


def kitti_raw_lidar_window_integration_json_schema() -> dict[str, Any]:
    """Return the standalone schema for raw window integrations."""

    return KittiRawLidarWindowIntegrationArtifact.model_json_schema()


def _read_imu_to_velodyne(path: Path) -> FloatArray:
    values = read_calibration_file(path)
    rotation = values.get("R")
    translation = values.get("T")
    if rotation is None or len(rotation) != 9:
        raise ValueError("calib_imu_to_velo.txt has no 3x3 R")
    if translation is None or len(translation) != 3:
        raise ValueError("calib_imu_to_velo.txt has no 3-vector T")
    velodyne_from_imu: FloatArray = np.eye(4, dtype=np.float64)
    velodyne_from_imu[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    velodyne_from_imu[:3, 3] = np.asarray(translation, dtype=np.float64)
    return np.linalg.inv(velodyne_from_imu)


def _motion_compensate_azimuth_weighted_scan(
    points: Float32Array,
    *,
    frame_id: int,
    poses: dict[int, FloatArray],
    velodyne_to_pose: FloatArray,
) -> Float32Array:
    values = np.asarray(points, dtype=np.float64)
    relative_pose: FloatArray = np.eye(4, dtype=np.float64)
    adjacent = int(_motion_adjacent_frame_id(f"{frame_id:010d}"))
    if frame_id not in poses or adjacent not in poses:
        raise ValueError(f"raw motion pose coverage is incomplete for frame {frame_id}")
    if frame_id in (0, 1):
        relative_pose = np.linalg.inv(poses[adjacent]) @ poses[frame_id]
    else:
        relative_pose = np.linalg.inv(poses[frame_id]) @ poses[adjacent]
    delta = np.linalg.inv(velodyne_to_pose) @ relative_pose @ velodyne_to_pose
    rotation_vector = _rotation_vector(delta[:3, :3])
    translation = delta[:3, 3]
    xyz = values[:, :3]
    fractions = 0.5 * np.arctan2(xyz[:, 1], xyz[:, 0]) / math.pi
    scaled = fractions[:, None] * rotation_vector[None, :]
    angles = np.linalg.norm(scaled, axis=1)
    compensated = xyz.copy()
    rotating = angles > 1.0e-10
    if np.any(rotating):
        axes = scaled[rotating] / angles[rotating, None]
        source = xyz[rotating]
        cosines = np.cos(angles[rotating])[:, None]
        sines = np.sin(angles[rotating])[:, None]
        compensated[rotating] = (
            source * cosines
            + np.cross(axes, source) * sines
            + axes
            * np.sum(axes * source, axis=1)[:, None]
            * (1.0 - cosines)
        )
    compensated += fractions[:, None] * translation[None, :]
    output = values.copy()
    output[:, :3] = compensated
    return output.astype(np.float32)


def _rotation_vector(rotation: FloatArray) -> FloatArray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle <= 1.0e-10:
        return np.zeros(3, dtype=np.float64)
    sine = math.sin(angle)
    if abs(sine) <= 1.0e-10:
        raise ValueError("near-pi raw inter-frame rotation is unsupported")
    axis = np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    ) / (2.0 * sine)
    return axis * angle


def _normalize_frame_ids(frame_ids: Sequence[str | int]) -> list[str]:
    values = [str(item).strip() for item in frame_ids]
    if not values or any(not item.isdigit() for item in values):
        raise ValueError("raw center frame IDs must be numeric")
    normalized = [f"{int(item):010d}" for item in values]
    if normalized != sorted(set(normalized), key=int):
        raise ValueError("raw center frame IDs must be unique and ordered")
    return normalized


def _motion_adjacent_frame_id(frame_id: str) -> str:
    value = int(frame_id)
    adjacent = value + 1 if value in (0, 1) else value - 1
    return f"{adjacent:010d}"


def _validate_windows(windows: dict[str, list[str]]) -> None:
    for center, frame_ids in windows.items():
        if center not in frame_ids:
            raise ValueError(f"raw window omits center {center}")
        if any(int(right) != int(left) + 1 for left, right in pairwise(frame_ids)):
            raise ValueError(f"raw window has an internal gap around {center}")


def _verified_member_path(
    raw_path: str,
    expected_digest: str,
    expected_size: int,
    base_path: Path,
) -> Path:
    path = _resolve_path(raw_path, base_path)
    if path.stat().st_size != expected_size or _required_sha256(path) != expected_digest:
        raise ValueError(f"raw selected member differs from manifest: {path}")
    return path


def _resolve_path(raw_path: str, base_path: Path) -> Path:
    path = Path(raw_path)
    return (path if path.is_absolute() else base_path / path).resolve()


def _xyzi_point_count(path: Path) -> int:
    item_size = 4 * np.dtype("<f4").itemsize
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"raw LiDAR scan is empty or missing: {path}")
    if path.stat().st_size % item_size:
        raise ValueError(f"raw LiDAR scan is not float32 XYZI: {path}")
    return path.stat().st_size // item_size


def _read_xyzi(path: Path) -> Float32Array:
    values = np.fromfile(path, dtype="<f4").reshape(_xyzi_point_count(path), 4)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"raw LiDAR scan contains non-finite values: {path}")
    return np.asarray(values, dtype=np.float32)


def _transform_xyzi(points: Float32Array, transform: FloatArray) -> Float32Array:
    values = np.asarray(points, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    output = values.copy()
    output[:, :3] = values[:, :3] @ matrix[:3, :3].T + matrix[:3, 3]
    return output.astype(np.float32)


def _voxel_centroids(points: Float32Array, resolution_m: float) -> Float32Array:
    if resolution_m == 0.0:
        return np.asarray(points, dtype=np.float32)
    keys = np.floor(
        np.asarray(points[:, :3], dtype=np.float64) / resolution_m
    ).astype(np.int64)
    _unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    sums: FloatArray = np.zeros((len(counts), 4), dtype=np.float64)
    np.add.at(sums, inverse, np.asarray(points, dtype=np.float64))
    return (sums / counts[:, None]).astype(np.float32)


def _write_xyzi_without_overwrite(
    path: Path,
    points: Float32Array,
    frame_id: str,
) -> KittiRawLidarScanReference:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.calibrex-partial")
    partial.unlink(missing_ok=True)
    np.asarray(points, dtype="<f4").tofile(partial)
    digest = _required_sha256(partial)
    size = partial.stat().st_size
    if path.exists():
        if path.stat().st_size != size or _required_sha256(path) != digest:
            partial.unlink(missing_ok=True)
            raise ValueError(f"refusing to overwrite different raw map: {path}")
        partial.unlink(missing_ok=True)
    else:
        partial.replace(path)
    return KittiRawLidarScanReference(
        frame_id=frame_id,
        path=str(path),
        sha256=digest,
        size_bytes=size,
        point_count=len(points),
    )


def _file_reference(path: Path) -> KittiRawIntegrationFileReference:
    resolved = path.resolve()
    return KittiRawIntegrationFileReference(
        path=str(resolved),
        sha256=_required_sha256(resolved),
        size_bytes=resolved.stat().st_size,
    )


def _required_sha256(path: Path) -> str:
    digest = sha256_path(path)
    if digest is None or not path.is_file():
        raise ValueError(f"required raw integration file is unreadable: {path}")
    return digest


def _scan_reference_set_sha256(
    references: Sequence[KittiRawLidarScanReference],
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


def _se3_to_matrix(transform: SE3) -> FloatArray:
    x_value, y_value, z_value, w_value = transform.rotation_quat_xyzw
    matrix: FloatArray = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(
        [
            [
                1.0 - 2.0 * (y_value**2 + z_value**2),
                2.0 * (x_value * y_value - z_value * w_value),
                2.0 * (x_value * z_value + y_value * w_value),
            ],
            [
                2.0 * (x_value * y_value + z_value * w_value),
                1.0 - 2.0 * (x_value**2 + z_value**2),
                2.0 * (y_value * z_value - x_value * w_value),
            ],
            [
                2.0 * (x_value * z_value - y_value * w_value),
                2.0 * (y_value * z_value + x_value * w_value),
                1.0 - 2.0 * (x_value**2 + y_value**2),
            ],
        ],
        dtype=np.float64,
    )
    matrix[:3, 3] = np.asarray(transform.translation_m, dtype=np.float64)
    return matrix


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )
