"""Build digest-pinned Camera--LiDAR problems from KITTI raw depth artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from calibrex import __version__
from calibrex.core.camera_lidar_artifacts import (
    CameraLidarArtifactProvenance,
    CameraLidarCalibrationProblem,
    CameraLidarObservationBinding,
)
from calibrex.core.geometry import SE3, quaternion_xyzw_from_rotation_matrix
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.result import (
    EstimateRole,
    TransformEstimateProvenance,
    TransformResult,
)
from calibrex.data.depth import (
    depth_file_reference,
    load_depth_provider,
    verify_depth_provider_files,
)
from calibrex.data.kitti import read_calibration_file, read_timestamps

FloatArray: TypeAlias = NDArray[np.float64]
KITTI_CAMERA_LIDAR_PROBLEM_BUILDER_VERSION = "calibrex.kitti_camera_lidar_problem/v0.2"


def uniform_kitti_frame_ids(
    sequence_count: int,
    selected_count: int,
) -> list[str]:
    """Select endpoint-inclusive uniform IDs with nearest-half-up rounding."""

    if selected_count < 2:
        raise ValueError("selected_count must be at least two")
    if sequence_count < selected_count:
        raise ValueError(f"sequence has {sequence_count} frames, cannot select {selected_count}")
    denominator = selected_count - 1
    return [
        f"{(index * (sequence_count - 1) + denominator // 2) // denominator:010d}"
        for index in range(selected_count)
    ]


def read_rectified_velodyne_to_camera_transform(
    calibration_root: str | Path,
    *,
    camera_stream: str = "image_02",
) -> SE3:
    """Read ``T_rectified_camera_lidar`` for a KITTI raw camera stream."""

    root = Path(calibration_root)
    suffix = _camera_suffix(camera_stream)
    velo = read_calibration_file(root / "calib_velo_to_cam.txt")
    camera = read_calibration_file(root / "calib_cam_to_cam.txt")
    rotation = velo.get("R")
    translation = velo.get("T")
    rectification = camera.get("R_rect_00")
    projection = camera.get(f"P_rect_{suffix}")
    if rotation is None or len(rotation) != 9:
        raise ValueError("calib_velo_to_cam.txt must contain a 3x3 R")
    if translation is None or len(translation) != 3:
        raise ValueError("calib_velo_to_cam.txt must contain a three-vector T")
    if rectification is None or len(rectification) != 9:
        raise ValueError("calib_cam_to_cam.txt must contain a 3x3 R_rect_00")
    if projection is None or len(projection) != 12:
        raise ValueError(f"calib_cam_to_cam.txt must contain a 3x4 P_rect_{suffix}")

    rotation_velo: FloatArray = np.asarray(rotation, dtype=float).reshape(3, 3)
    translation_velo: FloatArray = np.asarray(translation, dtype=float)
    rotation_rect: FloatArray = np.asarray(rectification, dtype=float).reshape(
        3,
        3,
    )
    projection_matrix: FloatArray = np.asarray(projection, dtype=float).reshape(
        3,
        4,
    )
    intrinsic = projection_matrix[:, :3]
    try:
        camera_origin_offset = np.linalg.solve(
            intrinsic,
            projection_matrix[:, 3],
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"P_rect_{suffix} has a singular intrinsic matrix") from exc

    final_rotation = rotation_rect @ rotation_velo
    final_translation = rotation_rect @ translation_velo + camera_origin_offset
    return SE3(
        translation_m=(
            float(final_translation[0]),
            float(final_translation[1]),
            float(final_translation[2]),
        ),
        rotation_quat_xyzw=quaternion_xyzw_from_rotation_matrix(final_rotation.reshape(-1)),
    )


def build_kitti_raw_camera_lidar_problem(
    sequence_path: str | Path,
    depth_provider_path: str | Path,
    *,
    camera_stream: str = "image_02",
    lidar_directory: str | Path | None = None,
    lidar_manifest_path: str | Path | None = None,
    dataset_id: str | None = None,
    problem_id: str | None = None,
    rotation_bound_deg: float = 20.0,
    translation_bound_m: float = 0.0,
    command: tuple[str, ...] = (),
) -> CameraLidarCalibrationProblem:
    """Bind a verified frozen depth artifact to matching KITTI Velodyne scans."""

    sequence = Path(sequence_path).resolve()
    provider_path = Path(depth_provider_path).resolve()
    if not sequence.is_dir():
        raise ValueError(f"KITTI raw sequence does not exist: {sequence}")
    expected_stream = sequence / camera_stream / "data"
    lidar_stream = (
        Path(lidar_directory).resolve()
        if lidar_directory is not None
        else sequence / "velodyne_points" / "data"
    )
    lidar_timestamps_path = sequence / "velodyne_points" / "timestamps.txt"
    imu_to_velodyne_path = sequence.parent / "calib_imu_to_velo.txt"
    if not expected_stream.is_dir():
        raise ValueError(f"KITTI camera stream does not exist: {expected_stream}")
    if not lidar_stream.is_dir():
        raise ValueError(f"KITTI Velodyne stream does not exist: {lidar_stream}")
    if (lidar_directory is None) != (lidar_manifest_path is None):
        raise ValueError(
            "generated KITTI raw LiDAR requires both lidar_directory and "
            "lidar_manifest_path"
        )

    provider = load_depth_provider(provider_path)
    provider_digest = _required_digest(provider_path)
    provider_issues = verify_depth_provider_files(
        provider,
        base_path=provider_path.parent,
    )
    if provider_issues:
        raise ValueError("depth provider verification failed: " + "; ".join(provider_issues))
    if len({item.frame_id for item in provider.observations}) != len(provider.observations):
        raise ValueError("depth provider frame IDs must be unique")
    integrated_outputs: dict[str, Path] | None = None
    integration_manifest: Path | None = None
    if lidar_manifest_path is not None:
        integration_manifest = Path(lidar_manifest_path).resolve()
        integrated_outputs = _verify_integrated_lidar_binding(
            integration_manifest,
            lidar_directory=lidar_stream,
            frame_ids=[item.frame_id for item in provider.observations],
            sequence_id=sequence.name,
            lidar_timestamps_path=lidar_timestamps_path,
            imu_to_velodyne_path=imu_to_velodyne_path,
        )

    lidar_timestamps = read_timestamps(lidar_timestamps_path)
    if not lidar_timestamps:
        raise ValueError(f"KITTI Velodyne timestamps are empty: {lidar_timestamps_path}")
    timestamp_by_index = {item.index: item.timestamp_ns for item in lidar_timestamps}
    bindings: list[CameraLidarObservationBinding] = []
    source_files = [
        provider_path,
        sequence.parent / "calib_cam_to_cam.txt",
        sequence.parent / "calib_velo_to_cam.txt",
        lidar_timestamps_path,
    ]
    if integration_manifest is not None:
        source_files.extend((integration_manifest, imu_to_velodyne_path))
    for observation in provider.observations:
        if not observation.frame_id.isdigit():
            raise ValueError(f"KITTI frame ID must be decimal: {observation.frame_id!r}")
        index = int(observation.frame_id)
        image_path = expected_stream / f"{observation.frame_id}.png"
        _verify_provider_image(
            provider_path,
            observation.image.path,
            observation.image.sha256,
            image_path,
        )
        lidar_path = (
            integrated_outputs[observation.frame_id]
            if integrated_outputs is not None
            else lidar_stream / f"{observation.frame_id}.bin"
        )
        if index not in timestamp_by_index:
            raise ValueError(f"Velodyne timestamp does not cover frame {observation.frame_id}")
        lidar_reference = depth_file_reference(
            lidar_path,
            encoding="kitti_velodyne_f32x4",
        )
        source_files.extend((image_path, lidar_path))
        bindings.append(
            CameraLidarObservationBinding(
                frame_id=observation.frame_id,
                split_id="evaluation",
                depth_observation_frame_id=observation.frame_id,
                lidar=lidar_reference,
                lidar_capture_time_ns=timestamp_by_index[index],
            )
        )

    calibration_root = sequence.parent
    reference = read_rectified_velodyne_to_camera_transform(
        calibration_root,
        camera_stream=camera_stream,
    )
    parent_frame = f"camera_{_camera_suffix(camera_stream)[-1]}"
    reference_result = _transform_result(
        reference,
        parent=parent_frame,
        source_path=calibration_root,
        role="selected_reference",
    )
    initial_result = _transform_result(
        reference,
        parent=parent_frame,
        source_path=calibration_root,
        role="initial",
    )
    resolved_dataset_id = dataset_id or f"kitti_raw_{sequence.name.removesuffix('_sync')}"
    return CameraLidarCalibrationProblem(
        problem_id=problem_id or f"{resolved_dataset_id}-{camera_stream}-d2d",
        dataset_id=resolved_dataset_id,
        dataset_family="KITTI raw",
        sequence_id=sequence.name,
        depth_provider_path=str(provider_path),
        depth_provider_sha256=provider_digest,
        observations=bindings,
        reference_transform_camera_lidar=reference_result,
        initial_transform_camera_lidar=initial_result,
        time_convention=(
            "KITTI per-stream capture timestamps in UTC nanoseconds; "
            "same numeric frame IDs are paired; "
            + (
                "Velodyne scans use provenance-pinned OXTS motion-compensated "
                "center-frame windows"
                if integrated_outputs is not None
                else "raw scans are not motion compensated"
            )
        ),
        rotation_bound_deg=rotation_bound_deg,
        translation_bound_m=translation_bound_m,
        provenance=CameraLidarArtifactProvenance(
            generator="calibrex.data.kitti_camera_lidar_problem",
            generator_version=KITTI_CAMERA_LIDAR_PROBLEM_BUILDER_VERSION,
            git_commit=git_commit(),
            command=list(command),
            config_sha256=provider_digest,
            source_sha256=_source_digest(source_files),
        ),
    )


def _verify_integrated_lidar_binding(
    manifest_path: Path,
    *,
    lidar_directory: Path,
    frame_ids: list[str],
    sequence_id: str,
    lidar_timestamps_path: Path,
    imu_to_velodyne_path: Path,
) -> dict[str, Path]:
    """Verify raw integrated outputs, sequence, centers, and timestamps."""

    from calibrex.data.kitti_raw_lidar_window_integration import (
        load_kitti_raw_lidar_window_integration,
        verify_kitti_raw_lidar_window_integration_files,
    )

    artifact = load_kitti_raw_lidar_window_integration(manifest_path)
    issues = verify_kitti_raw_lidar_window_integration_files(artifact)
    if issues:
        raise ValueError(
            "KITTI raw LiDAR integration verification failed: "
            + "; ".join(issues)
        )
    if artifact.sequence_id != sequence_id:
        raise ValueError(
            "KITTI raw LiDAR integration sequence mismatch: "
            f"{artifact.sequence_id} != {sequence_id}"
        )
    if artifact.imu_to_velodyne_calibration.sha256 != _required_digest(
        imu_to_velodyne_path
    ):
        raise ValueError(
            "KITTI raw LiDAR integration IMU calibration differs from problem"
        )
    normalized_ids = sorted(set(frame_ids), key=int)
    if artifact.center_frame_ids != normalized_ids:
        raise ValueError(
            "KITTI raw LiDAR integration centers differ from depth observations"
        )
    outputs = {
        item.center_frame_id: Path(item.output.path).resolve()
        for item in artifact.windows
    }
    expected_outputs = {
        frame_id: (lidar_directory / f"{frame_id}.bin").resolve()
        for frame_id in normalized_ids
    }
    if outputs != expected_outputs:
        raise ValueError(
            "KITTI raw LiDAR integration outputs do not match lidar_directory"
        )
    timestamp_digest = _required_digest(lidar_timestamps_path)
    pointcloud_timestamps = [
        item
        for item in artifact.timestamp_files
        if Path(item.path).parent.name == "velodyne_points"
    ]
    if len(pointcloud_timestamps) != 1:
        raise ValueError(
            "KITTI raw LiDAR integration does not identify one pointcloud timestamp file"
        )
    if pointcloud_timestamps[0].sha256 != timestamp_digest:
        raise ValueError(
            "KITTI raw LiDAR integration timestamps differ from the problem sequence"
        )
    return outputs


def _camera_suffix(camera_stream: str) -> str:
    if (
        not camera_stream.startswith("image_")
        or len(camera_stream) != 8
        or not camera_stream[-2:].isdigit()
    ):
        raise ValueError("camera_stream must use the KITTI image_XX form, for example image_02")
    return camera_stream[-2:]


def _verify_provider_image(
    provider_path: Path,
    provider_image_path: str,
    provider_image_digest: str,
    expected_image_path: Path,
) -> None:
    resolved = Path(provider_image_path)
    if not resolved.is_absolute():
        resolved = provider_path.parent / resolved
    if resolved.resolve() != expected_image_path.resolve():
        raise ValueError(
            "depth provider image does not bind to the requested KITTI stream: "
            f"{provider_image_path} != {expected_image_path}"
        )
    observed = _required_digest(expected_image_path)
    if observed != provider_image_digest:
        raise ValueError(f"depth provider image digest mismatch for {provider_image_path}")


def _transform_result(
    transform: SE3,
    *,
    parent: str,
    source_path: Path,
    role: EstimateRole,
) -> TransformResult:
    return TransformResult(
        parent=parent,
        child="lidar",
        translation_m=list(transform.translation_m),
        rotation_quat_xyzw=list(transform.rotation_quat_xyzw),
        provenance=TransformEstimateProvenance(
            producer="dataset_provider",
            execution_mode="imported",
            role_in_comparison=role,
            evidence_level="dataset_provided",
            source="KITTI raw calibration",
            source_path=str(source_path),
            tool_name="calibrex",
            tool_version=__version__,
            adapter_version=KITTI_CAMERA_LIDAR_PROBLEM_BUILDER_VERSION,
            notes=[
                "calib_velo_to_cam transformed through R_rect_00 and "
                "the selected P_rect_XX camera-origin offset"
            ],
        ),
    )


def _source_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted({item.resolve() for item in paths}, key=str):
        file_digest = _required_digest(path)
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_digest.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _required_digest(path: Path) -> str:
    digest = sha256_path(path)
    if digest is None:
        raise ValueError(f"required file is missing or unreadable: {path}")
    return digest
