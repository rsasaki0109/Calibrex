"""Build digest-pinned Camera--LiDAR problems from KITTI-360 artifacts."""

from __future__ import annotations

import hashlib
import math
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
    DepthCameraIntrinsics,
    depth_file_reference,
    load_depth_provider,
    verify_depth_provider_files,
)
from calibrex.data.kitti import read_timestamps

FloatArray: TypeAlias = NDArray[np.float64]
KITTI360_CAMERA_LIDAR_PROBLEM_BUILDER_VERSION = (
    "calibrex.kitti360_camera_lidar_problem/v0.2"
)


def read_kitti360_fisheye_intrinsics(
    calibration_root: str | Path,
    *,
    camera_stream: str = "image_03",
) -> DepthCameraIntrinsics:
    """Read an official KITTI-360 MEI fisheye calibration."""

    _validate_fisheye_stream(camera_stream)
    calibration_path = Path(calibration_root) / f"{camera_stream}.yaml"
    text = calibration_path.read_text(encoding="utf-8")
    if "model_type: MEI" not in text:
        raise ValueError(f"{camera_stream}.yaml must declare model_type: MEI")
    values = _read_opencv_yaml_scalars(calibration_path)
    required = (
        "image_width",
        "image_height",
        "xi",
        "k1",
        "k2",
        "p1",
        "p2",
        "gamma1",
        "gamma2",
        "u0",
        "v0",
    )
    missing = [name for name in required if name not in values]
    if missing:
        raise ValueError(
            f"{camera_stream}.yaml is missing required values: {', '.join(missing)}"
        )
    return DepthCameraIntrinsics(
        width=int(values["image_width"]),
        height=int(values["image_height"]),
        fx=values["gamma1"],
        fy=values["gamma2"],
        cx=values["u0"],
        cy=values["v0"],
        projection="mei",
        xi=values["xi"],
        distortion_model="radial-tangential",
        distortion=[
            values["k1"],
            values["k2"],
            values["p1"],
            values["p2"],
        ],
    )


def read_kitti360_velodyne_to_camera_transform(
    calibration_root: str | Path,
    *,
    camera_stream: str = "image_03",
) -> SE3:
    """Read ``T_camera_velodyne`` using the official KITTI-360 chain."""

    _validate_fisheye_stream(camera_stream)
    root = Path(calibration_root)
    camera_to_velodyne = _read_matrix_3x4(root / "calib_cam_to_velo.txt")
    camera_to_pose = _read_named_matrices(root / "calib_cam_to_pose.txt")
    if "image_00" not in camera_to_pose:
        raise ValueError("calib_cam_to_pose.txt is missing image_00")
    if camera_stream not in camera_to_pose:
        raise ValueError(f"calib_cam_to_pose.txt is missing {camera_stream}")

    camera_to_camera0 = (
        np.linalg.inv(camera_to_pose["image_00"]) @ camera_to_pose[camera_stream]
    )
    camera_to_velodyne_selected = camera_to_velodyne @ camera_to_camera0
    velodyne_to_camera = np.linalg.inv(camera_to_velodyne_selected)
    return _se3_from_matrix(velodyne_to_camera)


def read_kitti360_poses(path: str | Path) -> dict[int, FloatArray]:
    """Read KITTI-360 sparse ``T_world_pose`` matrices by frame ID."""

    pose_path = Path(path)
    if not pose_path.is_file():
        raise ValueError(f"pose file does not exist: {pose_path}")
    poses: dict[int, FloatArray] = {}
    for line in pose_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 13:
            raise ValueError("each KITTI-360 pose row must contain 13 values")
        poses[int(fields[0])] = _homogeneous(
            np.asarray([float(value) for value in fields[1:]], dtype=float)
        )
    return poses


def read_kitti360_velodyne_to_pose_matrix(
    calibration_root: str | Path,
) -> FloatArray:
    """Read ``T_pose_velodyne`` used by the official scan uncurling code."""

    root = Path(calibration_root)
    camera_to_velodyne = _read_matrix_3x4(root / "calib_cam_to_velo.txt")
    camera_to_pose = _read_named_matrices(root / "calib_cam_to_pose.txt")
    if "image_00" not in camera_to_pose:
        raise ValueError("calib_cam_to_pose.txt is missing image_00")
    return camera_to_pose["image_00"] @ np.linalg.inv(camera_to_velodyne)


def motion_compensate_kitti360_scan(
    points: NDArray[np.float32],
    *,
    frame_id: int,
    poses: dict[int, FloatArray],
    velodyne_to_pose: FloatArray,
) -> NDArray[np.float32]:
    """Apply the official KITTI-360 azimuth-weighted scan uncurling model."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("KITTI-360 Velodyne points must have shape Nx4")
    relative_pose: FloatArray = np.eye(4, dtype=float)
    if frame_id in poses:
        adjacent = frame_id + 1 if frame_id in (0, 1) else frame_id - 1
        if adjacent in poses:
            if frame_id in (0, 1):
                relative_pose = np.linalg.inv(poses[adjacent]) @ poses[frame_id]
            else:
                relative_pose = np.linalg.inv(poses[frame_id]) @ poses[adjacent]
    delta = np.linalg.inv(velodyne_to_pose) @ relative_pose @ velodyne_to_pose
    rotation_vector = _rotation_vector(delta[:3, :3])
    translation = delta[:3, 3]
    xyz = values[:, :3]
    fractions = 0.5 * np.arctan2(xyz[:, 1], xyz[:, 0]) / math.pi
    scaled_vectors = fractions[:, None] * rotation_vector[None, :]
    angles = np.linalg.norm(scaled_vectors, axis=1)
    compensated = xyz.copy()
    rotating = angles > 1.0e-10
    if np.any(rotating):
        axes = scaled_vectors[rotating] / angles[rotating, None]
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


def build_kitti360_camera_lidar_problem(
    sequence_path: str | Path,
    depth_provider_path: str | Path,
    *,
    calibration_root: str | Path | None = None,
    lidar_directory: str | Path | None = None,
    lidar_manifest_path: str | Path | None = None,
    camera_stream: str = "image_03",
    split_id: str = "evaluation",
    dataset_id: str | None = None,
    problem_id: str | None = None,
    rotation_bound_deg: float = 20.0,
    translation_bound_m: float = 0.0,
    command: tuple[str, ...] = (),
) -> CameraLidarCalibrationProblem:
    """Bind a verified depth artifact to matching KITTI-360 raw scans."""

    _validate_fisheye_stream(camera_stream)
    sequence = Path(sequence_path).resolve()
    provider_path = Path(depth_provider_path).resolve()
    calibration = (
        Path(calibration_root).resolve()
        if calibration_root is not None
        else sequence.parent / "calibration"
    )
    image_stream = sequence / camera_stream / "data_rgb"
    lidar_stream = (
        Path(lidar_directory).resolve()
        if lidar_directory is not None
        else sequence / "velodyne_points" / "data"
    )
    lidar_timestamps_path = sequence / "velodyne_points" / "timestamps.txt"
    camera_timestamps_path = sequence / camera_stream / "timestamps.txt"
    for directory, label in (
        (sequence, "sequence"),
        (calibration, "calibration"),
        (image_stream, "camera stream"),
        (lidar_stream, "Velodyne stream"),
    ):
        if not directory.is_dir():
            raise ValueError(f"KITTI-360 {label} does not exist: {directory}")
    if (lidar_directory is None) != (lidar_manifest_path is None):
        raise ValueError(
            "generated KITTI-360 LiDAR requires both lidar_directory and "
            "lidar_manifest_path"
        )

    provider = load_depth_provider(provider_path)
    provider_digest = _required_digest(provider_path)
    issues = verify_depth_provider_files(provider, base_path=provider_path.parent)
    if issues:
        raise ValueError("depth provider verification failed: " + "; ".join(issues))
    if len({item.frame_id for item in provider.observations}) != len(
        provider.observations
    ):
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
            calibration_directory=calibration,
        )

    timestamps = read_timestamps(lidar_timestamps_path)
    timestamp_by_index = {item.index: item.timestamp_ns for item in timestamps}
    bindings: list[CameraLidarObservationBinding] = []
    source_files = [
        provider_path,
        calibration / f"{camera_stream}.yaml",
        calibration / "calib_cam_to_velo.txt",
        calibration / "calib_cam_to_pose.txt",
        lidar_timestamps_path,
        camera_timestamps_path,
    ]
    if integration_manifest is not None:
        source_files.append(integration_manifest)
    for observation in provider.observations:
        if not observation.frame_id.isdigit():
            raise ValueError(
                f"KITTI-360 frame ID must be decimal: {observation.frame_id!r}"
            )
        index = int(observation.frame_id)
        image_path = image_stream / f"{observation.frame_id}.png"
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
            raise ValueError(
                f"Velodyne timestamps do not cover frame {observation.frame_id}"
            )
        bindings.append(
            CameraLidarObservationBinding(
                frame_id=observation.frame_id,
                split_id=split_id,
                depth_observation_frame_id=observation.frame_id,
                lidar=depth_file_reference(
                    lidar_path,
                    encoding="kitti_velodyne_f32x4",
                ),
                lidar_capture_time_ns=timestamp_by_index[index],
            )
        )
        source_files.extend((image_path, lidar_path))
    reference = read_kitti360_velodyne_to_camera_transform(
        calibration,
        camera_stream=camera_stream,
    )
    parent_frame = f"camera_{camera_stream[-1]}"
    resolved_dataset_id = dataset_id or (
        f"kitti360_{sequence.name.removesuffix('_sync')}"
    )
    return CameraLidarCalibrationProblem(
        problem_id=problem_id or f"{resolved_dataset_id}-{camera_stream}-d2d",
        dataset_id=resolved_dataset_id,
        dataset_family="KITTI-360",
        sequence_id=sequence.name,
        depth_provider_path=str(provider_path),
        depth_provider_sha256=provider_digest,
        observations=bindings,
        reference_transform_camera_lidar=_transform_result(
            reference,
            parent=parent_frame,
            source_path=calibration,
            role="selected_reference",
        ),
        initial_transform_camera_lidar=_transform_result(
            reference,
            parent=parent_frame,
            source_path=calibration,
            role="initial",
        ),
        time_convention=(
            "KITTI-360 per-stream capture timestamps in UTC nanoseconds; "
            "same numeric frame IDs are paired; "
            + (
                "Velodyne scans use the provenance-pinned official motion "
                "compensation adapter"
                if lidar_directory is not None
                else "raw scans are not motion compensated"
            )
        ),
        rotation_bound_deg=rotation_bound_deg,
        translation_bound_m=translation_bound_m,
        provenance=CameraLidarArtifactProvenance(
            generator="calibrex.data.kitti360_camera_lidar_problem",
            generator_version=KITTI360_CAMERA_LIDAR_PROBLEM_BUILDER_VERSION,
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
    calibration_directory: Path,
) -> dict[str, Path]:
    """Verify a KITTI-360 integration artifact and return its outputs."""

    # Local import avoids a module cycle: the integration adapter reuses the
    # official motion-compensation primitives defined in this module.
    from calibrex.data.kitti360_lidar_window_integration import (
        load_kitti360_lidar_window_integration,
        verify_kitti360_lidar_window_integration_files,
    )

    artifact = load_kitti360_lidar_window_integration(manifest_path)
    issues = verify_kitti360_lidar_window_integration_files(artifact)
    if issues:
        raise ValueError(
            "KITTI-360 LiDAR integration verification failed: "
            + "; ".join(issues)
        )
    if artifact.sequence_id != sequence_id:
        raise ValueError(
            "KITTI-360 LiDAR integration sequence mismatch: "
            f"{artifact.sequence_id} != {sequence_id}"
        )
    calibration_references = {
        Path(item.path).name: item for item in artifact.calibration_files
    }
    expected_calibration = {
        name: calibration_directory / name
        for name in ("calib_cam_to_velo.txt", "calib_cam_to_pose.txt")
    }
    if set(calibration_references) != set(expected_calibration):
        raise ValueError(
            "KITTI-360 LiDAR integration calibration files are incomplete"
        )
    for name, expected_path in expected_calibration.items():
        if calibration_references[name].sha256 != _required_digest(expected_path):
            raise ValueError(
                "KITTI-360 LiDAR integration calibration differs from problem: "
                + name
            )
    normalized_ids = sorted(set(frame_ids), key=int)
    if artifact.center_frame_ids != normalized_ids:
        raise ValueError(
            "KITTI-360 LiDAR integration centers differ from depth observations"
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
            "KITTI-360 LiDAR integration outputs do not match lidar_directory"
        )
    return outputs


def _read_opencv_yaml_scalars(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise ValueError(f"calibration file does not exist: {path}")
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("%") or ":" not in stripped:
            continue
        key, raw = stripped.split(":", 1)
        try:
            values[key.strip()] = float(raw.strip())
        except ValueError:
            continue
    return values


def _validate_fisheye_stream(camera_stream: str) -> None:
    if camera_stream not in {"image_02", "image_03"}:
        raise ValueError("KITTI-360 fisheye stream must be image_02 or image_03")


def _homogeneous(values: FloatArray) -> FloatArray:
    matrix: FloatArray = np.eye(4, dtype=float)
    matrix[:3, :] = values.reshape(3, 4)
    return matrix


def _read_matrix_3x4(path: Path) -> FloatArray:
    try:
        values = [float(value) for value in path.read_text().split()]
    except FileNotFoundError as exc:
        raise ValueError(f"calibration file does not exist: {path}") from exc
    if len(values) != 12:
        raise ValueError(f"{path.name} must contain one 3x4 matrix")
    return _homogeneous(np.asarray(values, dtype=float))


def _read_named_matrices(path: Path) -> dict[str, FloatArray]:
    if not path.is_file():
        raise ValueError(f"calibration file does not exist: {path}")
    matrices: dict[str, FloatArray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        name, raw = line.split(":", 1)
        values = [float(value) for value in raw.split()]
        if len(values) != 12:
            raise ValueError(f"{path.name} entry {name} must be a 3x4 matrix")
        matrices[name.strip()] = _homogeneous(np.asarray(values, dtype=float))
    return matrices


def _se3_from_matrix(matrix: FloatArray) -> SE3:
    return SE3(
        translation_m=(
            float(matrix[0, 3]),
            float(matrix[1, 3]),
            float(matrix[2, 3]),
        ),
        rotation_quat_xyzw=quaternion_xyzw_from_rotation_matrix(
            matrix[:3, :3].reshape(-1)
        ),
    )


def _rotation_vector(rotation: FloatArray) -> FloatArray:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle <= 1.0e-10:
        return np.zeros(3, dtype=float)
    sine = math.sin(angle)
    if abs(sine) <= 1.0e-10:
        raise ValueError("near-pi KITTI-360 inter-frame rotation is unsupported")
    axis = np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=float,
    ) / (2.0 * sine)
    return axis * angle


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
            "depth provider image does not bind to the requested KITTI-360 stream: "
            f"{provider_image_path} != {expected_image_path}"
        )
    if _required_digest(expected_image_path) != provider_image_digest:
        raise ValueError(
            f"depth provider image digest mismatch for {provider_image_path}"
        )


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
            source="KITTI-360 official calibration",
            source_path=str(source_path),
            tool_name="calibrex",
            tool_version=__version__,
            adapter_version=KITTI360_CAMERA_LIDAR_PROBLEM_BUILDER_VERSION,
            notes=[
                "T_camera_lidar = inverse(T_velo_camera0 @ "
                "inverse(T_pose_camera0) @ T_pose_camera_selected)"
            ],
        ),
    )


def _source_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted({item.resolve() for item in paths}, key=str):
        digest.update(str(path).encode())
        digest.update(b"\0")
        digest.update(_required_digest(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _required_digest(path: Path) -> str:
    digest = sha256_path(path)
    if digest is None:
        raise ValueError(f"required file is missing or unreadable: {path}")
    return digest
