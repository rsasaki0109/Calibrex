"""ROS-free conversion of KITTI-360 images, scans and declared motion inputs.

The input is the official split directory layout. Images are already rectified;
the selected camera's stereo baseline belongs in TF, not in CameraInfo twice.
Per-point times are an explicitly declared clockwise azimuth model, not measured
firing times. Published poses use image_00 timestamps and the GPS/IMU body frame;
their finite-difference twists are approximate and are not independent INS evidence.
The opt-in OXTS source uses packet timestamps and a recorded local Mercator origin,
without consuming published image/LiDAR-optimized poses. It is not metrology.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
from scipy.spatial.transform import Rotation

from calibrex import __version__
from calibrex.core.exceptions import DatasetError
from calibrex.core.provenance import git_commit
from calibrex.data.kitti import (
    kitti_oxts_pose_world_imu,
    read_calibration_file,
    read_oxts_packet,
    read_timestamps,
)
from calibrex.data.kitti_ins_lidar import se3_matrix
from calibrex.data.kitti_raw_to_rosbag2 import (
    BASE_FRAME,
    CAMERA_INFO_TOPIC,
    CLOUD_FIELDS,
    CONVERSION_FILENAME,
    CONVERSION_SCHEMA,
    IMAGE_TOPIC,
    IMU_FRAME,
    KITTI_LICENSE,
    LIDAR_FRAME,
    LIDAR_TOPIC,
    TF_STATIC_TOPIC,
    TIMED_CLOUD_FIELDS,
    FloatArray,
    KittiConversionResult,
    VelodyneTime,
    _file_sha256,
    _gray_rows,
    _quat,
    azimuth_time_offsets_s,
)
from calibrex.data.ros_cdr_writer import (
    Transform,
    encode_camera_info,
    encode_image,
    encode_odometry,
    encode_pointcloud2,
    encode_tf_message,
)
from calibrex.data.rosbag2_writer import Rosbag2Writer

ODOMETRY_TOPIC = "/kitti360/odometry"
WORLD_FRAME = "kitti360_world"
OXTS_WORLD_FRAME = "kitti360_oxts_world"
Kitti360MotionSource: TypeAlias = Literal["published_poses", "oxts"]


def _rigid(values: list[float], label: str) -> FloatArray:
    if len(values) != 12:
        raise DatasetError(f"{label}: expected a 3x4 transform")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3] = np.asarray(values, dtype=np.float64).reshape(3, 4)
    _rotation(matrix[:3, :3], label)
    if not np.isfinite(matrix).all():
        raise DatasetError(f"{label}: non-finite transform")
    return matrix


def _rotation(matrix: FloatArray, label: str) -> None:
    if (
        not np.isfinite(matrix).all()
        or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-5, rtol=0)
        or not math.isclose(float(np.linalg.det(matrix)), 1.0, abs_tol=1e-5)
    ):
        raise DatasetError(f"{label}: expected a proper rotation")


def _timestamps(path: Path) -> dict[int, int]:
    rows = read_timestamps(path)
    if not rows:
        raise DatasetError(f"{path}: no timestamps")
    stamps = [row.timestamp_ns for row in rows]
    if any(b <= a for a, b in pairwise(stamps)):
        raise DatasetError(f"{path}: timestamps must increase strictly")
    return {row.index: row.timestamp_ns for row in rows}


def _files(folder: Path, suffix: str, start: int, stop: int | None) -> dict[int, Path]:
    selected = {}
    for path in folder.glob(f"*{suffix}"):
        if not re.fullmatch(r"\d{10}", path.stem):
            raise DatasetError(f"{path}: expected a ten-digit frame ID")
        index = int(path.stem)
        if index >= start and (stop is None or index < stop):
            selected[index] = path
    if not selected:
        raise DatasetError(f"{folder}: no selected {suffix} files")
    return dict(sorted(selected.items()))


def _poses(path: Path, start: int, stop: int | None) -> dict[int, FloatArray]:
    selected = {}
    previous = -1
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 13 or not fields[0].isdigit():
            raise DatasetError(f"{path}:{number}: expected frame ID and twelve pose values")
        index = int(fields[0])
        if index <= previous:
            raise DatasetError(f"{path}:{number}: duplicate or unordered pose frame ID")
        previous = index
        if index >= start and (stop is None or index < stop):
            selected[index] = _rigid([float(v) for v in fields[1:]], f"{path}:{number}")
    if not selected:
        raise DatasetError(f"{path}: no poses in selected frame window")
    return selected


def _oxts_poses(
    folder: Path, start: int, stop: int | None,
) -> tuple[dict[int, FloatArray], set[Path], Path]:
    """Read selected OXTS packets with an explicitly recorded local origin."""
    available = _files(folder / "data", ".txt", 0, None)
    origin_path = next(iter(available.values()))
    origin = read_oxts_packet(origin_path)
    selected = {
        index: path for index, path in available.items()
        if index >= start and (stop is None or index < stop)
    }
    if not selected:
        raise DatasetError(f"{folder}: no OXTS packets in selected frame window")
    poses = {}
    for index, path in selected.items():
        matrix = se3_matrix(kitti_oxts_pose_world_imu(read_oxts_packet(path), origin))
        poses[index] = _rigid(matrix[:3].reshape(-1).tolist(), str(path))
    return poses, set(selected.values()) | {origin_path}, origin_path


def _transform(child: str, matrix: FloatArray) -> Transform:
    return (IMU_FRAME, child, _vector(matrix[:3, 3]), _quat(matrix[:3, :3]))


def _vector(values: FloatArray) -> tuple[float, float, float]:
    return float(values[0]), float(values[1]), float(values[2])


def convert_kitti360_to_rosbag2(
    dataset_root: str | Path,
    output: str | Path,
    *,
    sequence: str,
    calibration_dir: str | Path | None = None,
    camera: str = "image_00",
    velodyne_time: VelodyneTime = "azimuth",
    scan_period_s: float = 0.1,
    start_frame: int = 0,
    max_frames: int | None = None,
    command: list[str] | None = None,
    overwrite: bool = False,
    motion_source: Kitti360MotionSource = "published_poses",
) -> KittiConversionResult:
    """Convert one sequence/window in the official KITTI-360 split layout.

    ``oxts`` reads ``data_poses/SEQUENCE/oxts/{data,timestamps.txt}`` with
    its own timestamps. The earliest supplied packet defines the translation
    origin and is hashed even if outside the selected frame window. There is
    no automatic fallback between the two motion sources.

    ``start_frame`` is inclusive; ``max_frames`` is the numeric frame window's
    length, not a count of existing files. Missing scans/images are recorded and
    kept independent. Sparse pose rows are stamped by their original image_00
    frame IDs. No pose interpolation or scan motion compensation is performed.
    The default point-time model uses a declared 0.1 s scan period; ``none``
    explicitly omits the time field. Output replaces an existing bag only after
    every selected input has been converted successfully.
    """
    try:
        return _convert(
            Path(dataset_root).resolve(),
            Path(output).resolve(),
            sequence,
            Path(calibration_dir).resolve() if calibration_dir is not None else None,
            camera,
            velodyne_time,
            scan_period_s,
            start_frame,
            max_frames,
            list(command or []),
            overwrite,
            motion_source,
        )
    except (OSError, ValueError) as exc:
        raise DatasetError(f"KITTI-360 conversion failed: {exc}") from exc


def _convert(
    root: Path,
    output: Path,
    sequence: str,
    calibration: Path | None,
    camera: str,
    timing: VelodyneTime,
    period: float,
    start: int,
    count: int | None,
    command: list[str],
    overwrite: bool,
    motion_source: Kitti360MotionSource,
) -> KittiConversionResult:
    if not re.fullmatch(r"2013_05_28_drive_\d{4}_sync", sequence):
        raise DatasetError("sequence must be 2013_05_28_drive_NNNN_sync")
    if camera not in ("image_00", "image_01"):
        raise DatasetError("KITTI-360 perspective camera must be image_00 or image_01")
    if motion_source not in ("published_poses", "oxts"):
        raise DatasetError("motion_source must be published_poses or oxts")
    if timing not in ("none", "azimuth") or not math.isfinite(period) or period <= 0:
        raise DatasetError(
            "point time must be none or azimuth; scan period must be positive/finite"
        )
    if start < 0 or (count is not None and count < 1):
        raise DatasetError("start_frame must be nonnegative and max_frames must be positive")
    stop = None if count is None else start + count
    calibration = calibration or root / "calibration"
    image_root = root / "data_2d_raw" / sequence
    lidar_root = root / "data_3d_raw" / sequence / "velodyne_points"
    pose_path = root / "data_poses" / sequence / "poses.txt"
    for source in (root, calibration, image_root, lidar_root, pose_path.parent):
        if output == source or output in source.parents or source in output.parents:
            raise DatasetError("output must not overlap dataset/calibration input directories")
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise DatasetError(f"bag directory already exists and is not empty: {output}")
        if not (output / "metadata.yaml").is_file():
            raise DatasetError("overwrite requires an existing rosbag2 metadata.yaml")
    camera_stamp = image_root / camera / "timestamps.txt"
    oxts_root = root / "data_poses" / sequence / "oxts"
    body_stamp = (
        oxts_root / "timestamps.txt" if motion_source == "oxts"
        else image_root / "image_00" / "timestamps.txt"
    )
    lidar_stamp = lidar_root / "timestamps.txt"
    images_ns, body_ns, lidar_ns = (_timestamps(p) for p in (camera_stamp, body_stamp, lidar_stamp))
    images = _files(image_root / camera / "data_rect", ".png", start, stop)
    scans = _files(lidar_root / "data", ".bin", start, stop)
    origin_path = None
    if motion_source == "oxts":
        poses, motion_inputs, origin_path = _oxts_poses(oxts_root, start, stop)
    else:
        poses = _poses(pose_path, start, stop)
        motion_inputs = {pose_path}
    world_frame = OXTS_WORLD_FRAME if motion_source == "oxts" else WORLD_FRAME
    for selected, stamps, label in (
        (images, images_ns, "image"),
        (scans, lidar_ns, "Velodyne"),
        (poses, body_ns, "pose"),
    ):
        if set(selected) - stamps.keys():
            raise DatasetError(f"{label}: selected frame IDs lack timestamps")
    for index, path in scans.items():
        if path.stat().st_size == 0 or path.stat().st_size % 16:
            raise DatasetError(f"{path}: expected nonempty float32 Nx4 Velodyne scan ({index})")

    perspective_path = calibration / "perspective.txt"
    cam_pose_path = calibration / "calib_cam_to_pose.txt"
    cam_velo_path = calibration / "calib_cam_to_velo.txt"
    perspective = read_calibration_file(perspective_path)
    cam_poses = read_calibration_file(cam_pose_path)
    suffix = camera[-2:]
    required = {f"P_rect_{suffix}": 12, f"R_rect_{suffix}": 9, f"S_rect_{suffix}": 2}
    if any(len(perspective.get(key, [])) != n for key, n in required.items()):
        raise DatasetError(f"{perspective_path}: missing/malformed perspective calibration")
    projection = np.asarray(perspective[f"P_rect_{suffix}"], dtype=np.float64).reshape(3, 4)
    rect = np.eye(4, dtype=np.float64)
    rect[:3, :3] = np.asarray(perspective[f"R_rect_{suffix}"]).reshape(3, 3)
    _rotation(rect[:3, :3], str(perspective_path))
    size = perspective[f"S_rect_{suffix}"]
    if not all(math.isfinite(v) and v > 0 and float(v).is_integer() for v in size):
        raise DatasetError("rectified image dimensions must be positive integers")
    width, height = int(size[0]), int(size[1])
    k_matrix = projection[:, :3]
    if (
        not np.isfinite(projection).all()
        or k_matrix[0, 0] <= 0
        or k_matrix[1, 1] <= 0
        or not np.allclose(k_matrix[2], [0, 0, 1])
        or abs(float(np.linalg.det(k_matrix))) < 1e-12
    ):
        raise DatasetError("perspective intrinsics must be finite and nonsingular")
    k = tuple(float(v) for v in k_matrix.reshape(-1))
    t_pose_cam0 = _rigid(cam_poses.get("image_00", []), "image_00 to pose")
    t_pose_camera = _rigid(cam_poses.get(camera, []), f"{camera} to pose")
    t_velo_cam0 = _rigid([float(v) for v in cam_velo_path.read_text().split()], "camera0 to velo")
    t_pose_velo = t_pose_cam0 @ np.linalg.inv(t_velo_cam0)
    t_pose_rectified = t_pose_camera @ np.linalg.inv(rect)
    frame = f"cam{int(suffix)}_optical"
    transforms: list[Transform] = [
        (BASE_FRAME, IMU_FRAME, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        _transform(LIDAR_FRAME, t_pose_velo),
        _transform(frame, t_pose_rectified),
    ]
    inputs = {
        camera_stamp,
        body_stamp,
        lidar_stamp,
        *motion_inputs,
        perspective_path,
        cam_pose_path,
        cam_velo_path,
        *images.values(),
        *scans.values(),
    }
    hashes = {str(path): _file_sha256(path) for path in sorted(inputs)}
    digest = hashlib.sha256()
    for name, value in hashes.items():
        path = Path(name)
        label = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        digest.update(f"{label}\0{value}\n".encode())
    source_digest = digest.hexdigest()
    first_ns = min(
        min(images_ns[i] for i in images),
        min(lidar_ns[i] for i in scans),
        min(body_ns[i] for i in poses),
    )
    counts = dict.fromkeys(
        (TF_STATIC_TOPIC, IMAGE_TOPIC, CAMERA_INFO_TOPIC, LIDAR_TOPIC, ODOMETRY_TOPIC), 0
    )
    provenance: dict[str, Any] = {
        "schema_version": CONVERSION_SCHEMA,
        "generator": "calibrex convert kitti-360",
        "generator_version": __version__,
        "git_commit": git_commit(),
        "command": command,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset_family": "kitti_360",
            "sequence": sequence,
            "license": KITTI_LICENSE,
            "input_sha256": source_digest,
            "files_sha256": hashes,
        },
        "selection": {
            "start_frame": start,
            "stop_frame_exclusive": stop,
            "image_ids": list(images),
            "velodyne_ids": list(scans),
            "pose_ids": list(poses),
            "image_without_scan_ids": sorted(images.keys() - scans.keys()),
            "scan_without_image_ids": sorted(scans.keys() - images.keys()),
        },
        "camera": {
            "directory": camera,
            "rectified": True,
            "k": list(k),
            "image_size_px": [width, height],
            "t_camera_velodyne": (np.linalg.inv(t_pose_rectified) @ t_pose_velo).tolist(),
            "convention": "T_pose_rect = T_pose_camera * inverse(R_rect_selected); "
            "CameraInfo P=[K|0], stereo baseline represented once in TF",
        },
        "frames": {
            "base_link": "defined as KITTI-360 GPS/IMU pose frame; imu_link is identical",
            "world": world_frame,
        },
        "velodyne_time": {
            "mode": timing,
            "scan_period_s": period,
            "model": "-atan2(y,x)/(2*pi)*scan_period_s relative to header; "
            "clockwise, forward-facing mid-sweep; modelled, not measured",
        },
        "motion": {
            "mode": motion_source,
            "source": (
                "raw OXTS GNSS/INS Mercator positions and attitudes; "
                "not an image/LiDAR-optimized trajectory or independent metrology"
                if motion_source == "oxts" else
                "published sparse poses.txt (image/LiDAR/OXTS aided); "
                "not independent INS evidence"
            ),
            "origin_packet": str(origin_path.relative_to(root)) if origin_path else None,
            "timestamps": (
                "OXTS packet timestamps by original frame ID; no interpolation"
                if motion_source == "oxts" else
                "image_00 timestamps by original frame ID; no interpolation"
            ),
            "twist": "finite differences between neighboring selected poses in body frame; "
            "approximate interval averages; zero/unknown for a singleton",
            "twist_interval_s": [],
            "scan_motion_compensation": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as scratch:
        stage = Path(scratch) / output.name
        with Rosbag2Writer(
            stage,
            custom_data={
                "generator": provenance["generator"],
                "source_input_sha256": source_digest,
                "source_sequence": sequence,
            },
        ) as writer:
            writer.add_topic(TF_STATIC_TOPIC, "tf2_msgs/msg/TFMessage", latched=True)
            writer.write(
                TF_STATIC_TOPIC,
                first_ns,
                encode_tf_message(transforms, secs=first_ns // 10**9, nsecs=first_ns % 10**9),
            )
            counts[TF_STATIC_TOPIC] = 1
            for topic, message_type in (
                (IMAGE_TOPIC, "sensor_msgs/msg/Image"),
                (CAMERA_INFO_TOPIC, "sensor_msgs/msg/CameraInfo"),
                (LIDAR_TOPIC, "sensor_msgs/msg/PointCloud2"),
                (ODOMETRY_TOPIC, "nav_msgs/msg/Odometry"),
            ):
                writer.add_topic(topic, message_type)
            for index, path in images.items():
                h, w, pixels = _gray_rows(path)
                if (w, h) != (width, height):
                    raise DatasetError(f"{path}: image size differs from perspective.txt")
                stamp = images_ns[index]
                writer.write(
                    CAMERA_INFO_TOPIC,
                    stamp,
                    encode_camera_info(
                        frame_id=frame, timestamp_ns=stamp, width=w, height=h, k=k, d=(0.0,) * 5
                    ),
                )
                writer.write(
                    IMAGE_TOPIC,
                    stamp,
                    encode_image(
                        frame_id=frame,
                        timestamp_ns=stamp,
                        width=w,
                        height=h,
                        encoding="mono8",
                        step=w,
                        data=pixels,
                    ),
                )
                counts[IMAGE_TOPIC] += 1
                counts[CAMERA_INFO_TOPIC] += 1
            for index, path in scans.items():
                points = np.fromfile(path, dtype="<f4").reshape(-1, 4)
                if not np.isfinite(points).all():
                    raise DatasetError(f"{path}: non-finite Velodyne values")
                fields, step = CLOUD_FIELDS, 16
                if timing == "azimuth":
                    points = np.column_stack(
                        (points, azimuth_time_offsets_s(points[:, :3], period))
                    )
                    fields, step = TIMED_CLOUD_FIELDS, 20
                writer.write(
                    LIDAR_TOPIC,
                    lidar_ns[index],
                    encode_pointcloud2(
                        frame_id=LIDAR_FRAME,
                        timestamp_ns=lidar_ns[index],
                        fields=fields,
                        point_step=step,
                        data=np.asarray(points, dtype="<f4").tobytes(),
                    ),
                )
                counts[LIDAR_TOPIC] += 1
            pose_ids = list(poses)
            for position, index in enumerate(pose_ids):
                pose, stamp = poses[index], body_ns[index]
                velocity, rate = np.zeros(3), np.zeros(3)
                interval = 0.0
                if len(pose_ids) > 1:
                    neighbor = pose_ids[
                        position + 1 if position + 1 < len(pose_ids) else position - 1
                    ]
                    interval = (body_ns[neighbor] - stamp) * 1e-9
                    velocity = pose[:3, :3].T @ (poses[neighbor][:3, 3] - pose[:3, 3]) / interval
                    rate = Rotation.from_matrix(
                        pose[:3, :3].T @ poses[neighbor][:3, :3]
                    ).as_rotvec()
                    rate /= interval
                provenance["motion"]["twist_interval_s"].append(abs(interval))
                writer.write(
                    ODOMETRY_TOPIC,
                    stamp,
                    encode_odometry(
                        frame_id=world_frame,
                        child_frame_id=IMU_FRAME,
                        timestamp_ns=stamp,
                        position=_vector(pose[:3, 3]),
                        orientation_xyzw=_quat(pose[:3, :3]),
                        linear_velocity=_vector(velocity),
                        angular_velocity=_vector(rate),
                        twist_covariance=[1e12 if i % 7 == 0 else 0.0 for i in range(36)],
                    ),
                )
                counts[ODOMETRY_TOPIC] += 1
        provenance["message_counts"] = counts
        (stage / CONVERSION_FILENAME).write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if output.exists():
            shutil.rmtree(output)
        stage.replace(output)
    return KittiConversionResult(output, output / CONVERSION_FILENAME, counts, provenance)
