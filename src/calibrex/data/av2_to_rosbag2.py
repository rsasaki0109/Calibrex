"""Convert one Argoverse 2 sensor-dataset log to rosbag2 for ``calibrex check``.

An Argoverse 2 log distributes its LiDAR sweeps in the ego-vehicle frame (two stacked
VLP-32C merged into one sweep, coordinates in float16; ``laser_number`` 0-31 are the
``up_lidar``, 32-63 the upside-down ``down_lidar``) with each point's ``offset_ns`` from
the sweep timestamp, ring and stereo camera JPEGs with radial distortion ``k1, k2, k3``,
a per-log ``calibration/`` and the ``city_SE3_egovehicle`` poses.

The bag holds one camera (``/camera/image_raw`` mono8 and ``/camera/camera_info``
without distortion), one LiDAR's points moved back into that sensor's frame
(``/velodyne_points`` stamped at the sweep timestamp, a snapshot without per-point
times by default), the poses (``/av2/odometry`` with a finite-difference body twist) and
a static TF from the per-log calibration. The sensor frame matters: ``calibrex check``
recovers beams from each point's elevation, which a merged ego-frame cloud (origin
about 1.6 m below the sensors) does not preserve.

Images and sweeps follow the official ``av2-api`` conventions: its ``PinholeCamera``
projects with ``K`` alone, so the images are treated as undistorted, and its
motion-compensated projection moves a whole sweep with the single ego pose at the sweep
timestamp. ``point_time="offset_ns"`` instead keeps each point's ``offset_ns`` as a
``time`` field (development comparison only).

Reading feather files needs ``pyarrow`` (``pip install calibrex[av2]``); JPEG decoding
needs Pillow.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from calibrex import __version__
from calibrex.core.exceptions import DatasetError
from calibrex.core.provenance import git_commit
from calibrex.data.kitti_raw_to_rosbag2 import (
    BASE_FRAME,
    CAMERA_INFO_TOPIC,
    CLOUD_FIELDS,
    CONVERSION_FILENAME,
    CONVERSION_SCHEMA,
    IMAGE_TOPIC,
    IMU_FRAME,
    LIDAR_FRAME,
    LIDAR_TOPIC,
    TF_STATIC_TOPIC,
    TIMED_CLOUD_FIELDS,
    FloatArray,
    KittiConversionResult,
    _file_sha256,
    _quat,
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

ODOMETRY_TOPIC = "/av2/odometry"
WORLD_FRAME = "av2_city"
AV2_LICENSE = "CC BY-NC-SA 4.0"
CAMERAS = (
    "ring_front_center",
    "ring_front_left",
    "ring_front_right",
    "ring_rear_left",
    "ring_rear_right",
    "ring_side_left",
    "ring_side_right",
    "stereo_front_left",
    "stereo_front_right",
)
LIDARS = {"up_lidar": range(0, 32), "down_lidar": range(32, 64)}
POINT_TIMES = ("sweep", "offset_ns")
"""Each LiDAR's ``laser_number`` values in a merged Argoverse 2 sweep."""
TWIST_HALF_WINDOW_S = 0.05
"""Poses about this far before and after a pose give its finite-difference twist."""


def _feather() -> Any:
    try:
        return importlib.import_module("pyarrow.feather")
    except ImportError as exc:
        raise DatasetError(
            "reading Argoverse 2 feather files needs pyarrow: pip install 'calibrex[av2]'"
        ) from exc


def _image_module() -> Any:
    try:
        return importlib.import_module("PIL.Image")
    except ImportError as exc:
        raise DatasetError(
            "decoding Argoverse 2 JPEGs needs Pillow: pip install 'calibrex[av2]'"
        ) from exc


def _table(path: Path) -> dict[str, list[Any]]:
    if not path.is_file():
        raise DatasetError(f"missing Argoverse 2 file: {path}")
    columns: dict[str, list[Any]] = _feather().read_table(path).to_pydict()
    return columns


def _rows(table: dict[str, list[Any]]) -> list[dict[str, Any]]:
    names = list(table)
    return [dict(zip(names, values, strict=True)) for values in zip(*table.values(), strict=True)]


def _se3(row: dict[str, Any], label: str) -> FloatArray:
    quaternion = np.array([row["qx"], row["qy"], row["qz"], row["qw"]], dtype=np.float64)
    translation = np.array([row["tx_m"], row["ty_m"], row["tz_m"]], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(quaternion).all() or not np.isfinite(translation).all() or norm < 1e-9:
        raise DatasetError(f"{label}: non-finite or zero pose")
    if abs(norm - 1.0) > 1e-3:
        raise DatasetError(f"{label}: quaternion is not unit length ({norm:.6f})")
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(quaternion / norm).as_matrix()
    matrix[:3, 3] = translation
    return matrix


def _vector(values: FloatArray) -> tuple[float, float, float]:
    return float(values[0]), float(values[1]), float(values[2])


def _stamped(folder: Path, suffix: str) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in sorted(folder.glob(f"*{suffix}")):
        try:
            files[int(path.stem)] = path
        except ValueError as exc:
            raise DatasetError(f"{path}: file name is not a nanosecond timestamp") from exc
    return files


def _twists(
    stamps: np.ndarray, poses: list[FloatArray]
) -> list[tuple[FloatArray, FloatArray, float]]:
    """Body-frame velocity, angular rate and interval of each pose, by finite differences."""

    half = round(TWIST_HALF_WINDOW_S * 1e9)
    result = []
    for index, stamp in enumerate(stamps):
        before = int(np.searchsorted(stamps, stamp - half, side="right")) - 1
        after = int(np.searchsorted(stamps, stamp + half, side="left"))
        before = max(before, 0)
        after = min(after, len(stamps) - 1)
        if after == before:
            result.append((np.zeros(3), np.zeros(3), 0.0))
            continue
        # Differences of integer nanoseconds: absolute stamps (~3e17 ns) lose about 6e-8 s
        # as float seconds, a relative error near 1e-6 on a 0.1 s interval.
        interval = int(stamps[after] - stamps[before]) * 1e-9
        first, last = poses[before], poses[after]
        rotation = poses[index][:3, :3]
        velocity = rotation.T @ (last[:3, 3] - first[:3, 3]) / interval
        rate = Rotation.from_matrix(first[:3, :3].T @ last[:3, :3]).as_rotvec() / interval
        result.append((velocity, rate, float(interval)))
    return result


def convert_av2_to_rosbag2(
    log_dir: str | Path,
    output: str | Path,
    *,
    camera: str = "ring_front_center",
    lidar: str = "up_lidar",
    point_time: str = "sweep",
    start_s: float = 0.0,
    max_seconds: float | None = None,
    command: list[str] | None = None,
    overwrite: bool = False,
) -> KittiConversionResult:
    """Convert one Argoverse 2 sensor log (or a time window of it) to a rosbag2 directory.

    ``start_s`` and ``max_seconds`` select sweeps and images whose timestamps lie in
    ``[first sweep + start_s, that + max_seconds)``; poses cover the window with a
    margin for the twist. The output replaces an existing bag only with ``overwrite``
    and only after every selected input has converted.
    """

    root, out = Path(log_dir).resolve(), Path(output).resolve()
    if camera not in CAMERAS:
        raise DatasetError(f"camera must be one of {', '.join(CAMERAS)}")
    if lidar not in LIDARS:
        raise DatasetError(f"lidar must be one of {', '.join(LIDARS)}")
    if point_time not in POINT_TIMES:
        raise DatasetError(f"point_time must be one of {', '.join(POINT_TIMES)}")
    if not math.isfinite(start_s) or start_s < 0.0:
        raise DatasetError("start_s must be finite and nonnegative")
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0.0):
        raise DatasetError("max_seconds must be finite and positive")
    if out == root or root in out.parents or out in root.parents:
        raise DatasetError("output must not overlap the log directory")
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise DatasetError(f"bag directory already exists and is not empty: {out}")
        if not (out / "metadata.yaml").is_file():
            raise DatasetError("overwrite requires an existing rosbag2 metadata.yaml")
    try:
        return _convert(
            root, out, camera, lidar, point_time, start_s, max_seconds, list(command or [])
        )
    except (OSError, ValueError, KeyError) as exc:
        raise DatasetError(f"Argoverse 2 conversion failed: {exc}") from exc


def _convert(
    root: Path,
    output: Path,
    camera: str,
    lidar: str,
    point_time: str,
    start_s: float,
    max_seconds: float | None,
    command: list[str],
) -> KittiConversionResult:
    extrinsics_path = root / "calibration" / "egovehicle_SE3_sensor.feather"
    intrinsics_path = root / "calibration" / "intrinsics.feather"
    poses_path = root / "city_SE3_egovehicle.feather"
    extrinsics = {row["sensor_name"]: row for row in _rows(_table(extrinsics_path))}
    intrinsics = {row["sensor_name"]: row for row in _rows(_table(intrinsics_path))}
    if camera not in extrinsics or camera not in intrinsics or lidar not in extrinsics:
        raise DatasetError(f"{root}: no calibration for {camera} or {lidar}")
    ego_camera = _se3(extrinsics[camera], f"{camera} extrinsic")
    ego_lidar = _se3(extrinsics[lidar], f"{lidar} extrinsic")
    lidar_ego = np.linalg.inv(ego_lidar)
    lasers = LIDARS[lidar]
    intrinsic = intrinsics[camera]
    fx, fy = float(intrinsic["fx_px"]), float(intrinsic["fy_px"])
    cx, cy = float(intrinsic["cx_px"]), float(intrinsic["cy_px"])
    width, height = int(intrinsic["width_px"]), int(intrinsic["height_px"])
    source_distortion = [float(intrinsic[key]) for key in ("k1", "k2", "k3")]
    values = (fx, fy, cx, cy, *source_distortion)
    if not all(math.isfinite(v) for v in values) or fx <= 0 or fy <= 0 or width < 1 or height < 1:
        raise DatasetError(f"{camera}: invalid intrinsics")
    k = (fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0)

    sweeps = _stamped(root / "sensors" / "lidar", ".feather")
    images = _stamped(root / "sensors" / "cameras" / camera, ".jpg")
    if not sweeps or not images:
        raise DatasetError(f"{root}: needs sensors/lidar/*.feather and {camera} JPEGs")
    first = min(sweeps) + round(start_s * 1e9)
    stop = None if max_seconds is None else first + round(max_seconds * 1e9)

    def inside(stamp: int) -> bool:
        return stamp >= first and (stop is None or stamp < stop)

    sweeps = {s: p for s, p in sweeps.items() if inside(s)}
    images = {s: p for s, p in images.items() if inside(s)}
    if not sweeps or not images:
        raise DatasetError("the selected window holds no sweep or no image")

    pose_table = _table(poses_path)
    order = np.argsort(np.asarray(pose_table["timestamp_ns"], dtype=np.int64))
    pose_rows = [_rows(pose_table)[i] for i in order]
    stamps = np.asarray([row["timestamp_ns"] for row in pose_rows], dtype=np.int64)
    if len(stamps) < 2 or np.any(np.diff(stamps) <= 0):
        raise DatasetError(f"{poses_path}: needs at least two strictly increasing poses")
    poses = [_se3(row, f"pose {row['timestamp_ns']}") for row in pose_rows]
    twists = _twists(stamps, poses)
    margin = round(4 * TWIST_HALF_WINDOW_S * 1e9)
    last = max(*sweeps, *images)
    written = [i for i, s in enumerate(stamps) if min(sweeps) - margin <= s <= last + margin]
    if not written:
        raise DatasetError("no pose covers the selected window")

    frame = f"{camera}_optical"
    transforms: list[Transform] = [
        (BASE_FRAME, IMU_FRAME, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        (IMU_FRAME, LIDAR_FRAME, _vector(ego_lidar[:3, 3]), _quat(ego_lidar[:3, :3])),
        (IMU_FRAME, frame, _vector(ego_camera[:3, 3]), _quat(ego_camera[:3, :3])),
    ]
    inputs = [extrinsics_path, intrinsics_path, poses_path, *images.values(), *sweeps.values()]
    hashes = {str(path.relative_to(root)): _file_sha256(path) for path in sorted(inputs)}
    digest = hashlib.sha256()
    for name, value in hashes.items():
        digest.update(f"{name}\0{value}\n".encode())
    split = root.parent.name if root.parent.name in ("train", "val", "test") else None
    provenance: dict[str, Any] = {
        "schema_version": CONVERSION_SCHEMA,
        "generator": "calibrex convert av2",
        "generator_version": __version__,
        "git_commit": git_commit(),
        "command": command,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset_family": "argoverse2_sensor",
            "log_id": root.name,
            "split": split,
            "license": AV2_LICENSE,
            "input_sha256": digest.hexdigest(),
            "files_sha256": hashes,
        },
        "selection": {
            "start_s": start_s,
            "max_seconds": max_seconds,
            "first_ns": first,
            "stop_ns_exclusive": stop,
            "sweeps": len(sweeps),
            "images": len(images),
            "poses": len(written),
        },
        "lidar": {
            "sensor": lidar,
            "laser_numbers": [lasers.start, lasers.stop - 1],
            "points": "ego-frame points of these lasers moved into the sensor frame with "
            "inverse(egovehicle_SE3_sensor); float16 in the source",
        },
        "camera": {
            "directory": camera,
            "frame": frame,
            "rectified": False,
            "k": list(k),
            "distortion_model": "none",
            "d": [],
            "source_k1_k2_k3": source_distortion,
            "image_size_px": [width, height],
            "t_camera_velodyne": (np.linalg.inv(ego_camera) @ ego_lidar).tolist(),
            "t_pose_velodyne": ego_lidar.tolist(),
            "convention": "imu_link = base_link = egovehicle; both TF edges are "
            "egovehicle_SE3_sensor of calibration/; images treated as undistorted "
            "(av2-api PinholeCamera projects with K alone); source k1..k3 recorded only",
        },
        "frames": {"base_link": "Argoverse 2 egovehicle frame", "world": WORLD_FRAME},
        "velodyne_time": {
            "mode": point_time,
            "header": "sweep timestamp",
            "model": "a snapshot at the header; av2-api moves a sweep with one ego pose"
            if point_time == "sweep"
            else "time = offset_ns * 1e-9 after the header (development comparison)",
        },
        "motion": {
            "mode": "city_SE3_egovehicle",
            "twist": "finite differences between the poses nearest "
            f"{TWIST_HALF_WINDOW_S:g} s before and after, in the body frame",
            "scan_motion_compensation": False,
        },
    }
    counts = dict.fromkeys(
        (TF_STATIC_TOPIC, IMAGE_TOPIC, CAMERA_INFO_TOPIC, LIDAR_TOPIC, ODOMETRY_TOPIC), 0
    )
    image_module = _image_module()
    first_ns = min(min(sweeps), min(images), int(stamps[written[0]]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as scratch:
        stage = Path(scratch) / output.name
        with Rosbag2Writer(
            stage,
            custom_data={
                "generator": provenance["generator"],
                "source_input_sha256": digest.hexdigest(),
                "source_log": root.name,
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
            for stamp, path in images.items():
                with image_module.open(path) as image:
                    gray = image.convert("L")
                    if gray.size != (width, height):
                        raise DatasetError(f"{path}: image size differs from intrinsics.feather")
                    pixels = gray.tobytes()
                writer.write(
                    CAMERA_INFO_TOPIC,
                    stamp,
                    encode_camera_info(
                        frame_id=frame,
                        timestamp_ns=stamp,
                        width=width,
                        height=height,
                        k=k,
                        distortion_model="plumb_bob",
                        d=(0.0,) * 5,
                    ),
                )
                writer.write(
                    IMAGE_TOPIC,
                    stamp,
                    encode_image(
                        frame_id=frame,
                        timestamp_ns=stamp,
                        width=width,
                        height=height,
                        encoding="mono8",
                        step=width,
                        data=pixels,
                    ),
                )
                counts[IMAGE_TOPIC] += 1
                counts[CAMERA_INFO_TOPIC] += 1
            for stamp, path in sweeps.items():
                table = _table(path)
                laser = np.asarray(table["laser_number"], dtype=np.int64)
                mine = (laser >= lasers.start) & (laser < lasers.stop)
                ego = np.column_stack(
                    [np.asarray(table[name], dtype=np.float64)[mine] for name in ("x", "y", "z")]
                )
                columns = [
                    ego @ lidar_ego[:3, :3].T + lidar_ego[:3, 3],
                    np.asarray(table["intensity"], dtype=np.float64)[mine],
                ]
                fields, step = CLOUD_FIELDS, 16
                if point_time == "offset_ns":
                    columns.append(np.asarray(table["offset_ns"], dtype=np.float64)[mine] * 1e-9)
                    fields, step = TIMED_CLOUD_FIELDS, 20
                points = np.column_stack(columns)
                if not np.isfinite(points).all():
                    raise DatasetError(f"{path}: non-finite LiDAR values")
                writer.write(
                    LIDAR_TOPIC,
                    stamp,
                    encode_pointcloud2(
                        frame_id=LIDAR_FRAME,
                        timestamp_ns=stamp,
                        fields=fields,
                        point_step=step,
                        data=np.asarray(points, dtype="<f4").tobytes(),
                    ),
                )
                counts[LIDAR_TOPIC] += 1
            for index in written:
                pose = poses[index]
                velocity, rate, _interval = twists[index]
                stamp = int(stamps[index])
                writer.write(
                    ODOMETRY_TOPIC,
                    stamp,
                    encode_odometry(
                        frame_id=WORLD_FRAME,
                        child_frame_id=IMU_FRAME,
                        timestamp_ns=stamp,
                        position=(float(pose[0, 3]), float(pose[1, 3]), float(pose[2, 3])),
                        orientation_xyzw=_quat(pose[:3, :3]),
                        linear_velocity=(
                            float(velocity[0]),
                            float(velocity[1]),
                            float(velocity[2]),
                        ),
                        angular_velocity=(float(rate[0]), float(rate[1]), float(rate[2])),
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
