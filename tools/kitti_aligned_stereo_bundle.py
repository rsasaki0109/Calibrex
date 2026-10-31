"""Build a spent KITTI stereo/point bundle at the camera capture time.

Uses OXTS and IMU-to-LiDAR calibration for motion, never camera-to-LiDAR
reference extrinsics. Stereo depth uses only images and camera intrinsics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict
from scipy.spatial.transform import Rotation, Slerp

from calibrex.data.kitti import read_calibration_file, read_timestamps
from calibrex.data.ros_cdr import decode_ros2_odometry, decode_ros2_pointcloud2
from calibrex.data.rosbag2 import iter_messages


class BundleReport(BaseModel):
    """Input identities, fixed frontend settings and motion diagnostics."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    scope: str
    source_sha256: dict[str, str]
    output_sha256: dict[str, str]
    timing: dict[str, int | float]
    stereo: dict[str, Any]
    motion: dict[str, Any]


def align_points(
    points: np.ndarray,
    point_times: np.ndarray,
    reference_time: float,
    pose_times: np.ndarray,
    positions: np.ndarray,
    quaternions: np.ndarray,
    T_imu_lidar: np.ndarray,
) -> np.ndarray:
    """Move points to LiDAR at reference time using bounded pose interpolation.

    All times are seconds relative to a common integer-nanosecond origin.
    Positions interpolate linearly; orientations use shortest-path SLERP.
    """
    query = np.r_[point_times, reference_time]
    if (
        not np.isfinite(query).all()
        or not np.isfinite(points).all()
        or np.any(np.diff(pose_times) <= 0)
        or query.min() < pose_times[0]
        or query.max() > pose_times[-1]
    ):
        raise ValueError("finite points and complete strictly ordered pose coverage required")
    orientations = Slerp(pose_times, Rotation.from_quat(quaternions))(query)
    locations = np.column_stack([np.interp(query, pose_times, positions[:, i]) for i in range(3)])
    imu_points = points @ T_imu_lidar[:3, :3].T + T_imu_lidar[:3, 3]
    world = orientations[:-1].apply(imu_points) + locations[:-1]
    at_reference_imu = orientations[-1].inv().apply(world - locations[-1])
    return (at_reference_imu - T_imu_lidar[:3, 3]) @ T_imu_lidar[:3, :3]


def frame_timestamp(path: Path, frame_index: int) -> int:
    """Use the original line index; missing KITTI timestamp lines are not renumbered."""
    matches = [item.timestamp_ns for item in read_timestamps(path) if item.index == frame_index]
    if len(matches) != 1:
        raise ValueError(f"missing or ambiguous timestamp for frame {frame_index}: {path}")
    return matches[0]


def select_scan(scans: list[tuple[int, bytes]], timestamp_ns: int) -> bytes:
    """Select the exact timestamp, retaining original IDs in sparsely fetched bags."""
    matches = [data for stamp, data in scans if stamp == timestamp_ns]
    if len(matches) != 1:
        raise ValueError("expected exactly one cloud at the requested KITTI timestamp")
    return matches[0]


def depth_from_disparity(
    disparity: np.ndarray, left_projection: np.ndarray, right_projection: np.ndarray
) -> np.ndarray:
    """Recover left-camera optical z from horizontal disparity and full projections.

    Equal rectified K is required. For right-from-left translation (dx,dy,dz),
    z = ((u_right-cx)*dz - fx*dx) / disparity. Vertical offsets are not
    corrected here: the image matcher still assumes horizontal scanlines.
    """
    kl, kr = left_projection[:, :3], right_projection[:, :3]
    if not np.allclose(kl, kr, atol=1e-6, rtol=0):
        raise ValueError("matching rectified intrinsics required")
    delta = np.linalg.solve(kr, right_projection[:, 3]) - np.linalg.solve(kl, left_projection[:, 3])
    if delta[0] >= 0:
        raise ValueError("positive horizontal stereo baseline required")
    _, u = np.indices(disparity.shape)
    depth = np.full(disparity.shape, np.nan)
    valid = np.isfinite(disparity) & (disparity > 0)
    numerator = (u[valid] - disparity[valid] - kl[0, 2]) * delta[2] - kl[0, 0] * delta[0]
    depth[valid] = numerator / disparity[valid]
    depth[(depth < 0.1) | (depth > 80)] = np.nan
    return depth


def build(
    bag: Path,
    raw: Path,
    calibration: Path,
    right: Path,
    right_times: Path,
    frame_index: int,
    output: Path,
) -> BundleReport:
    """Construct one fixed-index bundle; reject missing timing instead of extrapolating."""
    import cv2

    output.mkdir(parents=True, exist_ok=False)
    poses, scans = [], []
    for connection, stamp, data in iter_messages(bag):
        if connection.topic == "/oxts/odometry":
            poses.append(decode_ros2_odometry(connection.topic, stamp, data))
        elif connection.topic == "/velodyne_points":
            scans.append((stamp, data))
    stamp = frame_timestamp(raw / "velodyne_points/timestamps.txt", frame_index)
    payload = select_scan(scans, stamp)
    scan = decode_ros2_pointcloud2("/velodyne_points", stamp, payload, point_time_field="time")
    if scan.nonfinite_xyz_count or scan.point_time_offsets_s is None or scan.intensity is None:
        raise ValueError("complete finite timed/intensity scan required")
    left_path = raw / f"image_02/data/{frame_index:010d}.png"
    left_times = raw / "image_02/timestamps.txt"
    image_stamp = frame_timestamp(left_times, frame_index)
    right_stamp = frame_timestamp(right_times, frame_index)
    if abs(image_stamp - right_stamp) > 1_000_000:
        raise ValueError("stereo image timestamps differ by more than 1 ms")
    origin = poses[0].timestamp_ns
    pose_times = np.array([(p.timestamp_ns - origin) * 1e-9 for p in poses])
    point_times = (scan.timestamp_ns - origin) * 1e-9 + scan.point_time_offsets_s
    imu_file = calibration / "calib_imu_to_velo.txt"
    imu = read_calibration_file(imu_file)
    T_lidar_imu = np.eye(4)
    T_lidar_imu[:3, :3] = np.array(imu["R"]).reshape(3, 3)
    T_lidar_imu[:3, 3] = imu["T"]
    points = align_points(
        scan.xyz,
        point_times,
        (image_stamp - origin) * 1e-9,
        pose_times,
        np.array([p.position for p in poses]),
        np.array([p.orientation_xyzw for p in poses]),
        np.linalg.inv(T_lidar_imu),
    )
    left_image = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(str(right), cv2.IMREAD_GRAYSCALE)
    if left_image is None or right_image is None or left_image.shape != right_image.shape:
        raise ValueError("matching rectified stereo images required")
    cam_file = calibration / "calib_cam_to_cam.txt"
    cam = read_calibration_file(cam_file)
    pl, pr = [np.array(cam["P_rect_" + s]).reshape(3, 4) for s in ("02", "03")]
    if not np.allclose(pl[:2, :3], pr[:2, :3], atol=1e-6):
        raise ValueError("this provider requires matching rectified intrinsics")
    baseline = float(pl[0, 3] / pl[0, 0] - pr[0, 3] / pr[0, 0])
    if baseline <= 0:
        raise ValueError("positive stereo baseline required")
    settings = {
        "minDisparity": 0,
        "numDisparities": 128,
        "blockSize": 5,
        "P1": 200,
        "P2": 800,
        "disp12MaxDiff": 1,
        "preFilterCap": 31,
        "uniquenessRatio": 10,
        "speckleWindowSize": 100,
        "speckleRange": 2,
        "mode": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    }
    disparity = (
        cv2.StereoSGBM_create(**settings).compute(left_image, right_image).astype(float) / 16
    )
    depth = depth_from_disparity(disparity, pl, pr)
    np.save(output / "depth.npy", depth)
    np.save(output / "points.npy", points)
    cv2.imwrite(str(output / "aligned.png"), cv2.equalizeHist(left_image))
    # Reuse Koide's intensity rank-equalization rule with 256 bins.
    order = np.argsort(scan.intensity, kind="stable")
    intensity = np.empty(len(order), dtype=np.float32)
    intensity[order] = np.floor(np.arange(len(order)) * 256 / len(order)) / 256
    packed = np.column_stack([points, intensity]).astype("<f4")
    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex "
        + str(len(points))
        + "\nproperty float x\nproperty float y\nproperty float z\n"
        + "property float intensity\nend_header\n"
    )
    (output / "aligned.ply").write_bytes(header.encode() + packed.tobytes())
    calib = {
        "camera": {
            "camera_model": "plumb_bob",
            "intrinsics": [float(pl[0, 0]), float(pl[1, 1]), float(pl[0, 2]), float(pl[1, 2])],
            "distortion_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
        },
        "meta": {"bag_names": ["aligned"]},
    }
    (output / "calib.json").write_text(json.dumps(calib, indent=2) + "\n")
    shifts = np.linalg.norm(points - scan.xyz, axis=1)
    files = [left_path, right, left_times, right_times, imu_file, cam_file, bag]
    source = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    source["driver"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = BundleReport(
        scope="spent-data stereo-assisted frontend; no camera-LiDAR reference used",
        source_sha256=source,
        output_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()
        },
        timing={
            "image_header_ns": image_stamp,
            "right_image_header_ns": right_stamp,
            "scan_header_ns": scan.timestamp_ns,
            "image_minus_scan_s": (image_stamp - scan.timestamp_ns) * 1e-9,
            "first_point_minus_image_s": float(point_times.min() - (image_stamp - origin) * 1e-9),
            "last_point_minus_image_s": float(point_times.max() - (image_stamp - origin) * 1e-9),
        },
        stereo={
            "provider": "OpenCV StereoSGBM",
            "opencv_version": cv2.__version__,
            "settings": settings,
            "baseline_m": float(
                (np.linalg.solve(pl[:, :3], pl[:, 3]) - np.linalg.solve(pr[:, :3], pr[:, 3]))[0]
            ),
            "legacy_effective_baseline_m": baseline,
            "depth_formula": "full-projection horizontal disparity/v1",
            "valid_pixels": int(np.isfinite(depth).sum()),
            "total_pixels": depth.size,
            "depth_convention": "optical-axis z metres",
            "depth_range_m": [0.1, 80.0],
        },
        motion={
            "model": "OXTS linear position and SLERP rotation; KITTI modeled point times",
            "point_count": len(points),
            "median_point_shift_m": float(np.median(shifts)),
            "max_point_shift_m": float(shifts.max()),
            "extrapolation": False,
            "auxiliary_calibration": "IMU-to-LiDAR only",
        },
    )
    (output / "frontend.json").write_text(report.model_dump_json(indent=2) + "\n")
    (output / "frontend.schema.json").write_text(
        json.dumps(BundleReport.model_json_schema(), indent=2) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["bag", "raw", "calibration", "right", "right_times", "output"]:
        parser.add_argument(name, type=Path)
    parser.add_argument("--frame-index", type=int, default=1)
    a = parser.parse_args()
    report = build(a.bag, a.raw, a.calibration, a.right, a.right_times, a.frame_index, a.output)
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
