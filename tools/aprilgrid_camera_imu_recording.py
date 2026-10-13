#!/usr/bin/env python3
"""Turn a Kalibr AprilGrid recording into a ``slac.camera_imu_motion_recording/v0.1``.

A data-prep adapter, not part of the Calibrex package; it needs OpenCV
(``pip install opencv-python-headless``), which Calibrex itself does not depend on.
For every image of the camera topic it detects the AprilGrid's tags (AprilTag 36h11),
solves the camera pose against the grid with the camchain's intrinsics, and writes the
camera orientations and positions (grid frame) together with the IMU's gyroscope and
accelerometer samples, for ``calibrex external-run evaluate-camera-imu``::

    python tools/aprilgrid_camera_imu_recording.py BAG \\
        --camchain camchain-imucam.yaml --target target.yaml \\
        --image-topic /dvs/image_raw --imu-topic /dvs/imu --output recording.json

Stamps are message header stamps, shifted by one common origin (the first IMU stamp),
so the camera-IMU clock offset is preserved. A frame is kept only when at least
``--min-tags`` tags are seen and the pose reprojects them within
``--max-reprojection-px`` (RMS). The camchain's intrinsics are used for the poses;
only its ``T_cam_imu`` and ``timeshift_cam_imu`` are under test afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from calibrex.core.external_camera_imu_evidence import (
    CameraImuMotionRecording,
    CameraImuMotionRecordingProvenance,
    CameraOrientationSample,
    ImuAccelSample,
    ImuGyroSample,
)
from calibrex.data.rosbag2 import decode_image, decode_imu, iter_messages

# Where OpenCV's four tag corners (in its detection order) sit in Kalibr's grid frame
# (tag i at column i % tagCols, row i // tagCols, in units of the tag size). Found by
# reprojection on a Kalibr grid: every other assignment reprojects tens of pixels off.
_TAG_CORNERS = ((1.0, 0.0), (0.0, 0.0), (0.0, 1.0), (1.0, 1.0))


def grid_corners(target: dict[str, Any]) -> dict[int, np.ndarray]:
    """The four 3D corners (z = 0) of every tag id of a Kalibr ``aprilgrid`` target."""

    if target.get("target_type") != "aprilgrid":
        raise ValueError("expected a Kalibr target.yaml with target_type 'aprilgrid'")
    size = float(target["tagSize"])
    pitch = size * (1.0 + float(target["tagSpacing"]))
    columns, rows = int(target["tagCols"]), int(target["tagRows"])
    corners: dict[int, np.ndarray] = {}
    for tag in range(columns * rows):
        origin = np.asarray([(tag % columns) * pitch, (tag // columns) * pitch])
        corners[tag] = np.asarray(
            [[*(origin + size * np.asarray(corner)), 0.0] for corner in _TAG_CORNERS]
        )
    return corners


def camera_model(camchain: dict[str, Any], camera: str) -> tuple[np.ndarray, np.ndarray, str]:
    entry = camchain[camera]
    if entry.get("camera_model") != "pinhole":
        raise ValueError(f"{camera}: only Kalibr's pinhole camera model is supported")
    fx, fy, cx, cy = (float(value) for value in entry["intrinsics"])
    matrix = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    distortion = np.asarray(entry.get("distortion_coeffs", []), dtype=float)
    model = str(entry.get("distortion_model", "none"))
    if model not in {"equidistant", "radtan", "none"}:
        raise ValueError(f"{camera}: unsupported distortion model {model!r}")
    return matrix, distortion, model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("bag", type=Path, help="ROS 1 .bag or rosbag2 with images and IMU")
    parser.add_argument("--camchain", type=Path, required=True, help="Kalibr camchain(-imucam)")
    parser.add_argument("--target", type=Path, required=True, help="Kalibr aprilgrid target.yaml")
    parser.add_argument("--camera", default="cam0")
    parser.add_argument("--image-topic", required=True)
    parser.add_argument("--imu-topic", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-tags", type=int, default=4)
    parser.add_argument("--max-reprojection-px", type=float, default=1.0)
    parser.add_argument(
        "--detect-scale",
        type=int,
        default=1,
        help="upsample images by this factor for tag detection (small sensors such as DAVIS)",
    )
    args = parser.parse_args(argv)
    try:
        import cv2
    except ImportError:
        parser.error("needs OpenCV: pip install opencv-python-headless")

    camchain = yaml.safe_load(args.camchain.read_text(encoding="utf-8"))
    matrix, distortion, model = camera_model(camchain, args.camera)
    corners = grid_corners(yaml.safe_load(args.target.read_text(encoding="utf-8")))
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()
    parameters.markerBorderBits = 2  # Kalibr's AprilGrid tags have a two-bit black border
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    def normalized(pixels: np.ndarray) -> np.ndarray:
        points = pixels.reshape(-1, 1, 2).astype(np.float64)
        if model == "equidistant":
            return cv2.fisheye.undistortPoints(points, matrix, distortion).reshape(-1, 2)
        coefficients = distortion if model == "radtan" else None
        return cv2.undistortPoints(points, matrix, coefficients).reshape(-1, 2)

    def project(points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        if model == "equidistant":
            projected, _ = cv2.fisheye.projectPoints(
                points.reshape(-1, 1, 3), rvec, tvec, matrix, distortion
            )
        else:
            coefficients = distortion if model == "radtan" else None
            projected, _ = cv2.projectPoints(points, rvec, tvec, matrix, coefficients)
        return projected.reshape(-1, 2)

    images: list[tuple[int, np.ndarray]] = []
    gyro: list[tuple[int, list[float]]] = []
    accel: list[tuple[int, list[float]]] = []
    for connection, timestamp_ns, payload in iter_messages(
        args.bag, topics={args.image_topic, args.imu_topic}
    ):
        if connection.topic == args.imu_topic:
            imu = decode_imu(connection.topic, timestamp_ns, payload)
            gyro.append((imu.timestamp_ns, list(imu.angular_velocity)))
            accel.append((imu.timestamp_ns, list(imu.linear_acceleration)))
            continue
        image = decode_image(connection.topic, timestamp_ns, payload)
        if image.encoding not in {"mono8", "8UC1"} or image.data is None:
            parser.error(f"expected mono8 images, got {image.encoding}")
        pixels = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.step)
        stamp = image.header_stamp_ns or image.timestamp_ns
        images.append((stamp, pixels[:, : image.width].copy()))
    if not gyro or not images:
        parser.error("the bag has no images or no IMU samples on the given topics")
    gyro.sort()
    accel.sort()
    origin_ns = gyro[0][0]

    samples: list[CameraOrientationSample] = []
    errors: list[float] = []
    rejected = {"too_few_tags": 0, "reprojection": 0}
    for stamp, pixels in sorted(images, key=lambda item: item[0]):
        scale = args.detect_scale
        if scale > 1:
            pixels = cv2.resize(pixels, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        detected, ids, _ = detector.detectMarkers(pixels)
        if ids is None:
            rejected["too_few_tags"] += 1
            continue
        pairs = [
            (corners[int(tag)], (found.reshape(4, 2) + 0.5) / scale - 0.5)
            for tag, found in zip(ids.reshape(-1), detected, strict=True)
            if int(tag) in corners
        ]
        if len(pairs) < args.min_tags:
            rejected["too_few_tags"] += 1
            continue
        object_points = np.concatenate([model_corners for model_corners, _ in pairs])
        image_points = np.concatenate([found for _, found in pairs])
        ok, rvec, tvec = cv2.solvePnP(
            object_points, normalized(image_points), np.eye(3), None, flags=cv2.SOLVEPNP_IPPE
        )
        if not ok:
            rejected["reprojection"] += 1
            continue
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            normalized(image_points),
            np.eye(3),
            None,
            rvec,
            tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        residual = project(object_points, rvec, tvec) - image_points
        rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        if not ok or rms > args.max_reprojection_px:
            rejected["reprojection"] += 1
            continue
        errors.append(rms)
        rotation_cam_grid, _ = cv2.Rodrigues(rvec)
        world_cam = rotation_cam_grid.T
        position = -world_cam @ tvec.reshape(3)
        quaternion = _quaternion_xyzw(world_cam)
        samples.append(
            CameraOrientationSample(
                timestamp_sec=(stamp - origin_ns) * 1.0e-9,
                rotation_quat_xyzw=quaternion,
                position_m=[float(value) for value in position],
            )
        )
    if len(samples) < 2:
        parser.error(f"only {len(samples)} usable frames: {rejected}")

    bag_sha = hashlib.sha256(args.bag.read_bytes()).hexdigest() if args.bag.is_file() else None
    recording = CameraImuMotionRecording(
        recording_id=f"aprilgrid:{args.bag.stem}:{args.camera}",
        camera_name=args.camera,
        camera_orientations=samples,
        imu_gyro=[
            ImuGyroSample(timestamp_sec=(stamp - origin_ns) * 1.0e-9, omega_rad_s=values)
            for stamp, values in _unique(gyro)
        ],
        imu_accel=[
            ImuAccelSample(timestamp_sec=(stamp - origin_ns) * 1.0e-9, specific_force_m_s2=values)
            for stamp, values in _unique(accel)
        ],
        provenance=CameraImuMotionRecordingProvenance(
            producer="tools/aprilgrid_camera_imu_recording.py",
            producer_version="0.1",
            source=(
                f"{args.bag.name} (sha256 {bag_sha}); {args.image_topic} poses against the "
                f"AprilGrid of {args.target.name} with {args.camchain.name} {args.camera} "
                f"intrinsics; {args.imu_topic}; header stamps minus {origin_ns} ns"
            ),
            synthetic=False,
        ),
    )
    recording.save(args.output)
    print(
        f"{len(samples)}/{len(images)} frames posed (median reprojection "
        f"{np.median(errors):.3f} px; rejected {rejected}); {len(recording.imu_gyro)} IMU "
        f"samples; wrote {args.output}"
    )
    return 0


def _unique(samples: list[tuple[int, list[float]]]) -> list[tuple[int, list[float]]]:
    """Drop repeated stamps (the recording requires strictly increasing ones)."""

    kept: list[tuple[int, list[float]]] = []
    for stamp, values in samples:
        if not kept or stamp > kept[-1][0]:
            kept.append((stamp, [float(value) for value in values]))
    return kept


def _quaternion_xyzw(rotation: np.ndarray) -> list[float]:
    from scipy.spatial.transform import Rotation

    return [float(value) for value in Rotation.from_matrix(rotation).as_quat()]


if __name__ == "__main__":
    sys.exit(main())
