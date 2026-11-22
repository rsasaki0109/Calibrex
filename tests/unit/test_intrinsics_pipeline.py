"""From a bag without camera intrinsics to the camera-IMU extrinsic, end to end.

A camera turned in place (with barrel distortion and no ``CameraInfo``) and an IMU on
the same rig are written to a bag. ``calibrex intrinsics rotate --bag`` calibrates the
camera from that bag and records its topic; ``calibrex check --intrinsics`` then judges
the camera-IMU mounting with those intrinsics, which it skips without them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from calibrex.check.intrinsics_override import IntrinsicsOverride
from calibrex.check.runner import CheckRunOptions, build_calibration_check
from calibrex.data.ros_cdr_writer import encode_image, encode_imu, encode_tf_message
from calibrex.data.rosbag2_writer import Rosbag2Writer

cv2 = pytest.importorskip("cv2")

WIDTH, HEIGHT = 320, 240
TRUE_K = np.array([[280.0, 0.0, 159.5], [0.0, 280.0, 119.5], [0.0, 0.0, 1.0]])
TRUE_D = np.array([-0.15, 0.03, 0.0005, -0.0004, 0.0])
R_IMU_CAM = Rotation.from_euler("xyz", [91.0, -2.0, 88.0], degrees=True).as_matrix()
"""The camera's axes in the IMU frame."""
IMAGE_TOPIC, IMU_TOPIC = "/cam/image_raw", "/imu/data"
CAMERA_FRAME, IMU_FRAME = "camera_optical", "imu_link"
EPOCH_NS = 1_700_000_000_000_000_000
ROOM_HALF_M, CELL_M = 8.0, 0.5
SUBSAMPLES = ((-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25))


def _rays() -> np.ndarray:
    u, v = np.meshgrid(np.arange(WIDTH, dtype=float), np.arange(HEIGHT, dtype=float))
    rays = []
    for du, dv in SUBSAMPLES:
        pixels = np.stack([u.ravel() + du, v.ravel() + dv], 1).reshape(-1, 1, 2)
        normalized = cv2.undistortPointsIter(
            pixels, TRUE_K, TRUE_D, None, None, (cv2.TERM_CRITERIA_COUNT, 20, 0)
        ).reshape(-1, 2)
        rays.append(np.column_stack([normalized, np.ones(len(normalized))]))
    return np.concatenate(rays)


def _hash(a: np.ndarray, b: np.ndarray, face: np.ndarray) -> np.ndarray:
    x = (a * 73856093) ^ (b * 19349663) ^ (face * 83492791)
    x = (x ^ (x >> 13)) * 1274126177
    return ((x ^ (x >> 16)) & 0xFFFF) / 65535.0


def _render(rays: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    directions = rays @ camera_to_world.T
    with np.errstate(divide="ignore", invalid="ignore"):
        exits = np.where(directions > 0, ROOM_HALF_M / directions, -ROOM_HALF_M / directions)
    exits[~np.isfinite(exits)] = np.inf
    axis = np.argmin(exits, axis=1)
    rows = np.arange(len(axis))
    hit = directions * exits[rows, axis][:, None]
    face = axis * 2 + (directions[rows, axis] > 0)
    a = np.where(axis == 0, hit[:, 1], hit[:, 0]) + 50
    b = np.where(axis == 2, hit[:, 1], hit[:, 2]) + 50
    coarse = _hash(
        np.floor(a / CELL_M).astype(np.int64), np.floor(b / CELL_M).astype(np.int64), face
    )
    fine = _hash(
        np.floor(3 * a / CELL_M).astype(np.int64) + 7,
        np.floor(3 * b / CELL_M).astype(np.int64) + 11,
        face + 9,
    )
    image = (40 + 150 * coarse + 50 * fine).reshape(len(SUBSAMPLES), HEIGHT, WIDTH).mean(axis=0)
    return np.clip(cv2.GaussianBlur(image, (0, 0), 0.5), 0, 255).astype(np.uint8)


def _camera_to_world(time_s: float) -> np.ndarray:
    rotvec = [
        0.45 * np.sin(0.9 * time_s),
        0.6 * np.sin(0.55 * time_s + 0.3),
        0.35 * np.sin(0.75 * time_s + 1.0),
    ]
    return Rotation.from_rotvec(rotvec).as_matrix()


def _write_bag(path: Path, seconds: float = 32.0) -> Path:
    """camera-imu cuts the camera's track into 10 s windows and needs three."""

    rays = _rays()
    quaternion = Rotation.from_matrix(R_IMU_CAM).as_quat()
    with Rosbag2Writer(path) as writer:
        writer.add_topic("/tf_static", "tf2_msgs/msg/TFMessage", latched=True)
        writer.add_topic(IMU_TOPIC, "sensor_msgs/msg/Imu")
        writer.add_topic(IMAGE_TOPIC, "sensor_msgs/msg/Image")
        writer.write(
            "/tf_static",
            EPOCH_NS,
            encode_tf_message([(IMU_FRAME, CAMERA_FRAME, (0.03, 0.01, 0.02), tuple(quaternion))]),
        )
        # The IMU at 200 Hz: the camera's body rate in the IMU's axes, and gravity
        # (world +y is down at the start, the camera's own y).
        step = 0.005
        for index in range(int(seconds / step)):
            time_s = index * step
            before, after = _camera_to_world(time_s - 1e-4), _camera_to_world(time_s + 1e-4)
            omega_camera = Rotation.from_matrix(before.T @ after).as_rotvec() / 2e-4
            imu_to_world = _camera_to_world(time_s) @ R_IMU_CAM.T
            specific_force = imu_to_world.T @ np.array([0.0, -9.81, 0.0])
            stamp = EPOCH_NS + round(time_s * 1e9)
            writer.write(
                IMU_TOPIC,
                stamp,
                encode_imu(
                    frame_id=IMU_FRAME,
                    timestamp_ns=stamp,
                    angular_velocity=tuple(R_IMU_CAM @ omega_camera),
                    linear_acceleration=tuple(specific_force),
                ),
            )
        for index in range(int(seconds * 10)):
            time_s = index * 0.1
            stamp = EPOCH_NS + round(time_s * 1e9)
            image = _render(rays, _camera_to_world(time_s))
            writer.write(
                IMAGE_TOPIC,
                stamp,
                encode_image(
                    frame_id=CAMERA_FRAME,
                    timestamp_ns=stamp,
                    height=HEIGHT,
                    width=WIDTH,
                    encoding="mono8",
                    step=WIDTH,
                    data=image.tobytes(),
                ),
            )
    return path


def _camera_imu(bag: Path, tmp_path: Path, intrinsics: tuple[IntrinsicsOverride, ...]):
    artifact = build_calibration_check(
        bag,
        run=CheckRunOptions(
            pairs=("camera-imu",),
            intrinsics=intrinsics,
            evidence_dir=tmp_path / "evidence",
            base_dir=tmp_path,
        ),
    )
    return next(record for record in artifact.pairs if record.pair == "camera-imu")


def test_intrinsics_from_the_bag_let_check_judge_the_camera_imu_mounting(
    tmp_path: Path,
) -> None:
    from calibrex.cli.main import main

    bag = _write_bag(tmp_path / "bag")
    chain_path = tmp_path / "intrinsics_camchain.yaml"

    # Without intrinsics the camera pair cannot run.
    skipped = _camera_imu(bag, tmp_path / "without", ())
    assert (skipped.status, skipped.reason_code) == ("skipped", "missing_intrinsics")
    assert "--intrinsics" in (skipped.reason or "")

    status = main(
        ["intrinsics", "rotate", "--bag", str(bag), "--topic", IMAGE_TOPIC, "--seconds", "8",
         "--output", str(chain_path)]
    )  # fmt: skip
    assert status == 0
    entry = yaml.safe_load(chain_path.read_text())["cam0"]
    assert entry["rostopic"] == IMAGE_TOPIC
    fx, fy, cx, cy = entry["intrinsics"]
    assert fy == pytest.approx(TRUE_K[1, 1], rel=0.01)
    assert cy == pytest.approx(TRUE_K[1, 2], abs=2.0)
    assert fx == pytest.approx(TRUE_K[0, 0], rel=0.01)
    assert cx == pytest.approx(TRUE_K[0, 2], abs=2.0)
    assert entry["distortion_coeffs"][0] == pytest.approx(TRUE_D[0], abs=0.02)

    judged = _camera_imu(bag, tmp_path / "with", (IntrinsicsOverride(chain_path),))
    assert judged.status == "pass", (judged.status, judged.reason)
    assert any(axis.status != "unobservable" for axis in judged.axes or [])
