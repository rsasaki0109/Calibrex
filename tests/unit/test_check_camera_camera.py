"""camera-camera in ``calibrex check`` on a synthetic rosbag2 of a fisheye stereo pair.

The bag holds eight synchronised image pairs of the textured room of ``stereo_scene``;
the candidate comes from a Kalibr camchain given with ``--tf`` (cam0 is the IMU frame),
which also supplies both cameras' equidistant intrinsics. The planner, the adapter, the
evidence artifact and the verdict all run for real.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation
from tests.unit.stereo_scene import LEFT, RIGHT, TRUTH, render_pairs

from calibrex.check.runner import CheckRunOptions, build_calibration_check
from calibrex.core.calibration_check import CHECK_PAIR_NAMES, CalibrationCheckArtifact
from calibrex.core.validation import validate_file
from calibrex.data.ros_cdr_writer import encode_image
from calibrex.data.rosbag2_writer import Rosbag2Writer

LEFT_TOPIC = "/left/image_raw"
RIGHT_TOPIC = "/right/image_raw"


def _write_bag(path: Path) -> Path:
    with Rosbag2Writer(path) as writer:
        writer.add_topic(LEFT_TOPIC, "sensor_msgs/msg/Image")
        writer.add_topic(RIGHT_TOPIC, "sensor_msgs/msg/Image")
        for index, (left, right) in enumerate(render_pairs(TRUTH)):
            stamp = 10_000_000_000 + index * 1_500_000_000
            for topic, image, frame in (
                (LEFT_TOPIC, left, "left_optical"),
                (RIGHT_TOPIC, right, "right_optical"),
            ):
                data = np.clip(np.rint(image), 0, 255).astype(np.uint8)
                writer.write(
                    topic,
                    stamp,
                    encode_image(
                        frame_id=frame,
                        timestamp_ns=stamp,
                        height=data.shape[0],
                        width=data.shape[1],
                        encoding="mono8",
                        step=data.shape[1],
                        data=data.tobytes(),
                    ),
                )
    return path


def _camchain(path: Path, left_right: np.ndarray) -> Path:
    """A camchain whose IMU frame is the left camera; ``left_right`` is ``T_left_right``."""

    right_imu = np.linalg.inv(left_right)
    entries = {}
    for key, camera, topic, t_cam_imu in (
        ("cam0", LEFT, LEFT_TOPIC, np.eye(4)),
        ("cam1", RIGHT, RIGHT_TOPIC, right_imu),
    ):
        entries[key] = {
            "T_cam_imu": t_cam_imu.tolist(),
            "camera_model": "pinhole",
            "intrinsics": [camera.fx, camera.fy, camera.cx, camera.cy],
            "distortion_model": "equidistant",
            "distortion_coeffs": list(camera.distortion),
            "resolution": [camera.width, camera.height],
            "rostopic": topic,
        }
    path.write_text(yaml.safe_dump(entries), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def bag(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_bag(tmp_path_factory.mktemp("stereo") / "bag")


def _run(bag: Path, camchain: Path, tmp_path: Path, **kwargs: object) -> CalibrationCheckArtifact:
    run = CheckRunOptions(
        pairs=("camera-camera",),
        evidence_dir=tmp_path / "evidence",
        base_dir=tmp_path,
        **kwargs,  # type: ignore[arg-type]
    )
    return build_calibration_check(bag, tf_files=[camchain], run=run)


def test_the_pair_is_planned_from_the_camchain(bag: Path, tmp_path: Path) -> None:
    assert "camera-camera" in CHECK_PAIR_NAMES
    plan = build_calibration_check(bag, tf_files=[_camchain(tmp_path / "c.yaml", TRUTH)])
    (record,) = [p for p in plan.pairs if p.pair == "camera-camera"]
    assert record.status == "planned"
    assert record.frames == ["left/image_raw", "right/image_raw"]
    assert record.candidate_transform is not None
    assert record.candidate_transform.translation_m == pytest.approx(TRUTH[:3, 3].tolist())


def test_the_true_camchain_passes_on_every_rotation_axis(bag: Path, tmp_path: Path) -> None:
    artifact = _run(bag, _camchain(tmp_path / "c.yaml", TRUTH), tmp_path)
    (record,) = [p for p in artifact.pairs if p.pair == "camera-camera"]
    assert record.status == "pass", record.reason
    assert [axis.name for axis in record.axes] == ["roll", "pitch", "yaw"]
    assert all(abs(axis.candidate_error) < 0.05 for axis in record.axes)
    assert record.evidence_artifact is not None
    assert validate_file(tmp_path / record.evidence_artifact).valid


def test_a_camchain_turned_in_pitch_fails_on_pitch(bag: Path, tmp_path: Path) -> None:
    turned = TRUTH.copy()
    turned[:3, :3] = Rotation.from_euler("y", 1.5, degrees=True).as_matrix() @ TRUTH[:3, :3]
    artifact = _run(bag, _camchain(tmp_path / "c.yaml", turned), tmp_path)
    (record,) = [p for p in artifact.pairs if p.pair == "camera-camera"]
    assert record.status == "fail"
    errors = {axis.name: axis.candidate_error for axis in record.axes}
    assert errors["pitch"] == pytest.approx(1.5, abs=0.1)
    assert abs(errors["roll"]) < 0.1 and abs(errors["yaw"]) < 0.1
    statuses = {axis.name: axis.status for axis in record.axes}
    assert statuses == {"roll": "pass", "pitch": "fail", "yaw": "pass"}


def test_camera_selection_keeps_a_pair_with_either_camera(bag: Path, tmp_path: Path) -> None:
    camchain = _camchain(tmp_path / "c.yaml", TRUTH)
    kept = _run(bag, camchain, tmp_path, camera=RIGHT_TOPIC)
    other = _run(bag, camchain, tmp_path, camera="/elsewhere/image_raw")
    (kept_record,) = [p for p in kept.pairs if p.pair == "camera-camera"]
    (other_record,) = [p for p in other.pairs if p.pair == "camera-camera"]
    assert kept_record.status == "pass"
    assert (other_record.status, other_record.reason_code) == ("skipped", "not_selected")
