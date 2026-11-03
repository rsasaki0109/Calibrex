"""Board poses from a bag and ``calibrex estimate --board`` on a synthetic rosbag2.

The bag holds a checkerboard still at nine poses (tilted left/right and up/down) for
1.6 s each, moving between them for 0.4 s, with 10 Hz mono images, a CameraInfo and
10 Hz PointCloud2 scans of the board, a wall and a floor (the scene of
``test_board_camera_lidar``). There is no /tf_static: the board path starts from the
LiDAR's axes alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation
from tests.unit.test_board_camera_lidar import (
    BOARD,
    CAMERA,
    SIZE,
    TRUE,
    _board_pose,
    _render,
    _scan,
)

from calibrex.cli.main import main
from calibrex.core.board_camera_lidar import load_board_camera_lidar
from calibrex.core.validation import validate_file
from calibrex.data.ros_cdr_writer import (
    POINT_FIELD_FLOAT32,
    encode_camera_info,
    encode_image,
    encode_pointcloud2,
)
from calibrex.data.rosbag2_writer import Rosbag2Writer
from calibrex.evaluation.board_camera_lidar import BoardView
from calibrex.evaluation.board_camera_lidar_bag import (
    BoardCaptureOptions,
    find_still_segments,
    read_board_poses,
)

cv2 = pytest.importorskip("cv2")

CAMERA_FRAME = "camera_optical"
LIDAR_FRAME = "lidar"
IMAGE_TOPIC = "/camera/image_raw"
LIDAR_TOPIC = "/lidar/points"
HOLD_S, MOVE_S, RATE_HZ = 1.6, 0.4, 10
ANGLES = [(y, p) for y in (-30.0, 0.0, 30.0) for p in (-25.0, 0.0, 25.0)]
FIELDS = (
    ("x", 0, POINT_FIELD_FLOAT32, 1),
    ("y", 4, POINT_FIELD_FLOAT32, 1),
    ("z", 8, POINT_FIELD_FLOAT32, 1),
)


def _view(centre: tuple[float, float, float], yaw: float = 0.0) -> BoardView:
    rotation, translation = _board_pose(yaw, 0.0, centre)
    return BoardView(rotation, translation, 0.1)


def test_still_segments_skip_motion_and_merge_a_lost_detection() -> None:
    options = BoardCaptureOptions(min_still_s=1.0)
    detections: list[tuple[float, BoardView | None]] = []
    # 0-1.5 s still, the board lost at 0.8 s for one image
    detections += [(0.2 * k, None if k == 4 else _view((0.0, 0.0, 2.0))) for k in range(8)]
    # moving 1.6-2.4 s: 5 cm per image
    detections += [(1.6 + 0.2 * i, _view((0.05 * (i + 1), 0.0, 2.0))) for i in range(5)]
    # still 2.6-3.2 s: too short
    detections += [(2.6 + 0.2 * i, _view((0.5, 0.0, 2.0), 20.0)) for i in range(4)]
    # still 3.4-5.0 s at a new pose
    detections += [(3.4 + 0.2 * i, _view((-0.3, 0.1, 2.5), -20.0)) for i in range(9)]

    segments = find_still_segments(detections, BOARD, options)

    assert [(round(s.start_s, 1), round(s.end_s, 1)) for s in segments] == [(0.0, 1.4), (3.4, 5.0)]
    assert segments[0].detections == 7  # the two halves of the first hold, merged


def _write_bag(path: Path) -> Path:
    rng = np.random.default_rng(5)
    poses = [
        _board_pose(yaw, pitch, (rng.uniform(-0.3, 0.3), rng.uniform(-0.15, 0.15), 2.0 + 0.2 * i))
        for i, (yaw, pitch) in enumerate(ANGLES)
    ]
    step_ns = 1_000_000_000 // RATE_HZ
    stamp = 10_000_000_000
    with Rosbag2Writer(path) as writer:
        writer.add_topic(IMAGE_TOPIC, "sensor_msgs/msg/Image")
        writer.add_topic("/camera/camera_info", "sensor_msgs/msg/CameraInfo")
        writer.add_topic(LIDAR_TOPIC, "sensor_msgs/msg/PointCloud2")
        for index, (rotation, translation) in enumerate(poses):
            following = poses[(index + 1) % len(poses)]
            for tick in range(round((HOLD_S + MOVE_S) * RATE_HZ)):
                held = tick < HOLD_S * RATE_HZ
                if held:
                    pose_rotation, pose_translation = rotation, translation
                else:  # gliding towards the next pose
                    share = (tick - HOLD_S * RATE_HZ + 1) / (MOVE_S * RATE_HZ + 1)
                    pose_rotation = rotation
                    pose_translation = (1 - share) * translation + share * following[1]
                image = _render(pose_rotation, pose_translation)
                writer.write(
                    "/camera/camera_info",
                    stamp,
                    encode_camera_info(
                        frame_id=CAMERA_FRAME,
                        timestamp_ns=stamp,
                        height=SIZE[1],
                        width=SIZE[0],
                        k=(CAMERA.fx, 0.0, CAMERA.cx, 0.0, CAMERA.fy, CAMERA.cy, 0.0, 0.0, 1.0),
                    ),
                )
                writer.write(
                    IMAGE_TOPIC,
                    stamp,
                    encode_image(
                        frame_id=CAMERA_FRAME,
                        timestamp_ns=stamp,
                        height=SIZE[1],
                        width=SIZE[0],
                        encoding="mono8",
                        step=SIZE[0],
                        data=image.tobytes(),
                    ),
                )
                points = _scan(pose_rotation, pose_translation, rng)
                # each scan sees a third of the board grid, as a sparse LiDAR would
                points = points[rng.random(len(points)) < 0.35]
                writer.write(
                    LIDAR_TOPIC,
                    stamp + 2_000_000,
                    encode_pointcloud2(
                        frame_id=LIDAR_FRAME,
                        timestamp_ns=stamp,
                        fields=FIELDS,
                        point_step=12,
                        data=np.ascontiguousarray(points, dtype=np.float32).tobytes(),
                    ),
                )
                stamp += step_ns
    return path


@pytest.fixture(scope="module")
def board_bag(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_bag(tmp_path_factory.mktemp("board") / "board")


def test_read_board_poses_finds_each_hold_and_its_scans(board_bag: Path) -> None:
    capture = read_board_poses(board_bag, IMAGE_TOPIC, LIDAR_TOPIC, BOARD, CAMERA)

    assert len(capture.poses) == len(ANGLES)
    assert capture.images == len(ANGLES) * (HOLD_S + MOVE_S) * RATE_HZ
    for segment in capture.segments:
        assert segment.end_s - segment.start_s >= 1.0
        # 1.6 s held at 10 Hz, less 0.2 s at either end
        assert 9 <= segment.scans <= 13


def test_estimate_board_measures_all_six_axes_without_a_prior(
    board_bag: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    code = main(
        [
            "estimate",
            str(board_bag),
            "--output",
            str(out),
            "--pairs",
            "camera-lidar",
            "--board",
            "7x5x0.08",
            "--no-cache",
        ]
    )

    assert code == 0
    artifact = json.loads((out / "bag_estimate.json").read_text())
    assert artifact["options"]["board"] == "7x5x0.08"
    (pair,) = [p for p in artifact["pairs"] if p["pair"] == "camera-lidar"]
    assert pair["status"] == "estimated", pair.get("reason")
    assert pair["fill"] == "board_axes"  # no prior: nothing filled from an identity
    assert pair["estimator"].startswith("planar_board")
    assert {a["name"] for a in pair["axes"] if a["status"] == "observed"} == {
        "roll",
        "pitch",
        "yaw",
        "x",
        "y",
        "z",
    }
    frames = yaml.safe_load((out / "frames.yaml").read_text())
    (entry,) = [f for f in frames["frames"] if f["name"] == LIDAR_FRAME]
    np.testing.assert_allclose(entry["translation_m"], TRUE[:3, 3], atol=0.01)
    rotation = Rotation.from_quat(entry["rotation_quat_xyzw"]).as_matrix()
    assert np.degrees(Rotation.from_matrix(rotation @ TRUE[:3, :3].T).magnitude()) < 0.2
    assert "NOT MEASURED" not in (out / "frames.yaml").read_text()
    assert not any("--board" in step for step in artifact["next_steps"])

    (evidence,) = [ref for ref in pair["evidence"] if ref["role"] == "board_planes"]
    board = load_board_camera_lidar(out / evidence["path"])
    assert board.policy_status == "pass"
    assert sum(pose.used for pose in board.poses) == len(ANGLES)
    # the rendered corners put the camera's board within about 1 cm at 3 m; a wrong plane would
    # be decimetres off
    assert all(abs(pose.offset_residual_m or 1.0) < 0.02 for pose in board.poses)
    validate_file(out / "bag_estimate.json")
    validate_file(out / evidence["path"])


def test_estimate_without_board_suggests_it_for_the_translation(tmp_path: Path) -> None:
    from calibrex.check.estimate import BOARD_HINT, next_steps
    from calibrex.core.bag_estimate import EstimateFrames, EstimatePairRecord

    axes = [
        {"name": n, "unit": "deg", "status": "observed", "value": 0.0, "std": 0.1}
        for n in ("roll", "pitch", "yaw")
    ] + [
        {"name": n, "unit": "m", "status": "unobservable", "reason": "edge"}
        for n in ("x", "y", "z")
    ]
    pair = EstimatePairRecord.model_validate(
        {
            "pair": "camera-lidar",
            "sensors": ["/camera/image_raw", "/lidar/points"],
            "topics": ["/camera/image_raw", "/lidar/points"],
            "frames": [CAMERA_FRAME, LIDAR_FRAME],
            "status": "partial",
            "estimator": "edge_alignment/v0.1 (camera-lidar)",
            "axes": axes,
        }
    )
    steps = next_steps([pair], EstimateFrames(), [], bag_name="b", output_dir=tmp_path)
    assert BOARD_HINT in steps
