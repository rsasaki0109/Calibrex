import struct
from pathlib import Path

import numpy as np

from calibrex.core.config import CalibrationConfig
from calibrex.core.frames import FrameGraph
from calibrex.data.inspect import DatasetInspection
from calibrex.pipelines.calibrate import CalibrationRunOptions, run_calibration
from calibrex.solvers.native_camera_lidar_capture_time_solver import (
    NativeCameraLidarCaptureTimeSolver,
    _decode_pose_stamped,
    _decode_ros1_header,
    _sample_scan,
    _uniform_indices,
)


def _config(dataset_path: Path) -> CalibrationConfig:
    return CalibrationConfig.model_validate(
        {
            "dataset": {"type": "rosbag1", "path": str(dataset_path)},
            "sensors": {
                "camera0": {"type": "camera", "topic": "/camera"},
                "lidar0": {
                    "type": "lidar",
                    "topic": "/points",
                    "fields": ["x", "y", "z", "time"],
                    "point_time_field": "time",
                },
            },
            "frames": {
                "camera0": {"root": True},
                "lidar0": {
                    "parent": "camera0",
                    "transform": {"estimate": False},
                },
            },
            "pipeline": {"factors": {"camera_lidar_capture_time": {"enabled": True}}},
            "solver": {"backend": "native_camera_lidar_capture_time"},
        }
    )


def test_native_capture_time_adapter_reports_missing_public_bag(tmp_path: Path) -> None:
    bag_path = tmp_path / "missing.bag"
    config = _config(bag_path)

    result = NativeCameraLidarCaptureTimeSolver().solve(
        config,
        FrameGraph.from_config(config),
        DatasetInspection("rosbag1", str(bag_path), False),
    )

    assert result.status == "missing_dataset"
    assert result.metrics["camera_lidar_capture_time_input_available"].grade == "fail"
    assert result.transforms == {}


def test_pipeline_does_not_emit_placeholder_spatial_calibration(tmp_path: Path) -> None:
    config = _config(tmp_path / "missing.bag")
    config_path = tmp_path / "config.json"
    config_path.write_text(config.model_dump_json(), encoding="utf-8")

    result = run_calibration(
        config_path,
        CalibrationRunOptions(output_dir=tmp_path / "output"),
    )

    assert result is not None
    assert result.transforms == {}
    assert result.candidate_extrinsics == {}
    assert result.run.provenance["solver_adapter"] == "native_camera_lidar_capture_time"


def test_ros1_header_and_pose_stamped_use_message_timestamp() -> None:
    frame = b"world"
    header = struct.pack("<IIII", 4, 12, 345, len(frame)) + frame
    pose = struct.pack("<7d", 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0)

    timestamp_ns, frame_id, offset = _decode_ros1_header(header, 99)
    sample = _decode_pose_stamped(header + pose, 99)

    assert timestamp_ns == 12_000_000_345
    assert frame_id == "world"
    assert offset == len(header)
    assert sample.timestamp_ns == timestamp_ns
    assert sample.pose.translation_m == (1.0, 2.0, 3.0)


def test_sample_scan_preserves_measured_time_endpoints() -> None:
    xyz = np.asarray(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    offsets = np.asarray([-0.1, -0.07, -0.03, 0.0], dtype=np.float64)

    scan = _sample_scan(123, xyz, offsets, 3, 0.5, 5.0)

    assert [point.capture_offset for point in scan.points] == [-0.1, -0.07, 0.0]
    assert scan.point_time_min_sec == -0.1
    assert scan.point_time_max_sec == 0.0
    assert _uniform_indices(10, 4).tolist() == [0, 3, 6, 9]


def test_ros1_header_rejects_truncated_payload() -> None:
    try:
        _decode_ros1_header(b"short", 0)
    except ValueError as exc:
        assert "truncated" in str(exc)
    else:
        raise AssertionError("truncated ROS header was accepted")
