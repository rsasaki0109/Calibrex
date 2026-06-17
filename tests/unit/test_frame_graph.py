import pytest

from calibrex.core.config import CalibrationConfig
from calibrex.core.exceptions import FrameGraphError
from calibrex.core.frames import FrameGraph


def test_frame_graph_from_minimal_config() -> None:
    config = CalibrationConfig.model_validate(
        {
            "schema_version": "calibrex.config/v0.1",
            "dataset": {"type": "filesystem", "path": "examples/synthetic_camera_lidar_imu"},
            "sensors": {"camera0": {"type": "camera"}},
            "frames": {
                "base": {"root": True},
                "camera0": {"parent": "base", "transform": {"estimate": True}},
            },
        }
    )
    graph = FrameGraph.from_config(config)
    assert graph.root == "base"
    assert graph.transform_to_root("camera0").translation_m == (0.0, 0.0, 0.0)


def test_frame_graph_rejects_cycle() -> None:
    config = CalibrationConfig.model_validate(
        {
            "schema_version": "calibrex.config/v0.1",
            "dataset": {"type": "filesystem", "path": "examples/synthetic_camera_lidar_imu"},
            "sensors": {"camera0": {"type": "camera"}},
            "frames": {
                "base": {"root": True},
                "camera0": {"parent": "camera0", "transform": {"estimate": True}},
            },
        }
    )
    with pytest.raises(FrameGraphError):
        FrameGraph.from_config(config)
