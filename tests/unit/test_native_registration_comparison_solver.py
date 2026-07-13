from pathlib import Path

from calibrex.core.config import load_config
from calibrex.core.frames import FrameGraph
from calibrex.data.inspect import inspect_dataset
from calibrex.solvers.native_registration_comparison_solver import (
    NativeRegistrationComparisonSolver,
    voxel_downsample_icp_points,
)


def test_voxel_downsample_uses_centroids_and_a_deterministic_cap() -> None:
    points = [
        (0.1, 0.1, 0.1),
        (0.3, 0.1, 0.1),
        (1.1, 0.0, 0.0),
        (2.1, 0.0, 0.0),
        (3.1, 0.0, 0.0),
    ]

    result = voxel_downsample_icp_points(points, "cloud", 1.0, 3)

    assert [point.point_id for point in result] == [
        "cloud:0:0:0",
        "cloud:1:0:0",
        "cloud:2:0:0",
    ]
    assert result[0].position_m == (0.2, 0.1, 0.1)


def test_registration_comparison_reports_missing_public_inputs(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """schema_version: slac.config/v0.1
project: {name: missing_registration_data}
dataset: {type: livox_pcd, path: missing, time_base: frame_index}
sensors:
  base_horizon: {type: lidar, frame_id: base_horizon}
  target_horizon: {type: lidar, frame_id: target_horizon}
frames:
  base_horizon: {root: true}
  target_horizon:
    parent: base_horizon
    transform: {estimate: true}
pipeline:
  factors: {registration_backend_comparison: {enabled: true}}
solver: {backend: native_registration_comparison}
""",
        encoding="utf-8",
    )
    config = load_config(config_path)
    frame_graph = FrameGraph.from_config(config)

    result = NativeRegistrationComparisonSolver().solve(
        config, frame_graph, inspect_dataset(config.dataset)
    )

    assert result.status == "missing_dataset"
    assert result.metrics["registration_dataset_available"].grade == "fail"
    assert result.transforms == {}
