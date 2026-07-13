from pathlib import Path

import numpy as np

from calibrex.core.config import load_config
from calibrex.core.frames import FrameGraph
from calibrex.core.geometry import SE3
from calibrex.data.inspect import inspect_dataset
from calibrex.solvers.native_registration_comparison_solver import (
    NativeRegistrationComparisonSolver,
    _metrics,
    voxel_downsample_icp_points,
)
from calibrex.solvers.registration_adapters import (
    NdtSubprocessAdapter,
    NdtSubprocessOptions,
    Open3DGeneralizedIcpAdapter,
)
from calibrex.solvers.robust_point_to_point_icp_solver import (
    IcpPoint,
    RobustPointToPointIcpOptions,
    RobustPointToPointIcpSolver,
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


def test_registration_adapters_expose_common_rematching_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    rng = np.random.default_rng(930)
    points = rng.uniform(-1.0, 1.0, size=(80, 3))
    source = [
        IcpPoint(f"source-{index:03d}", tuple(float(value) for value in point))
        for index, point in enumerate(points)
    ]
    target = [
        IcpPoint(f"target-{index:03d}", tuple(float(value) for value in point))
        for index, point in enumerate(points)
    ]
    options = RobustPointToPointIcpOptions(
        correspondence_distance_m=0.4,
        trim_fraction=1.0,
        mutual_correspondences=False,
        multi_start_diagnostics=False,
    )
    native = RobustPointToPointIcpSolver().solve(
        source, target, initial_transform=SE3.identity(), options=options
    )
    result_path = tmp_path / "ndt_result.yaml"
    result_path.write_text(
        "transform_target_source:\n"
        "  translation_m: [0.0, 0.0, 0.0]\n"
        "  rotation_quat_xyzw: [0.0, 0.0, 0.0, 1.0]\n",
        encoding="utf-8",
    )
    ndt = NdtSubprocessAdapter().solve(
        source,
        target,
        NdtSubprocessOptions(
            result_path=result_path,
            license_spdx="Apache-2.0",
            training_isolation_declared=True,
        ),
        common_options=options,
    )
    monkeypatch.setattr(
        "calibrex.solvers.registration_adapters.importlib.util.find_spec",
        lambda _name: None,
    )
    gicp = Open3DGeneralizedIcpAdapter().solve(source, target, common_options=options)

    metrics = _metrics(
        native,
        gicp,
        ndt,
        {
            "max_holdout_rmse_m": 0.1,
            "min_inlier_fraction": 0.9,
            "min_correspondence_jaccard": 0.99,
        },
    )

    assert metrics["registration_open3d_gicp_train_rmse_m"].grade == "warn"
    assert metrics["registration_external_ndt_train_rmse_m"].value is not None
    assert metrics["registration_external_ndt_train_rmse_m"].value < 1.0e-12
    assert metrics["registration_external_ndt_holdout_rmse_m"].grade == "pass"
    assert metrics["registration_external_ndt_inlier_fraction"].value == 1.0
    assert metrics["registration_external_ndt_min_correspondence_jaccard"].value == 1.0
    assert metrics["registration_external_ndt_weak_direction_count"].value == 0.0
    assert metrics["registration_backend_common_split_consistent"].value == 1.0
    assert metrics["registration_backend_common_split_consistent"].grade == "pass"
    assert metrics["registration_external_ndt_native_rotation_delta_deg"].value == 0.0
    assert metrics["registration_external_ndt_native_translation_delta_m"].value is not None
    assert metrics["registration_external_ndt_native_translation_delta_m"].value < 1.0e-12
