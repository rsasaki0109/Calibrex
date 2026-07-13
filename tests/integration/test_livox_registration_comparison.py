import importlib.metadata
import importlib.util
from pathlib import Path

import pytest

from calibrex.core.config import load_config
from calibrex.core.frames import FrameGraph
from calibrex.data.downloads import (
    LIVOX_BASE_PCD_NAME,
    LIVOX_TARGET_PCD_NAME,
)
from calibrex.data.inspect import inspect_dataset
from calibrex.solvers.native_registration_comparison_solver import (
    NativeRegistrationComparisonSolver,
)

_ROOT = Path("data/public/livox_horizon_horizon_pair")
_CONFIG = Path(
    "examples/public_datasets/livox_horizon_horizon_pcd_sample/icp_comparison_config.yaml"
)
_HAS_INPUTS = all((_ROOT / name).exists() for name in (LIVOX_BASE_PCD_NAME, LIVOX_TARGET_PCD_NAME))
_HAS_OPEN3D = importlib.util.find_spec("open3d") is not None
_OPEN3D_VERSION = importlib.metadata.version("open3d") if _HAS_OPEN3D else None


@pytest.mark.skipif(not _HAS_INPUTS, reason="public Livox Horizon pair is not downloaded")
@pytest.mark.skipif(not _HAS_OPEN3D, reason="optional Open3D GICP dependency is not installed")
@pytest.mark.skipif(_OPEN3D_VERSION != "0.19.0", reason="pinned public evidence uses Open3D 0.19.0")
def test_public_livox_open3d_gicp_common_rematching_evidence() -> None:
    config = load_config(_CONFIG)
    result = NativeRegistrationComparisonSolver().solve(
        config,
        FrameGraph.from_config(config),
        inspect_dataset(config.dataset),
    )

    assert result.status == "inconclusive"
    available = result.metrics["registration_open3d_gicp_available"]
    assert available.value == 1.0
    assert available.grade == "pass"
    train = result.metrics["registration_open3d_gicp_train_rmse_m"]
    holdout = result.metrics["registration_open3d_gicp_holdout_rmse_m"]
    inlier = result.metrics["registration_open3d_gicp_inlier_fraction"]
    jaccard = result.metrics["registration_open3d_gicp_min_correspondence_jaccard"]
    weak = result.metrics["registration_open3d_gicp_weak_direction_count"]
    assert train.value is not None and 0.48 < train.value < 0.49
    assert holdout.value is not None and 2.90 < holdout.value < 2.91
    assert holdout.grade == "fail"
    assert inlier.value is not None and 0.27 < inlier.value < 0.28
    assert inlier.grade == "pass"
    assert jaccard.value is not None and 0.68 < jaccard.value < 0.70
    assert jaccard.grade == "pass"
    assert weak.value == 3.0
    assert weak.grade == "warn"
    assert result.metrics["registration_backend_common_split_consistent"].value == 1.0
    tangent_rank = result.metrics["registration_chen_medioni_tangent_rank"]
    tangent_condition = result.metrics["registration_chen_medioni_tangent_condition_number"]
    surface_holdout = result.metrics[
        "registration_chen_medioni_holdout_point_to_plane_rmse_m"
    ]
    common_holdout = result.metrics["registration_chen_medioni_common_holdout_rmse_m"]
    assert tangent_rank.value == 6.0
    assert tangent_rank.grade == "pass"
    assert tangent_condition.value is not None and 90.0 < tangent_condition.value < 90.5
    assert surface_holdout.value is not None and 0.27 < surface_holdout.value < 0.29
    assert surface_holdout.grade == "pass"
    assert common_holdout.value is not None and 5.2 < common_holdout.value < 5.3
    assert common_holdout.grade == "fail"
    rotation_delta = result.metrics["registration_open3d_gicp_native_rotation_delta_deg"]
    translation_delta = result.metrics["registration_open3d_gicp_native_translation_delta_m"]
    assert rotation_delta.value is not None and 18.6 < rotation_delta.value < 18.8
    assert translation_delta.value is not None and 0.85 < translation_delta.value < 0.86
    comparison = result.provenance["native_registration_comparison"]
    point_to_plane = comparison["native_chen_medioni_point_to_plane"]
    assert point_to_plane["status"] == "converged"
    assert point_to_plane["valid_target_normal_count"] == 499
    assert len(point_to_plane["known_bad_probes"]) == 12
    assert point_to_plane["paper"]["conference_doi"] == "10.1109/ROBOT.1991.132043"
    assert "approximates" in point_to_plane["surface_specialization"]
    gicp = comparison["open3d_gicp"]
    assert gicp["status"] == "converged"
    assert gicp["provenance"]["tool_name"] == "Open3D"
    assert gicp["provenance"]["tool_version"] == _OPEN3D_VERSION
    assert gicp["provenance"]["license_spdx"] == "MIT"
    assert gicp["provenance"]["external_code_vendored"] is False
    assert gicp["common_evaluation"]["rematching_diagnostics"]["weak_directions"] == (
        "y",
        "roll",
        "yaw",
    )
