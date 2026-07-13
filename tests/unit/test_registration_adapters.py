from pathlib import Path

import numpy as np

from calibrex.core.geometry import SE3
from calibrex.solvers.registration_adapters import (
    NdtSubprocessAdapter,
    NdtSubprocessOptions,
    Open3DGeneralizedIcpAdapter,
)
from calibrex.solvers.robust_point_to_point_icp_solver import (
    IcpPoint,
    RobustPointToPointIcpOptions,
    evaluate_icp_candidate,
)


def _clouds(truth: SE3, count: int = 80) -> tuple[list[IcpPoint], list[IcpPoint]]:
    rng = np.random.default_rng(404)
    target_values = rng.uniform(-1.5, 1.5, size=(count, 3))
    source = [
        IcpPoint(f"source-{index:03d}", truth.inverse().transform_point(point))
        for index, point in enumerate(target_values)
    ]
    target = [
        IcpPoint(f"target-{index:03d}", tuple(point))
        for index, point in enumerate(target_values)
    ]
    return source, target


def test_common_candidate_evaluation_uses_spatial_holdout_and_rematching() -> None:
    truth = SE3((0.12, -0.07, 0.04), (0.01, -0.02, 0.03, 0.9993))
    source, target = _clouds(truth)

    result = evaluate_icp_candidate(
        source,
        target,
        truth,
        RobustPointToPointIcpOptions(
            correspondence_distance_m=0.4,
            trim_fraction=1.0,
            multi_start_diagnostics=False,
        ),
    )

    assert result.train_rmse_m is not None
    assert result.train_rmse_m < 1.0e-9
    assert result.holdout_rmse_m is not None
    assert result.holdout_rmse_m < 1.0e-9
    assert result.rematching_diagnostics is not None
    assert set(result.train_source_ids).isdisjoint(result.holdout_source_ids)


def test_open3d_gicp_reports_optional_boundary_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        "calibrex.solvers.registration_adapters.importlib.util.find_spec",
        lambda _name: None,
    )
    source, target = _clouds(SE3.identity())

    result = Open3DGeneralizedIcpAdapter().solve(source, target)

    assert result.status == "unavailable"
    assert result.available is False
    assert result.provenance["license_spdx"] == "MIT"
    assert "no Open3D code vendored" in str(result.provenance["adapter_boundary"])


def test_ndt_precomputed_transform_gets_common_holdout_evaluation(
    tmp_path: Path,
) -> None:
    truth = SE3((0.08, 0.03, -0.05), (-0.01, 0.025, 0.02, 0.9994))
    source, target = _clouds(truth)
    result_path = tmp_path / "ndt_result.yaml"
    result_path.write_text(
        "\n".join(
            (
                "transform_target_source:",
                f"  translation_m: {list(truth.translation_m)}",
                f"  rotation_quat_xyzw: {list(truth.rotation_quat_xyzw)}",
            )
        ),
        encoding="utf-8",
    )

    result = NdtSubprocessAdapter().solve(
        source,
        target,
        NdtSubprocessOptions(
            result_path=result_path,
            tool_name="autoware_ndt",
            tool_version="test",
            license_spdx="Apache-2.0",
            training_isolation_declared=True,
        ),
        common_options=RobustPointToPointIcpOptions(
            correspondence_distance_m=0.4,
            trim_fraction=1.0,
            multi_start_diagnostics=False,
        ),
    )

    assert result.status == "result_loaded"
    assert result.common_evaluation is not None
    assert result.common_evaluation.holdout_rmse_m is not None
    assert result.common_evaluation.holdout_rmse_m < 1.0e-9
    assert result.raw_metrics == {"returncode": None}
    assert result.provenance["tool_name"] == "autoware_ndt"
    assert result.provenance["external_code_vendored"] is False
    assert result.warnings == ()


def test_ndt_result_without_declared_train_isolation_is_warned(tmp_path: Path) -> None:
    source, target = _clouds(SE3.identity())
    result_path = tmp_path / "ndt_result.yaml"
    result_path.write_text(
        "transform_target_source:\n"
        "  translation_m: [0.0, 0.0, 0.0]\n"
        "  rotation_quat_xyzw: [0.0, 0.0, 0.0, 1.0]\n",
        encoding="utf-8",
    )

    result = NdtSubprocessAdapter().solve(
        source, target, NdtSubprocessOptions(result_path=result_path)
    )

    assert result.status == "result_loaded"
    assert result.warnings == ("external NDT train/holdout isolation is not declared",)
