from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from calibrex.pipelines.calibrate import CalibrationRunOptions, run_calibration

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATASET = _REPO_ROOT / "data" / "public" / "a2d2_pandey_mutual_information"
_CONFIG = (
    _REPO_ROOT
    / "examples"
    / "public_datasets"
    / "a2d2_pandey_mutual_information"
    / "levinson_online_config.yaml"
)

requires_a2d2_pair = pytest.mark.skipif(
    not (_DATASET / "20180810150607_camera_frontleft_000000060.png").exists(),
    reason=(
        "A2D2 camera/LiDAR pair is not downloaded; run "
        "tools/download_public_dataset.py a2d2_pandey_mutual_information"
    ),
)


@requires_a2d2_pair
@pytest.mark.parametrize("backend,workers", [("numpy", 1), ("numpy", 3), ("cuda", 3)])
def test_native_levinson_pipeline_records_honest_inconclusive_evidence(
    tmp_path: Path,
    backend: str,
    workers: int,
) -> None:
    if backend == "cuda":
        cp = pytest.importorskip("cupy")
        try:
            if cp.cuda.runtime.getDeviceCount() < 1:
                pytest.skip("requires a CUDA device")
        except cp.cuda.runtime.CUDARuntimeError:
            pytest.skip("requires a CUDA device")
    payload = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    payload["dataset"]["path"] = str(_DATASET)
    payload["project"]["output_dir"] = str(tmp_path / "outputs")
    options = payload["pipeline"]["factors"]["lidar_camera_depth_edge_alignment"]["options"]
    options["evaluation_backend"] = backend
    options["preparation_workers"] = workers
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = run_calibration(config_path, CalibrationRunOptions())

    assert result is not None
    assert result.run.provenance["solver_adapter"] == "native_levinson_thrun_online"
    provenance = result.run.provenance
    levinson = provenance["levinson_thrun_online"]
    assert provenance["paper_doi"] == "10.15607/RSS.2013.IX.029"
    assert provenance["a2d2_preregistered_camera_view"] is True
    assert provenance["paper_default_window_complete"] is False
    assert provenance["public_evidence_claim"].startswith("inconclusive")
    assert provenance["extraction"]["equation_2_recomputed"] is False
    assert levinson["status"] == "monitored"
    assert result.metrics["levinson_projected_boundary_point_count"].holdout > 1_800
    assert result.metrics["levinson_worsening_fraction"].value < 0.6
    assert result.metrics["levinson_calibrated_probability"].value < 1.0e-100
    assert result.metrics["levinson_objective_curvature_rank"].value == 6.0
    assert result.metrics["levinson_public_accuracy_independent"].grade == "warn"
    assert result.quality.grade == "warn"
    assert len(provenance["raw_input_files"]) == 5
    assert provenance["data_verified"] is True
    assert all(item["sha256"] for item in provenance["raw_input_files"])
    if workers != 1:
        assert provenance["extraction"]["preparation_workers"] == workers
    if backend == "cuda":
        assert provenance["extraction"]["edge_evaluation_backend"] == "cuda"
        assert levinson["execution"]["cuda_frame_pose_evaluations"] == 0
        assert levinson["execution"]["native_frame_pose_evaluations"] == 728
