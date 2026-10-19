from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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
@pytest.mark.parametrize("backend,workers", [("numpy", 1), ("numpy", 3), ("numba", 3), ("cuda", 3)])
def test_native_levinson_pipeline_records_honest_inconclusive_evidence(
    tmp_path: Path,
    backend: str,
    workers: int,
) -> None:
    if backend == "numba":
        pytest.importorskip("numba")
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
    if backend == "numba":
        assert provenance["extraction"]["edge_evaluation_backend"] == "numba"
        assert levinson["execution"]["backend"] == "numba"
        assert levinson["execution"]["numba_frame_pose_evaluations"] == 0
        assert levinson["execution"]["native_frame_pose_evaluations"] > 728


@requires_a2d2_pair
def test_array_loading_preserves_the_complete_public_calibration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.benchmark_png_luminance import comparable

    from calibrex.data.kitti import read_png_luminance
    from calibrex.solvers import native_levinson_thrun_online_solver as adapter

    payload = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    payload["dataset"]["path"] = str(_DATASET)
    options = payload["pipeline"]["factors"]["lidar_camera_depth_edge_alignment"]["options"]
    options["evaluation_backend"] = "numpy"
    options["preparation_workers"] = 3
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    original = adapter.read_png_luminance_array

    def rows_control(path: str | Path):
        image = read_png_luminance(path)
        return None if image is None else np.asarray(image.rows, dtype=np.float64)

    monkeypatch.setattr(adapter, "read_png_luminance_array", rows_control)
    baseline_path = tmp_path / "rows"
    baseline = run_calibration(config_path, CalibrationRunOptions(output_dir=baseline_path))
    monkeypatch.setattr(adapter, "read_png_luminance_array", original)
    candidate_path = tmp_path / "array"
    candidate = run_calibration(config_path, CalibrationRunOptions(output_dir=candidate_path))
    assert baseline is not None and candidate is not None
    # Normalize only run/source metadata and declared output paths; JSON keeps
    # signed zero and every serialized float in all metrics/controls/provenance.
    first = comparable(baseline.model_dump(mode="json"), str(baseline_path))
    second = comparable(candidate.model_dump(mode="json"), str(candidate_path))
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
