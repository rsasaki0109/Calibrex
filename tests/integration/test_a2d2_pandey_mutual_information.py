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
    / "config.yaml"
)

requires_a2d2_pandey_pair = pytest.mark.skipif(
    not (_DATASET / "20180810150607_camera_frontleft_000000060.png").exists(),
    reason=(
        "A2D2 MI pair is not downloaded; run "
        "tools/download_public_dataset.py a2d2_pandey_mutual_information"
    ),
)


@requires_a2d2_pandey_pair
def test_native_pandey_pipeline_records_honest_public_failure(tmp_path: Path) -> None:
    payload = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))
    payload["dataset"]["path"] = str(_DATASET)
    payload["project"]["output_dir"] = str(tmp_path / "outputs")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = run_calibration(config_path, CalibrationRunOptions())

    assert result is not None
    assert result.run.provenance["solver_adapter"] == "native_pandey_mutual_information"
    pandey = result.run.provenance["pandey_mutual_information"]
    assert pandey["paper_doi"] == "10.1609/aaai.v26i1.8379"
    assert result.run.provenance["a2d2_preregistered_camera_view"] is True
    assert result.metrics["pandey_projected_point_count"].holdout > 3_000
    assert result.metrics["pandey_objective_curvature_rank"].value == 6.0
    assert result.metrics["pandey_public_accuracy_independent"].grade == "warn"
    assert result.metrics["lidar_camera_mutual_information_score"].value is None
    assert result.metrics["lidar_camera_mutual_information_score"].holdout is not None
    assert result.quality.grade == "fail"
    assert len(result.run.provenance["raw_input_files"]) == 5
    assert all(item["sha256"] for item in result.run.provenance["raw_input_files"])
