"""CPU-only checks of the rejected GPU gyro trial's actual work and scopes."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from tools.audit_bspline_lidar_imu_benchmark import audit

ROOT = Path(__file__).resolve().parents[2] / "docs/assets/cuda_gyro_deskew"


def test_rejected_gyro_archive_retains_complete_comparable_odometry() -> None:
    result = audit(ROOT)
    assert result["workers"] == 6 and result["quality_passed"]
    assert result["gpu_observation_samples"] == 43
    assert result["foreign_gpu_samples"] == result["unknown_gpu_samples"] == 0
    assert 1 < result["whole_speedup"] < 1.1
    assert 1 < result["odometry_speedup"] < 1.1
    assert "preprocessing_speedup" not in result


@pytest.mark.parametrize(
    "mutation",
    [
        "negative_calls",
        "missing_rows",
        "zero_translations",
        "excess_repairs",
        "missing_gyro_timing",
        "invalid_gyro_timing",
        "missing_odometry_timing",
        "invalid_odometry_timing",
        "false_preprocessing_scope",
    ],
)
def test_gyro_archive_rejects_rebound_false_work_or_timing(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = tmp_path / "archive"
    shutil.copytree(ROOT, root)
    measurement = root / "00-current/measurement.json"
    record = json.loads(measurement.read_bytes())
    if mutation.endswith("timing"):
        scope = "gyro_preprocessing" if "gyro" in mutation else "odometry"
        if mutation.startswith("missing"):
            record.pop(scope + "_seconds")
            record.pop(scope + "_calls")
        else:
            record[scope + "_seconds"] = record["pipeline_seconds"] + 1
    elif mutation == "false_preprocessing_scope":
        record["preprocessing_scope"] = "complete preprocessing"
    else:
        result = root / "00-current/result.json"
        value = json.loads(result.read_bytes())
        reports = (
            value["options"]["initial_odometry_execution"]
            + value["options"]["segment_odometry_execution"]
        )
        counters = next(
            report["statistics"]
            for report in reports
            if report["statistics"]["gpu_gyro_rotation_calls"] > 0
        )
        if mutation == "negative_calls":
            counters["gpu_gyro_rotation_calls"] = -1
        elif mutation == "missing_rows":
            counters.pop("gpu_gyro_rotation_rows")
        elif mutation == "zero_translations":
            counters["gpu_gyro_translation_calls"] = 0
        else:
            counters["native_gyro_point_repairs"] = counters["gpu_preprocess_input_rows"] + 1
        result.write_text(json.dumps(value), encoding="utf-8")
        record["provenance"]["result_sha256"] = hashlib.sha256(result.read_bytes()).hexdigest()
    measurement.write_text(json.dumps(record), encoding="utf-8")
    path = root / "publication-index.json"
    index = json.loads(path.read_bytes())
    index["files"] = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file() and p != path
    }
    path.write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match=r"gyro|odometry|preprocessing"):
        audit(root)
