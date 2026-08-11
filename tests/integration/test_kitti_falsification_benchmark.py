import os
from pathlib import Path

import pytest

from calibrex.core.validation import validate_file
from calibrex.evaluation.kitti_falsification_benchmark import (
    KITTI_SEQUENCE,
    run_kitti_falsification_benchmark,
)

FIXTURE = Path(
    "examples/public_datasets/kitti_lidar_camera_evidence/2011_09_26"
) / KITTI_SEQUENCE
CONFIG = Path("examples/public_datasets/kitti_lidar_camera_evidence/config.yaml")


def test_kitti_falsification_benchmark_materializes_schema_valid_trials(
    tmp_path: Path,
) -> None:
    benchmark = run_kitti_falsification_benchmark(
        FIXTURE,
        config_path=CONFIG,
        output_dir=tmp_path / "benchmark",
        max_frames=2,
        projection_sample_points=800,
    )

    assert benchmark.status == "inconclusive"
    assert len(benchmark.selected_frame_ids) == 2
    assert {trial.candidate_id for trial in benchmark.trials} == {
        "dataset_reference",
        "known_bad",
    }
    assert {trial.candidate_id: trial.assessment_status for trial in benchmark.trials} == {
        "dataset_reference": "pass",
        "known_bad": "inconclusive",
    }
    assert all(trial.bundle_valid for trial in benchmark.trials)
    assert all(len(trial.artifacts) == 9 for trial in benchmark.trials)
    report = validate_file(
        tmp_path / "benchmark" / "benchmark.json",
        kind="kitti-falsification",
    )
    assert report.valid is True


def test_kitti_falsification_benchmark_rejects_missing_pinned_sequence(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match=KITTI_SEQUENCE):
        run_kitti_falsification_benchmark(
            tmp_path / "wrong-sequence",
            config_path=CONFIG,
            output_dir=tmp_path / "outputs",
        )


@pytest.mark.skipif(
    "CALIBREX_KITTI_RAW_0005" not in os.environ,
    reason="set CALIBREX_KITTI_RAW_0005 to the official KITTI raw sequence",
)
def test_official_kitti_falsification_benchmark(tmp_path: Path) -> None:
    benchmark = run_kitti_falsification_benchmark(
        Path(os.environ["CALIBREX_KITTI_RAW_0005"]),
        config_path=Path(
            "examples/public_datasets/kitti_raw_2011_09_26_drive_0005/config.yaml"
        ),
        output_dir=tmp_path / "official-kitti-benchmark",
    )

    assert len(benchmark.selected_frame_ids) == 50
    assert all(trial.bundle_valid for trial in benchmark.trials)
    assert benchmark.status in {"pass", "fail", "inconclusive"}
