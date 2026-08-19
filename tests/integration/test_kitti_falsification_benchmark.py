import json
import os
from pathlib import Path

import pytest

from calibrex.cli.main import main
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

    assert benchmark.status == "pass"
    assert len(benchmark.selected_frame_ids) == 2
    assert {trial.candidate_id for trial in benchmark.trials} == {
        "dataset_reference",
        "known_bad",
    }
    assert {trial.candidate_id: trial.assessment_status for trial in benchmark.trials} == {
        "dataset_reference": "pass",
        "known_bad": "fail",
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


def test_kitti_falsification_benchmark_cli_emits_falsification_flag(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "benchmark-cli"
    exit_code = main(
        [
            "demo",
            "kitti-falsification-benchmark",
            str(FIXTURE),
            "--config",
            str(CONFIG),
            "--output-dir",
            str(output_dir),
            "--max-frames",
            "2",
            "--projection-sample-points",
            "800",
            "--json",
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "pass"
    assert payload["falsification_passed"] is True


@pytest.mark.skipif(
    "CALIBREX_KITTI_RAW_0005" not in os.environ,
    reason="set CALIBREX_KITTI_RAW_0005 to the official KITTI raw sequence",
)
@pytest.mark.kitti
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
    by_id = {trial.candidate_id: trial for trial in benchmark.trials}
    assert by_id["dataset_reference"].assessment_status == "pass"
    assert by_id["known_bad"].assessment_status == "fail"
    assert benchmark.status == "pass"
