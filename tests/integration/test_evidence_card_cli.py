import json
from pathlib import Path

import pytest

from calibrex.cli.main import main
from calibrex.core.result import (
    CalibrationResult,
    FrameGraphSnapshot,
    MetricResult,
    QualitySummary,
    RunInfo,
)


def test_render_evidence_card_cli_writes_provenance_bound_svg(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result_path = tmp_path / "result.yaml"
    CalibrationResult(
        run=RunInfo(id="cli-card", slac_version="0.3.0"),
        frame_graph=FrameGraphSnapshot(root="base", frames={"base": None}),
        metrics={
            "reprojection_rmse_px": MetricResult(
                holdout=0.9,
                grade="pass",
                unit="px",
            )
        },
        quality=QualitySummary(grade="pass"),
    ).save(result_path)
    output_dir = tmp_path / "rendered"

    assert (
        main(
            [
                "render",
                str(result_path),
                "--format",
                "evidence-card",
                "--output-dir",
                str(output_dir),
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    card_path = output_dir / "evidence-card.svg"
    assert payload["output_format"] == "evidence-card"
    assert payload["evidence_card"] == str(card_path)
    assert payload["source_sha256_bound"] is True
    assert card_path.exists()
    assert "calibrex.evidence-card/v0.1" in card_path.read_text(encoding="utf-8")

    explicit_path = tmp_path / "custom" / "readme-card.svg"
    assert (
        main(
            [
                "render",
                str(result_path),
                "--format",
                "evidence-card",
                "--output",
                str(explicit_path),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert explicit_path.exists()


def test_compare_evidence_table_cli_writes_provenance_bound_svg(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "comparison.svg"
    exit_code = main(
        [
            "compare",
            "examples/precomputed/result.yaml",
            "examples/precomputed/known_bad_result.yaml",
            "--format",
            "evidence-table",
            "--left-label",
            "accepted",
            "--right-label",
            "known bad",
            "--output",
            str(output),
            "--json",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["output_format"] == "evidence-table"
    assert payload["comparison_table"] == str(output)
    assert payload["source_sha256_bound"] is True
    assert payload["protocol_compatibility"] == "not_comparable"
    assert output.exists()
    assert "calibrex.evidence-comparison-table/v0.1" in output.read_text(
        encoding="utf-8"
    )
