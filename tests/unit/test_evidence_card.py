import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

from calibrex.core.result import (
    CalibrationResult,
    FrameGraphSnapshot,
    MetricResult,
    ObservabilityResult,
    QualitySummary,
    RunInfo,
    load_result,
)
from calibrex.visualization.evidence_card import render_evidence_card, write_evidence_card


def _result() -> CalibrationResult:
    return CalibrationResult(
        run=RunInfo(
            id="readme-hero",
            slac_version="0.3.0",
            git_commit="abc123",
            provenance={"metrics_origin": "recomputed", "data_verified": True},
        ),
        frame_graph=FrameGraphSnapshot(
            root="base",
            frames={"base": None, "camera0": "base", "lidar0": "base"},
        ),
        metrics={
            "reprojection_rmse_px": MetricResult(
                holdout=0.86,
                grade="pass",
                unit="px",
            ),
            "lidar_point_to_plane_rmse_m": MetricResult(
                holdout=0.021,
                grade="pass",
                unit="m",
            ),
        },
        observability=ObservabilityResult(rank=6, grade="pass"),
        quality=QualitySummary(grade="pass"),
    )


def test_evidence_card_is_valid_svg_with_visible_evidence() -> None:
    svg = render_evidence_card(
        _result(),
        source_label="result.yaml",
        source_sha256="a" * 64,
    )

    root = ET.fromstring(svg)
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert "Calibration evidence" in svg
    assert "LiDAR holdout RMSE" in svg
    assert "0.021 m" in svg
    assert "Reprojection holdout RMSE" in svg
    assert "SCHEMA VALIDATED" in svg
    assert "sha256:aaaaaaaaaaaa" in svg


def test_evidence_card_embeds_machine_readable_provenance() -> None:
    svg = render_evidence_card(
        _result(),
        source_label="result.yaml",
        source_sha256="b" * 64,
    )
    root = ET.fromstring(svg)
    metadata = root.find("{http://www.w3.org/2000/svg}metadata")
    assert metadata is not None
    payload = json.loads(metadata.text or "")

    assert payload["artifact"] == "calibrex.evidence-card/v0.1"
    assert payload["result_schema"] == "slac.result/v0.1"
    assert payload["run_id"] == "readme-hero"
    assert payload["git_commit"] == "abc123"
    assert payload["source"] == {"label": "result.yaml", "sha256": "b" * 64}
    assert payload["run_provenance"] == {
        "data_verified": True,
        "metrics_origin": "recomputed",
    }


def test_write_evidence_card_binds_source_digest_without_leaking_absolute_path(
    tmp_path: Path,
) -> None:
    source = tmp_path / "private" / "result.yaml"
    source.parent.mkdir()
    source.write_text("schema_version: slac.result/v0.1\n", encoding="utf-8")
    output = tmp_path / "assets" / "card.svg"

    written = write_evidence_card(_result(), output, source_path=source)

    assert written == output
    svg = output.read_text(encoding="utf-8")
    assert hashlib.sha256(source.read_bytes()).hexdigest() in svg
    assert str(tmp_path) not in svg


def test_committed_readme_card_matches_schema_valid_source() -> None:
    source = Path("examples/precomputed/result.yaml")
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    expected = render_evidence_card(
        load_result(source),
        source_label=source.name,
        source_sha256=source_sha256,
    )
    committed = Path("docs/assets/readme-evidence-card.svg").read_text(encoding="utf-8")

    assert committed == expected
