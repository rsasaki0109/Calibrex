"""The HTML report of a ``calibrex check`` artifact."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from tests.unit.check_fixtures import default_tf, write_check_bag
from tests.unit.test_check_run import CLOSURE_PAIRS, _closure_run

from calibrex.check.runner import CheckRunOptions, build_calibration_check
from calibrex.cli.main import main
from calibrex.core.calibration_check import CalibrationCheckArtifact
from calibrex.visualization.check_report import render_check_html


@pytest.fixture
def bag(tmp_path: Path) -> Path:
    return write_check_bag(tmp_path / "bag.db3", tf_messages=[default_tf()])


def _artifact(
    tmp_path: Path, bag: Path, monkeypatch: pytest.MonkeyPatch
) -> CalibrationCheckArtifact:
    _closure_run(monkeypatch, 0.45, -0.45, -0.45)
    return build_calibration_check(
        bag,
        command=["calibrex", "check", str(bag)],
        run=CheckRunOptions(
            evidence_dir=tmp_path / "out" / "ev", base_dir=tmp_path / "out", pairs=CLOSURE_PAIRS
        ),
    )


def test_report_shows_verdict_pairs_closure_tree_and_provenance(
    tmp_path: Path, bag: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _artifact(tmp_path, bag, monkeypatch)
    page = render_check_html(
        artifact, artifact_dir=tmp_path / "out", html_dir=tmp_path / "out" / "html"
    )

    assert '<div class="banner warn"><strong>warn</strong>' in page
    for pair in ("imu-lidar", "lidar-lidar", "camera-imu"):
        assert pair in page
    assert "Skipped pairs" in page and "camera-imu" in page
    assert "Closure of the estimates" in page and 'class="badge fail"' in page
    assert "base_link" in page and "lidar_front" in page and "t=(1.000, 0.000, 0.500)" in page
    ref = (
        artifact.pairs[0].evidence[0]
        if artifact.pairs[0].evidence
        else next(p.evidence[0] for p in artifact.pairs if p.evidence)
    )
    assert f"sha256 {ref.sha256[:12]}" in page
    assert 'href="../ev/' in page  # relative link from html/ to ev/
    assert artifact.bag.input_sha256 in page
    assert artifact.provenance.generator_version in page
    assert "prefers-color-scheme: dark" in page
    assert "<script" not in page and "http://" not in page and "https://" not in page


def test_report_can_redact_absolute_paths(
    tmp_path: Path, bag: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _artifact(tmp_path, bag, monkeypatch)

    page = render_check_html(artifact, redact_paths=True)

    assert str(tmp_path) not in page
    assert not re.search(r"(?<![\w.<])/(tmp|home)/", page)
    assert ".../bag.db3" in page


def test_cli_writes_the_report_and_render_dispatches_on_the_schema(
    tmp_path: Path,
    bag: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _closure_run(monkeypatch, 0.0, 0.0, 0.0)
    out = tmp_path / "out" / "check.yaml"
    html = tmp_path / "out" / "check.html"
    args = ["check", str(bag), "--pairs", ",".join(CLOSURE_PAIRS), "--output", str(out)]

    assert main([*args, "--html", str(html), "--fail-on", "never"]) == 0
    assert "html report:" in capsys.readouterr().out
    first = html.read_text(encoding="utf-8")
    assert "overall verdict" in first and "<strong>pass</strong>" in first

    rendered = tmp_path / "again.html"
    assert main(["render", str(out), "--html", str(rendered)]) == 0
    capsys.readouterr()
    assert "<strong>pass</strong>" in rendered.read_text(encoding="utf-8")
    assert main(["render", str(out), "--format", "evidence-card"]) != 0
