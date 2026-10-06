"""The HTML report of a ``calibrex estimate`` artifact."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.unit.check_fixtures import write_check_bag
from tests.unit.test_check_estimate import (
    _estimate,
    _imu_lidar_full,
    _install,
    _run,
)

from calibrex.check.estimate import ARTIFACT_FILENAME
from calibrex.check.estimators import invert_transform
from calibrex.cli.main import main
from calibrex.core.bag_estimate import BagEstimateArtifact
from calibrex.visualization.estimate_report import render_estimate_html


@pytest.fixture
def bag(tmp_path: Path) -> Path:
    return write_check_bag(tmp_path / "bag.db3")


def _partial_artifact(
    tmp_path: Path, bag: Path, monkeypatch: pytest.MonkeyPatch
) -> BagEstimateArtifact:
    _install(
        monkeypatch,
        {
            "imu-lidar": lambda ctx: _run(
                invert_transform(ctx.candidate), translation=True, rotation_observed=False
            )
        },
    )
    return _estimate(bag, tmp_path / "est")


def test_report_has_every_pair_axis_status_and_the_omissions(
    tmp_path: Path, bag: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = _partial_artifact(tmp_path, bag, monkeypatch)
    page = render_estimate_html(artifact, artifact_dir=tmp_path / "est", html_dir=tmp_path)
    assert page.startswith("<!doctype html>")
    for pair in artifact.pairs:
        assert pair.pair in page
    for axis in ("roll", "pitch", "yaw", ">x<", ">y<", ">z<"):
        assert axis in page
    assert "unobservable" in page and "NOT MEASURED" in page
    assert "Omitted frames" in page and "lidar_front" in page
    assert "https://" not in page and "src=" not in page


def test_report_lists_exports_with_digests_and_commands(
    tmp_path: Path, bag: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, {"imu-lidar": _imu_lidar_full})
    artifact = _estimate(bag, tmp_path / "est")
    assert artifact.exports
    page = render_estimate_html(artifact, artifact_dir=tmp_path / "est", html_dir=tmp_path)
    for item in artifact.exports:
        assert item.sha256 in page
    assert "calibrex check &lt;other-bag&gt; --tf" in page
    assert "static_transform_publisher" in page
    assert "all six observed" in page
    assert 'href="est/frames.yaml"' in page


def test_report_escapes_text(tmp_path: Path, bag: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = _partial_artifact(tmp_path, bag, monkeypatch)
    artifact.next_steps.append("<script>alert(1)</script>")
    artifact.pairs[0].notes.append('"><img src=x onerror=alert(1)>')
    page = render_estimate_html(artifact)
    assert "<script>alert(1)" not in page
    assert "&lt;script&gt;alert(1)" in page
    assert "<img" not in page


def test_cli_estimate_writes_html_and_render_dispatches(
    tmp_path: Path,
    bag: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install(monkeypatch, {"imu-lidar": _imu_lidar_full})
    out = tmp_path / "est"
    html = tmp_path / "report.html"
    assert (
        main(["estimate", str(bag), "--output", str(out), "--no-cache", "--html", str(html)]) == 0
    )
    assert "html report:" in capsys.readouterr().out
    assert "calibrex estimate" in html.read_text(encoding="utf-8")

    again = tmp_path / "again.html"
    assert main(["render", str(out / ARTIFACT_FILENAME), "--html", str(again)]) == 0
    capsys.readouterr()
    assert "calibrex estimate" in again.read_text(encoding="utf-8")
    assert main(["render", str(out / ARTIFACT_FILENAME), "--format", "evidence-card"]) != 0
