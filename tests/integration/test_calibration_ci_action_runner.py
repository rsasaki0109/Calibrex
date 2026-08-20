"""Integration test for the GitHub Action runner script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_CANDIDATE = Path(
    "examples/public_datasets/livox_horizon_horizon_pcd_sample/"
    "cached_evidence_result.yaml"
)
_RUNNER = Path("tools/run_calibration_ci_action.py")


def test_run_calibration_ci_action_exports_github_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not _CANDIDATE.is_file():
        pytest.skip(f"missing fixture result: {_CANDIDATE}")

    output_dir = tmp_path / "calibrex-ci"
    github_output = tmp_path / "github_output.txt"
    monkeypatch.setenv("CALIBREX_ACTION_CANDIDATE", str(_CANDIDATE.resolve()))
    monkeypatch.setenv("CALIBREX_ACTION_BASELINE", str(_CANDIDATE.resolve()))
    monkeypatch.setenv("CALIBREX_ACTION_OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("CALIBREX_ACTION_ENFORCE", "false")
    monkeypatch.setenv("CALIBREX_ACTION_ENFORCE_PROTOCOL", "true")
    monkeypatch.setenv("GITHUB_OUTPUT", str(github_output))

    completed = subprocess.run(
        [sys.executable, str(_RUNNER.resolve())],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
    )

    assert completed.returncode == 0
    assert (output_dir / "calibration-ci.json").is_file()
    assert (output_dir / "summary.md").is_file()

    payload = github_output.read_text(encoding="utf-8")
    assert "status=inconclusive" in payload
    assert f"artifact={output_dir / 'calibration-ci.json'}" in payload
    assert f"summary={output_dir / 'summary.md'}" in payload
