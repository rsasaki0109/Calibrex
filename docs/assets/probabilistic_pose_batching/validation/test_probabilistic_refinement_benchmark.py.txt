"""Complete-result comparisons retain all calibration and trace fields."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from tools.benchmark_probabilistic_refinement import build_fixture, comparable

from calibrex.core.camera_lidar_artifacts import load_camera_lidar_problem
from calibrex.core.probabilistic_correspondence import load_probabilistic_correspondence


@pytest.mark.parametrize("field", ["trace", "acceptance", "options", "holdout"])
def test_comparison_only_removes_declared_run_provenance(field: str) -> None:
    value = {
        "provenance": {"created_at": "a", "git_commit": "b", "input_sha256": "c"},
        "trace": [],
        "acceptance": True,
        "options": {},
        "holdout": 1.0,
    }
    raw = deepcopy(value)
    normalized = comparable(value)
    assert value == raw
    changed = deepcopy(value)
    changed["provenance"].update(created_at="d", git_commit="e")
    assert normalized == comparable(changed)
    changed[field] = "changed"
    assert normalized != comparable(changed)


def test_projected_fixture_binds_raw_inputs_and_declares_execution_control(tmp_path: Path) -> None:
    sequence = tmp_path / "sequence"
    scans = sequence / "velodyne_points/data"
    scans.mkdir(parents=True)
    points = np.column_stack(
        (np.full(16, 10.0), np.linspace(-1.0, 1.0, 16), np.linspace(-0.3, 0.3, 16), np.ones(16))
    ).astype("<f4")
    for i in range(4):
        points.tofile(scans / f"{i:010d}.bin")
    output = tmp_path / "fixture"
    build_fixture(sequence, output, 8)
    artifact = load_probabilistic_correspondence(output / "correspondence.json")
    problem = load_camera_lidar_problem(output / "problem.json")
    protocol = json.loads((output / "fixture-protocol.json").read_bytes())
    assert len(artifact.frames) == len(problem.observations) == 4
    assert [len(f.correspondences) for f in artifact.frames] == [8] * 4
    assert artifact.provenance.input_sha256 == protocol["input_sha256"]
    assert "execution-control" in artifact.provider.provider
    assert "performance/parity claims only" in artifact.warnings[0]
