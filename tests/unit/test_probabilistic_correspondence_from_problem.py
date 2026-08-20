"""Unit tests for problem-to-correspondence projection."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from calibrex.core.validation import validate_file
from calibrex.data.probabilistic_correspondence_from_problem import (
    build_featdepth_correspondence_from_problem,
    build_probabilistic_correspondence_from_problem,
)

_SPEC = importlib.util.spec_from_file_location(
    "borer_rotation_benchmark_fixtures",
    Path(__file__).with_name("test_borer_rotation_benchmark.py"),
)
assert _SPEC is not None and _SPEC.loader is not None
_fixtures = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_fixtures)  # type: ignore[union-attr]
_write_problem = _fixtures._write_problem


def test_build_probabilistic_correspondence_from_problem(tmp_path: Path) -> None:
    problem_path = _write_problem(tmp_path)

    artifact = build_probabilistic_correspondence_from_problem(
        problem_path,
        max_points_per_frame=32,
        depth_relative_gate=1.0,
        seed=11,
    )

    assert len(artifact.frames) == 1
    assert len(artifact.frames[0].correspondences) >= 4
    output = tmp_path / "correspondence.yaml"
    artifact.save(output)
    assert validate_file(output, kind="probabilistic-correspondence").valid


def test_reference_perturbed_projection_differs_from_initial(tmp_path: Path) -> None:
    problem_path = _write_problem(tmp_path)

    initial = build_probabilistic_correspondence_from_problem(
        problem_path,
        max_points_per_frame=32,
        depth_relative_gate=1.0,
        seed=11,
    )
    perturbed = build_probabilistic_correspondence_from_problem(
        problem_path,
        projection_transform_source="reference_perturbed",
        max_points_per_frame=32,
        depth_relative_gate=1.0,
        seed=11,
    )

    initial_means = [
        tuple(item.image_mean_px)
        for frame in initial.frames
        for item in frame.correspondences
    ]
    perturbed_means = [
        tuple(item.image_mean_px)
        for frame in perturbed.frames
        for item in frame.correspondences
    ]
    assert initial_means != perturbed_means


def test_featdepth_correspondence_uses_depth_provider_identity(tmp_path: Path) -> None:
    problem_path = _write_problem(tmp_path)

    artifact = build_featdepth_correspondence_from_problem(
        problem_path,
        max_points_per_frame=32,
        depth_relative_gate=1.0,
        seed=11,
    )

    assert artifact.provider.provider == "synthetic"
    assert artifact.provider.model == "exact-range"
    assert artifact.provider.source_commit == "0123456789abcdef"
    assert artifact.provider.version == "v0.1"
