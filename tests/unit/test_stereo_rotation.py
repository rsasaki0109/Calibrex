"""A stereo pair's relative rotation from its images, on rendered fisheye pairs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tests.unit.stereo_scene import LEFT, RIGHT, TRUTH, render_pairs

from calibrex.check.estimators import rotation_errors_deg
from calibrex.core.stereo_rotation import StereoRotationProvenance
from calibrex.core.validation import validate_file
from calibrex.evaluation.stereo_rotation import (
    StereoRotationOptions,
    estimate_stereo_rotation,
)

OPTIONS = StereoRotationOptions(patches_per_pair=150, min_patches=300)
PROVENANCE = StereoRotationProvenance(
    generator="tests",
    generator_version="0",
    left_topic="/left",
    right_topic="/right",
    left_frame="left",
    right_frame="right",
    intrinsics_source="test",
)


@pytest.fixture(scope="module")
def pairs() -> list[tuple[np.ndarray, np.ndarray]]:
    return render_pairs(TRUTH)


def test_the_true_rotation_passes_with_every_axis_observed(
    pairs: list[tuple[np.ndarray, np.ndarray]], tmp_path: Path
) -> None:
    artifact = estimate_stereo_rotation(
        pairs, LEFT, RIGHT, TRUTH, provenance=PROVENANCE, options=OPTIONS
    )
    assert artifact.policy_status == "pass"
    assert [axis.name for axis in artifact.axes] == ["roll", "pitch", "yaw"]
    for axis in artifact.axes:
        assert axis.observable, axis.reason
        assert abs(axis.candidate_error_deg) < 0.05
        assert axis.std_deg < 0.05
        assert axis.control_response == pytest.approx(1.0, abs=0.2)
    artifact.save(tmp_path / "stereo.yaml")
    assert validate_file(tmp_path / "stereo.yaml").valid


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_turned_candidate_reads_its_turn_about_the_left_camera(
    pairs: list[tuple[np.ndarray, np.ndarray]], axis: int
) -> None:
    candidate = TRUTH.copy()
    turn = Rotation.from_rotvec(np.radians(0.6) * np.eye(3)[axis]).as_matrix()
    candidate[:3, :3] = turn @ TRUTH[:3, :3]
    expected = rotation_errors_deg(candidate[:3, :3], TRUTH[:3, :3])  # the check's convention
    artifact = estimate_stereo_rotation(
        pairs, LEFT, RIGHT, candidate, provenance=PROVENANCE, options=OPTIONS
    )
    errors = [entry.candidate_error_deg for entry in artifact.axes]
    assert errors == pytest.approx(list(expected), abs=0.06)


def test_too_few_pairs_are_inconclusive(pairs: list[tuple[np.ndarray, np.ndarray]]) -> None:
    artifact = estimate_stereo_rotation(
        pairs[:3], LEFT, RIGHT, TRUTH, provenance=PROVENANCE, options=OPTIONS
    )
    assert artifact.policy_status == "inconclusive"
    assert artifact.axes == []
