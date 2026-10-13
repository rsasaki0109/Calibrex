"""A stereo pair's relative rotation from its images, on rendered fisheye pairs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tests.unit.stereo_scene import LEFT, RIGHT, TRUTH, render_pairs, render_turning_pairs

from calibrex.check.estimators import rotation_errors_deg
from calibrex.core.stereo_rotation import StereoRotationProvenance
from calibrex.core.validation import validate_file
from calibrex.evaluation.stereo_rotation import (
    StereoRotationOptions,
    StereoTiming,
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


def _rates(count: int) -> np.ndarray:
    rng = np.random.default_rng(9)
    axes = rng.normal(size=(count, 3))
    return np.radians(25.0) * axes / np.linalg.norm(axes, axis=1, keepdims=True)


def test_a_stamped_delay_is_turned_out_with_the_gyro() -> None:
    rates = _rates(8)
    delays = np.full(8, 0.025)  # the right image one 40 Hz frame later, stamped so
    pairs = render_turning_pairs(TRUTH, delays, rates)
    timing = StereoTiming(gaps_s=delays, rates_rad_s=rates, source="test gyro")

    blind = estimate_stereo_rotation(
        pairs, LEFT, RIGHT, TRUTH, provenance=PROVENANCE, options=OPTIONS
    )
    timed = estimate_stereo_rotation(
        pairs, LEFT, RIGHT, TRUTH, provenance=PROVENANCE, options=OPTIONS, timing=timing
    )

    assert max(axis.std_deg for axis in blind.axes) > 0.1  # the turn scatters the pairs
    assert all(abs(axis.candidate_error_deg) < 0.05 for axis in timed.axes)
    assert timed.policy_status == "pass"
    assert timed.time_offset_s == pytest.approx(0.0, abs=0.001)
    assert timed.time_offset_status == "estimated"
    assert timed.max_stamp_gap_s == pytest.approx(0.025)


def test_a_clock_offset_between_the_cameras_is_measured() -> None:
    rates = _rates(8)
    pairs = render_turning_pairs(TRUTH, np.zeros(8), rates)  # taken together
    stamped = np.full(8, 0.005)  # but the right camera's clock runs 5 ms ahead
    timing = StereoTiming(gaps_s=stamped, rates_rad_s=rates, source="test gyro")

    artifact = estimate_stereo_rotation(
        pairs, LEFT, RIGHT, TRUTH, provenance=PROVENANCE, options=OPTIONS, timing=timing
    )

    assert artifact.time_offset_s == pytest.approx(-0.005, abs=0.001)
    assert artifact.time_offset_std_s is not None and artifact.time_offset_std_s < 0.001
    assert all(abs(axis.candidate_error_deg) < 0.05 for axis in artifact.axes)
