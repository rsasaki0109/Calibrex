"""A stereo pair's relative rotation from its images, on rendered fisheye pairs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tests.unit.stereo_scene import (
    LEFT,
    RIGHT,
    TRUTH,
    render,
    render_pairs,
    render_turning_pairs,
)

from calibrex.check.estimators import rotation_errors_deg
from calibrex.core.stereo_rotation import StereoRotationProvenance
from calibrex.core.validation import validate_file
from calibrex.evaluation.stereo_rotation import (
    StereoRotationOptions,
    StereoTiming,
    estimate_stereo_rotation,
    shared_view_fraction,
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


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_candidate_degrees_off_is_found_by_re_centring(
    pairs: list[tuple[np.ndarray, np.ndarray]], axis: int
) -> None:
    # 3 deg moves rows or columns ~8 px at 160 px: beyond the +-4 row search and, about y,
    # to negative disparity; the coarse pass finds it and the rectification is re-centred
    candidate = TRUTH.copy()
    turn = Rotation.from_rotvec(np.radians(3.0) * np.eye(3)[axis]).as_matrix()
    candidate[:3, :3] = turn @ TRUTH[:3, :3]
    expected = rotation_errors_deg(candidate[:3, :3], TRUTH[:3, :3])
    artifact = estimate_stereo_rotation(
        pairs, LEFT, RIGHT, candidate, provenance=PROVENANCE, options=OPTIONS
    )
    assert artifact.policy_status == "pass"
    assert [entry.candidate_error_deg for entry in artifact.axes] == pytest.approx(
        list(expected), abs=0.06
    )
    assert artifact.recentred_by_deg is not None
    assert artifact.recentred_by_deg[axis] == pytest.approx(3.0, abs=0.1)


def test_a_close_candidate_is_measured_without_re_centring(
    pairs: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    artifact = estimate_stereo_rotation(
        pairs, LEFT, RIGHT, TRUTH, provenance=PROVENANCE, options=OPTIONS
    )
    assert artifact.recentred_by_deg is None


def test_too_few_pairs_are_inconclusive(pairs: list[tuple[np.ndarray, np.ndarray]]) -> None:
    artifact = estimate_stereo_rotation(
        pairs[:3], LEFT, RIGHT, TRUTH, provenance=PROVENANCE, options=OPTIONS
    )
    assert artifact.policy_status == "inconclusive"
    assert artifact.axes == []


def _rates(count: int, deg_s: float = 25.0) -> np.ndarray:
    rng = np.random.default_rng(9)
    axes = rng.normal(size=(count, 3))
    return np.radians(deg_s) * axes / np.linalg.norm(axes, axis=1, keepdims=True)


@pytest.mark.parametrize("deg_s", [25.0, 80.0])
def test_a_stamped_delay_is_turned_out_with_the_gyro(deg_s: float) -> None:
    # at 80 deg/s a 25 ms gap turns the right camera 2 deg: rows move ~6 px, beyond the
    # matcher's +-4 px search unless the gap's turn is rectified away first
    rates = _rates(8, deg_s)
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


def test_the_shared_view_of_two_cameras_follows_their_angle() -> None:
    def turned(degrees: float) -> np.ndarray:
        transform = TRUTH.copy()
        transform[:3, :3] = Rotation.from_euler("y", degrees, degrees=True).as_matrix()
        return transform

    same = shared_view_fraction(LEFT, RIGHT, TRUTH)
    square = shared_view_fraction(LEFT, RIGHT, turned(90.0))
    opposite = shared_view_fraction(LEFT, RIGHT, turned(180.0))
    assert same > 0.85
    assert 0.0 < square < same
    assert opposite == 0.0


def _pairs_with_right_camera(
    truth: np.ndarray, right_true: object, count: int = 8, seed: int = 4
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pairs whose right images come from ``right_true``, not the model the fit is given."""

    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(count):
        world_left = Rotation.from_euler(
            "zyx",
            [rng.uniform(-180, 180), rng.uniform(-30, 30), rng.uniform(-20, 20)],
            degrees=True,
        ).as_matrix()
        pairs.append(
            (
                render(LEFT, world_left, np.zeros(3)),
                render(right_true, world_left @ truth[:3, :3], world_left @ truth[:3, 3]),  # type: ignore[arg-type]
            )
        )
    return pairs


SIDE = np.eye(4)
SIDE[:3, :3] = Rotation.from_euler("xyz", [0.3, 60.0, 0.2], degrees=True).as_matrix()
SIDE[:3, 3] = [0.08, 0.0, -0.05]


@pytest.mark.parametrize("truth", [TRUTH, SIDE], ids=["forward", "60 deg apart"])
def test_a_vertical_scale_mismatch_is_fitted_not_read_as_a_turn(truth: np.ndarray) -> None:
    # the right camera's true fy is 0.5% above its model: its rectified rows are 0.5% too
    # far from the centre. Read as a turn, that is ~0.3 to 0.7 deg; fitted, it is the scale.
    pairs = _pairs_with_right_camera(truth, replace(RIGHT, fy=RIGHT.fy * 1.005))
    options = StereoRotationOptions(patches_per_pair=150, min_patches=200)
    fitted = estimate_stereo_rotation(
        pairs, LEFT, RIGHT, truth, provenance=PROVENANCE, options=options
    )
    ignored = estimate_stereo_rotation(
        pairs,
        LEFT,
        RIGHT,
        truth,
        provenance=PROVENANCE,
        options=replace(options, fit_row_scale=False),
    )

    assert fitted.policy_status == "pass"
    assert all(abs(axis.candidate_error_deg) < 0.15 for axis in fitted.axes)
    assert all(axis.scale_std_deg is not None for axis in fitted.axes)
    assert fitted.row_scale_ratio == pytest.approx(0.005, abs=0.001)
    assert max(abs(axis.candidate_error_deg) for axis in ignored.axes) > 0.3
    assert ignored.row_scale_ratio is None
