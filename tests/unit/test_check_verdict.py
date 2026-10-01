"""The verdict rule of ``calibrex check``: tolerances, boundaries, detection power."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibrex.check.estimators import (
    classify_point_time,
    invert_transform,
    rotation_axis_estimates,
    rotation_errors_deg,
    translation_axis_estimates,
)
from calibrex.check.verdict import (
    AxisEstimate,
    VerdictOptions,
    judge_axis,
    judge_pair,
    worst_verdict,
)

OPTIONS = VerdictOptions()


def _rot(error: float, std: float, *, estimated: bool = True, name: str = "yaw") -> AxisEstimate:
    return AxisEstimate(name, "deg", error, std, estimated)  # type: ignore[arg-type]


def _trans(error: float, std: float, name: str = "x") -> AxisEstimate:
    return AxisEstimate(name, "m", error, std, True)  # type: ignore[arg-type]


def test_floor_dominates_small_sigma() -> None:
    axis = judge_axis(_rot(0.4, 0.02), OPTIONS)  # 3 * 0.02 = 0.06 < floor 0.5

    assert axis.tolerance == pytest.approx(0.5)
    assert axis.tolerance_source == "floor"
    assert axis.status == "pass"
    assert axis.ratio == pytest.approx(0.8)


def test_sigma_dominates_large_sigma() -> None:
    axis = judge_axis(_rot(1.2, 0.5), OPTIONS)  # 3 * 0.5 = 1.5 > floor

    assert axis.tolerance == pytest.approx(1.5)
    assert axis.tolerance_source == "sigma"
    assert axis.status == "pass"


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (0.5, "pass"),  # exactly at the tolerance passes
        (0.5001, "warn"),
        (-0.9, "warn"),
        (1.0, "warn"),  # exactly twice the tolerance is still a warning
        (1.0001, "fail"),
        (-3.0, "fail"),
    ],
)
def test_boundaries_use_absolute_error(error: float, status: str) -> None:
    assert judge_axis(_rot(error, 0.0), OPTIONS).status == status


def test_translation_uses_its_own_floor() -> None:
    assert judge_axis(_trans(0.019, 0.001), OPTIONS).status == "pass"
    assert judge_axis(_trans(0.03, 0.001), OPTIONS).status == "warn"
    assert judge_axis(_trans(0.05, 0.001), OPTIONS).status == "fail"
    assert judge_axis(_trans(0.03, 0.001), OPTIONS).detection_probe is None


def test_options_are_applied() -> None:
    wide = VerdictOptions(sigma_k=5.0, rotation_floor_deg=2.0)

    assert judge_axis(_rot(1.9, 0.1), wide).status == "pass"
    assert judge_axis(_rot(1.9, 0.1), OPTIONS).status == "fail"
    with pytest.raises(ValueError, match="positive"):
        VerdictOptions(sigma_k=0.0)


def test_detection_power_numbers() -> None:
    axis = judge_axis(_rot(0.2, 0.05), OPTIONS)  # tolerance 0.5

    assert axis.detectable_error == pytest.approx(0.7)
    assert axis.detection_probe == 1.0
    assert axis.detects_perturbation is True
    # The bound is exact for both signs: just above it flags either sign, just below it does not.
    tolerance, delta = axis.tolerance, abs(axis.candidate_error)
    for sign in (1, -1):
        assert abs(delta + sign * (axis.detectable_error + 1e-6)) > tolerance
    assert min(abs(delta + 0.69), abs(delta - 0.69)) <= tolerance

    blunt = judge_axis(_rot(0.2, 0.5), OPTIONS)  # tolerance 1.5
    assert blunt.detectable_error == pytest.approx(1.7)
    assert blunt.detects_perturbation is False


def test_pair_verdicts_follow_worst_axis() -> None:
    assert judge_pair([_rot(0.1, 0.1), _rot(0.2, 0.1, name="pitch")], OPTIONS).verdict == "pass"
    warn = judge_pair([_rot(0.1, 0.1), _rot(0.7, 0.1, name="pitch")], OPTIONS)
    assert warn.verdict == "warn"
    fail = judge_pair([_rot(0.7, 0.1), _rot(2.0, 0.1, name="pitch")], OPTIONS)
    assert fail.verdict == "fail"


def test_unobservable_axes_are_unchecked_not_judged() -> None:
    result = judge_pair(
        [
            _rot(0.1, 0.1),
            _rot(50.0, 4.0, estimated=False, name="pitch"),
            _trans(0.001, 0.001),
        ],
        OPTIONS,
    )

    assert result.verdict == "pass"
    assert [axis.name for axis in result.axes] == ["yaw", "x"]
    assert [axis.name for axis in result.unchecked] == ["pitch"]
    assert result.unchecked[0].std == 4.0


def test_all_unobservable_is_inconclusive() -> None:
    result = judge_pair(
        [_rot(0.0, 3.0, estimated=False, name="roll"), _rot(0.0, 3.0, estimated=False)],
        OPTIONS,
    )

    assert result.verdict == "inconclusive"
    assert result.axes == ()
    assert len(result.unchecked) == 2
    assert judge_pair([], OPTIONS).verdict == "inconclusive"


def test_overall_ordering() -> None:
    assert worst_verdict(["pass", "pass"]) == "pass"
    assert worst_verdict(["pass", "inconclusive"]) == "inconclusive"
    assert worst_verdict(["inconclusive", "warn", "pass"]) == "warn"
    assert worst_verdict(["warn", "fail", "inconclusive"]) == "fail"
    assert worst_verdict(["skipped", "planned"]) is None
    assert worst_verdict([]) is None


# ------------------------------------------------------------------ conversions


def test_rotation_errors_are_small_rotations_about_parent_axes() -> None:
    estimate = Rotation.from_euler("xyz", [10, 20, 30], degrees=True).as_matrix()
    perturbation = Rotation.from_rotvec(np.radians([0.0, 0.0, 1.0])).as_matrix()  # about parent z

    roll, pitch, yaw = rotation_errors_deg(perturbation @ estimate, estimate)

    assert (roll, pitch, yaw) == pytest.approx((0.0, 0.0, 1.0), abs=1e-9)


class _Record:
    def __init__(self, name: str, std: float, status: str = "estimated") -> None:
        self.name = name
        self.std_reported = std
        self.status = status


def test_axis_estimates_carry_status_and_std() -> None:
    estimate = np.eye(3)
    candidate = Rotation.from_rotvec(np.radians([0.0, 0.5, 0.0])).as_matrix()
    records = [_Record("roll", 0.1), _Record("pitch", 0.2), _Record("yaw", 2.0, "unobservable")]

    axes = rotation_axis_estimates(records, candidate, estimate)

    assert [a.name for a in axes] == ["roll", "pitch", "yaw"]
    assert axes[1].candidate_error == pytest.approx(0.5)
    assert [a.estimated for a in axes] == [True, True, False]
    assert axes[2].unchecked_reason and "2" in axes[2].unchecked_reason
    trans = translation_axis_estimates(
        [_Record("x", 0.01), _Record("y", 0.01), _Record("z", 1.0, "unobservable")],
        [0.1, 0.2, 0.3],
        [0.1, 0.25, 0.0],
    )
    assert trans[1].candidate_error == pytest.approx(-0.05)
    assert [a.estimated for a in trans] == [True, True, False]


def test_invert_transform_roundtrip() -> None:
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("xyz", [5, 6, 7], degrees=True).as_matrix()
    matrix[:3, 3] = [1.0, 2.0, 3.0]

    assert invert_transform(matrix) @ matrix == pytest.approx(np.eye(4), abs=1e-12)


def test_classify_point_time() -> None:
    header_ns = 1_650_529_602_387_975_000
    header_s = header_ns * 1e-9

    assert classify_point_time(np.array([0.0, 0.05, 0.099]), header_ns) == "offset_s"  # Ouster t
    assert classify_point_time(header_s + np.array([0.0, 0.05, 0.1]), header_ns) == "absolute_s"
    assert classify_point_time(header_ns + np.array([0.0, 5e7, 1e8]), header_ns) == "absolute_ns"
    assert classify_point_time(np.array([123456.0, 234567.0]), header_ns) is None
    assert classify_point_time(np.array([np.nan]), header_ns) is None
    assert classify_point_time(None, header_ns) is None


def test_coverage_and_control_not_detected() -> None:
    class Control:
        detected = False
        amount = 20.0
        unit = "mm"
        holdout_delta_chi2 = -15.3

    record = _Record("x", 0.009)
    record.known_bad_control = Control()  # type: ignore[attr-defined]
    ok = _Record("z", 0.005)
    ok.known_bad_control = None  # type: ignore[attr-defined]
    axes = translation_axis_estimates([record, ok], [0.0, 0.0, 0.0], [0.03, 0.0, 0.0])

    assert axes[0].estimated is False and axes[0].unchecked_code == "control_not_detected"
    result = judge_pair(axes, OPTIONS)
    assert result.unchecked[0].reason_code == "control_not_detected"
    assert result.coverage == "partial"
    assert judge_pair([_rot(0.1, 0.1)], OPTIONS).coverage == "full"
