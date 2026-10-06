"""``calibrex drift``: the consistency statistics, the artifact and the CLI."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tests.unit.check_fixtures import write_check_bag

from calibrex.check import estimators
from calibrex.check.drift import (
    ARTIFACT_NAME,
    DriftOptions,
    Observation,
    assess_axis,
    build_calibration_drift,
    compare_pair,
    format_drift_text,
    worst_drift_verdict,
)
from calibrex.check.estimators import (
    CheckSkipError,
    EstimatorRun,
    PairContext,
    invert_transform,
)
from calibrex.check.runner import CheckRunOptions
from calibrex.check.verdict import AxisEstimate
from calibrex.cli.main import main
from calibrex.core.calibration_check import CheckTimeOffset
from calibrex.core.calibration_drift import CalibrationDriftArtifact
from calibrex.core.schema_registry import kind_for_schema_version
from calibrex.core.validation import validate_file

OPTIONS = DriftOptions()


def _obs(bag: str, value: float | None, std: float | None, status: str = "observed") -> Observation:
    return Observation(bag, value, std, status)


# --------------------------------------------------------------------- statistics


def test_identical_bags_are_stable() -> None:
    axis = assess_axis("yaw", "deg", [_obs("a", 0.3, 0.05), _obs("b", 0.3, 0.05)], OPTIONS)
    assert axis.status == "stable"
    assert axis.chi2 == pytest.approx(0.0)
    assert axis.p_value == pytest.approx(1.0)
    assert axis.max_abs_difference == pytest.approx(0.0)


def test_small_std_does_not_flag_a_change_below_the_floor() -> None:
    # 0.3 deg is 4 combined sigma but under the 0.5 deg rotation floor
    axis = assess_axis("yaw", "deg", [_obs("a", 0.0, 0.05), _obs("b", 0.3, 0.05)], OPTIONS)
    assert axis.status == "stable"
    assert axis.pairwise[0].z == pytest.approx(-0.3 / math.hypot(0.05, 0.05))
    assert not axis.pairwise[0].exceeds
    assert axis.minimum_detectable_change == pytest.approx(0.5)
    assert axis.floor_source == "rotation"


def test_change_above_floor_and_sigma_is_drift_and_symmetric() -> None:
    forward = assess_axis("yaw", "deg", [_obs("a", 0.0, 0.05), _obs("b", 1.0, 0.05)], OPTIONS)
    backward = assess_axis("yaw", "deg", [_obs("b", 1.0, 0.05), _obs("a", 0.0, 0.05)], OPTIONS)
    assert forward.status == backward.status == "drift"
    assert forward.pairwise[0].exceeds
    assert forward.p_value is not None and forward.p_value < OPTIONS.chi2_alpha
    assert forward.leave_one_out == []  # two bags cannot attribute


def test_large_std_swallows_a_change_above_the_floor() -> None:
    axis = assess_axis("yaw", "deg", [_obs("a", 0.0, 0.5), _obs("b", 1.0, 0.5)], OPTIONS)
    assert axis.status == "stable"
    assert axis.pairwise[0].tolerance == pytest.approx(3.0 * math.hypot(0.5, 0.5))


def test_rigid_scan_and_translation_floors() -> None:
    rigid = assess_axis(
        "yaw", "deg", [_obs("a", 0.0, 0.05), _obs("b", 1.0, 0.05)], OPTIONS, rigid_scan=True
    )
    assert rigid.status == "stable"
    assert rigid.floor_source == "rigid_scan_rotation"
    assert rigid.floor == pytest.approx(1.5)
    moved = assess_axis("x", "m", [_obs("a", 0.0, 0.002), _obs("b", 0.05, 0.002)], OPTIONS)
    assert moved.status == "drift" and moved.floor_source == "translation"
    near = assess_axis("x", "m", [_obs("a", 0.0, 0.002), _obs("b", 0.015, 0.002)], OPTIONS)
    assert near.status == "stable"


def test_three_bags_attribute_the_odd_one() -> None:
    axis = assess_axis(
        "pitch",
        "deg",
        [_obs("a", 0.1, 0.05), _obs("b", 0.12, 0.05), _obs("c", 1.5, 0.05)],
        OPTIONS,
    )
    assert axis.status == "drift"
    assert [e.bag for e in axis.leave_one_out if e.deviates] == ["c"]
    assert axis.dof == 2


def test_chi_square_gate_can_overrule_one_marginal_pair() -> None:
    # Many agreeing bags and one pair just past 3 sigma: the pair exceeds but homogeneity holds.
    options = DriftOptions(sigma_k=1.0, rotation_floor_deg=0.01, chi2_alpha=0.001)
    values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.45]
    axis = assess_axis("yaw", "deg", [_obs(f"b{i}", v, 0.1) for i, v in enumerate(values)], options)
    assert any(p.exceeds for p in axis.pairwise)
    assert axis.status == "stable"
    assert axis.reason is not None and "chi-square" in axis.reason


def test_unobserved_axes_are_excluded_and_one_bag_is_inconclusive() -> None:
    axis = assess_axis(
        "roll",
        "deg",
        [
            _obs("a", 0.0, 0.05),
            _obs("b", 5.0, 0.05, "unobservable"),
            _obs("c", 5.0, 0.05, "control_not_detected"),
            _obs("d", None, None, "pair_not_estimated"),
        ],
        OPTIONS,
    )
    assert axis.status == "inconclusive"
    assert axis.bags_observed == 1
    assert [o.used for o in axis.observations] == [True, False, False, False]
    # an unobserved value never enters the test, however far from the others
    three = assess_axis(
        "roll",
        "deg",
        [_obs("a", 0.0, 0.05), _obs("b", 0.0, 0.05), _obs("c", 9.0, 0.05, "unobservable")],
        OPTIONS,
    )
    assert three.status == "stable" and three.bags_observed == 2


def test_pair_verdict_and_rotation_magnitude_of_the_deviating_bag() -> None:
    rotation = Rotation.from_rotvec(np.radians([0.2, -0.1, 0.05]))
    shift = Rotation.from_euler("z", 2.0, degrees=True)
    shifted = np.degrees((shift * rotation).as_rotvec())
    base = np.degrees(rotation.as_rotvec())
    axes: list[tuple[Any, Any, list[Observation]]] = []
    for i, name in enumerate(("roll", "pitch", "yaw")):
        axes.append(
            (
                name,
                "deg",
                [
                    _obs("a", float(base[i]), 0.05),
                    _obs("b", float(base[i]), 0.05),
                    _obs("c", float(shifted[i]), 0.05),
                ],
            )
        )
    axes.append(("x", "m", [_obs("a", 0.0, 0.01), _obs("b", None, None, "unobservable")]))
    pair = compare_pair("imu-lidar", "lidar", "imu", axes, ["a", "b", "c"], OPTIONS)
    assert pair.verdict == "drift"
    assert pair.deviating_bags == ["c"]
    assert pair.attribution == "bag"
    (change,) = pair.changes
    assert change.bag == "c"
    assert change.rotation_delta_deg == pytest.approx(2.0, abs=0.05)
    x_axis = next(a for a in pair.axes if a.name == "x")
    assert x_axis.status == "inconclusive"


def test_two_bag_drift_is_ambiguous_and_inconclusive_pairs() -> None:
    axes: list[tuple[Any, Any, list[Observation]]] = [
        ("yaw", "deg", [_obs("a", 0.0, 0.05), _obs("b", 1.2, 0.05)])
    ]
    pair = compare_pair("imu-lidar", None, None, axes, ["a", "b"], OPTIONS)
    assert pair.verdict == "drift" and pair.attribution == "ambiguous"
    assert pair.deviating_bags == []
    assert pair.changes[0].reference == "other_bag"
    none_observed: list[tuple[Any, Any, list[Observation]]] = [
        ("yaw", "deg", [_obs("a", 0.0, 0.05), _obs("b", 0.0, 0.05, "unobservable")])
    ]
    assert (
        compare_pair("imu-lidar", None, None, none_observed, ["a", "b"], OPTIONS).verdict
        == "inconclusive"
    )


def test_overall_verdict_ranking_and_options_validation() -> None:
    assert worst_drift_verdict(["stable", "inconclusive"]) == "inconclusive"
    assert worst_drift_verdict(["stable", "drift", "inconclusive"]) == "drift"
    assert worst_drift_verdict(["stable"]) == "stable"
    assert worst_drift_verdict([]) == "inconclusive"
    with pytest.raises(ValueError):
        DriftOptions(chi2_alpha=1.5)
    with pytest.raises(ValueError):
        DriftOptions(sigma_k=0.0)


# ------------------------------------------------- artifact and CLI on stubbed estimators

ERRORS_PER_BAG: list[tuple[float, float, float]] = []
#: (estimate_s, std_s, status) of the stub's clock offset per run; empty means 2 ms +- 1 ms
OFFSETS_PER_BAG: list[tuple[float, float, str]] = []


def _set_offsets(*per_bag: tuple[float, float, str]) -> None:
    OFFSETS_PER_BAG[:] = [offset for offset in per_bag for _ in range(2)]


def _set_errors(*per_bag: tuple[float, float, float]) -> None:
    """The fixture bag has two LiDARs, so the stub runs twice per bag."""

    ERRORS_PER_BAG[:] = [errors for errors in per_bag for _ in range(2)]


def _stub(ctx: PairContext) -> EstimatorRun:
    errors = ERRORS_PER_BAG.pop(0)
    value, std, status = OFFSETS_PER_BAG.pop(0) if OFFSETS_PER_BAG else (0.002, 0.001, "estimated")
    estimates = tuple(
        AxisEstimate(name, "deg", error, 0.05, True)  # type: ignore[arg-type]
        for name, error in zip(("roll", "pitch", "yaw"), errors, strict=True)
    )
    return EstimatorRun(
        estimator="stub/v0",
        artifacts=(),
        solved=True,
        policy_status="pass",
        policy_reasons=("ok",),
        estimates=estimates,
        compared=invert_transform(ctx.candidate),
        time_offset=CheckTimeOffset(
            estimate_s=value,
            std_s=std,
            status=status,  # type: ignore[arg-type]
        ),
    )


def _skip(_ctx: PairContext) -> EstimatorRun:
    raise CheckSkipError("unsupported_sensor", "stub: not in this test")


@pytest.fixture
def bags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    table: dict[str, Callable[[PairContext], EstimatorRun]] = dict.fromkeys(
        estimators.ESTIMATORS, _skip
    )
    table["imu-lidar"] = _stub
    monkeypatch.setattr(estimators, "ESTIMATORS", table)
    return [write_check_bag(tmp_path / name) for name in ("rec1.db3", "rec2.db3", "rec3.db3")]


def _run(bags: list[Path], out: Path) -> CalibrationDriftArtifact:
    return build_calibration_drift(
        bags,
        output_dir=out,
        run=CheckRunOptions(),
        command=["calibrex", "drift"],
    )


def test_drift_artifact_names_the_deviating_bag_and_is_schema_valid(
    tmp_path: Path, bags: list[Path]
) -> None:
    _set_errors((0.1, 0.0, 0.2), (0.12, 0.0, 0.18), (0.1, 0.0, 2.2))
    out = tmp_path / "drift"
    artifact = _run(bags, out)
    pair = next(p for p in artifact.pairs if p.pair == "imu-lidar")
    assert artifact.overall_verdict == "drift" and pair.verdict == "drift"
    assert pair.deviating_bags == ["rec3.db3"]
    assert pair.changes[0].rotation_delta_deg == pytest.approx(2.0, abs=0.1)
    assert artifact.summary.axes_drifting == 2  # yaw of each of the two LiDAR pairs
    # per-bag estimates are written and referenced by digest
    for ref in artifact.bags:
        assert (out / ref.estimate_path).is_file()
        assert len(ref.estimate_sha256) == 64
    assert "deviating" in format_drift_text(artifact, out)
    payload = json.loads(artifact.model_dump_json(exclude_none=True))
    assert kind_for_schema_version(payload["schema_version"]) == "calibration-drift"
    assert artifact.provenance.generator == "calibrex drift"
    assert len(artifact.provenance.input_sha256) == 64
    assert artifact.thresholds.sigma_k == 3.0


def test_cli_exit_codes_files_and_validation(tmp_path: Path, bags: list[Path]) -> None:
    out = tmp_path / "drift"
    args = ["drift", *map(str, bags[:2]), "--output", str(out), "--no-cache", "-q"]
    _set_errors((0.1, 0.0, 0.2), (0.1, 0.0, 0.2))
    assert main([*args, "--html", str(tmp_path / "r.html")]) == 0
    path = out / ARTIFACT_NAME
    assert validate_file(path, "calibration-drift").kind == "calibration-drift"
    assert validate_file(path).kind == "calibration-drift"
    html = (tmp_path / "r.html").read_text(encoding="utf-8")
    assert "Calibration drift" in html and "stable" in html
    _set_errors((0.1, 0.0, 0.2), (0.1, 0.0, 1.5))
    assert main(args) == 1  # drift fails by default
    _set_errors((0.1, 0.0, 0.2), (0.1, 0.0, 1.5))
    assert main([*args, "--fail-on", "never"]) == 0
    assert main(["drift", str(bags[0]), "--output", str(out)]) != 0  # needs two bags


def test_next_steps_for_an_ambiguous_two_bag_drift(tmp_path: Path, bags: list[Path]) -> None:
    _set_errors((0.0, 0.0, 0.0), (0.0, 0.0, 1.5))
    artifact = _run(bags[:2], tmp_path / "d")
    assert artifact.pairs[0].attribution == "ambiguous"
    assert any("third recording" in step for step in artifact.next_steps)


def test_attribution_uses_all_axes_of_the_pair_together() -> None:
    """Koide indoor_easy_01 / 02 and 01 with its IMU remounted by 2 deg about z.

    On yaw alone the middle bag is closer to the remounted bag than to the original, so a
    per-axis attribution blames the original; over the three rotation axes together the
    remounted bag is the odd one out.
    """

    data = {
        "roll": (-64.837, -64.822, -65.710),
        "pitch": (65.334, 64.654, 63.942),
        "yaw": (72.325, 71.416, 70.786),
    }
    stds = (0.25, 0.21, 0.25)
    axes: list[tuple[Any, Any, list[Observation]]] = [
        (name, "deg", [_obs(b, v, s) for b, v, s in zip("abc", values, stds, strict=True)])
        for name, values in data.items()
    ]
    yaw_alone = assess_axis("yaw", "deg", axes[2][2], OPTIONS, rigid_scan=True)
    assert [e.bag for e in yaw_alone.leave_one_out if e.deviates] == ["a"]
    pair = compare_pair("imu-lidar", None, None, axes, ["a", "b", "c"], OPTIONS, rigid_scan=True)
    assert pair.verdict == "drift"
    assert pair.deviating_bags == ["c"]
    assert [e.bag for a in pair.axes for e in a.leave_one_out if e.deviates] == ["c"] * 3


# ------------------------------------------------------------------ clock offset


def test_time_offset_floor_protects_a_small_std() -> None:
    # 1 ms apart with 0.1 ms stds: seven combined sigma but under the 2 ms floor
    obs = [_obs("a", 0.0020, 0.0001), _obs("b", 0.0030, 0.0001)]
    record = assess_axis("time_offset", "s", obs, OPTIONS, floor_override=0.002)
    assert record.status == "stable" and record.unit == "s"
    assert record.floor_source == "time_offset" and record.floor == pytest.approx(0.002)
    assert record.pairwise[0].exceeds is False
    assert record.pairwise[0].z == pytest.approx(-1.0 / (math.sqrt(2) * 0.1), rel=1e-6)
    bigger = [_obs("a", 0.0020, 0.0001), _obs("b", 0.0075, 0.0001)]
    assert assess_axis("time_offset", "s", bigger, OPTIONS, floor_override=0.002).status == "drift"


def test_compare_pair_flags_and_attributes_a_clock_offset_change_alone() -> None:
    axes: list[tuple[Any, Any, list[Observation]]] = [
        ("yaw", "deg", [_obs(b, 0.3, 0.05) for b in "abc"])
    ]
    clock = [_obs("a", 0.0018, 0.0002), _obs("b", 0.0019, 0.0002), _obs("c", 0.0118, 0.0002)]
    pair = compare_pair("camera-imu", None, None, axes, ["a", "b", "c"], OPTIONS, time_offset=clock)
    assert pair.verdict == "drift" and pair.attribution == "bag"
    assert pair.deviating_bags == ["c"]
    assert [a.status for a in pair.axes] == ["stable"]
    assert pair.time_offset is not None and pair.time_offset.status == "drift"
    assert [e.bag for e in pair.time_offset.leave_one_out if e.deviates] == ["c"]
    (change,) = pair.changes
    assert change.bag == "c" and change.time_offset_delta_s == pytest.approx(0.01, abs=2e-4)
    assert change.axes_delta["yaw"] == pytest.approx(0.0)
    # the clock offset alone with two bags cannot be attributed
    two_axes: list[tuple[Any, Any, list[Observation]]] = [
        ("yaw", "deg", [_obs("a", 0.3, 0.05), _obs("c", 0.3, 0.05)])
    ]
    two = compare_pair(
        "camera-imu", None, None, two_axes, ["a", "c"], OPTIONS, time_offset=[clock[0], clock[2]]
    )
    assert two.verdict == "drift" and two.attribution == "ambiguous"


def test_unobserved_clock_offsets_are_skipped_and_a_pair_can_be_stable_on_time_alone() -> None:
    clock = [
        _obs("a", 0.0018, 0.0002),
        _obs("b", 0.5, 0.2, "unobservable"),
        _obs("c", 0.0019, 0.0002),
    ]
    pair = compare_pair("imu-lidar", None, None, [], ["a", "b", "c"], OPTIONS, time_offset=clock)
    assert pair.time_offset is not None
    assert pair.time_offset.bags_observed == 2 and pair.verdict == "stable"
    assert pair.bags_compared == ["a", "c"]
    only_one = compare_pair("imu-lidar", None, None, [], ["a", "b"], OPTIONS, time_offset=clock[:2])
    assert only_one.verdict == "inconclusive"
    assert "clock offset" in (only_one.reason or "")


def test_time_offset_floor_options() -> None:
    assert OPTIONS.time_offset_floor_s("camera-imu") == pytest.approx(0.002)
    custom = DriftOptions(time_offset_floors_s={"default": 0.004, "camera-imu": 0.0005})
    assert custom.time_offset_floor_s("camera-imu") == pytest.approx(0.0005)
    assert custom.time_offset_floor_s("imu-lidar") == pytest.approx(0.004)
    with pytest.raises(ValueError):
        DriftOptions(time_offset_floors_s={"camera-imu": 0.001})
    with pytest.raises(ValueError):
        DriftOptions(time_offset_floors_s={"default": 0.0})


def _clock_change() -> None:
    _set_errors((0.1, 0.0, 0.2), (0.12, 0.0, 0.18), (0.1, 0.0, 0.2))
    _set_offsets(
        (0.0020, 0.0002, "estimated"),
        (0.0021, 0.0002, "estimated"),
        (0.0120, 0.0002, "estimated"),
    )


def test_drift_artifact_reports_a_clock_offset_change(tmp_path: Path, bags: list[Path]) -> None:
    _clock_change()
    out = tmp_path / "drift"
    artifact = _run(bags, out)
    pair = next(p for p in artifact.pairs if p.pair == "imu-lidar")
    assert [a.status for a in pair.axes] == ["stable"] * 3
    assert pair.verdict == "drift" and pair.deviating_bags == ["rec3.db3"]
    assert pair.time_offset is not None and pair.time_offset.status == "drift"
    assert artifact.summary.time_offsets_tested == 2  # two LiDAR pairs
    assert artifact.summary.time_offsets_drifting == 2
    assert artifact.thresholds.time_offset_floors_s["default"] == pytest.approx(0.002)
    assert any("clock offset" in step for step in artifact.next_steps)
    text = format_drift_text(artifact, out)
    assert " dt " in text and "time offset +9.950 ms" in text
    _clock_change()
    args = ["drift", *map(str, bags), "--output", str(out), "--no-cache", "-q"]
    html_path = tmp_path / "r.html"
    # a 20 ms floor swallows the 10 ms change: stable, so the exit code is 0 even by default
    assert main([*args, "--html", str(html_path), "--time-offset-floor-ms", "20"]) == 0
    assert "time_offset" in html_path.read_text(encoding="utf-8")
    assert validate_file(out / ARTIFACT_NAME).kind == "calibration-drift"
