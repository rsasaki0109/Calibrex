from __future__ import annotations

import functools
import importlib.util
import math
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibrex.check.hints import excitation_hints
from calibrex.core.calibration_check import CheckUncheckedAxis
from calibrex.core.excitation import AxisExcitation
from calibrex.core.gnss_lidar_lever_arm import GnssLidarDofRecord
from calibrex.core.imu_lidar_translation import ImuLidarTranslationAxisRecord
from calibrex.evaluation.gnss_lidar_lever_arm import (
    GnssLidarEvaluation,
    GnssLidarRunOptions,
    evaluate_gnss_lidar_lever_arm,
)
from calibrex.evaluation.imu_lidar_translation import (
    ImuLidarTranslationEvaluation,
    evaluate_imu_lidar_translation,
)
from calibrex.evaluation.translation_observability import (
    ExcitationInputs,
    diagnose_translation_axes,
    format_duration,
    ideal_information,
    relative_rotation_vectors,
    rotation_moment,
)
from calibrex.solvers.imu_lidar_translation_solver import (
    solve_translation,
    translation_std_m,
    window_systems,
)


def _load(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


IMU = _load("imu_translation_fixtures", "test_imu_lidar_translation.py")
GNSS = _load("gnss_lever_arm_fixtures", "test_gnss_lidar_lever_arm.py")


def _stack(rotation: Rotation) -> np.ndarray:
    return np.asarray(rotation.as_matrix(), dtype=np.float64).reshape(-1, 3, 3)


# ------------------------------------------------------------------ analytic math


def test_yaw_only_rotation_gives_no_information_on_the_z_lever_arm() -> None:
    yaws = np.linspace(0.1, 1.2, 12)
    first = np.broadcast_to(np.eye(3), (12, 3, 3))
    second = _stack(Rotation.from_rotvec(np.column_stack([np.zeros(12), np.zeros(12), yaws])))

    information = ideal_information(first, second)
    moment = rotation_moment(relative_rotation_vectors(first, second))

    assert information[2, 2] == pytest.approx(0.0, abs=1e-12)
    assert information[0, 0] > 1.0 and information[1, 1] > 1.0
    # |(R_z(a) - I) e_x|^2 = 2 - 2 cos a, summed over the rotations
    assert information[0, 0] == pytest.approx(float(np.sum(2.0 - 2.0 * np.cos(yaws))))
    assert np.allclose(moment, np.diag([0.0, 0.0, float(np.sum(yaws**2))]), atol=1e-12)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_rotation_about_one_axis_leaves_that_axis_unobservable(axis: int) -> None:
    angles = np.linspace(0.2, 0.9, 7)
    rotvec = np.zeros((7, 3))
    rotvec[:, axis] = angles
    first = np.broadcast_to(np.eye(3), (7, 3, 3))

    information = ideal_information(first, _stack(Rotation.from_rotvec(rotvec)))

    assert information[axis, axis] == pytest.approx(0.0, abs=1e-12)
    others = [index for index in range(3) if index != axis]
    assert all(information[index, index] > 0.5 for index in others)


def test_pure_translation_excites_nothing() -> None:
    first = np.broadcast_to(np.eye(3), (5, 3, 3))

    assert np.allclose(ideal_information(first, first.copy()), 0.0)
    assert np.allclose(rotation_moment(relative_rotation_vectors(first, first.copy())), 0.0)


def _inputs(
    std_m: list[float],
    *,
    moment_deg: tuple[float, float, float] = (20.0, 20.0, 20.0),
    kind: str = "imu_lidar",
    recording_s: float = 100.0,
    reported: list[float] | None = None,
    sensitivity: list[float] | None = None,
) -> ExcitationInputs:
    samples = 400
    moment = np.diag([math.radians(value) ** 2 * samples for value in moment_deg])
    std = np.array(std_m)
    return ExcitationInputs(
        kind=kind,  # type: ignore[arg-type]
        std_analytic_m=std,
        std_scaling_m=std,
        std_reported_m=np.array(reported) if reported is not None else std,
        bound_m=0.01,
        ideal_information=np.diag([4.0e4, 4.0e4, 4.0e4]),
        rotation_moment=moment,
        samples=samples,
        covered_s=80.0,
        recording_s=recording_s,
        interval_s=1.2,
        sensitivity_m=None if sensitivity is None else np.array(sensitivity),
    )


def test_needed_duration_scales_with_the_square_of_the_std_to_bound_ratio() -> None:
    result = diagnose_translation_axes(_inputs([0.02, 0.03, 0.005]))
    by_name = {item.name: item for item in result}

    assert by_name["x"].cause == "insufficient_duration"
    assert by_name["x"].needed_duration_s == pytest.approx(100.0 * 4.0)
    assert by_name["y"].needed_duration_s == pytest.approx(100.0 * 9.0)
    assert by_name["z"].cause == "observable"
    assert by_name["z"].needed_duration_s == pytest.approx(100.0 * 0.25)
    assert "about 7 min" in by_name["x"].recommendation or "7 min" in by_name["x"].recommendation


def test_std_shrinks_as_one_over_root_time_so_the_prediction_is_self_consistent() -> None:
    # Predict the std at the needed duration: it must equal the bound.
    std = 0.04
    needed = diagnose_translation_axes(_inputs([std, std, std]))[0].needed_duration_s
    assert needed is not None
    predicted_std_at_needed = std * math.sqrt(100.0 / needed)
    assert predicted_std_at_needed == pytest.approx(0.01)


def test_planar_yaw_only_motion_is_called_fundamentally_unobservable_for_z() -> None:
    result = diagnose_translation_axes(
        _inputs([0.004, 0.004, 0.2], moment_deg=(0.5, 0.4, 25.0), kind="gnss_lidar")
    )
    by_name = {item.name: item for item in result}

    assert by_name["x"].cause == "observable" and by_name["y"].cause == "observable"
    z = by_name["z"]
    assert z.cause == "no_excitation"
    assert z.missing_rotation_axes == ["x", "y"]
    assert z.perpendicular_rotation_rms_deg < 1.0
    assert "planar" in z.recommendation and "--tf" in z.recommendation
    assert "tilting for x or y" in z.recommendation


def test_model_limited_when_only_the_non_scaling_floor_is_over_the_bound() -> None:
    result = diagnose_translation_axes(
        _inputs([0.004] * 3, reported=[0.004, 0.004, 0.03], sensitivity=[0.0, 0.0, 0.03])
    )

    assert result[2].cause == "model_limited"
    assert "modelling error" in result[2].recommendation


def test_unsolved_axes_are_reported_as_such() -> None:
    result = diagnose_translation_axes(_inputs([math.inf, math.inf, math.inf]))

    assert all(item.cause == "unsolved" and item.needed_duration_s is None for item in result)


def test_format_duration_picks_a_readable_unit() -> None:
    assert format_duration(45.0) == "45 s"
    assert format_duration(600.0) == "10 min"
    assert format_duration(3 * 3600.0) == "3.0 h"
    assert format_duration(math.inf) == "an unbounded time"


# ------------------------------------------------------- accelerometer lever arm


@functools.cache
def _imu_evaluation(planar: bool) -> ImuLidarTranslationEvaluation:
    imu, windows = IMU.synthetic(planar=planar, seed=3)
    return evaluate_imu_lidar_translation(
        imu, windows, IMU.TRUE_ROTATION, IMU.TRUE_DT, reference_translation=IMU.TRUE_TRANSLATION
    )


def _excitation(
    records: tuple[ImuLidarTranslationAxisRecord, ...] | tuple[GnssLidarDofRecord, ...],
) -> dict[str, AxisExcitation]:
    return {r.name: r.excitation for r in records if r.excitation is not None}


def test_imu_yaw_only_motion_has_no_rotation_to_excite_z() -> None:
    evaluation = _imu_evaluation(planar=True)
    excitation = _excitation(evaluation.records)

    z = excitation["z"]
    assert z.cause != "observable"
    assert z.perpendicular_rotation_rms_deg < 1.0
    assert z.rotation_rms_deg[2] > 20.0 > z.rotation_rms_deg[0]
    assert z.missing_rotation_axes == ["x", "y"]
    # Yaw excites x and y (they have rotation perpendicular to them), so they are far
    # better conditioned than z.
    assert excitation["x"].perpendicular_rotation_rms_deg > 10.0
    assert (z.predicted_std_m or 0.0) > 3.0 * (excitation["x"].predicted_std_m or 0.0)


def test_imu_three_dimensional_motion_excites_every_axis() -> None:
    evaluation = _imu_evaluation(planar=False)
    excitation = _excitation(evaluation.records)

    assert {item.cause for item in excitation.values()} == {"observable"}
    for item in excitation.values():
        assert item.perpendicular_rotation_rms_deg > 5.0
        assert item.information_efficiency is not None and 0.0 < item.information_efficiency <= 1.0


def test_imu_ideal_information_has_no_z_for_constant_rate_yaw() -> None:
    imu, windows = IMU.synthetic(planar=True, seed=1)
    systems = window_systems(windows, imu, IMU.TRUE_ROTATION, IMU.TRUE_DT)

    ideal = sum((system.design.T @ system.design for system in systems), np.zeros((3, 3)))
    moment = sum((system.rotation_moment for system in systems), np.zeros((3, 3)))

    assert ideal[2, 2] < 1e-3 * ideal[0, 0]
    assert moment[2, 2] > 1e3 * moment[0, 0]


def test_imu_std_shrinks_with_more_windows_of_the_same_motion() -> None:
    imu, windows = IMU.synthetic(planar=False, seed=5)

    def std(count: int) -> np.ndarray:
        result = solve_translation(
            window_systems(windows[:count], imu, IMU.TRUE_ROTATION, IMU.TRUE_DT)
        )
        return translation_std_m(result)

    ratio = std(4) / std(9)
    # 9 / 4 times the data: the std shrinks by about 1.5.
    assert np.all((ratio > 1.15) & (ratio < 2.1)), ratio


# ----------------------------------------------------------------- GNSS lever arm


@functools.cache
def _gnss_evaluation(rotate: bool) -> GnssLidarEvaluation:
    track, windows = GNSS.synthetic(0, rotate=rotate)
    return evaluate_gnss_lidar_lever_arm(track, windows, GnssLidarRunOptions())


def test_gnss_rotating_motion_excites_every_axis_and_reports_efficiency() -> None:
    evaluation = _gnss_evaluation(rotate=True)
    excitation = _excitation(evaluation.records)

    assert set(excitation) == {"x", "y", "z"}
    for item in excitation.values():
        assert item.perpendicular_rotation_rms_deg > 5.0
        assert item.information_efficiency is not None and 0.0 < item.information_efficiency <= 1.0
        assert item.recording_s > 50.0


def test_gnss_pure_translation_has_no_rotation_to_excite_anything() -> None:
    evaluation = _gnss_evaluation(rotate=False)
    excitation = _excitation(evaluation.records)

    assert {item.cause for item in excitation.values()} == {"no_excitation"}
    assert all(max(item.rotation_rms_deg) < 1.0 for item in excitation.values())
    assert all(
        "needs rotation about the sensor" in item.recommendation for item in excitation.values()
    )


# -------------------------------------------------------------- hints and schema


def _axis_excitation(cause: str = "no_excitation") -> AxisExcitation:
    return AxisExcitation.model_validate(
        {
            "name": "z",
            "cause": cause,
            "predicted_std_m": 0.2,
            "bound_std_m": 0.01,
            "information_efficiency": 0.1,
            "rotation_rms_deg": [0.5, 0.4, 25.0],
            "perpendicular_rotation_rms_deg": 0.45,
            "recording_s": 300.0,
            "needed_duration_s": 120000.0,
            "missing_rotation_axes": ["x", "y"],
            "recommendation": "z: needs rotation about the sensor x or y axis.",
        }
    )


def test_unobservable_axes_become_next_steps_and_observable_ones_do_not() -> None:
    unobservable = CheckUncheckedAxis(
        name="z", unit="m", std=0.2, reason="not constrained", excitation=_axis_excitation()
    )
    observable = CheckUncheckedAxis(
        name="x", unit="m", std=0.002, reason="control", excitation=_axis_excitation("observable")
    )
    bare = CheckUncheckedAxis(name="y", unit="m", std=0.5, reason="not constrained")

    hints = excitation_hints("gnss-lidar", [unobservable, observable, bare])

    assert hints == ["gnss-lidar lever arm z: needs rotation about the sensor x or y axis."]


def test_excitation_round_trips_through_the_schemas() -> None:
    axis = ImuLidarTranslationAxisRecord(
        name="z",
        value=0.01,
        std_analytic=0.2,
        std_reported=0.2,
        status="unobservable",
        excitation=_axis_excitation(),
    )
    restored = ImuLidarTranslationAxisRecord.model_validate(axis.model_dump(mode="json"))
    assert restored.excitation == axis.excitation

    schema = ImuLidarTranslationAxisRecord.model_json_schema()
    assert "excitation" in schema["properties"]
    assert "excitation" not in schema.get("required", [])
    with pytest.raises(ValueError):
        AxisExcitation.model_validate({**axis.excitation.model_dump(), "cause": "bogus"})  # type: ignore[union-attr]


def test_a_duration_extrapolated_from_few_windows_says_so() -> None:
    from dataclasses import replace

    base = _inputs([0.03, 0.03, 0.03])

    few = diagnose_translation_axes(replace(base, windows=6))[0]
    many = diagnose_translation_axes(replace(base, windows=40))[0]

    assert few.fit_windows == 6 and "only 6 windows" in few.recommendation
    assert "only" not in many.recommendation


def test_a_segment_sensitivity_over_the_bound_is_flagged_as_modelling_error() -> None:
    result = diagnose_translation_axes(_inputs([0.03] * 3, sensitivity=[0.0, 0.0, 0.02]))

    assert "Caution" not in result[0].recommendation
    assert "segment-duration sensitivity 2 cm" in result[2].recommendation
