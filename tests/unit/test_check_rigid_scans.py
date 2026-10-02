"""imu-lidar on clouds without per-point time: deskew selection, floor, provenance."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tests.unit.check_fixtures import default_tf, write_check_bag
from tests.unit.test_check_cache import _imu_lidar_context, _rotation_artifact
from tests.unit.test_imu_lidar_rotation import TRUE_ROTATION

from calibrex.check import estimators
from calibrex.check.estimators import (
    CheckSkipError,
    EstimatorRun,
    PairContext,
    RunControls,
    select_imu_lidar_deskew,
)
from calibrex.check.runner import CheckRunOptions, build_calibration_check
from calibrex.check.verdict import (
    RIGID_SCAN_ROTATION_FLOOR_DEG,
    AxisEstimate,
    VerdictOptions,
    judge_axis,
)
from calibrex.cli.main import main
from calibrex.core.calibration_check import (
    CalibrationCheckArtifact,
    CheckOptions,
    CheckPairRecord,
)
from calibrex.core.imu_lidar_rotation import ImuLidarRotationArtifact
from calibrex.evaluation import imu_lidar_rotation as rotation_module
from calibrex.evaluation.imu_lidar_rotation import ImuLidarRunOptions


def _no_time(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def detect(bag: Path, topic: str) -> None:
        calls.append(topic)

    monkeypatch.setattr(estimators, "detect_point_time", detect)
    return calls


def test_auto_uses_per_point_time_and_rigid_scans_only_without_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bag = Path("unused")
    monkeypatch.setattr(estimators, "detect_point_time", lambda b, t: ("t", "offset_s"))
    assert select_imu_lidar_deskew(bag, "/l", "auto") == ("gyro", ("t", "offset_s"))
    assert select_imu_lidar_deskew(bag, "/l", "gyro") == ("gyro", ("t", "offset_s"))
    # Forced rigid mode is explicit even when the field exists, and never reads the cloud.
    assert select_imu_lidar_deskew(bag, "/l", "none") == ("none", None)

    calls = _no_time(monkeypatch)
    assert select_imu_lidar_deskew(bag, "/l", "auto") == ("none", None)
    assert calls == ["/l"]
    with pytest.raises(CheckSkipError) as skipped:
        select_imu_lidar_deskew(bag, "/l", "gyro")
    assert skipped.value.code == "unsupported_sensor"
    assert "--imu-lidar-deskew none" in skipped.value.reason


def _run_adapter(
    monkeypatch: pytest.MonkeyPatch,
    controls: RunControls,
    *,
    has_time: bool,
    candidate: np.ndarray | None = None,
) -> tuple[EstimatorRun, list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []

    def fake_rotation(*args: Any, **kwargs: Any) -> ImuLidarRotationArtifact:
        seen.append({"profile": args[1], "options": args[2] if len(args) > 2 else None})
        return _rotation_artifact(kwargs["reference_rotation"])

    monkeypatch.setattr(rotation_module, "run_livox_imu_lidar_rotation", fake_rotation)
    monkeypatch.setattr(
        estimators, "detect_point_time", lambda b, t: ("t", "offset_s") if has_time else None
    )
    truth = np.eye(4)
    truth[:3, :3] = TRUE_ROTATION.T
    ctx = dataclasses.replace(
        _imu_lidar_context(truth if candidate is None else candidate, None), controls=controls
    )
    return estimators.run_imu_lidar(ctx), seen


def test_cloud_without_time_runs_rigid_with_the_larger_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, seen = _run_adapter(monkeypatch, RunControls(imu_lidar_translation=False), has_time=False)
    assert run.deskew == "none"
    assert seen[0]["profile"].point_time_field is None
    assert seen[0]["options"].deskew == "none"
    rotation = [e for e in run.estimates if e.unit == "deg"]
    assert rotation and all(e.floor == RIGID_SCAN_ROTATION_FLOOR_DEG for e in rotation)
    assert any("no per-point time field" in note for note in run.notes)
    assert any("rigid snapshots" in note for note in run.notes)


def test_cloud_with_time_is_never_silently_rigid(monkeypatch: pytest.MonkeyPatch) -> None:
    run, seen = _run_adapter(monkeypatch, RunControls(imu_lidar_translation=False), has_time=True)
    assert run.deskew == "gyro"
    assert seen[0]["profile"].point_time_field == "t"
    assert seen[0]["options"].deskew == "gyro"
    assert all(e.floor is None for e in run.estimates)
    assert not any("rigid" in note for note in run.notes)


def test_forced_rigid_mode_ignores_the_time_field_and_honours_the_floor_option(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controls = RunControls(
        imu_lidar_translation=False, imu_lidar_deskew="none", rigid_scan_rotation_floor_deg=2.5
    )
    run, seen = _run_adapter(monkeypatch, controls, has_time=True)
    assert run.deskew == "none" and seen[0]["profile"].point_time_field is None
    assert seen[0]["options"].deskew == "none"
    assert {e.floor for e in run.estimates if e.unit == "deg"} == {2.5}
    assert any("forced by --imu-lidar-deskew none" in note for note in run.notes)


def test_gyro_mode_without_a_time_field_skips_the_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_time(monkeypatch)
    ctx = _imu_lidar_context(np.eye(4), None)
    ctx = dataclasses.replace(ctx, controls=RunControls(imu_lidar_deskew="gyro"))
    with pytest.raises(CheckSkipError, match="per-point time"):
        estimators.run_imu_lidar(ctx)


def test_rigid_cache_key_differs_from_the_gyro_key() -> None:
    from calibrex.check.cache import EstimatorCache

    cache = EstimatorCache(Path("unused"), code="x")
    gyro = {"options": ImuLidarRunOptions()}
    rigid = {"options": ImuLidarRunOptions(deskew="none")}
    assert cache.key("a" * 64, "imu_lidar_rotation", gyro) != cache.key(
        "a" * 64, "imu_lidar_rotation", rigid
    )


# ----------------------------------------------------------------------- floor


def _estimate(error: float, std: float, floor: float | None = None) -> AxisEstimate:
    return AxisEstimate("yaw", "deg", error, std, True, floor=floor)


def test_axis_floor_raises_the_tolerance_but_never_lowers_it() -> None:
    options = VerdictOptions()  # floor 0.5 deg
    assert judge_axis(_estimate(1.0, 0.05), options).status == "warn"
    rigid = judge_axis(_estimate(1.0, 0.05, RIGID_SCAN_ROTATION_FLOOR_DEG), options)
    assert rigid.status == "pass"
    assert rigid.tolerance == pytest.approx(RIGID_SCAN_ROTATION_FLOOR_DEG)
    assert rigid.tolerance_source == "floor"
    # A bigger sigma tolerance and a bigger global floor still win.
    assert judge_axis(_estimate(0.1, 1.0, 0.2), options).tolerance == pytest.approx(3.0)
    assert judge_axis(
        _estimate(0.1, 0.01, 0.2), VerdictOptions(rotation_floor_deg=2.0)
    ).tolerance == (pytest.approx(2.0))
    # A 1 deg probe is not detectable on rigid scans with the default floor.
    assert rigid.detects_perturbation is False
    # A candidate 4 deg off still fails.
    assert judge_axis(_estimate(4.0, 0.05, RIGID_SCAN_ROTATION_FLOOR_DEG), options).status == "fail"


# ------------------------------------------------------- record, options, CLI


def test_pair_record_and_options_round_trip_with_deskew() -> None:
    options = CheckOptions(
        sigma_k=3.0,
        rotation_floor_deg=0.5,
        translation_floor_m=0.02,
        detection_probe_deg=1.0,
        imu_lidar_deskew="none",
        rigid_scan_rotation_floor_deg=1.5,
    )
    assert CheckOptions.model_validate(options.model_dump(mode="json")) == options
    record = CheckPairRecord(
        pair="imu-lidar", status="skipped", reason_code="not_selected", reason="x"
    )
    assert record.deskew is None
    rigid = record.model_copy(update={"deskew": "none"})
    again = CheckPairRecord.model_validate(rigid.model_dump(mode="json"))
    assert again.deskew == "none"
    with pytest.raises(ValueError):
        CheckPairRecord.model_validate({**rigid.model_dump(mode="json"), "deskew": "maybe"})
    # Artifacts written before the fields existed still validate.
    legacy = options.model_dump(
        mode="json", exclude={"imu_lidar_deskew", "rigid_scan_rotation_floor_deg"}
    )
    assert CheckOptions.model_validate(legacy).imu_lidar_deskew is None


def test_runner_records_deskew_on_the_pair_and_the_cli_flags_reach_the_adapters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bag = write_check_bag(tmp_path / "bag.db3", tf_messages=[default_tf()])
    seen: list[RunControls] = []

    def adapter(ctx: PairContext) -> EstimatorRun:
        seen.append(ctx.controls)
        from tests.unit.test_check_cache import _stub_run

        return dataclasses.replace(_stub_run(), deskew="none")

    monkeypatch.setattr(
        estimators,
        "ESTIMATORS",
        {"imu-lidar": adapter, "lidar-lidar": adapter, "camera-imu": adapter},
    )
    artifact = build_calibration_check(
        bag,
        run=CheckRunOptions(
            base_dir=tmp_path, imu_lidar_deskew="none", rigid_scan_rotation_floor_deg=2.0
        ),
    )
    ran = [p for p in artifact.pairs if p.status in {"pass", "warn", "fail"}]
    assert ran and all(p.deskew == "none" for p in ran)
    assert artifact.options is not None
    assert artifact.options.imu_lidar_deskew == "none"
    assert artifact.options.rigid_scan_rotation_floor_deg == 2.0
    assert all(c.imu_lidar_deskew == "none" for c in seen)

    seen.clear()
    out = tmp_path / "result.json"
    args = ["check", str(bag), "--output", str(out), "--no-cache", "--fail-on", "never"]
    assert main([*args, "--imu-lidar-deskew", "none", "--rigid-scan-rotation-floor-deg", "3"]) == 0
    assert seen and all(
        c.imu_lidar_deskew == "none" and c.rigid_scan_rotation_floor_deg == 3.0 for c in seen
    )
    loaded = CalibrationCheckArtifact.model_validate_json(out.read_text(encoding="utf-8"))
    assert loaded.options is not None and loaded.options.rigid_scan_rotation_floor_deg == 3.0

    seen.clear()
    assert main([*args]) == 0
    assert all(
        c.imu_lidar_deskew == "auto"
        and c.rigid_scan_rotation_floor_deg == RIGID_SCAN_ROTATION_FLOOR_DEG
        for c in seen
    )


def test_rigid_mode_relaxes_only_the_rotation_observability_threshold() -> None:
    from calibrex.check.estimators import (
        RIGID_SCAN_OBSERVABLE_ROTATION_STD_DEG,
        imu_lidar_run_options,
    )

    assert imu_lidar_run_options("gyro") == ImuLidarRunOptions()  # default path untouched
    rigid = imu_lidar_run_options("none")
    default = ImuLidarRunOptions()
    assert rigid.deskew == "none"
    assert rigid.rotation_control_deg == RIGID_SCAN_ROTATION_FLOOR_DEG
    assert rigid.solver.observable_rotation_std_deg == RIGID_SCAN_OBSERVABLE_ROTATION_STD_DEG
    assert pytest.approx(
        default.solver.observable_rotation_std_deg * RIGID_SCAN_ROTATION_FLOOR_DEG / 0.5
    ) == RIGID_SCAN_OBSERVABLE_ROTATION_STD_DEG
    same = dataclasses.replace(
        rigid,
        deskew="gyro",
        solver=default.solver,
        rotation_control_deg=default.rotation_control_deg,
    )
    assert same == default  # nothing else differs
