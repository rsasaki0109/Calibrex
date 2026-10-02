"""lidar-lidar and gnss-lidar on clouds without per-point time: mode selection, pins, floors."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tests.unit.test_check_cache import BAG_SHA  # noqa: F401  (shared fixture constant)

from calibrex.check import estimators
from calibrex.check.cache import EstimatorCache
from calibrex.check.estimators import (
    GNSS_LIDAR_RIGID_TRANSLATION_FLOOR_M,
    LIDAR_LIDAR_RIGID_ROTATION_FLOOR_DEG,
    LIDAR_LIDAR_RIGID_TRANSLATION_FLOOR_M,
    CheckSkipError,
    PairContext,
    RunControls,
    gnss_lidar_run_options,
    lidar_lidar_run_options,
    select_odometry_deskew,
)
from calibrex.check.runner import CheckRunOptions, build_calibration_check
from calibrex.check.verdict import AxisEstimate, VerdictOptions, judge_axis
from calibrex.cli.main import main
from calibrex.core.calibration_check import (
    CalibrationCheckArtifact,
    CheckOptions,
    CheckPairRecord,
    CheckTransform,
)
from calibrex.core.lidar_lidar_extrinsic import LidarLidarExtrinsicArtifact
from calibrex.evaluation import lidar_lidar_map as lidar_module
from calibrex.evaluation.gnss_lidar_lever_arm import (
    RIGID_SCAN_LIMITATIONS,
    GnssLidarRunOptions,
    collect_windows,
)
from calibrex.evaluation.lidar_lidar_map import LidarLidarMapOptions, collect_map_samples
from calibrex.solvers.gnss_lever_arm_solver import GnssTrackModel

POINTCLOUD2 = "sensor_msgs/msg/PointCloud2"


# ------------------------------------------------------------------ selection


def test_auto_uses_per_point_time_and_rigid_scans_only_without_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bag = Path("unused")
    monkeypatch.setattr(estimators, "detect_point_time", lambda b, t: ("t", "offset_s"))
    assert select_odometry_deskew(bag, "/l", "auto", "--x") == (
        "constant_velocity",
        ("t", "offset_s"),
    )
    assert select_odometry_deskew(bag, "/l", "constant_velocity", "--x")[0] == "constant_velocity"
    assert select_odometry_deskew(bag, "/l", "none", "--x") == ("none", None)

    calls: list[str] = []

    def absent(b: Path, t: str) -> None:
        calls.append(t)

    monkeypatch.setattr(estimators, "detect_point_time", absent)
    assert select_odometry_deskew(bag, "/l", "auto", "--x") == ("none", None)
    assert calls == ["/l"]
    with pytest.raises(CheckSkipError) as skipped:
        select_odometry_deskew(bag, "/l", "constant_velocity", "--lidar-lidar-deskew")
    assert skipped.value.code == "unsupported_sensor"
    assert "--lidar-lidar-deskew none" in skipped.value.reason


# ------------------------------------------------ lidar-lidar: data path / pins


def _stream(count: int = 4, with_time: bool = True) -> list[tuple[float, Any, Any]]:
    rng = np.random.default_rng(0)
    return [
        (
            float(index) * 0.1,
            rng.normal(0.0, 5.0, (50, 3)),
            np.linspace(0.0, 0.1, 50) if with_time else None,
        )
        for index in range(count)
    ]


class _Recorder:
    def __init__(self) -> None:
        self.offsets: list[Any] = []
        self.poses: list[np.ndarray] = [np.eye(4)]

    def add(self, points: Any, time_s: Any, offsets: Any = None) -> None:
        self.offsets.append(offsets)
        self.poses.append(np.eye(4))


def _collect(monkeypatch: pytest.MonkeyPatch, options: LidarLidarMapOptions) -> tuple[Any, Any]:
    recorder = _Recorder()
    profiles: list[Any] = []

    def fake_points(bag: Any, profile: Any, *, max_seconds: Any = None) -> Any:
        profiles.append(profile)
        return iter(_stream(with_time=profile.point_time_field is not None))

    monkeypatch.setattr(lidar_module, "iter_livox_points", fake_points)
    monkeypatch.setattr(lidar_module, "IncrementalScanOdometry", lambda *a, **k: recorder)
    monkeypatch.setattr(lidar_module, "preprocess_scan", lambda points, opts: np.asarray(points))
    collect_map_samples(
        "bag",
        reference_topic="/a",
        target_topic="/b",
        point_time_field="t",
        options=options,
    )
    return recorder, profiles


def test_default_path_still_deskews_with_the_per_point_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert LidarLidarMapOptions().deskew == "constant_velocity"
    assert lidar_lidar_run_options("constant_velocity") == LidarLidarMapOptions()
    recorder, profiles = _collect(monkeypatch, LidarLidarMapOptions())
    assert all(item is not None and len(item) == 50 for item in recorder.offsets)
    assert [p.point_time_field for p in profiles] == ["t", "t"]


def test_rigid_mode_hands_the_odometry_no_point_time_and_reads_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder, profiles = _collect(monkeypatch, LidarLidarMapOptions(deskew="none"))
    assert recorder.offsets and all(item is None for item in recorder.offsets)
    assert [p.point_time_field for p in profiles] == [None, None]


def test_rigid_options_change_nothing_but_the_mode() -> None:
    assert lidar_lidar_run_options("none") == dataclasses.replace(
        LidarLidarMapOptions(), deskew="none"
    )


def test_rigid_cache_key_and_artifact_options_differ_from_the_default() -> None:
    cache = EstimatorCache(Path("unused"), code="x")
    gnss_default = {"options": gnss_lidar_run_options("constant_velocity")}
    gnss_rigid = {"options": gnss_lidar_run_options("none")}
    assert cache.key("a" * 64, "gnss_lidar_lever_arm", gnss_default) != cache.key(
        "a" * 64, "gnss_lidar_lever_arm", gnss_rigid
    )
    assert gnss_lidar_run_options("constant_velocity") == GnssLidarRunOptions()
    assert gnss_lidar_run_options("none").deskew == "none"
    assert RIGID_SCAN_LIMITATIONS


# ------------------------------------------------------------ lidar-lidar adapter


def _lidar_lidar_artifact(initial: np.ndarray) -> LidarLidarExtrinsicArtifact:
    from tests.unit.test_lidar_lidar_map import TRUE_X, samples

    from calibrex.evaluation.lidar_lidar_map import evaluate_lidar_lidar_map
    from calibrex.solvers.lidar_lidar_map_solver import perturb_extrinsic

    data = samples(np.random.default_rng(0))
    start = perturb_extrinsic(TRUE_X, "yaw", np.radians(1.0))
    evaluation = evaluate_lidar_lidar_map(
        data, start, LidarLidarMapOptions(), reference_transform=initial
    )
    from calibrex.core.lidar_lidar_extrinsic import LidarLidarProvenance, LidarLidarSamples

    assert evaluation.result.transform is not None
    from scipy.spatial.transform import Rotation

    from calibrex.core.lidar_lidar_extrinsic import LidarLidarTransform

    transform = LidarLidarTransform(
        parent_frame="a",
        child_frame="b",
        translation_m=[float(v) for v in evaluation.result.transform[:3, 3]],
        rotation_quat_xyzw=[
            float(v) for v in Rotation.from_matrix(evaluation.result.transform[:3, :3]).as_quat()
        ],
    )
    return LidarLidarExtrinsicArtifact(
        solver_status=evaluation.result.status,
        policy_status=evaluation.policy_status,
        policy_reasons=list(evaluation.policy_reasons),
        calibrated_dofs=[r.name for r in evaluation.records if r.status == "estimated"],
        transform=transform,
        dofs=list(evaluation.records),
        residual_sigma_m=evaluation.result.sigma_m,
        holdout_median_residual_m=evaluation.holdout_median_m,
        samples=LidarLidarSamples(
            reference_scans=10,
            samples=len(data),
            train_samples=evaluation.train_samples,
            holdout_samples=evaluation.holdout_samples,
            train_correspondences=evaluation.result.correspondences,
            holdout_correspondences=evaluation.holdout_correspondences,
        ),
        train_blocks=list(evaluation.train_blocks),
        holdout_blocks=list(evaluation.holdout_blocks),
        jackknife_fits=evaluation.jackknife_fits,
        options={},
        reference="stub",
        limitations=["stub"],
        provenance=LidarLidarProvenance(
            generator="pytest",
            generator_version="1",
            command=[],
            dataset_family="synthetic",
            sequence_ids=["s"],
            reference_topic="/a",
            target_topic="/b",
            input_sha256="b" * 64,
            input_digest_scope="synthetic",
            dataset_license="synthetic",
        ),
    )


def _lidar_lidar_context(controls: RunControls) -> PairContext:
    candidate = np.eye(4)
    return PairContext(
        bag=Path("unused-bag"),
        pair=CheckPairRecord(
            pair="lidar-lidar",
            sensors=["a", "b"],
            frames=["a", "b"],
            status="planned",
            candidate_transform=CheckTransform(
                parent_frame="a",
                child_frame="b",
                translation_m=[0.0, 0.0, 0.0],
                rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            ),
        ),
        topic_types={"/a": POINTCLOUD2, "/b": POINTCLOUD2},
        sensor_topics={"first": ("/a",), "second": ("/b",)},
        candidate=candidate,
        sources=[],
        controls=controls,
    )


def _run_lidar_lidar(
    monkeypatch: pytest.MonkeyPatch, controls: RunControls, times: dict[str, bool]
) -> tuple[Any, list[dict[str, Any]]]:
    seen: list[dict[str, Any]] = []

    def fake_run(*args: Any, **kwargs: Any) -> LidarLidarExtrinsicArtifact:
        seen.append(kwargs)
        return _lidar_lidar_artifact(kwargs["initial"])

    monkeypatch.setattr(lidar_module, "run_ros2_lidar_lidar_map", fake_run)
    monkeypatch.setattr(
        estimators,
        "detect_point_time",
        lambda b, topic: ("t", "offset_s") if times[topic] else None,
    )
    return estimators.run_lidar_lidar(_lidar_lidar_context(controls)), seen


def test_reference_with_time_is_deskewed_and_a_target_without_time_is_fine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, seen = _run_lidar_lidar(monkeypatch, RunControls(), {"/a": True, "/b": False})
    assert run.deskew == "constant_velocity"
    assert seen[0]["point_time_field"] == "t"
    assert seen[0]["target_point_time_field"] is None  # the target is never deskewed
    assert seen[0]["options"] == LidarLidarMapOptions()
    assert all(e.floor is None for e in run.estimates)
    assert not any("rigid" in note for note in run.notes)


def test_reference_without_time_runs_rigid_with_the_rigid_floors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run, seen = _run_lidar_lidar(monkeypatch, RunControls(), {"/a": False, "/b": True})
    assert run.deskew == "none"
    assert seen[0]["point_time_field"] is None
    assert seen[0]["options"].deskew == "none"
    floors = {e.unit: e.floor for e in run.estimates}
    assert floors == {
        "deg": LIDAR_LIDAR_RIGID_ROTATION_FLOOR_DEG,
        "m": LIDAR_LIDAR_RIGID_TRANSLATION_FLOOR_M,
    }
    assert any("no per-point time field" in note for note in run.notes)


def test_forced_rigid_mode_ignores_an_existing_time_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controls = RunControls(lidar_lidar_deskew="none")
    run, seen = _run_lidar_lidar(monkeypatch, controls, {"/a": True, "/b": True})
    assert run.deskew == "none" and seen[0]["point_time_field"] is None
    assert any("forced by --lidar-lidar-deskew none" in note for note in run.notes)


def test_constant_velocity_without_a_time_field_skips_the_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(estimators, "detect_point_time", lambda b, t: None)
    ctx = _lidar_lidar_context(RunControls(lidar_lidar_deskew="constant_velocity"))
    with pytest.raises(CheckSkipError, match="--lidar-lidar-deskew none"):
        estimators.run_lidar_lidar(ctx)


# ----------------------------------------------------------- gnss-lidar windows


def test_gnss_windows_drop_the_point_time_only_in_rigid_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from calibrex.evaluation import gnss_lidar_lever_arm as module

    seen: list[list[Any]] = []

    def fake_collect(scans: Any, options: Any, **kwargs: Any) -> str:
        seen.append([item[2] if len(item) > 2 else "absent" for item in scans])
        return "segmenter"

    monkeypatch.setattr(module, "collect_odometry_windows", fake_collect)
    track = GnssTrackModel(np.arange(3.0), np.zeros((3, 3)), np.full(3, 0.02))
    points = np.zeros((4, 3))
    offsets = np.linspace(0.0, 0.1, 4)
    scans = [(0.0, points, offsets), (0.1, points, offsets), (0.2, points)]
    collect_windows(track, scans, GnssLidarRunOptions())
    assert seen[0][0] is offsets and seen[0][2] == "absent"  # untouched default
    collect_windows(track, scans, GnssLidarRunOptions(deskew="none"))
    assert seen[1] == [None, None, None]


# ----------------------------------------------------- judging with the floors


def test_rigid_translation_floor_raises_the_tolerance_but_never_lowers_it() -> None:
    options = VerdictOptions()
    plain = AxisEstimate("y", "m", 0.05, 0.005, True)
    rigid = dataclasses.replace(plain, floor=GNSS_LIDAR_RIGID_TRANSLATION_FLOOR_M)
    assert judge_axis(plain, options).status == "fail"
    judged = judge_axis(rigid, options)
    assert judged.status == "pass" and judged.tolerance == pytest.approx(
        GNSS_LIDAR_RIGID_TRANSLATION_FLOOR_M
    )
    assert judged.tolerance_source == "floor"
    assert judge_axis(dataclasses.replace(rigid, floor=0.001), options).tolerance == pytest.approx(
        0.02
    )


# ------------------------------------------------------- record, options, CLI


def test_options_and_pair_record_round_trip_with_the_new_fields() -> None:
    options = CheckOptions(
        sigma_k=3.0,
        rotation_floor_deg=0.5,
        translation_floor_m=0.02,
        detection_probe_deg=1.0,
        lidar_lidar_deskew="none",
        gnss_lidar_deskew="constant_velocity",
    )
    assert CheckOptions.model_validate(options.model_dump(mode="json")) == options
    legacy = options.model_dump(mode="json", exclude={"lidar_lidar_deskew", "gnss_lidar_deskew"})
    assert CheckOptions.model_validate(legacy).lidar_lidar_deskew is None
    record = CheckPairRecord(
        pair="lidar-lidar", status="skipped", reason_code="not_selected", reason="x"
    )
    for mode in ("gyro", "constant_velocity", "none"):
        updated = record.model_copy(update={"deskew": mode})
        assert CheckPairRecord.model_validate(updated.model_dump(mode="json")).deskew == mode
    with pytest.raises(ValueError):
        CheckPairRecord.model_validate({**record.model_dump(mode="json"), "deskew": "maybe"})


def test_cli_flags_reach_the_adapters_and_the_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tests.unit.check_fixtures import default_tf, write_check_bag
    from tests.unit.test_check_cache import _stub_run

    bag = write_check_bag(tmp_path / "bag.db3", tf_messages=[default_tf()])
    seen: list[RunControls] = []

    def adapter(ctx: PairContext) -> Any:
        seen.append(ctx.controls)
        return dataclasses.replace(_stub_run(), deskew="none")

    monkeypatch.setattr(estimators, "ESTIMATORS", {"lidar-lidar": adapter, "gnss-lidar": adapter})
    artifact = build_calibration_check(
        bag, run=CheckRunOptions(base_dir=tmp_path, lidar_lidar_deskew="none")
    )
    assert artifact.options is not None
    assert artifact.options.lidar_lidar_deskew == "none"
    assert artifact.options.gnss_lidar_deskew == "auto"

    seen.clear()
    out = tmp_path / "result.json"
    args = ["check", str(bag), "--output", str(out), "--no-cache", "--fail-on", "never"]
    assert main([*args, "--lidar-lidar-deskew", "none", "--gnss-lidar-deskew", "none"]) == 0
    assert all(c.lidar_lidar_deskew == "none" and c.gnss_lidar_deskew == "none" for c in seen)
    loaded = CalibrationCheckArtifact.model_validate_json(out.read_text(encoding="utf-8"))
    assert loaded.options is not None and loaded.options.gnss_lidar_deskew == "none"

    seen.clear()
    assert main([*args, "--imu-lidar-deskew", "none"]) == 0  # #99's flag still works
    assert all(
        c.imu_lidar_deskew == "none"
        and c.lidar_lidar_deskew == "auto"
        and c.gnss_lidar_deskew == "auto"
        for c in seen
    )
