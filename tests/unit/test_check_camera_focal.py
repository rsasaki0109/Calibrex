"""camera-focal in ``calibrex check``: the focal scale of the deployed intrinsics.

The tracker and the bag are stubbed (the focal artifact is the committed Hilti exp21 cam1
evidence); the planner, the runner, the dependency on this run's camera-imu result, the
cache, the verdict rule, the schema and the HTML report run for real.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tests.unit.check_fixtures import IDENTITY, IMAGE_TYPE, IMU_TYPE, Transform, write_check_bag

from calibrex.check import estimators
from calibrex.check.cache import EstimatorCache
from calibrex.check.estimators import (
    CheckSkipError,
    EstimatorRun,
    EvidenceArtifact,
    PairContext,
    RunControls,
)
from calibrex.check.runner import CheckRunOptions, build_calibration_check, format_check_table
from calibrex.check.verdict import (
    AxisEstimate,
    FocalEstimate,
    FocalRecord,
    VerdictOptions,
    judge_focal,
)
from calibrex.cli.main import main
from calibrex.core.calibration_check import (
    CalibrationCheckArtifact,
    CheckPairRecord,
)
from calibrex.core.camera_focal_scale import CameraFocalScaleArtifact, load_camera_focal_scale
from calibrex.core.imu_lidar_rotation import load_imu_lidar_rotation
from calibrex.core.validation import validate_file
from calibrex.evaluation.visual_rotation import CameraModel
from calibrex.visualization.check_report import render_check_html

ASSETS = Path(__file__).resolve().parents[2] / "docs/assets/hilti2022_camera_imu"
ROTATION_ASSET = ASSETS / "cam1_exp21.yaml"
FOCAL_ASSET = ASSETS / "focal_cam1_exp21.yaml"  # ratios 0.994 / 1.002 / 1.004, std ~0.004
CAMERA_FRAME = "camera_optical"
IMU_FRAME = "imu_link"
BAG_SHA = "a" * 64


def _focal(*, scale: float, std: float, name: str = "fx") -> FocalEstimate:
    return FocalEstimate(
        name=name,  # type: ignore[arg-type]
        candidate_px=350.0,
        estimate_px=350.0 * scale,
        estimate_std_px=350.0 * std,
        scale=scale,
        scale_std=std,
        estimated=True,
    )


# ---------------------------------------------------------------------- the verdict rule


@pytest.mark.parametrize(
    ("scale", "status"),
    [
        (1.0, "pass"),
        (1.0049, "pass"),  # inside the 0.5 % floor
        (0.9951, "pass"),
        (1.0051, "warn"),  # just outside the floor
        (0.9901, "warn"),
        (1.0099, "warn"),  # up to twice the tolerance
        (1.0101, "fail"),
        (0.97, "fail"),
    ],
)
def test_floor_boundaries_with_a_precise_estimate(scale: float, status: str) -> None:
    judged = judge_focal(FocalRecord((_focal(scale=scale, std=0.0005),)), VerdictOptions())

    (item,) = judged.record.components
    assert item.tolerance == pytest.approx(0.005) and item.tolerance_source == "floor"
    assert (item.status, judged.verdict) == (status, status)
    assert item.detectable_error == pytest.approx(0.005 + abs(scale - 1.0))
    assert judged.coverage == "full"


def test_sigma_tolerance_dominates_when_the_estimate_is_noisy() -> None:
    # std 0.7 % -> tolerance 3 * 0.7 = 2.1 %: a 1.5 % scale error passes, 3 % warns, 5 % fails
    options = VerdictOptions()
    statuses = [
        judge_focal(FocalRecord((_focal(scale=s, std=0.007),)), options).verdict
        for s in (0.985, 0.97, 0.95)
    ]
    assert statuses == ["pass", "warn", "fail"]
    item = judge_focal(FocalRecord((_focal(scale=1.0, std=0.007),)), options).record.components[0]
    assert item.tolerance == pytest.approx(0.021) and item.tolerance_source == "sigma"
    # sigma_k and the floor are options
    wide = VerdictOptions(sigma_k=1.0, focal_scale_floor=0.03)
    assert judge_focal(FocalRecord((_focal(scale=0.975, std=0.007),)), wide).verdict == "pass"


def test_detectable_error_and_probe() -> None:
    sharp = judge_focal(FocalRecord((_focal(scale=1.001, std=0.001),)), VerdictOptions())
    assert sharp.record.components[0].detects_perturbation is True  # 0.6 % < the 1 % probe
    blunt = judge_focal(FocalRecord((_focal(scale=1.002, std=0.006),)), VerdictOptions())
    assert blunt.record.components[0].detectable_error == pytest.approx(0.018 + 0.002)
    assert blunt.record.components[0].detects_perturbation is False


def test_pair_verdict_is_the_worst_component_and_unchecked_is_partial() -> None:
    fy = dataclasses.replace(
        _focal(scale=0.9, std=0.002, name="fy"),
        estimated=False,
        unchecked_reason="the x-axis ratio std exceeds the bound",
        unchecked_code="unobservable",
    )
    judged = judge_focal(FocalRecord((_focal(scale=1.0, std=0.002), fy)), VerdictOptions())
    assert judged.verdict == "pass" and judged.coverage == "partial"
    assert [item.name for item in judged.record.unchecked] == ["fy"]
    both = judge_focal(
        FocalRecord((_focal(scale=1.0, std=0.002), _focal(scale=0.9, std=0.002, name="fy"))),
        VerdictOptions(),
    )
    assert both.verdict == "fail"

    nothing = judge_focal(FocalRecord((fy,)), VerdictOptions())
    assert nothing.verdict == "inconclusive" and not nothing.record.components


def test_verdict_options_reject_non_positive_focal_settings() -> None:
    with pytest.raises(ValueError):
        VerdictOptions(focal_scale_floor=0.0)


# ------------------------------------------------------------- the artifact -> record step


def test_focal_record_maps_ratios_to_focal_lengths_and_gates_unobservable_axes() -> None:
    artifact = load_camera_focal_scale(FOCAL_ASSET)

    record = estimators.focal_record_of(artifact, "camchain cam1")

    fx, fy = record.components
    # a rotation about camera y reads as fx_true / fx_used, about x as fy_true / fy_used
    assert (fx.name, fx.scale, fx.scale_std) == ("fx", artifact.rate_ratio[1], 0.004055778803364455)
    assert (fy.name, fy.scale) == ("fy", artifact.rate_ratio[0])
    assert fx.estimate_px == artifact.fx_estimate_px and fy.candidate_px == artifact.fy_used_px
    assert fx.estimated and fy.estimated
    assert record.optical_axis_ratio == artifact.rate_ratio[2]

    noisy = artifact.model_copy(update={"rate_ratio_std": [0.01, 0.004, 0.003]})
    loose_fx, loose_fy = estimators.focal_record_of(noisy).components
    assert (
        loose_fx.estimated and not loose_fy.estimated and loose_fy.unchecked_code == "unobservable"
    )

    off = artifact.model_copy(update={"policy_status": "warn"})
    assert all(
        not item.estimated and item.unchecked_code == "control_not_detected"
        for item in estimators.focal_record_of(off).components
    )


# ------------------------------------------------------------------------ the dependency


def _pair() -> CheckPairRecord:
    return CheckPairRecord(
        pair="camera-focal",
        sensors=[CAMERA_FRAME, IMU_FRAME],
        topics=["/camera/image_raw", "/imu"],
        frames=[CAMERA_FRAME, IMU_FRAME],
        status="planned",
        candidate_transform={  # type: ignore[arg-type]
            "parent_frame": CAMERA_FRAME,
            "child_frame": IMU_FRAME,
            "translation_m": [0.0, 0.0, 0.0],
            "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    )


def _camera_imu_run(*, std: float = 0.1, solved: bool = True, policy: str = "pass") -> EstimatorRun:
    rotation = load_imu_lidar_rotation(ROTATION_ASSET)
    return EstimatorRun(
        estimator="stub/camera-imu",
        artifacts=(EvidenceArtifact("rotation", rotation, rotation.policy_status),),
        solved=solved,
        policy_status=policy,
        policy_reasons=("stub",),
        estimates=tuple(
            AxisEstimate(name, "deg", 0.0, std, True)  # type: ignore[arg-type]
            for name in ("roll", "pitch", "yaw")
        ),
        compared=np.eye(4),
    )


def _context(
    memo: dict[tuple[object, ...], Any],
    *,
    cache: EstimatorCache | None = None,
    max_duration_s: float | None = None,
) -> PairContext:
    return PairContext(
        bag=Path("bag"),
        pair=_pair(),
        topic_types={"/camera/image_raw": IMAGE_TYPE, "/imu": IMU_TYPE},
        sensor_topics={"camera": ("/camera/image_raw",), "imu": ("/imu",)},
        candidate=np.eye(4),
        sources=[],
        controls=RunControls(
            memo=memo, cache=cache, bag_sha256=BAG_SHA, max_duration_s=max_duration_s
        ),
    )


def _memo(run: EstimatorRun | None) -> dict[tuple[object, ...], Any]:
    if run is None:
        return {}
    return {(estimators.PAIR_RUN_KEY, "camera-imu", (CAMERA_FRAME, IMU_FRAME)): run}


class Tracker:
    """Replaces the feature tracking and regression: returns the committed artifact."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, scale: float = 1.0) -> None:
        self.calls: list[dict[str, Any]] = []
        self.scale = scale
        import calibrex.evaluation.camera_focal as module

        monkeypatch.setattr(module, "run_camera_focal_check", self)
        monkeypatch.setattr(
            estimators,
            "resolve_camera_intrinsics",
            lambda ctx, topic, frame: (
                CameraModel(352.65, 352.86, 360.0, 250.0, "none", ()),
                "stub camchain cam1",
                "cam1",
            ),
        )

    def __call__(self, bag: str, **kwargs: Any) -> CameraFocalScaleArtifact:
        self.calls.append({"bag": bag, **kwargs})
        base = load_camera_focal_scale(FOCAL_ASSET)
        ratio = [r * self.scale for r in base.rate_ratio[:2]] + [base.rate_ratio[2]]
        return base.model_copy(
            update={
                "rate_ratio": ratio,
                "provenance": base.provenance.model_copy(
                    update={"rotation_artifact_sha256": kwargs["rotation_artifact_sha256"]}
                ),
            }
        )


@pytest.fixture
def tracker(monkeypatch: pytest.MonkeyPatch) -> Tracker:
    return Tracker(monkeypatch)


@pytest.mark.parametrize(
    ("memo", "needle"),
    [
        (_memo(None), "gave no result"),
        (_memo(_camera_imu_run(solved=False)), "no rotation estimate"),
        (_memo(_camera_imu_run(policy="fail")), "failed its own held-out check"),
        (_memo(_camera_imu_run(std=0.8)), "exceeds 0.5 deg"),
    ],
)
def test_missing_or_unusable_camera_imu_skips_with_missing_dependency(
    tracker: Tracker, memo: dict[tuple[object, ...], Any], needle: str
) -> None:
    with pytest.raises(CheckSkipError) as skipped:
        estimators.run_camera_focal(_context(memo))

    assert skipped.value.code == "missing_dependency"
    assert needle in skipped.value.reason
    assert not tracker.calls


def test_the_estimator_gets_camera_imus_estimate_not_the_candidate(tracker: Tracker) -> None:
    run = estimators.run_camera_focal(_context(_memo(_camera_imu_run(std=0.3)), max_duration_s=60))

    (call,) = tracker.calls
    rotation = load_imu_lidar_rotation(ROTATION_ASSET)
    assert call["rotation_artifact"].rotation_quat_xyzw == rotation.rotation_quat_xyzw
    assert call["max_seconds"] == 60 and call["camera"].fx == 352.65
    assert run.focal is not None and run.estimates == ()
    assert run.artifacts[0].role == "focal_scale"
    assert any("camera-imu's estimate" in note for note in run.notes)


# ---------------------------------------------------------------------------- the cache


def test_cache_key_has_camera_model_and_camera_imu_estimate_not_the_candidate(
    tracker: Tracker, tmp_path: Path
) -> None:
    cache = EstimatorCache(tmp_path, code="v1")
    first = estimators.run_camera_focal(_context(_memo(_camera_imu_run()), cache=cache))
    assert first.evidence_from_cache is False and len(tracker.calls) == 1

    # another candidate extrinsic (and verdict thresholds) reuse the artifact ...
    other = _context(_memo(_camera_imu_run(std=0.2)), cache=cache)
    other = dataclasses.replace(other, candidate=np.diag([1.0, -1.0, -1.0, 1.0]))
    again = estimators.run_camera_focal(other)
    assert again.evidence_from_cache is True and len(tracker.calls) == 1
    assert again.focal == first.focal

    # ... the camera-imu estimate (rotation, offset, bias) changes the key ...
    rotation = load_imu_lidar_rotation(ROTATION_ASSET)
    moved = dataclasses.replace(
        _camera_imu_run().artifacts[0],
        artifact=rotation.model_copy(update={"time_offset_s": rotation.time_offset_s + 0.01}),
    )
    run = dataclasses.replace(_camera_imu_run(), artifacts=(moved,))
    estimators.run_camera_focal(_context(_memo(run), cache=cache))
    assert len(tracker.calls) == 2

    # ... and so does the deployed focal length that normalizes the features, and the duration.
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(
            estimators,
            "resolve_camera_intrinsics",
            lambda ctx, topic, frame: (
                CameraModel(1.01 * 352.65, 352.86, 360.0, 250.0, "none", ()),
                "stub",
                "cam1",
            ),
        )
        estimators.run_camera_focal(_context(_memo(_camera_imu_run()), cache=cache))
    finally:
        monkey.undo()
    assert len(tracker.calls) == 3
    estimators.run_camera_focal(_context(_memo(_camera_imu_run()), cache=cache, max_duration_s=30))
    assert len(tracker.calls) == 4


def test_a_hit_points_the_provenance_at_this_runs_camera_imu_evidence(
    tracker: Tracker, tmp_path: Path
) -> None:
    cache = EstimatorCache(tmp_path, code="v1")
    estimators.run_camera_focal(_context(_memo(_camera_imu_run()), cache=cache))
    # a different candidate reference makes the saved camera-imu artifact (and its digest) differ
    rotation = load_imu_lidar_rotation(ROTATION_ASSET)
    variant = rotation.model_copy(update={"policy_reasons": [*rotation.policy_reasons, "other"]})
    item = dataclasses.replace(_camera_imu_run().artifacts[0], artifact=variant)
    hit = estimators.run_camera_focal(
        _context(_memo(dataclasses.replace(_camera_imu_run(), artifacts=(item,))), cache=cache)
    )

    assert hit.evidence_from_cache is True
    digest = estimators._rotation_digest(variant)
    assert hit.artifacts[0].artifact.provenance.rotation_artifact_sha256 == digest  # type: ignore[attr-defined]


# ----------------------------------------------------- the check run, schema, report, CLI


def _tf() -> list[Transform]:
    return [
        ("base_link", IMU_FRAME, (0.0, 0.0, 0.1), IDENTITY),
        ("base_link", CAMERA_FRAME, (0.5, 0.0, 0.8), IDENTITY),
    ]


@pytest.fixture
def bag(tmp_path: Path) -> Path:
    return write_check_bag(
        tmp_path / "bag.db3",
        tf_messages=[_tf()],
        sensor_frames={
            "/imu": (IMU_TYPE, IMU_FRAME),
            "/camera/image_raw": (IMAGE_TYPE, CAMERA_FRAME),
        },
    )


@pytest.fixture
def stubbed_run(monkeypatch: pytest.MonkeyPatch, tracker: Tracker) -> Tracker:
    camera_imu = _camera_imu_run()
    monkeypatch.setitem(estimators.ESTIMATORS, "camera-imu", lambda ctx: camera_imu)
    return tracker


def _check(tmp_path: Path, bag: Path, **kwargs: Any) -> CalibrationCheckArtifact:
    options = {
        "pairs": ("camera-imu", "camera-focal"),
        "evidence_dir": tmp_path / "out" / "ev",
        "base_dir": tmp_path / "out",
        **kwargs,
    }
    return build_calibration_check(bag, run=CheckRunOptions(**options))


def _pair_of(artifact: CalibrationCheckArtifact, name: str) -> CheckPairRecord:
    (record,) = [p for p in artifact.pairs if p.pair == name]
    return record


def test_check_judges_the_focal_scale_with_its_own_record(
    tmp_path: Path, bag: Path, stubbed_run: Tracker
) -> None:
    artifact = _check(tmp_path, bag)

    focal = _pair_of(artifact, "camera-focal")
    assert focal.status == "pass", focal.reason
    assert focal.axes == [] and focal.unchecked_axes == []  # not an extrinsic: no axes
    assert focal.compared_transform is None and focal.time_offset is None
    assert focal.focal_scale is not None
    assert [c.name for c in focal.focal_scale.components] == ["fx", "fy"]
    fx = focal.focal_scale.components[0]
    assert fx.scale == pytest.approx(1.0020041589867277)
    assert fx.tolerance == pytest.approx(3 * 0.004055778803364455)
    assert focal.focal_scale.floor == 0.005 and focal.coverage == "full"
    assert focal.focal_scale.intrinsics_source == "stub camchain cam1"
    assert focal.evidence[0].role == "focal_scale"
    assert focal.evidence[0].schema_version == "slac.camera_focal_scale/v0.1"
    assert artifact.options is not None and artifact.options.focal_scale_floor == 0.005
    # the focal pair is not an edge of the closure graph
    assert artifact.closures is None or all(
        e.pair != "camera-focal" for e in artifact.closures.edges
    )
    # the evidence file validates and records the digest of this run's camera-imu evidence
    evidence = tmp_path / "out" / focal.evidence[0].path
    assert validate_file(evidence).kind == "camera-focal-scale"
    rotation_ref = _pair_of(artifact, "camera-imu").evidence[0]
    assert (
        load_camera_focal_scale(evidence).provenance.rotation_artifact_sha256 == rotation_ref.sha256
    )


def test_a_biased_deployed_focal_is_flagged(
    tmp_path: Path, bag: Path, monkeypatch: pytest.MonkeyPatch, stubbed_run: Tracker
) -> None:
    stubbed_run.scale = 0.97  # the gyro says the deployed focal is 3 % too long
    artifact = _check(tmp_path, bag)

    focal = _pair_of(artifact, "camera-focal")
    assert (focal.status, focal.verdict) == ("fail", "fail")
    assert artifact.overall_verdict == "fail"
    assert all(c.status == "fail" for c in focal.focal_scale.components)  # type: ignore[union-attr]
    # the floor is an option
    lax = _check(tmp_path, bag, verdict=VerdictOptions(focal_scale_floor=0.05))
    assert _pair_of(lax, "camera-focal").status == "pass"
    assert lax.options is not None and lax.options.focal_scale_floor == 0.05


def test_missing_camera_imu_skips_focal_in_a_check_run(
    tmp_path: Path, bag: Path, tracker: Tracker, monkeypatch: pytest.MonkeyPatch
) -> None:
    only = _check(tmp_path, bag, pairs=("camera-focal",))
    record = _pair_of(only, "camera-focal")
    assert (record.status, record.reason_code) == ("skipped", "missing_dependency")
    assert "not selected with --pairs" in (record.reason or "")

    def no_intrinsics(ctx: PairContext) -> EstimatorRun:
        raise CheckSkipError("missing_intrinsics", "no intrinsics")

    monkeypatch.setitem(estimators.ESTIMATORS, "camera-imu", no_intrinsics)
    skipped = _check(tmp_path, bag)
    assert _pair_of(skipped, "camera-imu").reason_code == "missing_intrinsics"
    assert _pair_of(skipped, "camera-focal").reason_code == "missing_dependency"
    assert not tracker.calls


def test_camera_filter_applies_to_focal_too(
    tmp_path: Path, bag: Path, stubbed_run: Tracker
) -> None:
    artifact = _check(tmp_path, bag, camera="/other")

    assert _pair_of(artifact, "camera-focal").reason_code == "not_selected"


def test_estimator_failed_policy_is_inconclusive(
    tmp_path: Path, bag: Path, monkeypatch: pytest.MonkeyPatch, stubbed_run: Tracker
) -> None:
    base = load_camera_focal_scale(FOCAL_ASSET)
    failing = base.model_copy(update={"policy_status": "fail", "policy_reasons": ["held-out"]})
    monkeypatch.setattr(stubbed_run, "__call__", lambda *a, **k: failing, raising=False)
    import calibrex.evaluation.camera_focal as module

    monkeypatch.setattr(module, "run_camera_focal_check", lambda bag, **kw: failing)
    record = _pair_of(_check(tmp_path, bag), "camera-focal")

    assert (record.status, record.reason_code) == ("inconclusive", "estimator_failed")


def test_schema_round_trip_and_table(tmp_path: Path, bag: Path, stubbed_run: Tracker) -> None:
    artifact = _check(tmp_path, bag)
    payload = artifact.model_dump(mode="json", exclude_none=True)

    assert CalibrationCheckArtifact.model_validate(payload) == artifact
    assert json.loads(json.dumps(payload)) == payload
    schema = CalibrationCheckArtifact.model_json_schema()
    assert "focal_scale" in schema["$defs"]["CheckPairRecord"]["properties"]
    # an artifact without the new optional fields (older runs) still validates
    old = json.loads(json.dumps(payload))
    old["options"].pop("focal_scale_floor")
    for pair in old["pairs"]:
        pair.pop("focal_scale", None)
    CalibrationCheckArtifact.model_validate(old)

    table = format_check_table(artifact)
    assert "fx scale 1.0020" in table and "fy scale 0.9937" in table
    assert "optical-axis control ratio 1.0036" in table
    assert "fx 1.4 %" in table  # detectable error: 3 * 0.41 % + 0.2 %


def test_html_report_shows_the_focal_record(
    tmp_path: Path, bag: Path, stubbed_run: Tracker
) -> None:
    stubbed_run.scale = 0.985
    page = render_check_html(
        _check(tmp_path, bag),
        artifact_dir=tmp_path / "out",
        html_dir=tmp_path / "out" / "html",
    )

    assert "camera-focal" in page and "scale 0.9" in page
    assert "px against" in page and "deployed" in page
    assert "optical-axis control ratio" in page
    assert "slac.camera_focal_scale/v0.1" in page
    assert "axis judgement(s) outside the tolerance" in page or 'class="badge' in page


def test_cli_focal_scale_floor_and_run(
    tmp_path: Path, bag: Path, stubbed_run: Tracker, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "result" / "check.json"
    args = [
        "check",
        str(bag),
        "--pairs",
        "camera-imu,camera-focal",
        "--focal-scale-floor",
        "0.02",
        "--cache-dir",
        str(tmp_path / "cache"),
        "--output",
        str(output),
        "--html",
        str(tmp_path / "result" / "report.html"),
    ]
    assert main(args) == 0
    capsys.readouterr()
    payload = json.loads(output.read_text(encoding="utf-8"))
    focal = next(p for p in payload["pairs"] if p["pair"] == "camera-focal")
    assert payload["options"]["focal_scale_floor"] == 0.02
    assert focal["focal_scale"]["floor"] == 0.02 and focal["status"] == "pass"
    assert focal["evidence_from_cache"] is False
    assert validate_file(output, "calibration-check").kind == "calibration-check"

    assert main(args) == 0  # the second run is served by the estimator cache
    capsys.readouterr()
    again = json.loads(output.read_text(encoding="utf-8"))
    assert next(p for p in again["pairs"] if p["pair"] == "camera-focal")["evidence_from_cache"]
    assert len(stubbed_run.calls) == 1
    with pytest.raises(SystemExit):
        main(["check", str(bag), "--focal-scale-floor", "0"])
