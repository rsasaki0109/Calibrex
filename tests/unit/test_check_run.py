"""``calibrex check`` full run: stubbed estimators on the synthetic db3 fixture."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from tests.unit.check_fixtures import default_tf, write_check_bag

from calibrex.check import estimators
from calibrex.check.estimators import (
    CheckSkipError,
    EstimatorRun,
    EvidenceArtifact,
    PairContext,
)
from calibrex.check.runner import CheckRunOptions, build_calibration_check, format_check_table
from calibrex.check.verdict import AxisEstimate, check_exit_status
from calibrex.cli.main import main
from calibrex.core.calibration_check import CalibrationCheckArtifact, CheckTimeOffset
from calibrex.core.validation import validate_file


class FakeArtifact:
    schema_version = "slac.fake/v0.1"

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump({"tag": self.tag}), encoding="utf-8")


def _axes(
    errors: tuple[float, float, float], std: float = 0.05, *, estimated: bool = True
) -> tuple[AxisEstimate, ...]:
    return tuple(
        AxisEstimate(name, "deg", error, std, estimated, None if estimated else "unobservable")  # type: ignore[arg-type]
        for name, error in zip(("roll", "pitch", "yaw"), errors, strict=True)
    )


def _run(
    errors: tuple[float, float, float] = (0.0, 0.0, 0.0),
    *,
    policy: str = "pass",
    solved: bool = True,
    estimated: bool = True,
    tag: str = "x",
) -> EstimatorRun:
    return EstimatorRun(
        estimator="stub/v0",
        artifacts=(EvidenceArtifact("rotation", FakeArtifact(tag), policy),),  # type: ignore[arg-type]
        solved=solved,
        policy_status=policy,
        policy_reasons=("stub reason",) if policy != "pass" else ("ok",),
        estimates=_axes(errors, estimated=estimated),
        compared=np.eye(4),
        time_offset=CheckTimeOffset(estimate_s=0.002, std_s=0.001, status="estimated"),
        notes=("stub note",),
    )


@pytest.fixture
def bag(tmp_path: Path) -> Path:
    return write_check_bag(tmp_path / "bag.db3", tf_messages=[default_tf()])


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[PairContext]:
    recorded: list[PairContext] = []

    def wrap(result: EstimatorRun | Exception) -> Any:
        def adapter(ctx: PairContext) -> EstimatorRun:
            recorded.append(ctx)
            if isinstance(result, Exception):
                raise result
            return result

        return adapter

    monkeypatch.setattr(
        estimators,
        "ESTIMATORS",
        {
            "imu-lidar": wrap(_run((0.1, 0.0, 0.2), tag="il")),
            "lidar-lidar": wrap(_run((0.0, 0.9, 0.0), tag="ll")),
            "camera-imu": wrap(CheckSkipError("missing_intrinsics", "no intrinsics for stub")),
            "camera-focal": wrap(CheckSkipError("missing_dependency", "stub: camera-imu skipped")),
            "gnss-lidar": wrap(CheckSkipError("unsupported_sensor", "stub: no RTK-grade fixes")),
        },
    )
    return recorded


def _by(artifact: CalibrationCheckArtifact, pair: str) -> list[Any]:
    return [record for record in artifact.pairs if record.pair == pair]


def test_full_run_verdicts_evidence_and_options(
    tmp_path: Path, bag: Path, calls: list[PairContext]
) -> None:
    evidence = tmp_path / "out" / "check_evidence"
    messages: list[str] = []
    artifact = build_calibration_check(
        bag,
        run=CheckRunOptions(evidence_dir=evidence, base_dir=tmp_path / "out", max_duration_s=12.0),
        progress=messages.append,
    )

    assert artifact.plan_only is False
    assert artifact.options is not None
    assert (artifact.options.sigma_k, artifact.options.max_duration_s) == (3.0, 12.0)
    assert artifact.evidence_dir == "check_evidence"
    imu_lidar = _by(artifact, "imu-lidar")
    assert [r.status for r in imu_lidar] == ["pass", "pass"]  # two lidars
    assert _by(artifact, "lidar-lidar")[0].status == "warn"  # 0.9 deg vs tolerance 0.5
    camera = _by(artifact, "camera-imu")[0]
    assert (camera.status, camera.reason_code) == ("skipped", "missing_intrinsics")
    assert _by(artifact, "camera-focal")[0].reason_code == "missing_dependency"
    assert artifact.overall_verdict == "warn"
    assert any("running imu-lidar" in message for message in messages)

    record = _by(artifact, "lidar-lidar")[0]
    assert record.verdict == "warn"
    assert [axis.name for axis in record.axes] == ["roll", "pitch", "yaw"]
    assert record.axes[1].status == "warn"
    assert record.axes[1].detectable_error == pytest.approx(0.5 + 0.9)
    assert record.time_offset is not None and record.time_offset.estimate_s == 0.002
    assert record.runtime_s is not None
    ref = record.evidence[0]
    assert record.evidence_artifact == ref.path
    written = tmp_path / "out" / ref.path
    assert written.parent == evidence and written.is_file()
    assert hashlib.sha256(written.read_bytes()).hexdigest() == ref.sha256
    assert ref.schema_version == "slac.fake/v0.1"
    # The two imu-lidar records must not overwrite one another's evidence.
    names = {r.evidence[0].path for r in imu_lidar}
    assert len(names) == 2


def test_adapters_receive_candidate_and_topics(
    bag: Path, tmp_path: Path, calls: list[PairContext]
) -> None:
    build_calibration_check(bag, run=CheckRunOptions(evidence_dir=tmp_path / "ev"))

    first = next(
        c for c in calls if c.pair.pair == "imu-lidar" and c.pair.frames[1] == "lidar_front"
    )
    assert first.sensor_topics["imu"] == ("/imu",)
    assert first.sensor_topics["lidar"] == ("/lidar_front/points",)
    assert first.pair.frames == ["imu_link", "lidar_front"]
    # T_imu_lidar = inv(T_base_imu) T_base_lidar = translation (1, 0, 0.4)
    assert first.candidate[:3, 3] == pytest.approx([1.0, 0.0, 0.4])
    lidar_lidar = next(c for c in calls if c.pair.pair == "lidar-lidar")
    assert lidar_lidar.sensor_topics["first"] == ("/lidar_front/points",)
    assert lidar_lidar.sensor_topics["second"] == ("/lidar_rear/points",)
    assert lidar_lidar.topic_types["/imu"] == "sensor_msgs/msg/Imu"


def test_estimator_failures_become_inconclusive(
    tmp_path: Path, bag: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(ctx: PairContext) -> EstimatorRun:
        raise ValueError("tracking produced 1 windows")

    monkeypatch.setattr(
        estimators,
        "ESTIMATORS",
        {
            "imu-lidar": boom,
            "lidar-lidar": lambda ctx: _run(policy="fail"),
            "camera-imu": lambda ctx: _run(estimated=False, policy="inconclusive"),
        },
    )

    artifact = build_calibration_check(bag, run=CheckRunOptions(evidence_dir=tmp_path / "ev"))

    errored = _by(artifact, "imu-lidar")[0]
    assert (errored.status, errored.reason_code) == ("inconclusive", "estimator_error")
    assert "tracking produced 1 windows" in (errored.reason or "")
    assert errored.evidence == []
    failed = _by(artifact, "lidar-lidar")[0]
    assert (failed.status, failed.reason_code) == ("inconclusive", "estimator_failed")
    assert failed.axes  # shown for information
    unobservable = _by(artifact, "camera-imu")[0]
    assert (unobservable.status, unobservable.reason_code) == ("inconclusive", "no_judgeable_axes")
    assert [axis.name for axis in unobservable.unchecked_axes] == ["roll", "pitch", "yaw"]
    assert artifact.overall_verdict == "inconclusive"


def test_unsolved_estimator_is_inconclusive(
    tmp_path: Path, bag: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        estimators,
        "ESTIMATORS",
        {"lidar-lidar": lambda ctx: _run(solved=False, policy="inconclusive")},
    )

    artifact = build_calibration_check(bag, run=CheckRunOptions(evidence_dir=tmp_path / "ev"))

    record = _by(artifact, "lidar-lidar")[0]
    assert (record.status, record.reason_code) == ("inconclusive", "estimator_failed")


def test_pairs_and_camera_filters(tmp_path: Path, bag: Path, calls: list[PairContext]) -> None:
    artifact = build_calibration_check(
        bag,
        run=CheckRunOptions(
            evidence_dir=tmp_path / "ev", pairs=("lidar-lidar", "camera-imu"), camera="/other"
        ),
    )

    assert {c.pair.pair for c in calls} == {"lidar-lidar"}
    assert all(r.reason_code == "not_selected" for r in _by(artifact, "imu-lidar"))
    camera = _by(artifact, "camera-imu")[0]
    assert (camera.status, camera.reason_code) == ("skipped", "not_selected")
    assert artifact.options is not None and artifact.options.pairs == ["lidar-lidar", "camera-imu"]
    assert artifact.options.camera == "/other"


def test_nothing_ran_is_inconclusive(tmp_path: Path, bag: Path, calls: list[PairContext]) -> None:
    artifact = build_calibration_check(
        bag, run=CheckRunOptions(evidence_dir=tmp_path / "ev", pairs=("camera-focal",))
    )

    assert {c.pair.pair for c in calls} == {"camera-focal"}  # the stub skips it
    assert artifact.overall_verdict == "inconclusive"
    assert not (tmp_path / "ev").exists()


def test_plan_mode_never_calls_estimators(bag: Path, calls: list[PairContext]) -> None:
    artifact = build_calibration_check(bag)

    assert not calls
    assert artifact.plan_only is True
    assert artifact.overall_verdict is None and artifact.options is None
    assert {r.status for r in artifact.pairs} <= {"planned", "skipped"}


def test_table_lists_axes_unchecked_and_detectable(
    tmp_path: Path, bag: Path, calls: list[PairContext]
) -> None:
    artifact = build_calibration_check(bag, run=CheckRunOptions(evidence_dir=tmp_path / "ev"))

    table = format_check_table(artifact)

    assert "calibrex check (run)" in table
    assert "|delta|/tolerance" in table and "detectable" in table
    assert "pitch 0.9/0.5 deg [warn]" in table
    assert "overall verdict: warn" in table
    assert "time offset 2.00 ms" in table


# ------------------------------------------------------------------------ CLI


def test_cli_run_writes_artifact_and_evidence(
    tmp_path: Path,
    bag: Path,
    calls: list[PairContext],
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "result" / "check.json"
    exit_code = main(
        [
            "check",
            str(bag),
            "--pairs",
            "lidar-lidar",
            "--max-duration-s",
            "30",
            "--rotation-floor-deg",
            "1.5",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0  # warn does not fail by default... and 0.9 < 1.5 is a pass anyway
    assert "check: running lidar-lidar" in captured.err
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["options"]["rotation_floor_deg"] == 1.5
    assert payload["options"]["max_duration_s"] == 30.0
    assert payload["evidence_dir"] == "check_evidence"
    assert payload["overall_verdict"] == "pass"
    evidence = payload["pairs"][[p["pair"] for p in payload["pairs"]].index("lidar-lidar")][
        "evidence"
    ][0]
    assert (output.parent / evidence["path"]).is_file()
    assert validate_file(output, "calibration-check").kind == "calibration-check"
    reread = CalibrationCheckArtifact.model_validate(payload)
    assert reread.model_dump(mode="json", exclude_none=True) == payload


def test_cli_exit_codes_follow_fail_on(
    tmp_path: Path,
    bag: Path,
    calls: list[PairContext],
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = ["check", str(bag), "--pairs", "lidar-lidar", "--evidence-dir", str(tmp_path / "ev")]

    assert main(args) == 0  # warn, default --fail-on fail
    assert main([*args, "--fail-on", "warn"]) == 1
    assert main([*args, "--fail-on", "never"]) == 0
    capsys.readouterr()


def test_cli_rejects_unknown_pairs(bag: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["check", str(bag), "--pairs", "imu-lidarr"]) != 0
    assert "unknown pair" in capsys.readouterr().err


def test_cli_plan_is_unchanged_and_fast(
    bag: Path, calls: list[PairContext], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["check", str(bag), "--plan"]) == 0

    out = capsys.readouterr().out
    assert not calls
    assert "calibrex check (plan)" in out
    assert "overall verdict" not in out


def test_partial_coverage_is_recorded_and_printed(
    tmp_path: Path, bag: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    partial = _run((0.0, 0.0, 0.0))
    axes = list(partial.estimates)
    axes[2] = AxisEstimate("yaw", "deg", 0.0, 3.0, False, "x", "unobservable")
    partial = EstimatorRun(**{**partial.__dict__, "estimates": tuple(axes)})
    monkeypatch.setattr(estimators, "ESTIMATORS", {"lidar-lidar": lambda ctx: partial})

    artifact = build_calibration_check(
        bag, run=CheckRunOptions(evidence_dir=tmp_path / "ev", pairs=("lidar-lidar",))
    )

    record = _by(artifact, "lidar-lidar")[0]
    assert (record.status, record.coverage) == ("pass", "partial")
    assert artifact.overall_verdict == "pass" and artifact.summary.partial_pairs == 1
    table = format_check_table(artifact)
    assert "pass (partial: roll, pitch only)" in table and "WARNING: 1 pair(s)" in table
    assert "--fail-on partial" in table

    args = ["check", str(bag), "--pairs", "lidar-lidar", "--evidence-dir", str(tmp_path / "e2")]
    assert main(args) == 0  # a partial pass is a pass by default
    assert main([*args, "--fail-on", "inconclusive"]) == 0
    assert main([*args, "--fail-on", "partial"]) == 1  # but it vouches for roll and pitch only


def test_an_inconclusive_run_reads_plainly(
    tmp_path: Path, bag: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        estimators,
        "ESTIMATORS",
        {"camera-imu": lambda ctx: _run(estimated=False, policy="inconclusive")},
    )

    artifact = build_calibration_check(
        bag, run=CheckRunOptions(evidence_dir=tmp_path / "ev", pairs=("camera-imu",))
    )
    table = format_check_table(artifact)

    assert artifact.overall_verdict == "inconclusive" and artifact.summary.partial_pairs == 1
    # nothing passed, so no pass is over-trusted: the partial-coverage warning stays quiet
    assert "WARNING" not in table
    assert "says nothing about its calibration" in table
    # axes that share a reason share a line
    assert table.count("unchecked roll, pitch, yaw: ") == 1
    assert "unchecked pitch:" not in table


def test_the_overall_verdict_is_explained_and_absent_sensors_take_one_line(
    tmp_path: Path, bag: Path, calls: list[PairContext]
) -> None:
    run = build_calibration_check(bag, run=CheckRunOptions(evidence_dir=tmp_path / "ev"))
    plan = build_calibration_check(bag)

    table = format_check_table(run)
    absent = [p.pair for p in run.pairs if p.reason_code == "missing_topic" and not p.sensors]
    assert absent  # the fixture bag has no radar, among others

    assert "beyond its tolerance but within twice it" in table  # overall verdict: warn
    assert f"no topic in the bag: {', '.join(dict.fromkeys(absent))}" in table
    rows = [line.split()[0] for line in table.splitlines() if " skipped " in line]
    assert not set(rows) & set(absent)
    assert "--plan lists what each expects" in table
    # the plan keeps one row and one hint per pair: it is where those are read
    planned = format_check_table(plan)
    assert "no topic in the bag" not in planned
    assert all(f"\n{name} " in planned for name in absent)


def test_pairs_without_an_estimate_get_a_next_step(
    tmp_path: Path, bag: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(ctx: PairContext) -> EstimatorRun:
        raise ValueError("tracking produced 1 windows")

    monkeypatch.setattr(
        estimators,
        "ESTIMATORS",
        {"imu-lidar": boom, "lidar-lidar": lambda ctx: _run(solved=False)},
    )

    artifact = build_calibration_check(
        bag,
        run=CheckRunOptions(evidence_dir=tmp_path / "ev", pairs=("imu-lidar", "lidar-lidar")),
    )
    table = format_check_table(artifact)

    assert "  - lidar-lidar: the estimator did not converge" in table
    assert "  - imu-lidar: the estimator crashed" in table
    assert table.count("the estimator crashed") == 1  # two IMU-LiDAR records, one line


def test_the_plan_says_when_a_run_would_skip_the_camera_for_opencv(
    bag: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from calibrex.check import hints

    monkeypatch.setattr(hints, "opencv_missing", lambda: True)
    without = format_check_table(build_calibration_check(bag))
    monkeypatch.setattr(hints, "opencv_missing", lambda: False)
    with_opencv = format_check_table(build_calibration_check(bag))

    assert "a run would skip them here: OpenCV is not installed" in without
    assert "OpenCV is not installed" not in with_opencv


def test_notes_wrap_and_a_wide_verdict_table_is_stacked(
    tmp_path: Path, bag: Path, calls: list[PairContext]
) -> None:
    artifact = build_calibration_check(bag, run=CheckRunOptions(evidence_dir=tmp_path / "ev"))

    whole = format_check_table(artifact)
    narrow = format_check_table(artifact, width=80)

    header = next(line for line in whole.splitlines() if line.startswith("pair "))
    assert len(header) > 80 and header in format_check_table(artifact, width=1000)
    # at 80 columns the verdict table becomes one block per pair, and its cells wrap
    assert header not in narrow
    assert "imu-lidar: pass" in narrow.splitlines()
    assert any(line.startswith("    sensors: ") for line in narrow.splitlines())
    notes = [line for line in narrow.splitlines() if line.startswith(" ")]
    assert notes and max(len(line) for line in notes) <= 80
    # only the two unindented one-line facts (the bag path, the summary) are left whole
    long = [line for line in narrow.splitlines() if len(line) > 80]
    assert all(line.startswith(("bag:", "summary:")) for line in long)
    # every cell of the table is still there
    row = next(line for line in whole.splitlines() if line.startswith("imu-lidar "))
    assert all(word in narrow.split() for word in row.split())


def test_a_long_reason_is_cut_in_the_table_only(
    tmp_path: Path, bag: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(ctx: PairContext) -> EstimatorRun:
        raise ValueError("frame skipped; " * 60)

    monkeypatch.setattr(estimators, "ESTIMATORS", {"imu-lidar": boom})

    artifact = build_calibration_check(
        bag, run=CheckRunOptions(evidence_dir=tmp_path / "ev", pairs=("imu-lidar",))
    )
    table = format_check_table(artifact)

    assert len(_by(artifact, "imu-lidar")[0].reason) > 800
    assert "the JSON artifact has the full text" in table
    assert max(len(line) for line in table.splitlines()) < 400


@pytest.mark.parametrize(
    ("overall", "partial", "fail_on", "status"),
    [
        ("pass", 0, "partial", 0),
        ("pass", 1, "partial", 1),
        ("pass", 1, "inconclusive", 0),
        ("inconclusive", 0, "partial", 1),
        ("warn", 0, "inconclusive", 1),
        ("warn", 1, "fail", 0),
        ("fail", 0, "fail", 1),
        ("fail", 2, "never", 0),
        (None, 1, "partial", 0),
    ],
)
def test_exit_status_levels(overall: str | None, partial: int, fail_on: str, status: int) -> None:
    assert check_exit_status(overall, partial, fail_on) == status


def test_exit_status_rejects_an_unknown_level() -> None:
    with pytest.raises(ValueError, match="--fail-on"):
        check_exit_status("pass", 0, "sometimes")


def test_cli_topic_kind_classifies_a_twist_topic_and_is_recorded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    twist_bag = write_check_bag(
        tmp_path / "twist.db3",
        tf_messages=[default_tf()],
        sensor_frames={
            "/vehicle/twist": ("geometry_msgs/msg/TwistStamped", "imu_link"),
            "/lidar_front/points": ("sensor_msgs/msg/PointCloud2", "lidar_front"),
        },
    )
    args = ["check", str(twist_bag), "--plan", "--json", "--vehicle-frame", "base_link"]

    assert main(args) == 0
    plain = json.loads(capsys.readouterr().out)
    wheel = next(p for p in plain["pairs"] if p["pair"] == "lidar-wheel_odometry")
    assert (wheel["status"], wheel["reason_code"]) == ("skipped", "missing_topic")
    assert "--topic-kind" in wheel["reason"]

    assert main([*args, "--topic-kind", "/vehicle/twist=wheel"]) == 0
    forced = json.loads(capsys.readouterr().out)
    wheel = next(p for p in forced["pairs"] if p["pair"] == "lidar-wheel_odometry")
    assert wheel["status"] == "planned"
    twist = next(t for t in forced["topics"] if t["topic"] == "/vehicle/twist")
    assert (twist["role"], twist["odometry_kind"]) == ("twist", "wheel")
    assert main([*args, "--topic-kind", "/vehicle/twist=gps"]) == 2
    assert "--topic-kind" in capsys.readouterr().err


def test_cli_topic_kind_is_recorded_in_the_run_options(
    tmp_path: Path, calls: list[PairContext]
) -> None:
    out = tmp_path / "check.json"
    twist_bag = write_check_bag(
        tmp_path / "twist.db3",
        tf_messages=[default_tf()],
        sensor_frames={"/vehicle/twist": ("geometry_msgs/msg/TwistStamped", "imu_link")},
    )

    assert (
        main(
            [
                "check",
                str(twist_bag),
                "--topic-kind",
                "/vehicle/twist=ins",
                "--output",
                str(out),
                "--no-cache",
                "--fail-on",
                "never",
            ]
        )
        == 0
    )

    artifact = json.loads(out.read_text())
    assert artifact["options"]["topic_kinds"] == {"/vehicle/twist": "ins"}
    validate_file(out)


CLOSURE_PAIRS = ("imu-lidar", "lidar-lidar")


def _closure_run(monkeypatch: pytest.MonkeyPatch, front: float, rear: float, lidar: float) -> Any:
    errors = {"lidar_front": (0.0, 0.0, front), "lidar_rear": (0.0, 0.0, rear)}

    def imu_lidar(ctx: PairContext) -> EstimatorRun:
        return _run(errors[ctx.pair.frames[1]])

    monkeypatch.setattr(
        estimators,
        "ESTIMATORS",
        {"imu-lidar": imu_lidar, "lidar-lidar": lambda ctx: _run((0.0, 0.0, lidar))},
    )


def test_closure_over_the_estimates_floors_the_overall_verdict_at_warn(
    tmp_path: Path, bag: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # every pair is inside its 0.5 deg tolerance, but the three estimates do not close
    _closure_run(monkeypatch, 0.45, -0.45, -0.45)
    artifact = build_calibration_check(
        bag, run=CheckRunOptions(evidence_dir=tmp_path / "ev", pairs=CLOSURE_PAIRS)
    )

    assert {r.status for r in artifact.pairs if r.axes} == {"pass"}
    assert artifact.closures is not None
    loop = artifact.closures.loops[0]
    assert loop.kind == "cycle" and loop.verdict == "fail"
    assert artifact.overall_verdict == "warn"
    assert "closure of the estimates" in format_check_table(artifact)
    # the schema round trip keeps the optional field
    again = CalibrationCheckArtifact.model_validate(artifact.model_dump(mode="json"))
    assert again.closures == artifact.closures


def test_consistent_estimates_do_not_change_the_overall_verdict(
    tmp_path: Path, bag: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _closure_run(monkeypatch, 0.0, 0.0, 0.0)
    artifact = build_calibration_check(
        bag, run=CheckRunOptions(evidence_dir=tmp_path / "ev", pairs=CLOSURE_PAIRS)
    )

    assert artifact.closures is not None and artifact.closures.verdict == "pass"
    assert artifact.overall_verdict == "pass"


def test_errors_shared_through_one_frame_are_attributed_to_it(
    tmp_path: Path, bag: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # imu-lidar compares T_lidar_imu, so the IMU is the child of both pairs: its candidate
    # turned by w reads w on both, while the LiDARs agree with each other
    _closure_run(monkeypatch, -2.0, -2.0, 0.0)
    artifact = build_calibration_check(
        bag, run=CheckRunOptions(evidence_dir=tmp_path / "ev", pairs=CLOSURE_PAIRS)
    )

    assert artifact.rotation_attribution is not None
    (group,) = artifact.rotation_attribution.groups
    assert group.status == "attributed", group.summary
    assert group.frame == "imu_link"
    assert group.hypotheses[0].rotation_deg == pytest.approx([0.0, 0.0, -2.0], abs=1.0e-6)
    assert group.pairs[:2] == ["imu-lidar:lidar_front>imu_link", "imu-lidar:lidar_rear>imu_link"]
    assert "imu_link alone explains the pair errors" in format_check_table(artifact)
    again = CalibrationCheckArtifact.model_validate(artifact.model_dump(mode="json"))
    assert again.rotation_attribution == artifact.rotation_attribution


def test_an_unchecked_rotation_axis_says_what_to_record(
    tmp_path: Path, bag: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from calibrex.check import runner

    partial = _run((0.0, 0.0, 0.0))
    axes = list(partial.estimates)
    axes[1] = AxisEstimate("pitch", "deg", 0.0, 0.4, False, "x", "unobservable")
    # R_cam_imu: the IMU's z axis is the camera's y axis
    compared = np.eye(4)
    compared[:3, :3] = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
    partial = EstimatorRun(**{**partial.__dict__, "estimates": tuple(axes), "compared": compared})
    monkeypatch.setattr(estimators, "ESTIMATORS", {"camera-imu": lambda ctx: partial})
    gyro = np.zeros((200, 3))
    gyro[:, 2] = np.radians(40.0)  # a walk: the rig only pans, about the IMU's z
    reads: list[str] = []
    duration = [100.0]  # seconds of IMU the fake reader returns

    def fake_gyro(_bag: Path, topic: str, *, max_duration_s: float | None = None) -> Any:
        reads.append(topic)
        return list(np.arange(200) * duration[0] / 199), gyro

    monkeypatch.setattr(runner, "read_gyro", fake_gyro)

    artifact = build_calibration_check(
        bag, run=CheckRunOptions(evidence_dir=tmp_path / "ev", pairs=("camera-imu",))
    )

    record = _by(artifact, "camera-imu")[0]
    assert (record.status, record.coverage) == ("pass", "partial")
    reason = record.unchecked_axes[0].reason
    assert "; to judge pitch, rotate the rig about the x or z axis of camera_optical" in reason
    assert "x 0, y 40, z 0 deg/s" in reason
    table = format_check_table(artifact)
    assert "next steps:" in table
    assert "camera-imu (camera_optical): to judge pitch, rotate the rig" in table
    assert reads == ["/imu"]

    # the same pair on a rig that did rotate about x and z: no motion is asked for
    gyro[:, 0] = np.radians(30.0)
    moved = build_calibration_check(
        bag, run=CheckRunOptions(evidence_dir=tmp_path / "e3", pairs=("camera-imu",))
    )
    assert "to judge pitch, other motion is not what is missing" in format_check_table(moved)
    duration[0] = 30.0  # the same motion, but a short recording: ask for more of it
    short = build_calibration_check(
        bag, run=CheckRunOptions(evidence_dir=tmp_path / "e4", pairs=("camera-imu",))
    )
    assert "to judge pitch, record the same motion for longer" in format_check_table(short)

    # a pair with every axis judged reads no IMU and gets no advice
    reads.clear()
    monkeypatch.setattr(estimators, "ESTIMATORS", {"camera-imu": lambda ctx: _run()})
    full = build_calibration_check(
        bag, run=CheckRunOptions(evidence_dir=tmp_path / "e2", pairs=("camera-imu",))
    )
    assert not reads and "to judge" not in format_check_table(full)


def test_an_external_camera_joins_the_topics_and_only_camera_imu_reads_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from calibrex.data.external_camera import parse_camera_source

    imu_only = write_check_bag(
        tmp_path / "imu.db3",
        tf_messages=[default_tf()],
        sensor_frames={
            "/imu": ("sensor_msgs/msg/Imu", "imu_link"),
            "/lidar_front/points": ("sensor_msgs/msg/PointCloud2", "lidar_front"),
        },
    )
    (tmp_path / "frames").mkdir()
    source = parse_camera_source(f"camera_optical={tmp_path / 'frames'}")
    seen: list[tuple[str, tuple[str, ...]]] = []

    def camera_imu(ctx: PairContext) -> EstimatorRun:
        seen.append((ctx.pair.pair, ctx.sensor_topics["camera"]))
        assert ctx.controls.external_cameras["/external/camera_optical"] is source
        return _run()

    monkeypatch.setattr(
        estimators,
        "ESTIMATORS",
        {**estimators.ESTIMATORS, "camera-imu": camera_imu},
    )

    artifact = build_calibration_check(
        imu_only,
        run=CheckRunOptions(
            evidence_dir=tmp_path / "ev",
            pairs=("camera-imu", "camera-lidar"),
            external_cameras=(source,),
        ),
    )

    topic = next(t for t in artifact.topics if t.topic == "/external/camera_optical")
    assert (topic.role, topic.mapped_frame) == ("camera", "camera_optical")
    assert "camera outside the bag (images)" in topic.notes[0]
    assert seen == [("camera-imu", ("/external/camera_optical",))]
    assert _by(artifact, "camera-imu")[0].status == "pass"
    lidar = _by(artifact, "camera-lidar")[0]
    assert (lidar.status, lidar.reason_code) == ("skipped", "unsupported_sensor")
    assert "only camera-focal and camera-imu read external cameras" in (lidar.reason or "")


def test_a_camera_that_cannot_be_placed_on_the_imu_clock_is_inconclusive(
    tmp_path: Path, bag: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_clock(ctx: PairContext) -> EstimatorRun:
        raise estimators.CheckInconclusiveError(
            "estimator_failed", "cannot place the camera on the IMU clock: the motion repeats"
        )

    monkeypatch.setattr(estimators, "ESTIMATORS", {"camera-imu": no_clock})

    artifact = build_calibration_check(
        bag, run=CheckRunOptions(evidence_dir=tmp_path / "ev", pairs=("camera-imu",))
    )

    record = _by(artifact, "camera-imu")[0]
    # not skipped: a skipped pair would not count towards the overall verdict
    assert (record.status, record.reason_code) == ("inconclusive", "estimator_failed")
    assert artifact.overall_verdict == "inconclusive"
