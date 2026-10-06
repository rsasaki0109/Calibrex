"""Progress reporting of ``calibrex check``: renderers, wiring and result invariance."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tests.unit import vehicle_fixtures as vf
from tests.unit.check_fixtures import default_tf, write_check_bag

from calibrex.check import estimators, vehicle_inputs
from calibrex.check.estimators import EstimatorRun, EvidenceArtifact, PairContext
from calibrex.check.progress import (
    CLEAR_LINE,
    CheckProgress,
    LegacyProgress,
    PlainProgress,
    TtyProgress,
    eta_seconds,
    format_duration,
    make_progress,
    paint_verdict,
    use_color,
)
from calibrex.check.runner import CheckRunOptions, build_calibration_check, format_check_table
from calibrex.cli.main import main
from calibrex.core.calibration_check import CalibrationCheckArtifact, CheckTimeOffset
from calibrex.core.progress import emit_stage, emit_tick, progress_sink


class FakeTty(io.StringIO):
    def isatty(self) -> bool:
        return True


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


# ------------------------------------------------------------------ helpers


def test_format_duration_and_eta() -> None:
    assert format_duration(4.4) == "4s"
    assert format_duration(185) == "3m05s"
    assert format_duration(3720) == "1h02m"
    assert eta_seconds(10.0, 25, 100) == pytest.approx(30.0)
    assert eta_seconds(10.0, 0, 100) is None
    assert eta_seconds(10.0, 5, None) is None
    assert eta_seconds(10.0, 120, 100) is None


def test_hooks_do_nothing_without_a_sink_and_restore_the_previous_sink() -> None:
    emit_stage("no sink")
    emit_tick(1, 2)
    events: list[Any] = []
    with progress_sink(events.append):
        emit_stage("pass 1")
        emit_tick(3, 10)
    emit_tick(4)
    assert [(e.kind, e.text, e.done, e.total) for e in events] == [
        ("stage", "pass 1", None, None),
        ("tick", "", 3, 10),
    ]


# ---------------------------------------------------------------- renderers


def test_use_color_needs_a_tty_and_no_no_color() -> None:
    assert use_color(FakeTty(), {}) is True
    assert use_color(FakeTty(), {"NO_COLOR": "1"}) is False
    assert use_color(FakeTty(), {"NO_COLOR": ""}) is True  # empty counts as unset
    assert use_color(FakeTty(), {"TERM": "dumb"}) is False
    assert use_color(io.StringIO(), {}) is False


def test_make_progress_picks_the_renderer() -> None:
    assert type(make_progress(FakeTty(), environ={})) is TtyProgress
    assert type(make_progress(io.StringIO(), environ={})) is PlainProgress
    assert type(make_progress(FakeTty(), plain=True, environ={})) is PlainProgress  # --json
    assert type(make_progress(FakeTty(), environ={"TERM": "dumb"})) is PlainProgress
    assert type(make_progress(FakeTty(), quiet=True, environ={})) is CheckProgress


def test_plain_renderer_writes_lines_without_control_codes() -> None:
    out, clock = io.StringIO(), Clock()
    progress = PlainProgress(out, clock, tick_interval_s=10.0)
    progress.run_started(2)
    progress.pair_started(1, 2, "imu-lidar (a / b)")
    progress.set_scan_total(200)
    progress.stage("odometry + deskew pass 1 of up to 3")
    clock.now += 4
    progress.tick(20)  # inside the tick interval: no line
    clock.now += 20
    progress.tick(100)
    progress.stage("reusing cached imu_lidar_rotation estimate")
    progress.pair_finished("imu-lidar (a / b)", "pass", 24.0, True)
    progress.pair_started(2, 2, "lidar-lidar (a / b)")
    progress.pair_skipped("lidar-lidar (a / b)", "missing_topic", "no topic")
    text = out.getvalue()
    lines = text.splitlines()
    assert "\r" not in text and "\x1b" not in text
    assert lines[0] == "check: 2 pairs to run"
    assert lines[1] == "check: running imu-lidar (a / b) [pair 1/2]"
    assert any("100/200 msgs (50%)" in line and "eta 24s" in line for line in lines)
    assert not any("20/200" in line for line in lines)
    assert "check: imu-lidar (a / b) -> pass (24 s) [cache hit] [pair 1/2 done]" in lines
    assert lines[-1] == "check: lidar-lidar (a / b) skipped (missing_topic): no topic"


def test_tty_renderer_rewrites_one_line_and_keeps_finished_pairs() -> None:
    out, clock = FakeTty(), Clock()
    progress = TtyProgress(out, clock, refresh_s=0.0, color=True)
    progress.run_started(1)
    progress.pair_started(1, 3, "lidar-lidar (a / b)")
    progress.stage("reference LiDAR odometry")
    clock.now += 30
    progress.set_scan_total(300)
    progress.tick(100)
    progress.pair_finished("lidar-lidar (a / b)", "warn", 31.0, None)
    progress.run_finished()
    text = out.getvalue()
    assert text.count("\n") == 1  # only the finished pair is a permanent line
    assert CLEAR_LINE in text
    expected = "check: pair 1/3 [100/300 msgs (33%), 30s elapsed, eta 1m00s] lidar-lidar: "
    assert expected + "reference LiDAR odometry" in text
    assert "check: lidar-lidar (a / b) -> \x1b[33mwarn\x1b[0m (31 s) [pair 1/3 done]\n" in text
    assert text.endswith(CLEAR_LINE) is False  # nothing left drawn after the last pair


def test_tty_renderer_without_color_has_no_sgr_codes() -> None:
    out = FakeTty()
    progress = TtyProgress(out, Clock(), refresh_s=0.0, color=False)
    progress.run_started(1)
    progress.pair_started(1, 1, "x")
    progress.pair_finished("x", "pass", 1.0, False)
    progress.run_finished()
    assert "\x1b[3" not in out.getvalue()


def test_legacy_adapter_keeps_the_original_messages() -> None:
    messages: list[str] = []
    progress = LegacyProgress(messages.append)
    progress.pair_started(1, 1, "lidar-lidar (a / b)")
    progress("lidar-lidar: a -> b")
    progress.pair_finished("lidar-lidar (a / b)", "pass", 3.2, True)
    assert messages == [
        "check: running lidar-lidar (a / b)",
        "lidar-lidar: a -> b",
        "check: lidar-lidar (a / b) -> pass (3 s)",
    ]


def test_verdict_colors() -> None:
    assert paint_verdict("pass") == "\x1b[32mpass\x1b[0m"
    assert paint_verdict("fail  ") == "\x1b[31mfail  \x1b[0m"
    assert paint_verdict("warn (partial: roll only)").startswith("\x1b[33m")
    assert paint_verdict("inconclusive").startswith("\x1b[2m")
    assert paint_verdict("skipped").startswith("\x1b[2m")
    assert paint_verdict("planned") == "planned"


# ------------------------------------------------- wiring and invariance


@pytest.fixture(scope="module")
def simulation() -> vf.Simulation:
    return vf.simulate(duration_s=60.0)


def _check(
    bag: Path, out: Path, simulation: vf.Simulation, progress: Any
) -> CalibrationCheckArtifact:
    patch = pytest.MonkeyPatch()
    patch.setattr(
        vehicle_inputs,
        "read_lidar_odometry",
        lambda bag, topic, options=None, max_duration_s=None: vf.lidar_track(simulation, topic),
    )
    try:
        return build_calibration_check(
            bag,
            vehicle_frame="base_link",
            run=CheckRunOptions(
                pairs=("lidar-vehicle", "imu-vehicle"),
                evidence_dir=out / "evidence",
                base_dir=out,
            ),
            progress=progress,
        )
    finally:
        patch.undo()


def _comparable(artifact: CalibrationCheckArtifact) -> str:
    payload = artifact.model_dump(mode="json", exclude_none=True)
    payload["provenance"].pop("created_at")
    for record in payload["pairs"]:
        record.pop("runtime_s", None)
        record.pop("evidence", None)  # the written artifacts carry their own timestamps
    return json.dumps(payload, sort_keys=True)


def test_results_are_identical_with_progress_on_off_or_legacy(
    tmp_path: Path, simulation: vf.Simulation
) -> None:
    bag = vf.write_vehicle_bag(tmp_path / "bag", simulation)
    out = io.StringIO()
    off = _check(bag, tmp_path / "off", simulation, lambda _m: None)
    plain = _check(bag, tmp_path / "plain", simulation, PlainProgress(out))
    tty = _check(bag, tmp_path / "tty", simulation, TtyProgress(FakeTty()))
    quiet = _check(bag, tmp_path / "quiet", simulation, CheckProgress())
    assert _comparable(off) == _comparable(plain) == _comparable(tty) == _comparable(quiet)
    lines = out.getvalue().splitlines()
    assert lines[0] == "check: 2 pairs to run"
    assert any("running" in line and "[pair 1/2]" in line for line in lines)
    assert any("[pair 2/2 done]" in line for line in lines)


def test_cache_hit_is_reported(tmp_path: Path, simulation: vf.Simulation) -> None:
    from calibrex.check.cache import EstimatorCache

    bag = vf.write_vehicle_bag(tmp_path / "bag", simulation)
    patch = pytest.MonkeyPatch()
    patch.setattr(
        vehicle_inputs,
        "read_lidar_odometry",
        lambda bag, topic, options=None, max_duration_s=None: vf.lidar_track(simulation, topic),
    )
    outputs: list[str] = []
    try:
        for name in ("first", "second"):
            stream = io.StringIO()
            build_calibration_check(
                bag,
                vehicle_frame="base_link",
                run=CheckRunOptions(
                    pairs=("imu-vehicle",),
                    evidence_dir=tmp_path / name,
                    base_dir=tmp_path,
                    cache_dir=tmp_path / "cache",
                ),
                progress=PlainProgress(stream),
            )
            outputs.append(stream.getvalue())
    finally:
        patch.undo()
    assert EstimatorCache(tmp_path / "cache").directory.exists()
    assert "[cache hit]" not in outputs[0]
    assert "reusing cached" in outputs[1] and "[cache hit]" in outputs[1]


def test_lidar_odometry_reports_scans_without_changing_the_track(tmp_path: Path) -> None:
    from tests.unit.test_ins_lidar_hand_eye import _motion, _scene, _view

    from calibrex.data.ros_cdr_writer import POINT_FIELD_FLOAT32, encode_pointcloud2
    from calibrex.data.rosbag2_writer import Rosbag2Writer

    scene = _scene()
    step = _motion(0.9, 1.0)
    fields = tuple((name, 4 * i, POINT_FIELD_FLOAT32, 1) for i, name in enumerate("xyz"))
    base = 1_700_000_000 * 10**9
    with Rosbag2Writer(tmp_path / "bag") as writer:
        writer.add_topic("/points", "sensor_msgs/msg/PointCloud2")
        for k in range(5):
            scan = _view(scene, np.linalg.matrix_power(step, k)).astype(np.float32)
            stamp = base + k * 10**8
            writer.write(
                "/points",
                stamp,
                encode_pointcloud2(
                    frame_id="lidar",
                    timestamp_ns=stamp,
                    fields=fields,
                    point_step=12,
                    data=np.ascontiguousarray(scan[:, :3], dtype="<f4").tobytes(),
                ),
            )
    plain = vehicle_inputs.read_lidar_odometry(tmp_path / "bag", "/points")
    events: list[Any] = []
    with progress_sink(events.append):
        reported = vehicle_inputs.read_lidar_odometry(tmp_path / "bag", "/points")
    assert [e.done for e in events] == [1, 2, 3, 4, 5]
    assert np.array_equal(np.asarray(plain.poses), np.asarray(reported.poses))
    assert plain.times_s == reported.times_s


# ----------------------------------------------------------------------- CLI


def _stub_estimator(monkeypatch: pytest.MonkeyPatch) -> None:
    class Fake:
        schema_version = "slac.fake/v0.1"

        def save(self, path: str | Path) -> None:
            Path(path).write_text("tag: x\n", encoding="utf-8")

    def adapter(ctx: PairContext) -> EstimatorRun:
        ctx.controls.progress("lidar-lidar: stub stage")
        return EstimatorRun(
            estimator="stub/v0",
            artifacts=(EvidenceArtifact("rotation", Fake(), "pass"),),  # type: ignore[arg-type]
            solved=True,
            policy_status="pass",
            policy_reasons=("ok",),
            estimates=tuple(
                estimators.AxisEstimate(n, "deg", 0.0, 0.05, True, None)  # type: ignore[arg-type]
                for n in ("roll", "pitch", "yaw")
            ),
            compared=np.eye(4),
            time_offset=CheckTimeOffset(estimate_s=0.002, std_s=0.001, status="estimated"),
        )

    monkeypatch.setitem(estimators.ESTIMATORS, "lidar-lidar", adapter)


def test_cli_progress_goes_to_stderr_only_and_quiet_silences_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_estimator(monkeypatch)
    bag = write_check_bag(tmp_path / "bag.db3", tf_messages=[default_tf()])
    args = ["check", str(bag), "--pairs", "lidar-lidar", "--no-cache", "--evidence-dir"]

    assert main([*args, str(tmp_path / "e1")]) == 0
    loud = capsys.readouterr()
    assert main([*args, str(tmp_path / "e2"), "--quiet"]) == 0
    quiet = capsys.readouterr()

    assert "check: 1 pair to run" in loud.err
    assert "check: running lidar-lidar" in loud.err and "[pair 1/1]" in loud.err
    assert "stub stage" in loud.err
    assert quiet.err == ""
    assert "check:" not in loud.out
    assert "\x1b" not in loud.out and "\x1b" not in loud.err  # not a TTY: no colour, no \r
    assert loud.out.splitlines()[0] == quiet.out.splitlines()[0]  # same report either way


def test_cli_json_stdout_is_unchanged_by_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _stub_estimator(monkeypatch)
    bag = write_check_bag(tmp_path / "bag.db3", tf_messages=[default_tf()])
    args = ["check", str(bag), "--pairs", "lidar-lidar", "--no-cache", "--json"]
    assert main([*args, "--evidence-dir", str(tmp_path / "e")]) == 0
    loud = capsys.readouterr()
    assert main([*args, "--evidence-dir", str(tmp_path / "e"), "--quiet"]) == 0
    quiet = capsys.readouterr()
    assert "check: running lidar-lidar" in loud.err
    assert quiet.err == ""

    def scrub(text: str) -> dict[str, Any]:
        payload = json.loads(text)
        payload["provenance"].pop("created_at")
        payload["provenance"].pop("command")
        for record in payload["pairs"]:
            record.pop("runtime_s", None)
        return payload  # type: ignore[no-any-return]

    assert scrub(loud.out) == scrub(quiet.out)


# ------------------------------------------------------------- colored table


def test_table_color_only_changes_the_verdict_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_estimator(monkeypatch)
    bag = write_check_bag(tmp_path / "bag.db3", tf_messages=[default_tf()])
    artifact = build_calibration_check(
        bag,
        run=CheckRunOptions(pairs=("lidar-lidar",), evidence_dir=tmp_path / "e", base_dir=tmp_path),
    )
    plain = format_check_table(artifact)
    colored = format_check_table(artifact, color=True)
    assert "\x1b" not in plain
    assert "\x1b[32mpass" in colored
    assert colored.count("\x1b[0m") >= 2
    stripped = colored
    for code in ("\x1b[32m", "\x1b[33m", "\x1b[31m", "\x1b[2m"):
        stripped = stripped.replace(code, "")
    stripped = stripped.replace("\x1b[0m", "")
    assert [line.rstrip() for line in stripped.splitlines()] == [
        line.rstrip() for line in plain.splitlines()
    ]
