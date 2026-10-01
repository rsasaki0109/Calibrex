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
from calibrex.check.verdict import AxisEstimate
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
    assert _by(artifact, "gnss-lidar")[0].reason_code == "method_not_wired"
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

    first = next(c for c in calls if c.pair.pair == "imu-lidar" and c.pair.frames[1] == "lidar_front")
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


def test_pairs_and_camera_filters(
    tmp_path: Path, bag: Path, calls: list[PairContext]
) -> None:
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

    assert not calls
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
