"""Schema, artifact builder and CLI for ``calibrex check`` (Phase A)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from tests.unit.check_fixtures import IDENTITY, default_tf, write_check_bag

from calibrex.check.runner import bag_input_digest, build_calibration_check, parse_frame_map
from calibrex.cli.main import main
from calibrex.core.calibration_check import (
    CalibrationCheckArtifact,
    CheckPairRecord,
    CheckTransform,
    summarize_pairs,
)
from calibrex.core.exceptions import DatasetError
from calibrex.core.schema_registry import kind_for_schema_version
from calibrex.core.validation import validate_file

TRANSFORM = CheckTransform(
    parent_frame="a", child_frame="b", translation_m=[0, 0, 0], rotation_quat_xyzw=[0, 0, 0, 1]
)


def _bag(tmp_path: Path) -> Path:
    return write_check_bag(tmp_path / "rig.db3", tf_messages=[default_tf()])


# ------------------------------------------------------------------ schema


def test_artifact_roundtrip_and_kind_lookup(tmp_path: Path) -> None:
    artifact = build_calibration_check(_bag(tmp_path), command=["calibrex", "check", "x"])

    reloaded = CalibrationCheckArtifact.model_validate(artifact.model_dump(mode="json"))

    assert reloaded == artifact
    assert artifact.schema_version == "slac.calibration_check/v0.1"
    assert kind_for_schema_version(artifact.schema_version) == "calibration-check"
    assert artifact.plan_only is True
    assert artifact.provenance.input_sha256 == artifact.bag.input_sha256
    assert artifact.provenance.command == ["calibrex", "check", "x"]


def test_pair_record_consistency_rules() -> None:
    with pytest.raises(ValidationError, match="reason_code and reason"):
        CheckPairRecord(pair="imu-lidar", status="skipped")
    with pytest.raises(ValidationError, match="candidate_transform"):
        CheckPairRecord(pair="imu-lidar", status="planned")
    with pytest.raises(ValidationError, match="must not carry"):
        CheckPairRecord(
            pair="imu-lidar",
            status="planned",
            candidate_transform=TRANSFORM,
            reason_code="missing_topic",
        )
    # later phases: verdict statuses need no extra fields in v0.1
    done = CheckPairRecord(pair="imu-lidar", status="pass", evidence_artifact="e.json")
    assert done.verdict is None


def test_summary_must_match_pairs(tmp_path: Path) -> None:
    artifact = build_calibration_check(_bag(tmp_path))
    payload = artifact.model_dump(mode="json")
    payload["summary"]["runnable_count"] += 1

    with pytest.raises(ValidationError, match="summary does not match"):
        CalibrationCheckArtifact.model_validate(payload)


def test_summarize_pairs_counts() -> None:
    pairs = [
        CheckPairRecord(pair="imu-lidar", status="planned", candidate_transform=TRANSFORM),
        CheckPairRecord(
            pair="camera-imu", status="skipped", reason_code="missing_topic", reason="x"
        ),
        CheckPairRecord(
            pair="camera-focal", status="skipped", reason_code="missing_topic", reason="x"
        ),
    ]

    summary = summarize_pairs(pairs)

    assert summary.pair_count == 3
    assert summary.runnable_count == 1
    assert summary.status_counts == {"planned": 1, "skipped": 2}
    assert summary.skipped_by_reason == {"missing_topic": 2}


def test_extra_fields_are_rejected(tmp_path: Path) -> None:
    payload = build_calibration_check(_bag(tmp_path)).model_dump(mode="json")
    payload["surprise"] = 1

    with pytest.raises(ValidationError):
        CalibrationCheckArtifact.model_validate(payload)


# ------------------------------------------------------------------ builder


def test_build_plans_synthetic_rig(tmp_path: Path) -> None:
    artifact = build_calibration_check(_bag(tmp_path))

    roles = {topic.topic: topic.role for topic in artifact.topics}
    assert roles == {
        "/camera/image_raw": "camera",
        "/gnss/fix": "gnss",
        "/imu": "imu",
        "/lidar_front/points": "lidar",
        "/lidar_rear/points": "lidar",
        "/tf_static": "tf_static",
    }
    assert all(
        topic.mapped_frame is not None for topic in artifact.topics if topic.role != "tf_static"
    )
    assert artifact.frame_tree.roots == ["base_link"]
    assert artifact.candidate_sources[0].kind == "bag_tf_static"
    assert artifact.candidate_sources[0].frame_count == 5
    statuses = {(pair.pair, pair.status) for pair in artifact.pairs}
    assert ("lidar-lidar", "planned") in statuses
    assert ("imu-lidar", "planned") in statuses
    assert ("lidar-vehicle", "planned") in statuses
    assert ("ins-lidar", "skipped") in statuses
    assert "1 bag topic(s) without a sensor role" in artifact.provenance.notes[1]


def test_vehicle_frame_option_and_frame_map(tmp_path: Path) -> None:
    bag = write_check_bag(
        tmp_path / "rig.db3",
        tf_messages=[default_tf()],
        sensor_frames={
            "/imu": ("sensor_msgs/msg/Imu", "weird_imu"),
            "/points": ("sensor_msgs/msg/PointCloud2", "lidar_front"),
        },
    )

    plain = build_calibration_check(bag, vehicle_frame="chassis")
    mapped = build_calibration_check(bag, frame_overrides={"/imu": "imu_link"})

    assert {p.reason_code for p in plain.pairs if p.pair == "imu-vehicle"} == {"no_vehicle_frame"}
    assert {p.reason_code for p in plain.pairs if p.pair == "imu-lidar"} == {"frame_not_in_tree"}
    (imu_lidar,) = [p for p in mapped.pairs if p.pair == "imu-lidar"]
    assert imu_lidar.status == "planned"
    assert {t.topic: t.frame_source for t in mapped.topics}["/imu"] == "override"


def test_tf_file_supplies_candidates_when_bag_has_none(tmp_path: Path) -> None:
    bag = write_check_bag(tmp_path / "rig.db3", include_tf_topic=False)
    frames = tmp_path / "frames.yaml"
    frames.write_text(
        yaml.safe_dump(
            {
                "schema_version": "slac.check_frames/v0.1",
                "frames": [
                    {"name": "imu_link", "parent": "base_link", "translation_m": [0, 0, 0]},
                    {"name": "lidar_front", "parent": "imu_link", "translation_m": [1, 0, 0]},
                ],
            }
        ),
        encoding="utf-8",
    )

    without = build_calibration_check(bag)
    with_file = build_calibration_check(bag, tf_files=[frames])

    assert without.candidate_sources == []
    assert {p.reason_code for p in without.pairs if p.pair == "imu-lidar"} == {
        "no_candidate_calibration"
    }
    assert with_file.candidate_sources[0].kind == "frames_yaml"
    assert with_file.candidate_sources[0].sha256 is not None
    (pair,) = [p for p in with_file.pairs if p.pair == "imu-lidar" and p.status == "planned"]
    assert pair.status == "planned" and pair.candidate_transform is not None
    assert pair.candidate_transform.translation_m == [1.0, 0.0, 0.0]


def test_bag_digest_changes_with_content(tmp_path: Path) -> None:
    first = write_check_bag(tmp_path / "a.db3", tf_messages=[default_tf()])
    second = write_check_bag(
        tmp_path / "b.db3",
        tf_messages=[[("base_link", "imu_link", (0.0, 0.0, 9.0), IDENTITY)]],
    )

    digest_a, scope, storage = bag_input_digest(first)

    assert storage == "sqlite3"
    assert "first 64 MiB" in scope
    assert digest_a != bag_input_digest(second)[0]
    assert digest_a == bag_input_digest(first)[0]


def test_parse_frame_map() -> None:
    assert parse_frame_map(["/a=/x", "/b = y"]) == {"/a": "x", "/b": "y"}
    with pytest.raises(DatasetError, match="TOPIC=FRAME"):
        parse_frame_map(["nonsense"])


# ---------------------------------------------------------------------- CLI


def test_cli_plan_table_output_and_validate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bag = _bag(tmp_path)
    output = tmp_path / "check.json"

    exit_code = main(["check", str(bag), "--plan", "--output", str(output)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "calibrex check (plan)" in captured.out
    assert "lidar-lidar" in captured.out and "planned" in captured.out
    assert "missing_topic" in captured.out
    assert "not wired" not in captured.err
    report = validate_file(output, "calibration-check")
    assert report.kind == "calibration-check"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["provenance"]["command"][:3] == ["calibrex", "check", str(bag)]
    # auto-detected kind
    assert validate_file(output, "auto").kind == "calibration-check"


def test_cli_without_plan_says_solvers_are_not_wired(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["check", str(_bag(tmp_path))])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "solvers are not wired yet" in captured.err
    assert "calibrex check (plan)" in captured.out


def test_cli_json_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        ["check", str(_bag(tmp_path)), "--plan", "--json", "--vehicle-frame", "base_link"]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["schema_version"] == "slac.calibration_check/v0.1"
    assert payload["summary"]["runnable_count"] > 0


def test_cli_reports_conflicting_tf_static(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bag = write_check_bag(
        tmp_path / "bad.db3",
        tf_messages=[
            [("base_link", "imu_link", (0.0, 0.0, 0.0), IDENTITY)],
            [("chassis", "imu_link", (0.0, 0.0, 0.0), IDENTITY)],
        ],
    )

    exit_code = main(["check", str(bag), "--plan"])

    assert exit_code == 2
    assert "conflicting parents" in capsys.readouterr().err


def test_cli_missing_bag_and_bad_frame_map(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["check", str(tmp_path / "nope"), "--plan"]) == 2
    assert main(["check", str(_bag(tmp_path)), "--plan", "--frame-map", "oops"]) == 2
    assert "error" in capsys.readouterr().err
