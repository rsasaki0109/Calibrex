"""Next-step hints and the frames template of ``calibrex check``."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from tests.unit.check_fixtures import write_check_bag

from calibrex.check import build_calibration_check, format_check_table, load_tf_file
from calibrex.check.hints import (
    ALL_REASON_CODES,
    REASON_HINTS,
    collect_next_steps,
    frames_template,
    pair_hint,
    topic_hint,
)
from calibrex.cli.main import main
from calibrex.core.calibration_check import CheckPairRecord, CheckTopicRecord
from calibrex.core.exceptions import DatasetError


def _skipped(code: str, reason: str = "x") -> CheckPairRecord:
    return CheckPairRecord(
        pair="ins-lidar",
        status="skipped",
        reason_code=code,  # type: ignore[arg-type]
        reason=reason,
    )


def test_every_reason_code_has_a_hint() -> None:
    assert set(REASON_HINTS) == set(ALL_REASON_CODES)
    assert all(hint.strip() for hint in REASON_HINTS.values())
    for code in ALL_REASON_CODES:
        assert pair_hint(_skipped(code))


def test_pair_hint_specifics() -> None:
    assert "--tf FILE" in (pair_hint(_skipped("no_candidate_calibration")) or "")
    assert "--vehicle-frame base_link" in (pair_hint(_skipped("no_vehicle_frame")) or "")
    missing = pair_hint(_skipped("missing_topic", "no topic for: gnss"))
    assert missing is not None and "sensor_msgs/msg/NavSatFix" in missing
    unknown = CheckTopicRecord(
        topic="/odom", message_type="x", role="odometry", odometry_kind="unknown"
    )
    classify = pair_hint(_skipped("missing_topic", "no topic for: ins"), [unknown])
    assert classify is not None and "--topic-kind /odom=wheel|ins" in classify
    assert pair_hint(_skipped("estimator_failed").model_copy(update={"status": "fail"})) is None


def test_topic_hint_suggests_frame_map() -> None:
    topic = CheckTopicRecord(topic="/imu", message_type="x", role="imu", header_frame_id="imu_link")
    hint = topic_hint(topic, tree_frames=["base_link", "imu"])
    assert hint is not None and "--frame-map /imu=base_link" in hint and "imu_link" in hint
    mapped = topic.model_copy(update={"mapped_frame": "imu"})
    assert topic_hint(mapped, tree_frames=["imu"]) is None


def test_plan_text_ends_with_deduplicated_next_steps(tmp_path: Path) -> None:
    bag = write_check_bag(tmp_path / "bag.db3", include_tf_topic=False)
    artifact = build_calibration_check(bag)

    steps = collect_next_steps(artifact)
    assert len(steps) == len(set(steps))
    assert sum("--tf FILE" in step for step in steps) == 1
    table = format_check_table(artifact)
    assert table.index("next steps:") > table.index("summary:")
    assert "--write-frames-template" in table
    assert "imu_link" in table


def test_template_round_trip_and_refusal(tmp_path: Path) -> None:
    bag = write_check_bag(tmp_path / "bag.db3", include_tf_topic=False)
    template = tmp_path / "frames.yaml"
    assert main(["check", str(bag), "--plan", "--write-frames-template", str(template)]) == 0
    text = template.read_text(encoding="utf-8")
    assert "TODO" in text and "NOT a calibration" in text
    assert yaml.safe_load(text)["schema_version"] == "slac.check_frames_template/v0.1"

    with pytest.raises(DatasetError, match="TODO placeholder values"):
        load_tf_file(template)

    # editing only the version line is not enough: TODO values are refused
    version_only = tmp_path / "version_only.yaml"
    version_only.write_text(
        text.replace("slac.check_frames_template/v0.1", "slac.check_frames/v0.1"),
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match=r"TODO placeholder values for frame.*imu_link"):
        load_tf_file(version_only)
    # and so is a filled-in file that kept the template version marker
    payload = yaml.safe_load(text)
    for entry in payload["frames"]:
        entry["translation_m"] = [0.1, 0.0, 0.2]
        entry["rotation_quat_xyzw"] = [0.0, 0.0, 0.0, 1.0]
    marked = tmp_path / "marked.yaml"
    marked.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(DatasetError, match="template, not a calibration"):
        load_tf_file(marked)

    payload["schema_version"] = "slac.check_frames/v0.1"
    edited = tmp_path / "edited.yaml"
    edited.write_text(yaml.safe_dump(payload), encoding="utf-8")
    source = load_tf_file(edited)
    assert source.kind == "frames_yaml"
    assert {edge.child for edge in source.edges} >= {"imu_link", "lidar_front", "camera_optical"}
    assert source.hints.topic_frames["/imu"] == "imu_link"
    assert build_calibration_check(bag, tf_files=[edited]).candidate_sources


def test_template_without_header_frames_names_a_placeholder() -> None:
    assert "sensor_frame" in frames_template([])
