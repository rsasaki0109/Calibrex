"""The browser check page's Python side: the plan API, the sample bag and the read-only open."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from calibrex.check.browser import (
    EventProgress,
    check_command,
    locate_bag,
    plan_bag_for_browser,
    plan_request_json,
    run_request_json,
)
from calibrex.check.runner import BAG_DIGEST_SCOPE, bag_input_digest
from calibrex.core.calibration_check import CalibrationCheckArtifact
from calibrex.core.exceptions import DatasetError
from calibrex.data.rosbag2 import iter_messages, open_sqlite_readonly

ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import build_check_sample_bag as sample  # noqa: E402

COMMITTED_SAMPLE = ROOT / "docs" / "app" / "samples" / "check_sample"


def _messages(bag: Path) -> list[tuple[str, str, int, bytes]]:
    return [
        (conn.topic, conn.message_type, stamp, bytes(data))
        for conn, stamp, data in iter_messages(bag)
    ]


def test_sample_bag_is_deterministic_and_small(tmp_path: Path) -> None:
    first = sample.build_sample_bag(tmp_path / "check_sample")
    second = sample.build_sample_bag(tmp_path / "again" / "check_sample")
    assert _messages(first) == _messages(second)
    assert (first / "metadata.yaml").read_bytes() == (second / "metadata.yaml").read_bytes()
    assert sum(path.stat().st_size for path in first.iterdir()) < 400 * 1024


def test_committed_sample_matches_the_generator(tmp_path: Path) -> None:
    rebuilt = sample.build_sample_bag(tmp_path / "check_sample")
    assert _messages(COMMITTED_SAMPLE) == _messages(rebuilt)
    assert (COMMITTED_SAMPLE / "metadata.yaml").read_bytes() == (
        rebuilt / "metadata.yaml"
    ).read_bytes()


def test_plan_of_the_sample_bag() -> None:
    result = plan_bag_for_browser(COMMITTED_SAMPLE, bag_label="check_sample")
    artifact = CalibrationCheckArtifact.model_validate(result["artifact"])
    assert artifact.plan_only and artifact.bag.path == "check_sample"
    assert [source.kind for source in artifact.candidate_sources] == ["bag_tf_static"]
    assert artifact.candidate_sources[0].path == "check_sample"
    assert artifact.frame_tree.roots == ["base_link"]
    assert {t.topic: t.role for t in artifact.topics} == {
        "/camera/image_raw": "camera",
        "/gnss/fix": "gnss",
        "/imu/data": "imu",
        "/lidar_front/points": "lidar",
        "/lidar_rear/points": "lidar",
        "/tf_static": "tf_static",
    }
    planned = sorted((p.pair, *p.sensors) for p in artifact.pairs if p.status == "planned")
    assert planned == [
        ("camera-focal", "camera_optical", "imu_link"),
        ("camera-imu", "camera_optical", "imu_link"),
        ("gnss-imu", "gnss_link", "imu_link"),
        ("gnss-lidar", "gnss_link", "lidar_front"),
        ("gnss-lidar", "gnss_link", "lidar_rear"),
        ("imu-lidar", "imu_link", "lidar_front"),
        ("imu-lidar", "imu_link", "lidar_rear"),
        ("lidar-lidar", "lidar_front", "lidar_rear"),
    ]
    skipped = {p.pair: p.reason_code for p in artifact.pairs if p.status == "skipped"}
    assert skipped == {
        "lidar-vehicle": "no_vehicle_frame",
        "imu-vehicle": "no_vehicle_frame",
        "ins-lidar": "missing_topic",
        "lidar-wheel_odometry": "missing_topic",
    }
    assert artifact.summary.runnable_count == 8
    assert artifact.provenance.command == ["calibrex", "check", "check_sample", "--plan"]
    assert any("browser" in note for note in artifact.provenance.notes)
    assert result["command"] == "calibrex check check_sample --output check.json --html check.html"
    assert result["plan_command"] == "calibrex check check_sample --plan"


def test_vehicle_frame_enables_vehicle_pairs_and_shows_in_the_command() -> None:
    result = plan_bag_for_browser(
        COMMITTED_SAMPLE,
        vehicle_frame=" base_link ",
        frame_map=["/imu/data = imu_link"],
        bag_label="my bag",
    )
    artifact = CalibrationCheckArtifact.model_validate(result["artifact"])
    assert artifact.vehicle_frame == "base_link"
    assert {p.pair for p in artifact.pairs if p.status == "planned"} >= {
        "lidar-vehicle",
        "imu-vehicle",
    }
    assert result["command"] == (
        "calibrex check 'my bag' --vehicle-frame base_link --frame-map '/imu/data = imu_link'"
        " --output check.json --html check.html"
    )


def test_plan_matches_the_cli_plan_apart_from_labels() -> None:
    from calibrex.check import build_calibration_check

    reference = build_calibration_check(COMMITTED_SAMPLE).model_dump(mode="json", exclude_none=True)
    result = plan_bag_for_browser(COMMITTED_SAMPLE, bag_label=str(COMMITTED_SAMPLE))
    artifact = result["artifact"]
    for key in ("topics", "pairs", "frame_tree", "summary", "bag"):
        assert artifact[key] == reference[key], key
    assert artifact["provenance"]["input_sha256"] == reference["provenance"]["input_sha256"]


def test_calibration_files_keep_their_names_and_override_tf_static(tmp_path: Path) -> None:
    frames = tmp_path / "x" / "rig.yaml"
    frames.parent.mkdir()
    frames.write_text(
        yaml.safe_dump(
            {
                "schema_version": "slac.check_frames/v0.1",
                "frames": [
                    {
                        "name": "lidar_front",
                        "parent": "base_link",
                        "translation_m": [1.0, 0.0, 0.5],
                        "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = plan_bag_for_browser(
        COMMITTED_SAMPLE, tf_files=[frames], tf_labels=["rig.yaml"], bag_label="check_sample"
    )
    sources = result["artifact"]["candidate_sources"]
    assert [(s["kind"], s["path"]) for s in sources] == [
        ("bag_tf_static", "check_sample"),
        ("frames_yaml", "rig.yaml"),
    ]
    assert result["artifact"]["frame_tree"]["overrides"]
    assert "--tf rig.yaml" in result["command"]
    with pytest.raises(DatasetError, match="one entry per tf file"):
        plan_bag_for_browser(COMMITTED_SAMPLE, tf_files=[frames], tf_labels=[])


def test_check_command_quotes_labels() -> None:
    assert check_command("a b", tf_labels=["c'd.yaml"], plan=True) == [
        "calibrex",
        "check",
        "a b",
        "--tf",
        "c'd.yaml",
        "--plan",
    ]


def test_locate_bag_directory_and_bare_files(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="no rosbag2 data"):
        locate_bag(tmp_path)
    (tmp_path / "a.mcap").write_bytes(b"")
    assert locate_bag(tmp_path) == tmp_path / "a.mcap"
    (tmp_path / "b.db3").write_bytes(b"")
    with pytest.raises(DatasetError, match="several storage files"):
        locate_bag(tmp_path)
    (tmp_path / "metadata.yaml").write_text("x: 1\n", encoding="utf-8")
    assert locate_bag(tmp_path) == tmp_path


def test_plan_request_json_reports_errors_instead_of_raising(tmp_path: Path) -> None:
    ok = json.loads(
        plan_request_json(json.dumps({"bag_dir": str(COMMITTED_SAMPLE), "bag_label": "s"}))
    )
    assert ok["ok"] and ok["artifact"]["summary"]["runnable_count"] == 8
    empty = json.loads(plan_request_json(json.dumps({"bag_dir": str(tmp_path)})))
    assert not empty["ok"] and "no rosbag2 data" in empty["error"]
    (tmp_path / "bad.db3").write_bytes(b"this is not a sqlite database" * 20)
    bad = json.loads(plan_request_json(json.dumps({"bag_dir": str(tmp_path)})))
    assert not bad["ok"] and bad["error"]
    mapping = json.loads(
        plan_request_json(
            json.dumps({"bag_dir": str(COMMITTED_SAMPLE), "frame_map": ["no-equals-sign"]})
        )
    )
    assert not mapping["ok"] and "TOPIC=FRAME" in mapping["error"]


def test_plan_of_a_bare_mcap(tmp_path: Path) -> None:
    from tests.unit.test_rosbag2 import _write_mcap_bag

    messages = []
    for conn, stamp, data in iter_messages(COMMITTED_SAMPLE):
        schema_id = {"sensor_msgs/msg/Imu": 1}.get(conn.message_type, 2 + conn.topic_id)
        messages.append(
            (conn.topic_id, schema_id, conn.topic, conn.message_type, stamp, bytes(data))
        )
    _write_mcap_bag(tmp_path / "sample.mcap", messages=messages)
    result = json.loads(plan_request_json(json.dumps({"bag_dir": str(tmp_path)})))
    assert result["ok"] and result["artifact"]["bag"]["storage_identifier"] == "mcap"
    assert result["artifact"]["summary"]["runnable_count"] == 8


def test_plan_path_imports_without_opencv_open3d_or_ros(tmp_path: Path) -> None:
    """The Pyodide plan path must not need opencv, open3d, matplotlib or any ROS package."""

    script = (
        "import importlib.abc, sys\n"
        "class Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path, target=None):\n"
        "        if name.split('.')[0] in {'cv2', 'open3d', 'matplotlib', 'rclpy', 'rosbag2_py',\n"
        "                                  'PIL', 'mcap'}:\n"
        "            raise ImportError('blocked ' + name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "from calibrex.check.browser import plan_request_json\n"
        "import json\n"
        f"r = json.loads(plan_request_json(json.dumps({{'bag_dir': {str(COMMITTED_SAMPLE)!r}}})))\n"
        "assert r['ok'], r\n"
        "assert r['artifact']['summary']['runnable_count'] == 8\n"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env, check=False
    )
    assert completed.returncode == 0, completed.stderr


def test_open_sqlite_readonly_matches_plain_open_and_handles_odd_paths(tmp_path: Path) -> None:
    odd = tmp_path / "a b#c?d%41"
    odd.mkdir()
    db = odd / "x.db3"
    connection = sqlite3.connect(db)
    connection.execute("CREATE TABLE t(v INTEGER)")
    connection.execute("INSERT INTO t VALUES (7)")
    connection.commit()
    connection.close()
    opened = open_sqlite_readonly(db)
    try:
        assert opened.execute("SELECT v FROM t").fetchall() == [(7,)]
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            opened.execute("INSERT INTO t VALUES (8)")
    finally:
        opened.close()
    corrupt = tmp_path / "corrupt.db3"
    corrupt.write_bytes(b"not a database" * 100)
    with pytest.raises(sqlite3.DatabaseError):
        open_sqlite_readonly(corrupt)


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0, reason="needs an unprivileged user"
)
def test_open_sqlite_readonly_reads_a_wal_bag_on_a_read_only_directory(tmp_path: Path) -> None:
    """The browser mounts the bag read-only (WORKERFS): no -shm file can be created there."""

    directory = tmp_path / "ro"
    directory.mkdir()
    db = directory / "wal.db3"
    connection = sqlite3.connect(db)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("CREATE TABLE t(v INTEGER)")
    connection.execute("INSERT INTO t VALUES (7)")
    connection.commit()
    connection.close()  # a clean close checkpoints and removes -wal and -shm
    assert not list(directory.glob("*-wal")) and not list(directory.glob("*-shm"))
    db.chmod(0o444)
    directory.chmod(0o555)
    try:
        opened = open_sqlite_readonly(db)
        try:
            assert opened.execute("SELECT v FROM t").fetchall() == [(7,)]
        finally:
            opened.close()
    finally:
        directory.chmod(0o755)


def test_bag_digest_reads_in_chunks_but_hashes_the_same_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from calibrex.check import runner

    storage = tmp_path / "bag.mcap"
    storage.write_bytes(bytes(range(256)) * 100)
    monkeypatch.setattr(runner, "BAG_DIGEST_PREFIX_BYTES", 10_000)
    monkeypatch.setattr(runner, "_DIGEST_CHUNK_BYTES", 777)
    digest, scope, storage_id = bag_input_digest(storage)
    expected = hashlib.sha256()
    expected.update(b"bag.mcap")
    expected.update(b"25600")
    expected.update(storage.read_bytes()[:10_000])
    assert digest == expected.hexdigest()
    assert (scope, storage_id) == (BAG_DIGEST_SCOPE, "mcap")


def test_run_path_imports_without_opencv_open3d_or_ros() -> None:
    """The non-camera run path (check and estimate) needs only numpy, scipy and the bag."""

    script = (
        "import importlib.abc, sys\n"
        "class Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path, target=None):\n"
        "        if name.split('.')[0] in {'cv2', 'open3d', 'matplotlib', 'rclpy', 'rosbag2_py',\n"
        "                                  'PIL', 'mcap'}:\n"
        "            raise ImportError('blocked ' + name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "from calibrex.check.browser import run_request_json\n"
        "import json\n"
        "for mode in ('check', 'estimate'):\n"
        f"    request = {{'bag_dir': {str(COMMITTED_SAMPLE)!r}, 'mode': mode,\n"
        "               'pairs': ['lidar-lidar', 'gnss-lidar'], 'max_duration_s': 5}\n"
        "    r = json.loads(run_request_json(json.dumps(request)))\n"
        "    assert r['ok'], r\n"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env, check=False
    )
    assert completed.returncode == 0, completed.stderr


def test_run_request_check_mode_reports_progress_and_validates() -> None:
    events: list[str] = []
    request = {
        "bag_dir": str(COMMITTED_SAMPLE),
        "bag_label": "check_sample",
        "vehicle_frame": "base_link",
        "pairs": ["lidar-lidar", "lidar-vehicle"],
        "max_duration_s": 5,
    }
    reply = json.loads(run_request_json(json.dumps(request), events.append))
    assert reply["ok"] and reply["mode"] == "check"
    artifact = CalibrationCheckArtifact.model_validate(reply["artifact"])
    assert not artifact.plan_only and artifact.bag.path == "check_sample"
    assert artifact.options is not None and artifact.options.max_duration_s == 5.0
    assert any("browser" in note for note in artifact.provenance.notes)
    assert "calibrex check (run)" in reply["text"] and "<html" in reply["html"].lower()
    assert reply["command"] == (
        "calibrex check check_sample --vehicle-frame base_link --pairs lidar-lidar,lidar-vehicle "
        "--max-duration-s 5 --output check.json --html check.html"
    )
    kinds = [json.loads(item)["event"] for item in events]
    assert kinds[0] == "run_started" and kinds[-1] == "run_finished"
    assert "pair_started" in kinds
    ran = {pair.pair for pair in artifact.pairs if pair.status != "skipped"}
    assert ran <= {"lidar-lidar", "lidar-vehicle"}


def test_run_request_estimate_mode_returns_the_exports() -> None:
    request = {
        "bag_dir": str(COMMITTED_SAMPLE),
        "mode": "estimate",
        "pairs": ["lidar-lidar"],
        "max_duration_s": 5,
    }
    reply = json.loads(run_request_json(json.dumps(request)))
    assert reply["ok"] and reply["mode"] == "estimate"
    assert reply["artifact"]["schema_version"] == "slac.bag_estimate/v0.1"
    assert reply["command"].startswith("calibrex estimate ")
    assert reply["text"].startswith("calibrex estimate")
    assert all(isinstance(text, str) for text in reply["files"].values())


def test_run_request_reports_errors_instead_of_raising(tmp_path: Path) -> None:
    reply = json.loads(run_request_json(json.dumps({"bag_dir": str(tmp_path)})))
    assert reply["ok"] is False and "no rosbag2 data" in reply["error"]
    assert json.loads(run_request_json("not json"))["ok"] is False


def test_event_progress_throttles_ticks() -> None:
    now = [0.0]
    events: list[dict[str, object]] = []
    progress = EventProgress(
        lambda text: events.append(json.loads(text)), clock=lambda: now[0], tick_interval_s=1.0
    )
    progress.pair_started(1, 2, "lidar-vehicle (a / b)")
    progress.set_scan_total(100)
    progress.stage("odometry")
    for step in range(10):
        now[0] += 0.3
        progress.tick(step, None)
    ticks = [event for event in events if event["event"] == "tick"]
    assert 1 <= len(ticks) <= 4 and ticks[0]["total"] == 100
    progress.pair_finished("lidar-vehicle (a / b)", "pass", 3.0, None)
    assert events[-1] == {
        "event": "pair_finished",
        "label": "lidar-vehicle (a / b)",
        "status": "pass",
        "runtime_s": 3.0,
    }


def test_topic_kind_and_camera_reach_the_command_and_reject_bad_values() -> None:
    from calibrex.check.browser import run_command

    command = run_command(
        "check",
        "kitti",
        vehicle_frame="base_link",
        topic_kind=["/oxts/twist=wheel"],
        camera="/cam0/image_raw",
        pairs=["lidar-wheel_odometry"],
    )
    assert "--topic-kind /oxts/twist=wheel" in command
    assert "--camera /cam0/image_raw" in command
    plan = plan_request_json(
        json.dumps({"bag_dir": str(COMMITTED_SAMPLE), "topic_kind": ["/x=nonsense"]})
    )
    assert json.loads(plan)["ok"] is False
    request = {
        "bag_dir": str(COMMITTED_SAMPLE),
        "vehicle_frame": "base_link",
        "topic_kind": ["/sample/odom=wheel"],
        "camera": " /sample/camera ",
        "pairs": ["imu-vehicle"],
        "max_duration_s": 2,
    }
    # The sample bag has no odometry topic: the override reaches the planner, which says so.
    reply = json.loads(run_request_json(json.dumps(request)))
    assert reply["ok"] is False and "/sample/odom" in reply["error"]
    del request["topic_kind"]
    reply = json.loads(run_request_json(json.dumps(request)))
    assert reply["ok"], reply
    assert "--camera /sample/camera" in reply["command"]
