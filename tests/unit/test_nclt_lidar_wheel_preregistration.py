"""The NCLT lidar-wheel pre-registration agrees with the frozen harness it names."""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from tools import nclt_lidar_wheel_audit as harness
from tools import nclt_to_rosbag2 as converter

ROOT = Path(__file__).resolve().parents[2]
PLAN = yaml.safe_load(
    (ROOT / "docs/benchmarks/nclt_lidar_wheel_preregistration.yaml").read_text(encoding="utf-8")
)


def test_every_session_is_pinned_as_the_harness_fetches_it() -> None:
    sessions = PLAN["sessions"]["evaluation"]
    # YAML 1.1 reads an unquoted 2012-02-02 as a date: every key must stay a string
    assert list(sessions) == list(harness.SESSIONS)
    for session, pins in sessions.items():
        sensor_size, sensor_etag, velodyne_size, velodyne_etag = harness.SESSIONS[session]
        assert pins["sensor"] == {
            "archive": f"sensor_data/{session}_sen.tar.gz",
            "size_bytes": sensor_size,
            "etag": sensor_etag,
        }
        assert pins["velodyne"] == {
            "archive": f"velodyne_data/{session}_vel.tar.gz",
            "size_bytes": velodyne_size,
            "etag": velodyne_etag,
        }
        assert re.fullmatch(r"[0-9a-f]{32}-\d+", velodyne_etag)
    assert PLAN["sessions"]["development_spent"] == ["2012-01-08"]
    assert "2012-01-08" not in sessions


def test_the_gates_are_the_harness_constants() -> None:
    accuracy = PLAN["claims"]["nclt-lidar-wheel-accuracy"]["gates"]
    assert accuracy["max_rotation_error_deg"] == harness.MAX_ROTATION_ERROR_DEG == 0.5
    assert accuracy["max_position_error_m"] == harness.MAX_POSITION_ERROR_M == 0.20
    assert accuracy["sessions_required"] == harness.SESSIONS_REQUIRED == 5
    honesty = PLAN["claims"]["nclt-lidar-wheel-check-honesty"]["gates"]
    assert honesty["sigma_k"] == harness.SIGMA_K
    assert honesty["rotation_floor_deg"] == harness.ROTATION_FLOOR_DEG
    assert honesty["fail_factor"] == harness.FAIL_FACTOR
    assert honesty["min_converged_sessions"] == harness.MIN_CONVERGED_SESSIONS


def test_the_method_is_frozen_with_its_commit() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", str(PLAN["frozen_code_commit"]))
    method = PLAN["method"]
    assert "60 s" in method["window"] and harness.START_S == 60.0
    assert "3600 s" in method["window"] and harness.DURATION_S == 3600.0
    assert "every 4th hit" in method["conversion"] and harness.DECIMATE == 4
    command = " ".join(harness.check_command(Path("BAG"), Path("o"), Path("e"))[3:9])
    assert command in method["estimate"].replace("\n", " ").replace("  ", " ")
    assert "0.002 m" in PLAN["claims"]["nclt-lidar-wheel-accuracy"]["text"]
    assert converter.reference_base_velodyne()[0, 3] == 0.002
