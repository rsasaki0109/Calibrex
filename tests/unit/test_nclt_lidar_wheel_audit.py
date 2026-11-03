"""The NCLT lidar-wheel audit harness's scoring and pass rule, on made-up evidence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tools.nclt_lidar_wheel_audit import (
    SESSIONS,
    check_command,
    score_session,
    verdicts,
)
from tools.nclt_to_rosbag2 import reference_base_velodyne


def _evidence(rotation_deg: float, lever_m: float, status: str = "converged") -> dict:
    reference = reference_base_velodyne()
    error = Rotation.from_rotvec([np.radians(rotation_deg), 0.0, 0.0])
    turned = error * Rotation.from_matrix(reference[:3, :3])
    return {
        "solver_status": status,
        "policy_status": "inconclusive",
        "rotation_quat_xyzw": turned.as_quat().tolist(),
        "lever_m": lever_m,
        "speed_scale": 0.98,
        "time_offset_s": 0.09,
        "parameters": [{"name": "roll", "std_reported": 0.3}],
    }


def test_score_against_nclt_calibration() -> None:
    inconclusive = {"pair": "lidar-wheel_odometry", "status": "inconclusive", "axes": []}

    good = score_session(_evidence(0.3, 0.1), inconclusive)
    assert good["accurate"] and good["rotation_error_deg"] == pytest.approx(0.3, abs=1e-6)
    assert good["lever_error_m"] == pytest.approx(0.098)
    assert good["judged_axes"] == []
    assert not score_session(_evidence(0.6, 0.0), inconclusive)["accurate"]
    assert not score_session(_evidence(0.0, 0.25), inconclusive)["accurate"]
    unsolved = score_session(_evidence(0.0, 0.0, status="insufficient_motions"), inconclusive)
    assert not unsolved["converged"] and not unsolved["accurate"]


def test_verdicts() -> None:
    passed = {"accurate": True, "check_status": "pass", "judged_axes": ["yaw"]}
    all_good = dict.fromkeys(SESSIONS, passed)
    assert verdicts(all_good)["accuracy"]["verdict"] == "supported"
    assert verdicts(all_good)["check_honesty"]["verdict"] == "supported"
    first = next(iter(SESSIONS))
    one_failed = {**all_good, first: {"accurate": False, "check_status": "fail"}}
    assert verdicts(one_failed)["accuracy"]["verdict"] == "refuted"
    assert verdicts(one_failed)["check_honesty"]["verdict"] == "refuted"
    crashed = {**all_good, first: {"error": "RuntimeError()", "accurate": False}}
    assert verdicts(crashed)["accuracy"]["verdict"] == "refuted"
    assert verdicts(crashed)["check_honesty"]["verdict"] == "supported"


def test_the_check_runs_only_the_pair_without_a_cache() -> None:
    command = check_command(Path("bag"), Path("out.json"), Path("evidence"))
    assert command[command.index("--pairs") + 1] == "lidar-wheel_odometry"
    assert "--no-cache" in command and "--tf" not in command
    assert command[command.index("--topic-kind") + 1] == "/wheel_odom=wheel"
