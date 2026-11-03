"""The NCLT lidar-wheel audit harness's scoring and pass rule, on made-up evidence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tools.nclt_lidar_wheel_audit import (
    FAIL_FACTOR,
    ROTATION_FLOOR_DEG,
    SESSIONS,
    SIGMA_K,
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
        "parameters": [{"name": name, "std_reported": 0.3} for name in ("roll", "pitch", "yaw")],
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
    # a 0.3 deg std: check fails an axis beyond 2 x max(3 x 0.3, 0.5) = 1.8 deg
    assert score_session(_evidence(1.2, 0.0), inconclusive)["beyond_fail_bound"] == []
    assert score_session(_evidence(1.9, 0.0), inconclusive)["beyond_fail_bound"] == ["roll"]


def test_verdicts() -> None:
    passed = {
        "accurate": True,
        "converged": True,
        "check_status": "pass",
        "judged_axes": ["yaw"],
        "beyond_fail_bound": [],
    }
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
    off = {
        **all_good,
        first: {**passed, "check_status": "inconclusive", "beyond_fail_bound": ["pitch"]},
    }
    assert verdicts(off)["check_honesty"]["verdict"] == "refuted"
    sessions = list(SESSIONS)
    unconverged = {**all_good, **{s: {"accurate": False, "converged": False} for s in sessions[:3]}}
    assert verdicts(unconverged)["check_honesty"]["verdict"] == "refuted"  # 2 < 3 converged


def test_the_honesty_bound_is_check_s_fail_rule() -> None:
    from calibrex.check import verdict

    assert FAIL_FACTOR == verdict.FAIL_FACTOR
    assert (SIGMA_K, ROTATION_FLOOR_DEG) == (3.0, 0.5)


def test_the_check_runs_only_the_pair_without_a_cache() -> None:
    command = check_command(Path("bag"), Path("out.json"), Path("evidence"))
    assert command[command.index("--pairs") + 1] == "lidar-wheel_odometry"
    assert "--no-cache" in command and "--tf" not in command
    assert command[command.index("--topic-kind") + 1] == "/wheel_odom=wheel"
