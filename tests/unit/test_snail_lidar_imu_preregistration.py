"""The Snail-Radar imu-lidar pre-registration agrees with the frozen harness it names."""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from tools import snail_lidar_imu_audit as harness

ROOT = Path(__file__).resolve().parents[2]
PLAN = yaml.safe_load(
    (ROOT / "docs/benchmarks/snail_lidar_imu_preregistration.yaml").read_text(encoding="utf-8")
)


def test_every_sequence_is_pinned_as_the_harness_reads_it() -> None:
    sessions = PLAN["sessions"]
    assert list(sessions["evaluation"]) == list(harness.EVALUATION)
    for sequence, pins in sessions["evaluation"].items():
        size, etag = harness.EVALUATION[sequence]
        assert pins == {"zip": f"{sequence}.zip", "size_bytes": size, "etag": etag}
    assert len(harness.EVALUATION) == 32
    assert tuple(sessions["development_spent"]) == harness.DEVELOPMENT_SPENT
    assert list(sessions["excluded"]) == [harness.REFERENCE_FITTING]
    unseen = set(harness.EVALUATION)
    assert not unseen & {*harness.DEVELOPMENT_SPENT, harness.REFERENCE_FITTING}
    assert sessions["share"] == harness.SHARE_URL


def test_the_gates_are_the_harness_constants() -> None:
    claims = PLAN["claims"]
    honesty = claims["snail-imu-lidar-check-honesty"]["gates"]
    assert honesty["max_reference_rotation_fails"] == harness.MAX_REFERENCE_ROTATION_FAILS
    assert honesty["min_reference_pass_fraction"] == harness.MIN_REFERENCE_PASS_FRACTION
    coverage = claims["snail-imu-lidar-check-coverage"]["gates"]
    assert coverage["min_reference_judged_fraction"] == harness.MIN_REFERENCE_JUDGED_FRACTION
    detection = claims["snail-imu-lidar-check-detection"]["gates"]
    assert detection["max_judged_turns_missed"] == harness.MAX_JUDGED_TURNS_MISSED
    assert detection["min_detecting_case_fraction"] == harness.MIN_DETECTING_CASE_FRACTION
    assert "60 cases" in claims["snail-imu-lidar-check-honesty"]["text"]
    assert harness.expected_cases(list(harness.EVALUATION)) == 60


def test_the_method_is_frozen_with_its_commit() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", str(PLAN["frozen_code_commit"]))
    method = PLAN["method"]
    assert "every 4th firing" in method["conversion"] and harness.POINT_STRIDE == 4
    assert "first 300 s" in method["conversion"] and harness.MAX_DURATION_S == 300.0
    assert "3 deg" in method["candidates"] and harness.TURN_DEG == 3.0
    command = harness.check_command(
        Path("BAG"), Path("CANDIDATE"), Path("o"), Path("WORK/cache-SEQ"), list(harness.IMUS)
    )
    text = " ".join(method["check"].split())
    for flag in ("--pairs", "--max-duration-s", "--fail-on"):
        assert f"{flag} {command[command.index(flag) + 1]}" in text
    assert "--evaluation --jobs 3" in PLAN["run_recipe"]["command"]
