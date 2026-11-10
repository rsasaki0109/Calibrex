"""The radar-imu v2 pre-registration agrees with the frozen harness it names."""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from tools import doer_radar_imu_v2_audit as harness

ROOT = Path(__file__).resolve().parents[2]
PLAN = yaml.safe_load(
    (ROOT / "docs/benchmarks/doer_radar_imu_v2_preregistration.yaml").read_text(encoding="utf-8")
)


def test_the_archive_is_pinned_as_the_harness_reads_it() -> None:
    archive = PLAN["sessions"]["evaluation"]
    assert archive["google_drive_id"] == harness.ARCHIVE_ID
    assert archive["archive"] == harness.ARCHIVE_NAME
    assert archive["size_bytes"] == harness.ARCHIVE_SIZE
    assert archive["expected_bags"] == harness.EXPECTED_BAGS
    assert archive["expected_radars"] == harness.EXPECTED_RADARS


def test_the_gates_are_the_harness_constants() -> None:
    claims = PLAN["claims"]
    honesty = claims["doer-radar-imu-check-v2-honesty"]["gates"]
    assert honesty["max_reference_fails"] == harness.MAX_REFERENCE_FAILS == 0
    assert honesty["min_reference_pass_fraction"] == harness.MIN_REFERENCE_PASS_FRACTION
    coverage = claims["doer-radar-imu-check-v2-coverage"]["gates"]
    assert coverage["min_reference_judged_fraction"] == harness.MIN_REFERENCE_JUDGED_FRACTION
    detection = claims["doer-radar-imu-check-v2-detection"]["gates"]
    assert detection["max_turned_unflagged"] == harness.MAX_TURNED_UNFLAGGED == 0
    assert detection["max_judged_turns_missed"] == harness.MAX_JUDGED_TURNS_MISSED == 0
    assert detection["min_detecting_case_fraction"] == harness.MIN_DETECTING_CASE_FRACTION


def test_the_method_is_frozen_with_its_commit() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", str(PLAN["frozen_code_commit"]))
    check = PLAN["method"]["check"].replace("\n", " ")
    assert f"--pairs {harness.PAIRS}" in check and f"--fail-on {harness.FAIL_ON}" in check
    assert f"{harness.IMU_TOPIC}=imu" in check
    assert f"turned {harness.TURN_DEG:g} deg" in PLAN["method"]["candidates"]
