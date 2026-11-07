"""The TUM-VI camera-imu v2 pre-registration agrees with the frozen harness it names."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml
from tools import tumvi_camera_imu_audit as v1
from tools import tumvi_camera_imu_v2_audit as harness

ROOT = Path(__file__).resolve().parents[2]
PLAN = yaml.safe_load(
    (ROOT / "docs/benchmarks/tumvi_camera_imu_v2_preregistration.yaml").read_text(encoding="utf-8")
)


def test_every_sequence_is_pinned_as_the_harness_fetches_it() -> None:
    sessions = PLAN["sessions"]["evaluation"]
    assert list(sessions) == list(harness.SESSIONS)
    for name, pin in sessions.items():
        size, etag = harness.SESSIONS[name]
        assert pin == {
            "bag": f"calibrated/512_16/dataset-{name}_512_16.bag",
            "size_bytes": size,
            "etag": etag,
        }
    spent = set(PLAN["sessions"]["development_spent"])
    assert set(v1.SESSIONS) <= spent and not spent & set(sessions)
    assert [size for size, _etag in harness.SESSIONS.values()] == sorted(
        size for size, _etag in harness.SESSIONS.values()
    )


def test_the_reference_is_the_pinned_camchain() -> None:
    camchain = ROOT / PLAN["reference"]["camchain"]
    digest = hashlib.sha256(camchain.read_bytes()).hexdigest()
    assert digest == PLAN["reference"]["camchain_sha256"] == v1.CAMCHAIN_SHA256


def test_the_gates_are_the_harness_constants() -> None:
    honesty = PLAN["claims"]["tumvi-camera-imu-check-v2-honesty"]["gates"]
    assert honesty["max_kalibr_fails"] == harness.MAX_KALIBR_FAILS == 0
    assert honesty["min_kalibr_full_pass_fraction"] == harness.MIN_KALIBR_FULL_PASS_FRACTION
    detection = PLAN["claims"]["tumvi-camera-imu-check-v2-detection"]["gates"]
    assert detection["max_turned_unflagged"] == harness.MAX_TURNED_UNFLAGGED == 0
    assert detection["min_turned_detected_fraction"] == harness.MIN_TURNED_DETECTED_FRACTION


def test_the_method_is_frozen_with_its_commit() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", str(PLAN["frozen_code_commit"]))
    check = PLAN["method"]["check"].replace("\n", " ")
    assert f"--pairs {harness.PAIRS}" in check and f"--fail-on {harness.FAIL_ON}" in check
