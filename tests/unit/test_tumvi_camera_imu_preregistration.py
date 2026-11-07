"""The TUM-VI camera-imu pre-registration agrees with the frozen harness it names."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml
from tools import tumvi_camera_imu_audit as harness

ROOT = Path(__file__).resolve().parents[2]
PLAN = yaml.safe_load(
    (ROOT / "docs/benchmarks/tumvi_camera_imu_preregistration.yaml").read_text(encoding="utf-8")
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
    spent = PLAN["sessions"]["development_spent"]
    assert spent == [f"room{k}" for k in range(1, 7)]
    assert not set(spent) & set(sessions)


def test_the_reference_is_the_pinned_camchain() -> None:
    reference = PLAN["reference"]
    camchain = ROOT / reference["camchain"]
    digest = hashlib.sha256(camchain.read_bytes()).hexdigest()
    assert digest == reference["camchain_sha256"] == harness.CAMCHAIN_SHA256


def test_the_gates_are_the_harness_constants() -> None:
    honesty = PLAN["claims"]["tumvi-camera-imu-check-honesty"]["gates"]
    assert honesty["max_kalibr_fails"] == harness.MAX_KALIBR_FAILS == 0
    assert honesty["min_kalibr_judged_fraction"] == harness.MIN_KALIBR_JUDGED_FRACTION
    detection = PLAN["claims"]["tumvi-camera-imu-check-detection"]["gates"]
    assert detection["max_perturbed_passes"] == harness.MAX_PERTURBED_PASSES == 0
    assert detection["min_perturbed_detected_fraction"] == harness.MIN_PERTURBED_DETECTED_FRACTION
    assert harness.ERROR_DEG == 2.0 and "2 deg" in PLAN["method"]["candidates"]
    assert harness.VARIANTS == {"kalibr": None, "rx2": "roll", "ry2": "pitch", "rz2": "yaw"}


def test_the_method_is_frozen_with_its_commit() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", str(PLAN["frozen_code_commit"]))
    assert harness.PAIRS in PLAN["method"]["check"].replace("\n", " ")
    command = harness.check_command(Path("BAG"), Path("C"), Path("O"))
    assert "--no-cache" in command and command[command.index("--pairs") + 1] == harness.PAIRS
