"""The radar-imu pre-registration agrees with the frozen harness it names."""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from tools import doer_radar_imu_audit as harness

ROOT = Path(__file__).resolve().parents[2]
PLAN = yaml.safe_load(
    (ROOT / "docs/benchmarks/doer_radar_imu_preregistration.yaml").read_text(encoding="utf-8")
)


def test_the_archive_is_pinned_as_the_harness_reads_it() -> None:
    archive = PLAN["sessions"]["evaluation"]
    assert archive["google_drive_id"] == harness.ARCHIVE_ID
    assert archive["archive"] == harness.ARCHIVE_NAME
    assert archive["size_bytes"] == harness.ARCHIVE_SIZE
    assert archive["expected_bags"] == harness.EXPECTED_ARCHIVE_BAGS


def test_the_gates_are_the_harness_constants() -> None:
    honesty = PLAN["claims"]["doer-radar-imu-check-honesty"]["gates"]
    assert honesty["max_reference_fails"] == harness.MAX_REFERENCE_FAILS == 0
    assert honesty["min_reference_pass_fraction"] == harness.MIN_REFERENCE_PASS_FRACTION
    detection = PLAN["claims"]["doer-radar-imu-check-detection"]["gates"]
    assert detection["max_turned_unflagged"] == harness.MAX_TURNED_UNFLAGGED == 0
    assert detection["max_judged_turns_missed"] == harness.MAX_JUDGED_TURNS_MISSED == 0
    assert detection["min_detecting_bag_fraction"] == harness.MIN_DETECTING_BAG_FRACTION


def test_the_method_is_frozen_with_its_commit() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", str(PLAN["frozen_code_commit"]))
    check = PLAN["method"]["check"].replace("\n", " ")
    assert f"--pairs {harness.PAIRS}" in check and f"--fail-on {harness.FAIL_ON}" in check
    assert f"{harness.RADAR_TOPIC}=radar" in check and f"{harness.IMU_TOPIC}=imu" in check
    assert f"turned {harness.TURN_DEG:g} deg" in PLAN["method"]["candidates"]


def test_the_candidates_turn_t_radar_imu_about_the_radar_axes(tmp_path: Path) -> None:
    import numpy as np
    from scipy.spatial.transform import Rotation

    calib = tmp_path / "calib.yaml"
    calib.write_text(
        "l_b_r_x: 0.06\nl_b_r_y: 0.04\nl_b_r_z: -0.04\nq_b_r_x: 0.699462156\n"
        "q_b_r_y: 0.714611909\nq_b_r_z: 0.007341567\nq_b_r_w: 0.005349194\n"
    )
    paths = harness.write_candidates(calib, tmp_path / "candidates")

    def t_radar_imu(name: str) -> Rotation:
        frame = yaml.safe_load(paths[name].read_text())["frames"][0]
        assert (frame["name"], frame["parent"]) == ("radar", "imu")
        return Rotation.from_quat(frame["rotation_quat_xyzw"]).inv()

    reference = t_radar_imu("reference")
    for name, axis in harness.VARIANTS.items():
        if axis is None:
            continue
        turn = (t_radar_imu(name) * reference.inv()).as_rotvec(degrees=True)
        expected = np.zeros(3)
        expected[harness.AXIS_INDEX[axis]] = harness.TURN_DEG
        assert np.allclose(turn, expected, atol=1e-9)
