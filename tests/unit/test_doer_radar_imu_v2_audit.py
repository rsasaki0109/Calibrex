"""The radar-imu v2 audit harness: archive layout, candidates and scoring."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from tools import doer_radar_imu_v2_audit as harness

CALIB = (
    "topic_radar_scan: /sensor_platform/radar_{name}/scan\n"
    "l_b_r_x: 0.04\nl_b_r_y: 0.03\nl_b_r_z: -0.03\n"
    "q_b_r_x: 0.030573233\nq_b_r_y: -0.031111548\nq_b_r_z: -0.712571593\nq_b_r_w: 0.700242154\n"
)


def test_the_archive_scan_takes_the_top_folder_only(tmp_path: Path) -> None:
    path = tmp_path / "a.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("top/", b"")
        for name in ("left", "right", "center"):
            archive.writestr(f"top/calib_{name}.yaml", CALIB.format(name=name))
        archive.writestr("top/workshop.bag", b"x")
        archive.writestr("top/basement.bag", b"y")
        archive.writestr("top/results/run/calib_left.yaml", b"not the reference")
        archive.writestr("top/results/run/other.bag", b"z")

    with zipfile.ZipFile(path) as archive:
        texts, bags = harness.archive_members(archive)

    assert sorted(texts) == ["center", "left", "right"]
    assert b"radar_left" in texts["left"]
    assert [Path(item.filename).name for item in bags] == ["basement.bag", "workshop.bag"]


def test_every_radar_is_turned_about_its_own_axes(tmp_path: Path) -> None:
    calibs = {name: yaml.safe_load(CALIB.format(name=name)) for name in ("left", "right")}
    paths = harness.write_candidates(calibs, tmp_path)

    def frames(variant: str) -> dict[str, Rotation]:
        doc = yaml.safe_load(paths[variant].read_text())
        return {f["name"]: Rotation.from_quat(f["rotation_quat_xyzw"]).inv() for f in doc["frames"]}

    reference = frames("reference")
    assert sorted(reference) == ["radar_left", "radar_right"]
    for variant, axis in harness.VARIANTS.items():
        if axis is None:
            continue
        for name, turned in frames(variant).items():
            turn = (turned * reference[name].inv()).as_rotvec(degrees=True)
            expected = np.zeros(3)
            expected[harness.AXIS_INDEX[axis]] = harness.TURN_DEG
            assert np.allclose(turn, expected, atol=1e-9)
    command = harness.check_command(Path("b"), paths["reference"], Path("o"), Path("c"), calibs)
    assert "/sensor_platform/radar_left/scan=radar_left" in command
    assert f"{harness.IMU_TOPIC}=imu" in command


def _record(status: str, judged: list[str], axis_status: dict[str, str]) -> dict:
    return {
        "status": status,
        "coverage": "partial",
        "exit_status": 1,
        "judged_axes": judged,
        "axes": {k: {"status": v} for k, v in axis_status.items()},
    }


def test_the_claims_count_cases_per_radar() -> None:
    radars = ["center", "left", "right"]
    passed = _record("pass", ["yaw"], {"yaw": "pass"})
    blind = _record("inconclusive", [], {})
    caught = _record("fail", ["yaw"], {"yaw": "fail"})
    scores = {}
    for bag in ("a", "b", "c", "d"):
        scores[bag] = {
            "reference": {f"radar_{r}": passed if r != "right" else blind for r in radars},
            "rx3": {f"radar_{r}": passed if r != "right" else blind for r in radars},
            "ry3": {f"radar_{r}": passed if r != "right" else blind for r in radars},
            "rz3": {f"radar_{r}": caught if r != "right" else blind for r in radars},
        }

    found = harness.verdicts(scores, radars)

    assert found["complete"]
    assert found["honesty"]["reference_pass_fraction"] == 8 / 12
    assert found["coverage"]["reference_judged_fraction"] == 8 / 12
    assert found["detection"]["detecting_case_fraction"] == 8 / 12
    assert found["detection"]["judged_turns_missed"] == 0
    assert [found[c]["verdict"] for c in ("honesty", "coverage", "detection")] == ["supported"] * 3
