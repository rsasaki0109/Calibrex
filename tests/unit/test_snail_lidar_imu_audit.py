"""The Snail-Radar imu-lidar audit harness: transforms, thinning, candidates and scoring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from tools import snail_lidar_imu_audit as harness

LAUNCH = """<launch>
  <node name="Body_T_X36d" pkg="tf" type="static_transform_publisher" args="-0.06 0.0 -0.16 0.0 0.0 0.0 1.0 /base_link /x36d 100" />
  <node name="Body_T_Xt32" pkg="tf" type="static_transform_publisher" args="0.0 0.0 0.0 0.0 0.0 0.7071067811865475 0.7071067811865475 /base_link /PandarXT-32 100" />
</launch>
"""  # noqa: E501


def test_static_transforms_are_read_by_child_frame() -> None:
    found = harness.static_transforms(LAUNCH)

    assert set(found) == {"x36d", "PandarXT-32"}
    translation, rotation = found["x36d"]
    assert np.allclose(translation, [-0.06, 0.0, -0.16])
    assert rotation.magnitude() == 0.0
    assert np.isclose(found["PandarXT-32"][1].as_rotvec(degrees=True)[2], 90.0)


def test_thinning_keeps_whole_firings() -> None:
    dtype = np.dtype(
        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4"), ("timestamp", "<f8")]
    )
    firings = np.repeat(np.arange(10) * 1e-4 + 1.7e9, 32)
    points = np.zeros(len(firings), dtype)
    points["timestamp"] = firings
    points["x"] = np.arange(len(firings))

    kept = harness.thinned_scan(points)

    assert kept.dtype == harness.POINT_DTYPE
    assert len(np.unique(kept["timestamp"])) == 3  # firings 0, 4 and 8
    assert len(kept) == 3 * 32


def test_a_turned_candidate_turns_the_lidar_about_its_own_axis(tmp_path: Path) -> None:
    transforms = harness.static_transforms(LAUNCH)

    paths = harness.write_candidates(transforms, ["x36d"], tmp_path)

    def lidar_in_imu(variant: str) -> Rotation:
        frames = {
            f["name"]: Rotation.from_quat(f["rotation_quat_xyzw"])
            for f in yaml.safe_load(paths[variant].read_text())["frames"]
        }
        return frames["x36d"].inv() * frames["PandarXT-32"]

    delta = lidar_in_imu("reference").inv() * lidar_in_imu("ry3")  # in the LiDAR frame
    assert np.allclose(delta.as_rotvec(degrees=True), [0.0, harness.TURN_DEG, 0.0])


def _record(status: str, axes: dict[str, str]) -> dict:
    return {
        "status": status,
        "judged_axes": [a for a in axes if a in harness.AXIS_INDEX],
        "axes": {a: {"status": s} for a, s in axes.items()},
    }


def test_the_verdicts_score_rotation_and_leave_translation_out() -> None:
    good = {"roll": "pass", "pitch": "pass", "yaw": "pass", "z": "fail"}
    score = {
        "imus": ["x36d"],
        "reference": {"x36d": _record("fail", good)},
        "rx3": {"x36d": _record("fail", {**good, "roll": "fail"})},
        "ry3": {"x36d": _record("fail", {**good, "pitch": "fail"})},
        "rz3": {"x36d": _record("warn", {**good, "yaw": "warn"})},
    }

    found = harness.verdicts({"a/data1": score}, expected_cases=1)

    assert found["honesty"]["reference_rotation_fails"] == 0  # the z fail is not scored
    assert found["honesty"]["reference_pass_fraction"] == 1.0
    assert found["detection"]["judged_turns_missed"] == 1  # yaw judged, only warned
    assert found["detection"]["detecting_case_fraction"] == 1.0
    assert found["detection"]["verdict"] == "refuted"
    assert found["complete"]


def test_a_missing_case_leaves_the_run_incomplete() -> None:
    score = {"imus": ["x36d", "mti3dk"], **{v: {} for v in harness.VARIANTS}}

    found = harness.verdicts({"a/data1": score}, expected_cases=2)

    assert not found["complete"]
    assert {found[c]["verdict"] for c in ("honesty", "coverage", "detection")} == {"refuted"}
