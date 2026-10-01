"""Candidate static extrinsic sources for ``calibrex check``."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml
from tests.unit.check_fixtures import IDENTITY, write_check_bag

from calibrex.check.frame_tree import StaticEdge, StaticFrameTree
from calibrex.check.tf_sources import (
    load_bag_tf_static,
    load_tf_file,
    merge_hints,
    merge_sources,
    quaternion_from_rpy,
)
from calibrex.core.exceptions import DatasetError, FrameGraphError
from calibrex.core.geometry import SE3

SQRT_HALF = math.sqrt(0.5)


def _edge(parent: str, child: str, x: float = 0.0, source: str = "frames_yaml") -> StaticEdge:
    return StaticEdge(parent, child, SE3((x, 0.0, 0.0), IDENTITY), source)  # type: ignore[arg-type]


# --------------------------------------------------------------------- frame tree


def test_tree_lookup_composes_through_common_ancestor() -> None:
    tree = StaticFrameTree([_edge("base", "a", 1.0), _edge("base", "b", 3.0), _edge("b", "c", 0.5)])

    transform = tree.lookup("a", "c")

    assert transform is not None
    assert transform.translation_m == pytest.approx((2.5, 0.0, 0.0))
    assert tree.roots == ["base"]
    assert tree.frames == ["a", "b", "base", "c"]


def test_tree_forest_has_no_cross_component_lookup() -> None:
    tree = StaticFrameTree([_edge("r1", "a"), _edge("r2", "b")])

    assert tree.roots == ["r1", "r2"]
    assert tree.lookup("a", "b") is None
    assert not tree.connected("a", "missing")


def test_tree_rejects_cycles_conflicting_parents_and_self_edges() -> None:
    with pytest.raises(FrameGraphError, match="cycle"):
        StaticFrameTree([_edge("a", "b"), _edge("b", "a")])
    with pytest.raises(FrameGraphError, match="conflicting parents"):
        StaticFrameTree([_edge("a", "c"), _edge("b", "c")])
    with pytest.raises(FrameGraphError, match="own parent"):
        StaticFrameTree([_edge("a", "a")])


# --------------------------------------------------------------------- bag tf_static


def test_bag_tf_static_takes_latest_per_child(tmp_path: Path) -> None:
    bag = write_check_bag(
        tmp_path / "bag.db3",
        tf_messages=[
            [("base_link", "imu", (0.0, 0.0, 0.1), IDENTITY)],
            [
                ("base_link", "imu", (0.0, 0.0, 0.2), IDENTITY),
                ("base_link", "lidar", (1.0, 0.0, 0.0), IDENTITY),
            ],
        ],
    )

    source = load_bag_tf_static(bag)

    assert source is not None
    assert source.kind == "bag_tf_static"
    assert source.sha256 is not None and len(source.sha256) == 64
    edges = {edge.child: edge for edge in source.edges}
    assert set(edges) == {"imu", "lidar"}
    assert edges["imu"].transform.translation_m == (0.0, 0.0, 0.2)
    assert any("changed a transform value" in note for note in source.notes)


def test_bag_tf_static_normalizes_leading_slash(tmp_path: Path) -> None:
    bag = write_check_bag(
        tmp_path / "bag.db3", tf_messages=[[("/base_link", "/imu", (0.0, 0.0, 0.0), IDENTITY)]]
    )

    source = load_bag_tf_static(bag)

    assert source is not None
    assert (source.edges[0].parent, source.edges[0].child) == ("base_link", "imu")


def test_bag_tf_static_rejects_conflicting_parents(tmp_path: Path) -> None:
    bag = write_check_bag(
        tmp_path / "bag.db3",
        tf_messages=[
            [("base_link", "imu", (0.0, 0.0, 0.0), IDENTITY)],
            [("chassis", "imu", (0.0, 0.0, 0.0), IDENTITY)],
        ],
    )

    with pytest.raises(FrameGraphError, match="conflicting parents 'base_link' and 'chassis'"):
        load_bag_tf_static(bag)


def test_bag_without_tf_static_returns_none(tmp_path: Path) -> None:
    bag = write_check_bag(tmp_path / "bag.db3", include_tf_topic=False)

    assert load_bag_tf_static(bag) is None


# ---------------------------------------------------------------------------- URDF

URDF = """<?xml version="1.0"?>
<robot name="rig">
  <link name="base_link"/>
  <link name="lidar"/>
  <link name="wheel"/>
  <joint name="lidar_joint" type="fixed">
    <parent link="base_link"/>
    <child link="lidar"/>
    <origin xyz="1 2 3" rpy="0 0 1.5707963267948966"/>
  </joint>
  <joint name="wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="wheel"/>
  </joint>
  <joint name="imu_joint" type="fixed">
    <parent link="base_link"/>
    <child link="imu"/>
  </joint>
</robot>
"""


def test_urdf_reads_fixed_joints_only(tmp_path: Path) -> None:
    path = tmp_path / "rig.urdf"
    path.write_text(URDF, encoding="utf-8")

    source = load_tf_file(path)

    assert source.kind == "urdf"
    assert source.sha256 is not None
    edges = {edge.child: edge for edge in source.edges}
    assert set(edges) == {"lidar", "imu"}
    lidar = edges["lidar"].transform
    assert lidar.translation_m == (1.0, 2.0, 3.0)
    assert lidar.rotation_quat_xyzw == pytest.approx((0.0, 0.0, SQRT_HALF, SQRT_HALF))
    assert edges["imu"].transform == SE3.identity()
    assert any("1 non-fixed" in note for note in source.notes)


def test_rpy_matches_fixed_axis_convention() -> None:
    roll, pitch, yaw = 0.3, -0.4, 0.5
    quaternion = SE3((0, 0, 0), quaternion_from_rpy(roll, pitch, yaw))

    # R = Rz(yaw) Ry(pitch) Rx(roll): rotate x axis by roll about x first.
    point = quaternion.transform_point((0.0, 1.0, 0.0))
    expected_y = math.cos(roll)
    expected_z = math.sin(roll)
    # undo pitch and yaw by composing the inverses
    undo = SE3((0, 0, 0), quaternion_from_rpy(0.0, 0.0, yaw)).compose(
        SE3((0, 0, 0), quaternion_from_rpy(0.0, pitch, 0.0))
    )
    recovered = undo.inverse().transform_point(point)
    assert recovered == pytest.approx((0.0, expected_y, expected_z), abs=1e-9)


def test_urdf_errors_are_clear(tmp_path: Path) -> None:
    broken = tmp_path / "broken.urdf"
    broken.write_text("<robot><joint", encoding="utf-8")
    xacro = tmp_path / "xacro.urdf"
    xacro.write_text(
        '<robot><joint name="j" type="fixed"><parent link="a"/><child link="b"/>'
        '<origin xyz="${x} 0 0"/></joint></robot>',
        encoding="utf-8",
    )
    empty = tmp_path / "empty.urdf"
    empty.write_text("<robot/>", encoding="utf-8")

    with pytest.raises(DatasetError, match="invalid URDF"):
        load_tf_file(broken)
    with pytest.raises(DatasetError, match="xacro"):
        load_tf_file(xacro)
    with pytest.raises(DatasetError, match="no fixed joints"):
        load_tf_file(empty)


# ------------------------------------------------------------------- frames YAML


def test_frames_yaml_with_hints(tmp_path: Path) -> None:
    path = tmp_path / "frames.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "slac.check_frames/v0.1",
                "frames": [
                    {"name": "imu", "parent": "base_link", "translation_m": [0, 0, 0.1]},
                    {
                        "name": "/lidar",
                        "parent": "imu",
                        "translation_m": [1, 0, 0],
                        "rotation_quat_xyzw": [0, 0, SQRT_HALF, SQRT_HALF],
                    },
                ],
                "topic_frames": {"/points": "/lidar"},
                "role_frames": {"imu": "imu"},
            }
        ),
        encoding="utf-8",
    )

    source = load_tf_file(path)

    assert source.kind == "frames_yaml"
    assert {edge.child for edge in source.edges} == {"imu", "lidar"}
    assert source.hints.topic_frames == {"/points": "lidar"}
    assert source.hints.role_frames == {"imu": "imu"}


def test_frames_yaml_validation_error(tmp_path: Path) -> None:
    path = tmp_path / "frames.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": "slac.check_frames/v0.1",
                "frames": [{"name": "a", "parent": "b", "translation_m": [0, 0]}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DatasetError, match=r"invalid slac\.check_frames"):
        load_tf_file(path)


# -------------------------------------------------------------------- Kalibr


def _camchain(path: Path, topic: str | None = "/cam/image_raw") -> Path:
    # T_cam_imu: rotate 90 deg about z and translate; inverse is easy to check.
    matrix = [[0.0, -1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0, 0, 0, 1]]
    entry: dict[str, object] = {"T_cam_imu": matrix, "camera_model": "pinhole"}
    if topic is not None:
        entry["rostopic"] = topic
    path.write_text(yaml.safe_dump({"cam0": entry}), encoding="utf-8")
    return path


def test_kalibr_camchain_inverts_t_cam_imu(tmp_path: Path) -> None:
    source = load_tf_file(_camchain(tmp_path / "camchain.yaml"))

    assert source.kind == "kalibr_camchain"
    (edge,) = source.edges
    assert (edge.parent, edge.child) == ("imu", "cam/image_raw")
    # T_imu_cam = inv(T_cam_imu): origin of cam in imu frame is R^T * -t
    assert edge.transform.translation_m == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)
    assert edge.transform.rotation_quat_xyzw == pytest.approx((0, 0, -SQRT_HALF, SQRT_HALF))
    assert source.hints.topic_frames == {"/cam/image_raw": "cam/image_raw"}
    assert source.hints.role_frames == {"imu": "imu"}


def test_kalibr_camchain_without_rostopic_uses_key(tmp_path: Path) -> None:
    source = load_tf_file(_camchain(tmp_path / "camchain.yaml", topic=None))

    assert source.edges[0].child == "cam0"
    assert any("no rostopic" in note for note in source.notes)


def test_kalibr_rejects_non_rotation(tmp_path: Path) -> None:
    path = tmp_path / "camchain.yaml"
    path.write_text(
        yaml.safe_dump(
            {"cam0": {"T_cam_imu": [[2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]}}
        ),
        encoding="utf-8",
    )

    with pytest.raises(DatasetError, match="not orthonormal"):
        load_tf_file(path)


# ------------------------------------------------------------------- RTK-SLAM


def test_rtk_slam_calib(tmp_path: Path) -> None:
    path = tmp_path / "calib.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "cam0": {
                    "T_cam_imu": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                    "rostopic": "/camera/image_raw/compressed",
                },
                "lidar0": {
                    "T_lidar_imu": [
                        [1, 0, 0, -0.011],
                        [0, 1, 0, -0.023],
                        [0, 0, 1, 0.044],
                        [0, 0, 0, 1],
                    ]
                },
                "reference_offsets": {
                    "gnss_antenna_phase_center": [0.023, -0.023, 0.09],
                    "base_center": [0, 0, 0],
                },
            }
        ),
        encoding="utf-8",
    )

    source = load_tf_file(path)

    assert source.kind == "rtk_slam_calib"
    edges = {edge.child: edge for edge in source.edges}
    assert set(edges) == {"camera/image_raw/compressed", "lidar0", "gnss_antenna"}
    assert edges["lidar0"].parent == "imu"
    assert edges["lidar0"].transform.translation_m == (-0.011, -0.023, 0.044)
    assert source.hints.role_frames == {"imu": "imu", "lidar": "lidar0", "gnss": "gnss_antenna"}
    assert any("base_center" in note for note in source.notes)


# ---------------------------------------------------------------------- Hilti


def test_hilti_sensor_list(tmp_path: Path) -> None:
    path = tmp_path / "lidar_calibration.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "sensors": {
                    "imu": {
                        "parent": "base_link",
                        "extrinsics": {
                            "quaternion": [0, 0, 0, 1],
                            "translation": [0, 0, 0],
                        },
                    },
                    "Pandar": {
                        "parent": "imu",
                        "extrinsics": {
                            "quaternion": [SQRT_HALF, -SQRT_HALF, 0, 0],
                            "translation": [-0.001, -0.00855, 0.055],
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    source = load_tf_file(path)

    assert source.kind == "hilti_sensors"
    edges = {edge.child: edge for edge in source.edges}
    assert edges["Pandar"].parent == "imu"
    assert edges["Pandar"].transform.translation_m == (-0.001, -0.00855, 0.055)
    assert source.hints.role_frames == {"imu": "imu"}


def test_unknown_and_missing_files(tmp_path: Path) -> None:
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text("a: 1\n", encoding="utf-8")

    with pytest.raises(DatasetError, match="unrecognized calibration format"):
        load_tf_file(unknown)
    with pytest.raises(DatasetError, match="does not exist"):
        load_tf_file(tmp_path / "nope.yaml")


# ----------------------------------------------------------------------- merging


def test_files_override_bag_and_conflicting_files_are_rejected(tmp_path: Path) -> None:
    bag = write_check_bag(
        tmp_path / "bag.db3",
        tf_messages=[
            [
                ("base_link", "imu", (0.0, 0.0, 0.1), IDENTITY),
                ("base_link", "lidar", (1.0, 0.0, 0.0), IDENTITY),
            ]
        ],
    )
    bag_source = load_bag_tf_static(bag)
    frames = tmp_path / "frames.yaml"
    frames.write_text(
        yaml.safe_dump(
            {
                "schema_version": "slac.check_frames/v0.1",
                "frames": [{"name": "lidar", "parent": "base_link", "translation_m": [9, 0, 0]}],
            }
        ),
        encoding="utf-8",
    )
    other = tmp_path / "other.yaml"
    other.write_text(
        yaml.safe_dump(
            {
                "schema_version": "slac.check_frames/v0.1",
                "frames": [{"name": "lidar", "parent": "base_link", "translation_m": [8, 0, 0]}],
            }
        ),
        encoding="utf-8",
    )

    tree, overrides = merge_sources(bag_source, [load_tf_file(frames)])

    lidar = tree.lookup("base_link", "lidar")
    assert lidar is not None and lidar.translation_m == (9.0, 0.0, 0.0)
    assert tree.lookup("base_link", "imu") is not None
    assert len(overrides) == 1 and "overrides bag tf_static" in overrides[0]
    with pytest.raises(FrameGraphError, match="defined differently"):
        merge_sources(None, [load_tf_file(frames), load_tf_file(other)])


def test_merge_hints_later_wins(tmp_path: Path) -> None:
    first = load_tf_file(_camchain(tmp_path / "a.yaml", "/a"))
    second = load_tf_file(_camchain(tmp_path / "b.yaml", "/b"))

    hints = merge_hints([first, second])

    assert set(hints.topic_frames) == {"/a", "/b"}
