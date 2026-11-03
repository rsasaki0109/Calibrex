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


# ------------------------------------------------------------------- FAST-LIO

# FAST-LIVO's config/avia.yaml, as used for the hku1/hku2 recordings.
FAST_LIVO_AVIA = """\
common:
    lid_topic:  "/livox/lidar"
    imu_topic:  "/livox/imu"
mapping:
    extrinsic_T: [ 0.04165, 0.02326, -0.0284 ]
    extrinsic_R: [ 1, 0, 0,
                   0, 1, 0,
                   0, 0, 1]
camera:
    img_topic: /left_camera/image
    Rcl: [0.00162756,-0.999991,0.00390957,
         -0.0126748,-0.00392989,-0.999912,
          0.999918,0.00157786,-0.012681]
    Pcl: [0.0409257, 0.0318424, -0.0927219]
"""


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_fast_lio_extrinsic_is_the_lidar_in_the_imu_frame(tmp_path: Path) -> None:
    rotation = [0, -1, 0, 1, 0, 0, 0, 0, 1]  # the LiDAR turned +90 deg about z in the IMU
    path = _write(
        tmp_path / "mid360.yaml",
        yaml.safe_dump(
            {
                "common": {"lid_topic": "/livox/lidar", "imu_topic": "/livox/imu"},
                "mapping": {
                    "extrinsic_est_en": True,
                    "extrinsic_T": [-0.011, -0.02329, 0.04412],
                    "extrinsic_R": rotation,
                },
            }
        ),
    )

    source = load_tf_file(path)

    assert source.kind == "fast_lio_config"
    (edge,) = source.edges
    assert (edge.parent, edge.child) == ("imu", "lidar")
    assert edge.transform.translation_m == pytest.approx((-0.011, -0.02329, 0.04412))
    assert edge.transform.rotation_quat_xyzw == pytest.approx((0, 0, SQRT_HALF, SQRT_HALF))
    assert source.hints.topic_frames == {"/livox/lidar": "lidar", "/livox/imu": "imu"}
    assert source.hints.role_frames == {"imu": "imu", "lidar": "lidar"}
    assert any("extrinsic_est_en is true" in note for note in source.notes)


def test_fast_livo_camera_matches_the_hand_built_camchain(tmp_path: Path) -> None:
    """The FAST-LIVO hku1/hku2 candidate, built by hand earlier, read from avia.yaml."""

    source = load_tf_file(_write(tmp_path / "avia.yaml", FAST_LIVO_AVIA))
    tree, _ = merge_sources(None, [source])

    assert source.hints.topic_frames["/left_camera/image"] == "left_camera/image"
    t_imu_cam = tree.lookup("imu", "left_camera/image")
    assert t_imu_cam is not None
    # docs: T_cam_imu = [Rcl|Pcl] * inv([I|extrinsic_T]), its translation as written there
    t_cam_imu = t_imu_cam.inverse()
    assert t_cam_imu.translation_m == pytest.approx(
        (0.064228735414406, 0.004064213659206686, -0.13476534186780031), abs=1e-6
    )
    t_cam_lidar = tree.lookup("left_camera/image", "lidar")
    assert t_cam_lidar is not None
    assert t_cam_lidar.translation_m == pytest.approx((0.0409257, 0.0318424, -0.0927219))


def test_fast_livo2_keeps_its_extrinsics_under_extrin_calib(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "avia2.yaml",
        yaml.safe_dump(
            {
                "common": {"img_topic": "/left_camera/image", "lid_topic": "/livox/lidar"},
                "extrin_calib": {
                    "extrinsic_T": [0.04165, 0.02326, -0.0284],
                    "extrinsic_R": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                    "Rcl": [0, -1, 0, 0, 0, -1, 1, 0, 0],
                    "Pcl": [0.0, 0.0, 0.0],
                },
            }
        ),
    )

    source = load_tf_file(path)

    assert [(edge.parent, edge.child) for edge in source.edges] == [
        ("imu", "lidar"),
        ("lidar", "left_camera/image"),
    ]
    # the camera looks along the LiDAR's x axis: its z axis is the LiDAR's x
    t_lidar_cam = source.edges[1].transform
    assert t_lidar_cam.rotation_quat_xyzw == pytest.approx((-0.5, 0.5, -0.5, 0.5))


def test_fast_lio_rejects_a_matrix_that_is_not_a_rotation(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "bad.yaml",
        yaml.safe_dump(
            {"mapping": {"extrinsic_T": [0, 0, 0], "extrinsic_R": [1, 0, 0, 0, 1, 0, 0, 0, -1]}}
        ),
    )
    with pytest.raises(DatasetError, match="reflection"):
        load_tf_file(path)
    path.write_text(
        yaml.safe_dump({"mapping": {"extrinsic_T": [0, 0], "extrinsic_R": [1, 0, 0]}}),
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="9 rotation and 3 translation"):
        load_tf_file(path)


def test_a_camchain_may_carry_intrinsics_only(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "intrinsics.yaml",
        yaml.safe_dump(
            {
                "cam0": {
                    "camera_model": "pinhole",
                    "intrinsics": [863.6, 863.1, 622.2, 534.5],
                    "distortion_model": "radtan",
                    "distortion_coeffs": [-0.094, 0.095, -0.008, 0.0001],
                    "resolution": [1280, 1024],
                    "rostopic": "/left_camera/image",
                }
            }
        ),
    )

    source = load_tf_file(path)

    assert source.kind == "kalibr_camchain"
    assert source.edges == ()
    assert source.hints.topic_frames == {"/left_camera/image": "left_camera/image"}
    assert any("intrinsics only" in note for note in source.notes)
    tree, _ = merge_sources(
        None, [load_tf_file(_write(tmp_path / "a.yaml", FAST_LIVO_AVIA)), source]
    )
    assert "left_camera/image" in tree
    path.write_text(yaml.safe_dump({"cam0": {"rostopic": "/x"}}), encoding="utf-8")
    with pytest.raises(DatasetError, match=r"no camN\.T_cam_imu or camN\.intrinsics"):
        load_tf_file(path)


def test_livox_topics_map_to_the_config_frames_without_frame_map(tmp_path: Path) -> None:
    """The LiDAR and its IMU share the header frame livox_frame, which is not in the tree."""

    from calibrex.check.roles import map_topics_to_frames
    from calibrex.core.calibration_check import CheckTopicRecord

    source = load_tf_file(_write(tmp_path / "avia.yaml", FAST_LIVO_AVIA))
    tree, _ = merge_sources(None, [source])
    records = [
        CheckTopicRecord(
            topic="/livox/lidar",
            message_type="livox_ros_driver/msg/CustomMsg",
            message_count=10,
            role="lidar",
        ),
        CheckTopicRecord(
            topic="/livox/imu", message_type="sensor_msgs/msg/Imu", message_count=10, role="imu"
        ),
        CheckTopicRecord(
            topic="/left_camera/image/compressed",
            message_type="sensor_msgs/msg/CompressedImage",
            message_count=10,
            role="camera",
        ),
    ]
    header_frames = {
        "/livox/lidar": "livox_frame",
        "/livox/imu": "livox_frame",
        "/left_camera/image/compressed": "",
    }

    mapped = map_topics_to_frames(records, header_frames, tree, merge_hints([source]))

    assert [(r.mapped_frame, r.frame_source) for r in mapped] == [
        ("lidar", "source_topic_hint"),
        ("imu", "source_topic_hint"),
        ("left_camera/image", "source_topic_hint"),
    ]


# ------------------------------------------------- FAST-LIVO camera file (vikit)

# FAST-LIVO's config/camera_pinhole.yaml: the hku1/hku2 camera at 640 x 512.
FAST_LIVO_CAMERA = """\
cam_model: Pinhole
cam_width: 640
cam_height: 512
cam_fx: 431.795259219
cam_fy: 431.550090267
cam_cx: 310.833037316
cam_cy: 266.985989326
cam_d0: -0.0944205499243979
cam_d1: 0.0946727677776504
cam_d2: -0.00807970960613932
cam_d3: 8.07461209775283e-05
"""


def test_a_fast_livo_camera_file_is_a_source_without_frames(tmp_path: Path) -> None:
    source = load_tf_file(_write(tmp_path / "camera_pinhole.yaml", FAST_LIVO_CAMERA))

    assert source.kind == "vikit_camera"
    assert source.edges == ()
    assert any("640x512" in note for note in source.notes)


def test_scaling_matches_the_hand_doubled_camchain(tmp_path: Path) -> None:
    """The hku1/hku2 camchain doubled these intrinsics for the 1280 x 1024 JPEGs."""

    from calibrex.check.tf_sources import scale_camera_entry, vikit_camera_entry

    entry = vikit_camera_entry(_write(tmp_path / "camera_pinhole.yaml", FAST_LIVO_CAMERA))
    scaled, scale = scale_camera_entry(entry, 1280, 1024)

    assert scale == 2.0
    assert scaled["intrinsics"] == pytest.approx(
        [863.590518438, 863.100180534, 622.166074632, 534.471978652]
    )
    assert scaled["distortion_coeffs"] == entry["distortion_coeffs"]
    assert scale_camera_entry(entry, 640, 512)[0]["intrinsics"] == entry["intrinsics"]
    with pytest.raises(ValueError, match="another aspect ratio"):
        scale_camera_entry(entry, 1280, 720)


def test_a_camera_file_with_another_model_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path / "camera.yaml", FAST_LIVO_CAMERA.replace("Pinhole", "ATAN"))
    with pytest.raises(DatasetError, match="only Pinhole"):
        load_tf_file(path)


def _image_bag(path: Path, width: int, height: int) -> Path:
    import numpy as np

    from calibrex.data.ros_cdr_writer import encode_compressed_image
    from calibrex.data.rosbag2_writer import Rosbag2Writer
    from calibrex.visualization.camera_lidar_check_overlay import encode_png

    png = encode_png(np.zeros((height, width, 3), dtype=np.uint8))
    with Rosbag2Writer(path) as writer:
        writer.add_topic("/left_camera/image/compressed", "sensor_msgs/msg/CompressedImage")
        writer.write(
            "/left_camera/image/compressed",
            10**9,
            encode_compressed_image(frame_id="", timestamp_ns=10**9, format="png", data=png),
        )
    return path


def test_the_fast_livo_camera_gets_the_camera_file_scaled_to_its_images(tmp_path: Path) -> None:
    pytest.importorskip("PIL.Image")
    from calibrex.check import estimators

    config = load_tf_file(_write(tmp_path / "avia.yaml", FAST_LIVO_AVIA))
    camera = load_tf_file(
        _write(
            tmp_path / "camera.yaml",
            FAST_LIVO_CAMERA.replace("640", "32").replace("512", "24"),
        )
    )
    topic = "/left_camera/image/compressed"
    ctx = estimators.PairContext(
        bag=_image_bag(tmp_path / "bag", 64, 48),
        pair=None,  # type: ignore[arg-type]
        topic_types={topic: "sensor_msgs/msg/CompressedImage"},
        sensor_topics={"camera": (topic,)},
        candidate=IDENTITY,  # type: ignore[arg-type]
        sources=[config, camera],
        controls=estimators.RunControls(),
    )

    model, description, _ = estimators.resolve_camera_intrinsics(ctx, topic, "left_camera/image")

    assert description == "camera.yaml (scaled x2 to 64x48)"
    assert (model.fx, model.cx) == pytest.approx((2 * 431.795259219, 2 * 310.833037316 + 0.5))
    # a camera the FAST-LIVO config does not name is left to the other sources
    with pytest.raises(estimators.CheckSkipError, match="no intrinsics"):
        estimators.resolve_camera_intrinsics(ctx, topic, "other_camera")
