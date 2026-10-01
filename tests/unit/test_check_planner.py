"""Topic roles, frame mapping and pair planning for ``calibrex check``."""

from __future__ import annotations

import pytest

from calibrex.check.frame_tree import FrameHints, StaticEdge, StaticFrameTree
from calibrex.check.planner import plan_pairs
from calibrex.check.roles import (
    classify_message_type,
    classify_odometry_kind,
    classify_topics,
    map_topics_to_frames,
)
from calibrex.core.calibration_check import CheckPairRecord, CheckTopicRecord
from calibrex.core.geometry import SE3
from calibrex.data.rosbag2 import Rosbag2Connection

IDENTITY = (0.0, 0.0, 0.0, 1.0)


def _tree(*edges: tuple[str, str, float]) -> StaticFrameTree:
    return StaticFrameTree(
        StaticEdge(parent, child, SE3((x, 0.0, 0.0), IDENTITY), "frames_yaml")
        for parent, child, x in edges
    )


def _topic(
    topic: str,
    role: str,
    frame: str | None,
    *,
    header: str | None = None,
    kind: str | None = None,
) -> CheckTopicRecord:
    return CheckTopicRecord(
        topic=topic,
        message_type="x",
        role=role,  # type: ignore[arg-type]
        odometry_kind=kind,  # type: ignore[arg-type]
        header_frame_id=header,
        mapped_frame=frame,
        frame_source="header" if frame else None,
    )


def _by_pair(records: list[CheckPairRecord], pair: str) -> list[CheckPairRecord]:
    return [record for record in records if record.pair == pair]


# ---------------------------------------------------------------------- roles


@pytest.mark.parametrize(
    ("message_type", "topic", "role"),
    [
        ("sensor_msgs/msg/PointCloud2", "/a", "lidar"),
        ("livox_ros_driver2/msg/CustomMsg", "/a", "lidar"),
        ("sensor_msgs/msg/Imu", "/a", "imu"),
        ("sensor_msgs/msg/Image", "/a", "camera"),
        ("sensor_msgs/msg/CompressedImage", "/a", "camera"),
        ("sensor_msgs/msg/NavSatFix", "/a", "gnss"),
        ("nav_msgs/msg/Odometry", "/a", "odometry"),
        ("geometry_msgs/msg/TwistStamped", "/a", "twist"),
        ("geometry_msgs/msg/TwistWithCovarianceStamped", "/a", "twist"),
        ("tf2_msgs/msg/TFMessage", "/tf_static", "tf_static"),
        ("tf2_msgs/msg/TFMessage", "/robot/tf_static", "tf_static"),
        ("tf2_msgs/msg/TFMessage", "/tf", None),
        ("rcl_interfaces/msg/Log", "/rosout", None),
    ],
)
def test_classify_message_type(message_type: str, topic: str, role: str | None) -> None:
    assert classify_message_type(message_type, topic) == role


def test_odometry_kind_by_name_tokens() -> None:
    assert classify_odometry_kind("/wheel_odom") == "wheel"
    assert classify_odometry_kind("/vehicle/encoder/odometry") == "wheel"
    assert classify_odometry_kind("/gnss_ins/odometry") == "ins"
    assert classify_odometry_kind("/inspect/odom") == "unknown"
    assert classify_odometry_kind("/lio/odometry") == "unknown"


def test_classify_topics_skips_unroled_and_sorts() -> None:
    connections = [
        (Rosbag2Connection(1, "/rosout", "rcl_interfaces/msg/Log"), 5),
        (Rosbag2Connection(2, "/b/imu", "sensor_msgs/msg/Imu"), 10),
        (Rosbag2Connection(3, "/a/odom", "nav_msgs/msg/Odometry"), None),
    ]

    records = classify_topics(connections)

    assert [(r.topic, r.role, r.message_count) for r in records] == [
        ("/a/odom", "odometry", None),
        ("/b/imu", "imu", 10),
    ]
    assert records[0].odometry_kind == "unknown"


# ------------------------------------------------------------ frame mapping


def test_frame_mapping_priority_and_role_default() -> None:
    tree = _tree(("base", "imu", 0.0), ("base", "lidar0", 1.0), ("base", "cam", 2.0))
    records = [
        CheckTopicRecord(topic="/lidar_a", message_type="x", role="lidar"),
        CheckTopicRecord(topic="/lidar_b", message_type="x", role="lidar"),
        CheckTopicRecord(topic="/imu", message_type="x", role="imu"),
        CheckTopicRecord(topic="/cam", message_type="x", role="camera"),
        CheckTopicRecord(topic="/gnss", message_type="x", role="gnss"),
        CheckTopicRecord(topic="/tf_static", message_type="x", role="tf_static"),
    ]
    headers = {
        "/lidar_a": "driver_lidar",
        "/lidar_b": "driver_lidar",
        "/imu": "imu",
        "/cam": "cam_optical",
        "/gnss": "gps",
    }
    hints = FrameHints(
        topic_frames={"/cam": "cam"},
        role_frames={"lidar": "lidar0", "gnss": "antenna_not_in_tree"},
    )

    mapped = {
        r.topic: r for r in map_topics_to_frames(records, headers, tree, hints, {"/gnss": "base"})
    }

    assert (mapped["/imu"].mapped_frame, mapped["/imu"].frame_source) == ("imu", "header")
    assert (mapped["/cam"].mapped_frame, mapped["/cam"].frame_source) == (
        "cam",
        "source_topic_hint",
    )
    # two topics share one header frame: the role default applies to both
    assert mapped["/lidar_a"].mapped_frame == mapped["/lidar_b"].mapped_frame == "lidar0"
    assert mapped["/lidar_a"].frame_source == "role_default"
    assert (mapped["/gnss"].mapped_frame, mapped["/gnss"].frame_source) == ("base", "override")
    assert mapped["/tf_static"].mapped_frame is None


def test_role_default_is_ambiguous_with_distinct_unmapped_sensors() -> None:
    tree = _tree(("base", "lidar0", 1.0))
    records = [
        CheckTopicRecord(topic="/l1", message_type="x", role="lidar"),
        CheckTopicRecord(topic="/l2", message_type="x", role="lidar"),
    ]

    mapped = map_topics_to_frames(
        records,
        {"/l1": "a", "/l2": "b"},
        tree,
        FrameHints(role_frames={"lidar": "lidar0"}),
    )

    assert [r.mapped_frame for r in mapped] == [None, None]


# ------------------------------------------------------------------ planner


def _rig_topics() -> list[CheckTopicRecord]:
    return [
        _topic("/imu", "imu", "imu"),
        _topic("/lidar_f", "lidar", "lf"),
        _topic("/lidar_r", "lidar", "lr"),
        _topic("/cam", "camera", "cam"),
        _topic("/gnss", "gnss", "gps"),
        _topic("/wheel/odom", "odometry", "base", kind="wheel"),
        _topic("/ins/odom", "odometry", "ins", kind="ins"),
    ]


def _rig_tree() -> StaticFrameTree:
    return _tree(
        ("base", "imu", 0.0),
        ("base", "lf", 1.0),
        ("base", "lr", -1.0),
        ("base", "cam", 2.0),
        ("base", "gps", 3.0),
        ("base", "ins", 4.0),
    )


def test_all_pairs_planned_with_composed_transforms() -> None:
    records = plan_pairs(_rig_topics(), _rig_tree(), vehicle_frame="base")

    assert {r.status for r in records} == {"planned"}
    names = [r.pair for r in records]
    assert names == [
        "imu-lidar",
        "imu-lidar",
        "lidar-lidar",
        "camera-imu",
        "camera-focal",
        "gnss-lidar",
        "gnss-lidar",
        "gnss-imu",
        "lidar-vehicle",
        "lidar-vehicle",
        "imu-vehicle",
        "ins-lidar",
        "ins-lidar",
        "lidar-wheel_odometry",
        "lidar-wheel_odometry",
    ]
    (lidar_lidar,) = _by_pair(records, "lidar-lidar")
    assert lidar_lidar.candidate_transform is not None
    assert lidar_lidar.candidate_transform.translation_m == pytest.approx([-2.0, 0.0, 0.0])
    assert (lidar_lidar.candidate_transform.parent_frame, lidar_lidar.frames) == (
        "lf",
        ["lf", "lr"],
    )
    imu_lidar = _by_pair(records, "imu-lidar")[0]
    assert imu_lidar.candidate_transform is not None
    assert imu_lidar.candidate_transform.translation_m == pytest.approx([1.0, 0.0, 0.0])
    assert imu_lidar.topics == ["/imu", "/lidar_f"]


def test_skip_missing_topic() -> None:
    topics = [t for t in _rig_topics() if t.role not in {"gnss", "camera"}]

    records = plan_pairs(topics, _rig_tree(), vehicle_frame="base")

    for pair in ("camera-imu", "camera-focal", "gnss-lidar", "gnss-imu"):
        (record,) = _by_pair(records, pair)
        assert (record.status, record.reason_code) == ("skipped", "missing_topic")
    assert "camera" in (_by_pair(records, "camera-imu")[0].reason or "")


def test_skip_lidar_lidar_needs_two_distinct_sensors() -> None:
    topics = [t for t in _rig_topics() if t.topic != "/lidar_r"]

    (record,) = _by_pair(plan_pairs(topics, _rig_tree(), vehicle_frame="base"), "lidar-lidar")

    assert (record.status, record.reason_code) == ("skipped", "missing_topic")


def test_two_topics_of_one_lidar_are_one_sensor() -> None:
    topics = [_topic("/pc", "lidar", "lf"), _topic("/custom", "lidar", "lf")]

    (record,) = _by_pair(plan_pairs(topics, _rig_tree()), "lidar-lidar")

    assert record.reason_code == "missing_topic"


def test_skip_no_candidate_calibration() -> None:
    records = plan_pairs(_rig_topics(), _tree(), vehicle_frame="base")

    for record in _by_pair(records, "imu-lidar") + _by_pair(records, "lidar-vehicle"):
        assert (record.status, record.reason_code) == ("skipped", "no_candidate_calibration")


def test_vehicle_pairs_are_opt_in() -> None:
    records = plan_pairs(_rig_topics(), _rig_tree())

    for pair in ("lidar-vehicle", "imu-vehicle"):
        (record,) = _by_pair(records, pair)
        assert (record.status, record.reason_code) == ("skipped", "no_vehicle_frame")
        assert "--vehicle-frame" in (record.reason or "")
    assert _by_pair(records, "imu-lidar")[0].status == "planned"


def test_vehicle_frame_absent_from_tree_is_frame_not_in_tree() -> None:
    records = plan_pairs(_rig_topics(), _rig_tree(), vehicle_frame="base_footprint")

    for pair in ("lidar-vehicle", "imu-vehicle"):
        assert {r.reason_code for r in _by_pair(records, pair)} == {"frame_not_in_tree"}
        assert {r.status for r in _by_pair(records, pair)} == {"skipped"}


def test_skip_frame_not_in_tree_unmapped_and_absent() -> None:
    topics = _rig_topics()
    topics[1] = _topic("/lidar_f", "lidar", None, header="driver_frame")
    topics[2] = _topic("/lidar_r", "lidar", "ghost")

    records = plan_pairs(topics, _rig_tree(), vehicle_frame="base")

    imu_lidar = _by_pair(records, "imu-lidar")
    assert [(r.status, r.reason_code) for r in imu_lidar] == [("skipped", "frame_not_in_tree")] * 2
    reasons = " ".join(r.reason or "" for r in imu_lidar)
    assert "driver_frame" in reasons and "ghost" in reasons
    # reported once per sensor, not once per combination
    assert len(imu_lidar) == 2
    assert _by_pair(records, "camera-imu")[0].status == "planned"


def test_skip_frames_not_connected() -> None:
    tree = _tree(("base", "imu", 0.0), ("other", "lf", 1.0))
    topics = [_topic("/imu", "imu", "imu"), _topic("/lidar_f", "lidar", "lf")]

    (record,) = _by_pair(plan_pairs(topics, tree), "imu-lidar")

    assert (record.status, record.reason_code) == ("skipped", "frames_not_connected")


def test_skip_method_not_wired() -> None:
    records = plan_pairs(
        _rig_topics(),
        _rig_tree(),
        vehicle_frame="base",
        wired_pairs=frozenset({"imu-lidar"}),
    )

    (record,) = _by_pair(records, "camera-imu")
    assert (record.status, record.reason_code) == ("skipped", "method_not_wired")
    assert _by_pair(records, "imu-lidar")[0].status == "planned"


def test_unknown_odometry_kind_is_reported() -> None:
    topics = [_topic("/lio/odometry", "odometry", "x", kind="unknown"), _topic("/l", "lidar", "lf")]

    (record,) = _by_pair(plan_pairs(topics, _rig_tree()), "ins-lidar")

    assert record.reason_code == "missing_topic"
    assert "/lio/odometry" in (record.reason or "")
