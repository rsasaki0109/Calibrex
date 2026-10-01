"""The vehicle pairs of ``calibrex check`` on a simulated vehicle bag.

The real estimators run (lidar-vehicle, imu-vehicle, ins-lidar, lidar-wheel_odometry);
only the LiDAR scan odometry is injected, because it needs scans, and its
bag reader is covered by the KITTI evaluation. With the true mounts in
``/tf_static`` every pair passes; a 3 degree yaw error on the LiDAR mount is
failed by every pair that involves the LiDAR and not by the pair that does not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tests.unit import vehicle_fixtures as vf

from calibrex.check import estimators, vehicle_inputs
from calibrex.check.cache import EstimatorCache
from calibrex.check.estimators import CheckSkipError, PairContext, RunControls
from calibrex.check.runner import CheckRunOptions, build_calibration_check
from calibrex.core.calibration_check import CalibrationCheckArtifact, CheckPairRecord
from calibrex.core.ins_lidar_hand_eye import load_ins_lidar_hand_eye
from calibrex.core.lidar_wheel_odometry import load_lidar_wheel_odometry
from calibrex.core.vehicle_frame_rotation import load_vehicle_frame_rotation

VEHICLE_PAIRS = ("lidar-vehicle", "imu-vehicle", "ins-lidar", "lidar-wheel_odometry")


@pytest.fixture(scope="module")
def simulation() -> vf.Simulation:
    return vf.simulate(duration_s=60.0)


@pytest.fixture
def odometry_calls(monkeypatch: pytest.MonkeyPatch, simulation: vf.Simulation) -> list[str]:
    calls: list[str] = []

    def fake(
        bag: Path, topic: str, options: Any = None, *, max_duration_s: float | None = None
    ) -> vehicle_inputs.LidarOdometryTrack:
        calls.append(topic)
        return vf.lidar_track(simulation, topic)

    monkeypatch.setattr(vehicle_inputs, "read_lidar_odometry", fake)
    return calls


def run_check(
    bag: Path,
    out: Path,
    *,
    pairs: tuple[str, ...] = VEHICLE_PAIRS,
    cache: Path | None = None,
    topic_kinds: dict[str, Any] | None = None,
    messages: list[str] | None = None,
) -> CalibrationCheckArtifact:
    return build_calibration_check(
        bag,
        vehicle_frame="base_link",
        topic_kinds=topic_kinds,
        run=CheckRunOptions(
            pairs=pairs, evidence_dir=out / "evidence", base_dir=out, cache_dir=cache
        ),
        progress=(messages.append if messages is not None else (lambda _m: None)),
    )


def pair(artifact: CalibrationCheckArtifact, name: str) -> CheckPairRecord:
    (record,) = [item for item in artifact.pairs if item.pair == name]
    return record


@pytest.fixture(scope="module")
def good_artifact(tmp_path_factory: pytest.TempPathFactory, simulation: vf.Simulation) -> Any:
    root = tmp_path_factory.mktemp("good")
    bag = vf.write_vehicle_bag(root / "bag", simulation)
    patch = pytest.MonkeyPatch()
    patch.setattr(
        vehicle_inputs,
        "read_lidar_odometry",
        lambda bag, topic, options=None, max_duration_s=None: vf.lidar_track(simulation, topic),
    )
    try:
        return run_check(bag, root), root
    finally:
        patch.undo()


def test_true_mounts_pass_on_every_vehicle_pair(good_artifact: Any) -> None:
    artifact, root = good_artifact
    for name in VEHICLE_PAIRS:
        record = pair(artifact, name)
        assert record.status == "pass", (name, record.reason, record.estimator_policy_reasons)
        assert record.axes, name
        assert all(axis.status == "pass" and axis.ratio < 1.0 for axis in record.axes), name
        assert record.runtime_s is not None
        assert (root / record.evidence[0].path).is_file()
    # lidar-vehicle accounts for all three rotations of the sensor against base_link: judged,
    # or listed as unchecked (roll needs turns the drive may not have)
    lidar = pair(artifact, "lidar-vehicle")
    assert {a.name for a in lidar.axes} >= {"pitch", "yaw"}
    assert {a.name for a in lidar.axes} | {a.name for a in lidar.unchecked_axes} == {
        "roll",
        "pitch",
        "yaw",
    }
    # ins-lidar also judges the lever arm, and lists the vertical axis the planar drive cannot see
    ins = pair(artifact, "ins-lidar")
    assert {a.name for a in ins.axes} == {"roll", "pitch", "yaw", "x", "y"}
    assert [(a.name, a.reason_code) for a in ins.unchecked_axes] == [("z", "unobservable")]
    assert ins.coverage == "partial"
    assert artifact.overall_verdict == "pass"


def test_estimates_are_in_the_documented_conventions(good_artifact: Any) -> None:
    artifact, _root = good_artifact
    # vehicle pairs report the sensor's pose in the vehicle frame
    for name in ("lidar-vehicle", "imu-vehicle", "lidar-wheel_odometry"):
        record = pair(artifact, name)
        assert record.compared_transform is not None
        assert record.compared_transform.parent_frame == "base_link"
        assert record.compared_transform.child_frame == record.frames[0]
    lidar = pair(artifact, "lidar-vehicle")
    assert lidar.compared_transform is not None
    np.testing.assert_allclose(
        Rotation.from_quat(lidar.compared_transform.rotation_quat_xyzw).as_matrix(),
        vf.R_BASE_VELO,
        atol=1e-9,
    )
    # ins-lidar is T_ins_lidar, as planned
    ins = pair(artifact, "ins-lidar")
    assert ins.compared_transform is not None
    assert (ins.compared_transform.parent_frame, ins.compared_transform.child_frame) == (
        "imu_link",
        "velo_link",
    )
    assert ins.time_offset is not None
    wheel = pair(artifact, "lidar-wheel_odometry")
    assert wheel.time_offset is not None


def test_evidence_artifacts_are_native_schema_artifacts_with_provenance(good_artifact: Any) -> None:
    artifact, root = good_artifact
    loaders = {
        "lidar-vehicle": load_vehicle_frame_rotation,
        "imu-vehicle": load_vehicle_frame_rotation,
        "ins-lidar": load_ins_lidar_hand_eye,
        "lidar-wheel_odometry": load_lidar_wheel_odometry,
    }
    for name, loader in loaders.items():
        evidence = pair(artifact, name).evidence[0]
        loaded = loader(root / evidence.path)
        assert loaded.provenance.input_sha256 == artifact.bag.input_sha256
        assert loaded.provenance.generator == "calibrex.check.estimators"
        assert loaded.provenance.git_commit is None or loaded.provenance.git_commit
    assert (
        load_vehicle_frame_rotation(
            root / pair(artifact, "lidar-vehicle").evidence[0].path
        ).sensor_modality
        == "lidar"
    )
    assert (
        load_vehicle_frame_rotation(
            root / pair(artifact, "imu-vehicle").evidence[0].path
        ).sensor_modality
        == "ins"
    )


def test_a_yaw_error_on_the_lidar_mount_fails_every_pair_that_uses_the_lidar(
    tmp_path: Path, simulation: vf.Simulation, odometry_calls: list[str]
) -> None:
    yaw = Rotation.from_euler("z", 3.0, degrees=True).as_matrix()
    bad = vf.write_vehicle_bag(
        tmp_path / "bad",
        simulation,
        tf_edges=[
            ("base_link", "imu_link", vf.T_BASE_IMU),
            ("base_link", "velo_link", vf._transform(yaw @ vf.R_BASE_VELO, vf.LEVER_VELO)),
        ],
    )

    artifact = run_check(bad, tmp_path)

    for name in ("lidar-vehicle", "ins-lidar", "lidar-wheel_odometry"):
        record = pair(artifact, name)
        assert record.status == "fail", name
        failed = {a.name: a for a in record.axes if a.status == "fail"}
        assert "yaw" in failed, name
        assert abs(abs(failed["yaw"].candidate_error) - 3.0) < 0.1, name
    # the IMU mount is untouched, so the pair that does not involve the LiDAR still passes
    assert pair(artifact, "imu-vehicle").status == "pass"
    assert artifact.overall_verdict == "fail"
    # one LiDAR odometry pass is shared by the three LiDAR pairs
    assert odometry_calls == ["/velodyne_points"]


def test_cache_skips_the_estimators_and_odometry_on_the_second_run(
    tmp_path: Path, simulation: vf.Simulation, odometry_calls: list[str]
) -> None:
    bag = vf.write_vehicle_bag(tmp_path / "bag", simulation)
    cache = tmp_path / "cache"
    first = run_check(bag, tmp_path / "a", pairs=("lidar-vehicle", "imu-vehicle"), cache=cache)
    assert [pair(first, n).evidence_from_cache for n in ("lidar-vehicle", "imu-vehicle")] == [
        False,
        False,
    ]
    assert odometry_calls == ["/velodyne_points"]

    second = run_check(bag, tmp_path / "b", pairs=("lidar-vehicle", "imu-vehicle"), cache=cache)

    assert [pair(second, n).evidence_from_cache for n in ("lidar-vehicle", "imu-vehicle")] == [
        True,
        True,
    ]
    assert odometry_calls == ["/velodyne_points"]  # not read again
    assert pair(second, "lidar-vehicle").axes == pair(first, "lidar-vehicle").axes
    assert EstimatorCache(cache).directory.exists()


# ------------------------------------------------- inputs and skip reasons


def context(bag: Path, tmp_path: Path, name: str) -> PairContext:
    artifact = build_calibration_check(bag, vehicle_frame="base_link")
    record = pair(artifact, name)
    assert record.candidate_transform is not None
    from calibrex.check.runner import _sensor_topics

    return PairContext(
        bag=bag,
        pair=record,
        topic_types={t.topic: t.message_type for t in artifact.topics},
        sensor_topics=_sensor_topics(record, artifact.topics),
        candidate=estimators.transform_matrix(
            record.candidate_transform.translation_m,
            record.candidate_transform.rotation_quat_xyzw,
        ),
        sources=[],
        controls=RunControls(),
        topics=artifact.topics,
    )


def test_imu_vehicle_needs_an_ins_velocity_source(
    tmp_path: Path, simulation: vf.Simulation
) -> None:
    # a raw IMU alone carries no velocity: the only odometry topic has an unknown kind
    bag = vf.write_vehicle_bag(tmp_path / "bag", simulation, ins_topic="/lio/odometry")
    with pytest.raises(CheckSkipError) as skipped:
        estimators.run_imu_vehicle(context(bag, tmp_path, "imu-vehicle"))
    assert skipped.value.code == "missing_topic"
    assert "--topic-kind" in skipped.value.reason

    artifact = run_check(
        bag,
        tmp_path / "out",
        pairs=("imu-vehicle",),
        topic_kinds={"/lio/odometry": "ins"},
    )
    assert pair(artifact, "imu-vehicle").status == "pass"
    (topic,) = [t for t in artifact.topics if t.topic == "/lio/odometry"]
    assert topic.odometry_kind == "ins" and "--topic-kind" in topic.notes[0]


def test_imu_vehicle_skips_when_the_ins_is_not_in_the_imu_frame(
    tmp_path: Path, simulation: vf.Simulation
) -> None:
    bag = vf.write_vehicle_bag(tmp_path / "bag", simulation, ins_frame="velo_link")
    artifact = build_calibration_check(
        bag,
        vehicle_frame="base_link",
        run=CheckRunOptions(pairs=("imu-vehicle",), base_dir=tmp_path),
    )
    record = pair(artifact, "imu-vehicle")
    assert (record.status, record.reason_code) == ("skipped", "unsupported_sensor")
    assert "imu_link" in (record.reason or "")


def test_imu_vehicle_skips_a_zero_twist(tmp_path: Path, simulation: vf.Simulation) -> None:
    bag = vf.write_vehicle_bag(tmp_path / "bag", simulation, zero_ins_twist=True)
    artifact = run_check(bag, tmp_path / "out", pairs=("imu-vehicle",))
    record = pair(artifact, "imu-vehicle")
    assert (record.status, record.reason_code) == ("skipped", "unsupported_sensor")
    assert "zero velocity" in (record.reason or "")


def test_wheel_pair_requires_a_wheel_classified_topic(
    tmp_path: Path, simulation: vf.Simulation
) -> None:
    bag = vf.write_vehicle_bag(tmp_path / "bag", simulation, wheel_topic="/vehicle/twist")
    plan = build_calibration_check(bag, vehicle_frame="base_link")
    record = pair(plan, "lidar-wheel_odometry")
    assert (record.status, record.reason_code) == ("skipped", "missing_topic")
    assert "--topic-kind" in (record.reason or "")

    forced = build_calibration_check(
        bag, vehicle_frame="base_link", topic_kinds={"/vehicle/twist": "wheel"}
    )
    assert pair(forced, "lidar-wheel_odometry").status == "planned"


def test_unknown_topic_kind_target_is_rejected(tmp_path: Path, simulation: vf.Simulation) -> None:
    from calibrex.core.exceptions import DatasetError

    bag = vf.write_vehicle_bag(tmp_path / "bag", simulation)
    with pytest.raises(DatasetError, match="--topic-kind"):
        build_calibration_check(bag, topic_kinds={"/oxts/imu": "ins"})
    with pytest.raises(DatasetError, match="--topic-kind"):
        build_calibration_check(bag, topic_kinds={"/not/a/topic": "wheel"})


def test_lidar_pairs_need_pointcloud2(tmp_path: Path, simulation: vf.Simulation) -> None:
    bag = vf.write_vehicle_bag(tmp_path / "bag", simulation)
    ctx = context(bag, tmp_path, "lidar-vehicle")
    object.__setattr__(ctx, "topic_types", {"/velodyne_points": "livox_ros_driver2/msg/CustomMsg"})
    for adapter in (estimators.run_lidar_vehicle,):
        with pytest.raises(CheckSkipError) as skipped:
            adapter(ctx)
        assert skipped.value.code == "unsupported_sensor"


def test_ins_lidar_needs_an_odometry_pose_not_a_twist(
    tmp_path: Path, simulation: vf.Simulation
) -> None:
    bag = vf.write_vehicle_bag(tmp_path / "bag", simulation)
    ctx = context(bag, tmp_path, "lidar-wheel_odometry")
    # the planner gave this pair the wheel twist; ins-lidar cannot use a twist as a pose
    ins_ctx = PairContext(
        bag=bag,
        pair=ctx.pair,
        topic_types=ctx.topic_types,
        sensor_topics={"lidar": ("/velodyne_points",), "ins": ("/wheel/twist",)},
        candidate=np.eye(4),
        sources=[],
        controls=RunControls(),
        topics=ctx.topics,
    )
    with pytest.raises(CheckSkipError) as skipped:
        estimators.run_ins_lidar(ins_ctx)
    assert skipped.value.code == "unsupported_sensor"
    assert "nav_msgs/Odometry" in skipped.value.reason


def test_vehicle_inputs_read_the_bag(tmp_path: Path, simulation: vf.Simulation) -> None:
    bag = vf.write_vehicle_bag(tmp_path / "bag", simulation)

    ins = vehicle_inputs.read_ins_track(bag, "/oxts/odometry")
    assert ins.body_frame == "imu_link"
    assert len(ins.times_s) == len(simulation.times_s)
    np.testing.assert_allclose(
        ins.poses[5], simulation.vehicle_poses[5] @ vf.T_BASE_IMU, atol=1e-12
    )
    np.testing.assert_allclose(ins.twist.times_s, ins.times_s)

    twist = vehicle_inputs.read_twist_track(
        bag, "/wheel/twist", "geometry_msgs/msg/TwistStamped", max_duration_s=5.0
    )
    assert twist.frame == "base_link"
    assert twist.times_s[-1] - twist.times_s[0] <= 5.0 + 1e-9
    assert np.all(twist.linear_mps[:, 0] > 5.0)  # forward speed
    odometry_twist = vehicle_inputs.read_twist_track(bag, "/oxts/odometry", "nav_msgs/msg/Odometry")
    assert odometry_twist.frame == "imu_link"
    with pytest.raises(ValueError, match="carries no twist"):
        vehicle_inputs.read_twist_track(bag, "/oxts/imu", "sensor_msgs/msg/Imu")


def test_streams_are_split_at_gaps() -> None:
    times = [0.0, 0.1, 0.2, 0.7, 0.8, 300.0, 300.1]

    pieces = vehicle_inputs.split_at_gaps(times)

    assert [(p.start, p.stop) for p in pieces] == [(0, 5), (5, 7)]  # 0.5 s dropouts are not gaps
    assert vehicle_inputs.split_at_gaps([]) == []
    assert [(p.start, p.stop) for p in vehicle_inputs.split_at_gaps(times, 0.3)] == [
        (0, 3),
        (3, 5),
        (5, 7),
    ]


def test_a_pooled_bag_keeps_motions_and_blocks_inside_each_segment(
    tmp_path: Path, simulation: vf.Simulation
) -> None:
    track = vf.lidar_track(simulation)
    half = len(track.times_s) // 2
    pooled = vehicle_inputs.LidarOdometryTrack(
        topic=track.topic,
        times_s=track.times_s[:half] + tuple(t + 600.0 for t in track.times_s[half:]),
        poses=track.poses[:half]
        + tuple(np.linalg.inv(track.poses[half]) @ p for p in track.poses[half:]),
        registrations=(),
        options=track.options,
        segments=((0, half), (half, len(track.times_s))),
    )

    motions = estimators._lidar_motions(pooled, step=2, block_duration_s=10.0)

    assert len(pooled.segment_slices()) == 2
    blocks = {m.block for m in motions}
    assert min(blocks) == 0 and max(blocks) >= estimators.SEGMENT_BLOCK_STRIDE
    assert not any(m.end_s - m.start_s > 1.0 for m in motions)  # nothing spans the 600 s gap
    assert len(motions) == 2 * (half - 2)


def test_lidar_odometry_from_a_bag_equals_the_loader_path_and_splits_gaps(tmp_path: Path) -> None:
    from tests.unit.test_ins_lidar_hand_eye import _motion, _scene, _view

    from calibrex.data.ros_cdr_writer import POINT_FIELD_FLOAT32, encode_pointcloud2
    from calibrex.data.rosbag2_writer import Rosbag2Writer
    from calibrex.solvers.scan_to_scan_odometry import odometry_from_loader

    scene = _scene()
    step = _motion(0.9, 1.0)
    poses = [np.linalg.matrix_power(step, number) for number in range(5)]
    scans = [_view(scene, pose).astype(np.float32) for pose in poses]
    fields = tuple((name, 4 * i, POINT_FIELD_FLOAT32, 1) for i, name in enumerate("xyz"))
    base = 1_700_000_000 * 10**9
    stamps = [base + k * 10**8 for k in range(5)] + [
        base + 600 * 10**9 + k * 10**8 for k in range(5)
    ]
    with Rosbag2Writer(tmp_path / "bag") as writer:
        writer.add_topic("/points", "sensor_msgs/msg/PointCloud2")
        for stamp, scan in zip(stamps, scans + scans, strict=True):
            writer.write(
                "/points",
                stamp,
                encode_pointcloud2(
                    frame_id="lidar",
                    timestamp_ns=stamp,
                    fields=fields,
                    point_step=12,
                    data=np.ascontiguousarray(scan[:, :3], dtype="<f4").tobytes(),
                ),
            )

    track = vehicle_inputs.read_lidar_odometry(tmp_path / "bag", "/points")
    reference = odometry_from_loader(
        lambda index: scans[index].astype(np.float64),
        5,
        track.options,
        times_s=[stamp * 1e-9 for stamp in stamps[:5]],
    )

    assert track.segments == ((0, 5), (5, 10))
    for ours, theirs in zip(track.poses[:5], reference.poses, strict=True):
        np.testing.assert_array_equal(ours, theirs)
    for ours, theirs in zip(track.poses[5:], reference.poses, strict=True):
        np.testing.assert_array_equal(ours, theirs)  # a new segment restarts at the identity
    np.testing.assert_array_equal(track.poses[5], np.eye(4))
    assert len(track.registrations) == 8  # 4 per segment, none across the gap

    limited = vehicle_inputs.read_lidar_odometry(tmp_path / "bag", "/points", max_duration_s=0.25)
    assert len(limited.times_s) == 3
