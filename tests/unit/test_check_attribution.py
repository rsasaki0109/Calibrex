"""Which frame's candidate rotation the pair errors of a check run point to."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibrex.check.attribution import (
    AttributionObservation,
    build_rotation_attribution,
    observations_from_run,
)
from calibrex.check.closure import edge_from_estimates
from calibrex.check.verdict import AxisEstimate, VerdictOptions
from calibrex.core.calibration_check import (
    CalibrationCheckArtifact,
    CheckPairRecord,
    CheckRotationAttribution,
    CheckTransform,
)

OPTIONS = VerdictOptions()
# A Hilti-like head: two forward cameras, one down, one to either side, and the IMU.
RIG = {
    "imu": Rotation.identity(),
    "cam0": Rotation.from_euler("xyz", (-90.0, 0.0, -90.0), degrees=True),
    "cam1": Rotation.from_euler("xyz", (-90.0, 0.0, -90.5), degrees=True),
    "cam2": Rotation.from_euler("xyz", (180.0, 0.0, -90.0), degrees=True),
    "cam3": Rotation.from_euler("xyz", (-90.0, 0.0, 0.0), degrees=True),
    "cam4": Rotation.from_euler("xyz", (-90.0, 0.0, 180.0), degrees=True),
}
CAMERAS = ("cam0", "cam1", "cam2", "cam3", "cam4")


def _observations(
    moved: dict[str, tuple[float, float, float]],
    pairs: list[tuple[str, str]],
    *,
    noise_deg: float = 0.03,
    std_deg: float = 0.03,
    observed: tuple[bool, bool, bool] = (True, True, True),
    seed: int = 0,
) -> list[AttributionObservation]:
    """Pair errors of a rig whose ``moved`` frames' candidates are turned by ``w``."""

    rng = np.random.default_rng(seed)
    truth = {
        frame: rotation * Rotation.from_rotvec(-np.radians(moved.get(frame, (0.0, 0.0, 0.0))))
        for frame, rotation in RIG.items()
    }
    observations = []
    for parent, child in pairs:
        candidate = (RIG[parent].inv() * RIG[child]).as_matrix()
        estimate = (
            Rotation.from_rotvec(np.radians(rng.normal(0.0, noise_deg, 3)))
            * truth[parent].inv()
            * truth[child]
        ).as_matrix()
        observations.append(
            AttributionObservation(
                label=f"pair:{parent}>{child}",
                parent=parent,
                child=child,
                candidate_rotation=candidate,
                error_deg=np.degrees(Rotation.from_matrix(candidate @ estimate.T).as_rotvec()),
                std_deg=np.full(3, std_deg),
                observed=observed,
            )
        )
    return observations


ALL_CAMERA_PAIRS = list(itertools.combinations(CAMERAS, 2))
CAMERA_IMU_PAIRS = [(camera, "imu") for camera in CAMERAS]


def _only_group(report: CheckRotationAttribution | None):  # type: ignore[no-untyped-def]
    assert report is not None
    (group,) = report.groups
    return group


def test_a_rig_with_nothing_off_is_consistent() -> None:
    group = _only_group(
        build_rotation_attribution(_observations({}, ALL_CAMERA_PAIRS + CAMERA_IMU_PAIRS), OPTIONS)
    )
    assert group.status == "consistent"
    assert group.observations == 15 * 3
    assert group.frame is None


@pytest.mark.parametrize("frame", ["cam1", "cam3", "imu"])
def test_one_turned_frame_is_named_with_its_turn(frame: str) -> None:
    # cam1 is a child of cam0 and a parent of cam2-4; the IMU is a child only
    turn = (0.8, -0.5, 0.3)
    report = build_rotation_attribution(
        _observations({frame: turn}, ALL_CAMERA_PAIRS + CAMERA_IMU_PAIRS), OPTIONS
    )
    group = _only_group(report)
    assert group.status == "attributed", group.summary
    assert group.frame == frame
    best = group.hypotheses[0]
    assert best.frame == frame
    assert best.rotation_deg == pytest.approx(list(turn), abs=0.05)
    assert all(std is not None and std < 0.1 for std in best.std_deg)
    assert group.null_p_value < 1.0e-6


def test_a_turn_below_the_verdicts_resolution_is_consistent() -> None:
    # 0.1 deg is under the 0.5 deg rotation floor: the pairs pass and nothing is named
    group = _only_group(
        build_rotation_attribution(
            _observations({"cam1": (0.1, 0.0, 0.0)}, ALL_CAMERA_PAIRS), OPTIONS
        )
    )
    assert group.status == "consistent"


def test_two_turned_frames_are_not_blamed_on_one() -> None:
    group = _only_group(
        build_rotation_attribution(
            _observations({"cam1": (1.0, 0.0, 0.0), "cam3": (0.0, 0.0, -1.0)}, ALL_CAMERA_PAIRS),
            OPTIONS,
        )
    )
    assert group.status == "unexplained", group.summary


def test_a_single_pair_cannot_tell_its_frames_apart() -> None:
    group = _only_group(
        build_rotation_attribution(
            _observations({"cam1": (1.0, 0.0, 0.0)}, [("cam0", "cam1")]), OPTIONS
        )
    )
    assert group.status == "ambiguous"
    assert sorted(group.candidates) == ["cam0", "cam1"]


def test_an_axis_no_pair_observes_is_left_undetermined() -> None:
    # cam0 and cam1 share their axes; only roll is observed on their pairs
    observations = _observations({}, [("cam0", "cam2"), ("cam1", "cam2")])
    observations += _observations(
        {"cam1": (1.0, 0.0, 0.0)}, [("cam0", "cam1")], observed=(True, False, False)
    )
    group = _only_group(build_rotation_attribution(observations, OPTIONS))
    hypothesis = next(item for item in group.hypotheses if item.frame == "cam1")
    assert hypothesis.rotation_deg[0] is not None
    assert hypothesis.rotation_deg[1] is not None  # cam1-cam2 observes all three
    lone = _only_group(
        build_rotation_attribution(
            _observations(
                {"cam1": (1.0, 0.0, 0.0)}, [("cam0", "cam1")], observed=(True, False, False)
            ),
            OPTIONS,
        )
    )
    cam1 = next(item for item in lone.hypotheses if item.frame == "cam1")
    assert cam1.rotation_deg[0] == pytest.approx(1.0, abs=0.1)
    assert cam1.rotation_deg[1:] == [None, None]
    assert cam1.std_deg[1:] == [None, None]


def test_separate_sensor_groups_are_attributed_separately() -> None:
    cameras = _observations({"cam3": (0.0, 1.2, 0.0)}, ALL_CAMERA_PAIRS)
    lidars = _observations({}, [("imu", "cam0")])  # stands in for an unrelated group
    lidars = [
        AttributionObservation(
            label="lidar-lidar:lidar_a>lidar_b",
            parent="lidar_a",
            child="lidar_b",
            candidate_rotation=np.eye(3),
            error_deg=np.zeros(3),
            std_deg=np.full(3, 0.05),
            observed=(True, True, True),
        )
        for _ in lidars
    ]
    report = build_rotation_attribution(cameras + lidars, OPTIONS)
    assert report is not None
    statuses = {tuple(group.frames): group.status for group in report.groups}
    assert statuses == {tuple(sorted(CAMERAS)): "attributed", ("lidar_a", "lidar_b"): "consistent"}


def _record(pair: str, parent: str, child: str, status: str = "fail") -> CheckPairRecord:
    quat = (RIG[parent].inv() * RIG[child]).as_quat()
    return CheckPairRecord.model_validate(
        {
            "pair": pair,
            "frames": [parent, child],
            "status": status,
            "compared_transform": CheckTransform(
                parent_frame=parent,
                child_frame=child,
                translation_m=[0.0, 0.0, 0.0],
                rotation_quat_xyzw=[float(v) for v in quat],
            ).model_dump(),
        }
    )


def test_run_records_give_the_errors_their_estimators_reported() -> None:
    errors = {"roll": 0.9, "pitch": -0.2, "yaw": 0.4}
    records = [
        _record("camera-camera", "cam0", "cam1"),
        _record("camera-imu", "cam0", "imu", status="inconclusive"),
        _record("camera-focal", "cam0", "cam0"),
    ]
    edges = {}
    for record in records:
        transform = record.compared_transform
        assert transform is not None
        compared = np.eye(4)
        compared[:3, :3] = Rotation.from_quat(transform.rotation_quat_xyzw).as_matrix()
        estimates = [
            AxisEstimate(name, "deg", value, 0.05, estimated=(name != "yaw"))
            for name, value in errors.items()  # type: ignore[misc]
        ]
        edges[(record.pair, tuple(record.frames))] = edge_from_estimates(
            record.pair, transform.parent_frame, transform.child_frame, compared, estimates
        )
    (observation,) = observations_from_run(records, edges)  # inconclusive, focal left out
    assert observation.label == "camera-camera:cam0>cam1"
    assert observation.observed == (True, True, False)
    assert observation.error_deg[:2] == pytest.approx([0.9, -0.2], abs=1.0e-6)


def test_the_report_is_part_of_a_valid_check_artifact() -> None:
    schema = CalibrationCheckArtifact.model_json_schema()
    assert "rotation_attribution" in schema["properties"]
    report = build_rotation_attribution(
        _observations({"cam1": (0.8, 0.0, 0.0)}, ALL_CAMERA_PAIRS), OPTIONS
    )
    assert report is not None
    assert CheckRotationAttribution.model_validate(report.model_dump(mode="json")) == report
    assert report.sigma_floor_deg == pytest.approx(0.5 / 3.0)


def test_a_barely_constrained_direction_is_not_fitted() -> None:
    # b is seen through one pitch row nearly orthogonal to its own y axis (cos 89.5 deg)
    # and one roll row: its y turn is pinned down only to ~20 deg, so it stays undetermined
    # instead of absorbing the errors with a far-fetched turn
    nearly_square = Rotation.from_euler("x", 89.5, degrees=True).as_matrix()
    observations = [
        AttributionObservation(
            label="p:a>b",
            parent="a",
            child="b",
            candidate_rotation=nearly_square,
            error_deg=np.array([0.0, 0.3, 0.0]),
            std_deg=np.full(3, 0.05),
            observed=(False, True, False),
        ),
        AttributionObservation(
            label="p:b>c",
            parent="b",
            child="c",
            candidate_rotation=np.eye(3),
            error_deg=np.array([0.1, 0.0, 0.0]),
            std_deg=np.full(3, 0.05),
            observed=(True, False, False),
        ),
    ]
    group = _only_group(build_rotation_attribution(observations, OPTIONS))
    b = next(item for item in group.hypotheses if item.frame == "b")
    assert b.rotation_deg[1] is None and b.std_deg[1] is None
    assert all(value is None or abs(value) < 1.0 for value in b.rotation_deg)


# A LiDAR rig with lever arms: an IMU, two LiDARs a metre or so away, and a GNSS antenna.
LIDAR_RIG = {
    "imu": (Rotation.identity(), np.zeros(3)),
    "lidar_a": (
        Rotation.from_euler("xyz", (0.0, 0.0, 90.0), degrees=True),
        np.array([1.2, 0.0, 0.8]),
    ),
    "lidar_b": (
        Rotation.from_euler("xyz", (0.0, 10.0, -90.0), degrees=True),
        np.array([-0.4, 0.9, 0.6]),
    ),
    "gnss": (Rotation.identity(), np.array([0.3, -0.5, 1.4])),
}
LIDAR_PAIRS = [("imu", "lidar_a"), ("imu", "lidar_b"), ("lidar_a", "lidar_b"), ("gnss", "lidar_a")]


def _rig_observations(
    turned: dict[str, tuple[float, float, float]],
    moved: dict[str, tuple[float, float, float]],
    pairs: list[tuple[str, str]],
    *,
    std_deg: float = 0.03,
    std_m: float = 0.002,
    seed: int = 0,
) -> list[AttributionObservation]:
    """Exact pair errors of a rig whose candidates are turned by ``w`` and moved by ``u``."""

    rng = np.random.default_rng(seed)
    truth = {}
    for frame, (rotation, position) in LIDAR_RIG.items():
        w = np.radians(turned.get(frame, (0.0, 0.0, 0.0)))
        u = np.array(moved.get(frame, (0.0, 0.0, 0.0)))
        truth[frame] = (rotation * Rotation.from_rotvec(-w), position - rotation.apply(u))

    def relative(poses: dict, parent: str, child: str) -> tuple[Rotation, np.ndarray]:  # type: ignore[type-arg]
        (rp, pp), (rc, pc) = poses[parent], poses[child]
        return rp.inv() * rc, rp.inv().apply(pc - pp)

    observations = []
    for parent, child in pairs:
        r_cand, t_cand = relative(LIDAR_RIG, parent, child)
        r_true, t_true = relative(truth, parent, child)
        r_est = Rotation.from_rotvec(np.radians(rng.normal(0.0, 0.3 * std_deg, 3))) * r_true
        t_est = t_true + rng.normal(0.0, 0.3 * std_m, 3)
        observations.append(
            AttributionObservation(
                label=f"lidar:{parent}>{child}",
                parent=parent,
                child=child,
                candidate_rotation=r_cand.as_matrix(),
                error_deg=np.degrees((r_cand * r_est.inv()).as_rotvec()),
                std_deg=np.full(3, std_deg),
                observed=(True, True, True),
                candidate_translation=t_cand,
                translation_error_m=t_cand - t_est,
                translation_std_m=np.full(3, std_m),
                translation_observed=(True, True, True),
            )
        )
    return observations


def test_a_moved_frame_is_named_with_its_shift() -> None:
    shift = (0.08, -0.05, 0.03)
    group = _only_group(
        build_rotation_attribution(_rig_observations({}, {"lidar_b": shift}, LIDAR_PAIRS), OPTIONS)
    )
    assert group.status == "attributed", group.summary
    assert group.frame == "lidar_b"
    assert group.translation_observations == 12
    best = group.hypotheses[0]
    assert best.translation_m == pytest.approx(list(shift), abs=0.003)
    assert best.rotation_deg == pytest.approx([0.0, 0.0, 0.0], abs=0.05)
    assert "moved by x +0.080" in group.summary


def test_a_parent_s_turn_swings_its_children_on_their_lever_arms() -> None:
    # 1 deg about the IMU's z moves lidar_a (1.2 m out) by 2 cm in the imu-lidar pair: that
    # is the IMU turned, not the LiDAR moved
    turn = (0.0, 0.0, 1.0)
    observations = _rig_observations({"imu": turn}, {}, LIDAR_PAIRS)
    assert np.linalg.norm(observations[0].translation_error_m) == pytest.approx(0.021, abs=0.002)
    group = _only_group(build_rotation_attribution(observations, OPTIONS))
    assert group.status == "attributed", group.summary
    assert group.frame == "imu"
    best = group.hypotheses[0]
    assert best.rotation_deg == pytest.approx(list(turn), abs=0.05)
    assert best.translation_m == pytest.approx([0.0, 0.0, 0.0], abs=0.003)


def test_a_turned_and_moved_frame_is_named_with_both() -> None:
    turn, shift = (0.0, 0.8, 0.0), (0.0, 0.0, -0.06)
    group = _only_group(
        build_rotation_attribution(
            _rig_observations({"lidar_a": turn}, {"lidar_a": shift}, LIDAR_PAIRS), OPTIONS
        )
    )
    assert group.status == "attributed", group.summary
    best = group.hypotheses[0]
    assert best.frame == "lidar_a"
    assert best.rotation_deg == pytest.approx(list(turn), abs=0.05)
    assert best.translation_m == pytest.approx(list(shift), abs=0.003)


def test_a_shift_below_the_translation_floor_is_consistent() -> None:
    group = _only_group(
        build_rotation_attribution(
            _rig_observations({}, {"lidar_b": (0.004, 0.0, 0.0)}, LIDAR_PAIRS), OPTIONS
        )
    )
    assert group.status == "consistent", group.summary
    assert "rotation or translation" in group.summary


def test_two_moved_frames_are_not_blamed_on_one() -> None:
    group = _only_group(
        build_rotation_attribution(
            _rig_observations(
                {}, {"lidar_a": (0.06, 0.0, 0.0), "lidar_b": (0.0, 0.0, 0.07)}, LIDAR_PAIRS
            ),
            OPTIONS,
        )
    )
    assert group.status == "unexplained", group.summary


def test_a_frame_no_translation_pair_reaches_has_no_shift() -> None:
    # gnss>lidar_a is the only pair with translation; imu's shift is not constrained
    rotation_only = [
        AttributionObservation(
            label=item.label,
            parent=item.parent,
            child=item.child,
            candidate_rotation=item.candidate_rotation,
            error_deg=item.error_deg,
            std_deg=item.std_deg,
            observed=item.observed,
        )
        for item in _rig_observations({}, {}, LIDAR_PAIRS[:3])
    ]
    observations = rotation_only + _rig_observations({}, {}, [("gnss", "lidar_a")])
    group = _only_group(build_rotation_attribution(observations, OPTIONS))
    imu = next(item for item in group.hypotheses if item.frame == "imu")
    assert imu.translation_m == [None, None, None]
    gnss = next(item for item in group.hypotheses if item.frame == "gnss")
    assert all(value is not None for value in gnss.translation_m or [])
    # without any translation pair the shift is absent, not undetermined
    lone = _only_group(build_rotation_attribution(rotation_only, OPTIONS))
    assert all(item.translation_m is None for item in lone.hypotheses)
    assert lone.translation_observations == 0


def test_run_records_give_the_translation_their_estimators_reported() -> None:
    record = CheckPairRecord.model_validate(
        {
            "pair": "imu-lidar",
            "frames": ["imu", "lidar_a"],
            "status": "pass",
            "compared_transform": CheckTransform(
                parent_frame="imu",
                child_frame="lidar_a",
                translation_m=[1.2, 0.0, 0.8],
                rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
            ).model_dump(),
        }
    )
    compared = np.eye(4)
    compared[:3, 3] = [1.2, 0.0, 0.8]
    estimates = [
        AxisEstimate(name, "deg", 0.1, 0.05, estimated=True) for name in ("roll", "pitch", "yaw")
    ]
    estimates += [
        AxisEstimate("x", "m", 0.03, 0.004, estimated=True),
        AxisEstimate("y", "m", -0.01, 0.004, estimated=True),
        AxisEstimate("z", "m", 0.0, 0.5, estimated=False),
    ]
    edge = edge_from_estimates("imu-lidar", "imu", "lidar_a", compared, estimates)
    (observation,) = observations_from_run([record], {("imu-lidar", ("imu", "lidar_a")): edge})
    assert observation.translation_observed == (True, True, False)
    assert observation.translation_error_m is not None
    assert observation.translation_error_m[:2] == pytest.approx([0.03, -0.01])
    assert observation.candidate_translation == pytest.approx([1.2, 0.0, 0.8])
    report = build_rotation_attribution([observation], OPTIONS)
    assert report is not None
    assert report.sigma_floor_m == pytest.approx(0.02 / 3.0)
    assert CheckRotationAttribution.model_validate(report.model_dump(mode="json")) == report
