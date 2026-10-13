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
