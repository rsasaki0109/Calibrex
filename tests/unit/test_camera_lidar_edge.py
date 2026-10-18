"""Targetless camera-LiDAR edge alignment on a synthetic scene with known extrinsics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tests.unit.camera_lidar_scene import (
    CX,
    CY,
    FOCAL,
    cast_points,
    make_scene,
    perturbed,
    render_image,
    solid_state_directions,
    true_transform,
)

from calibrex.core.camera_lidar_edge import (
    CameraLidarEdgeArtifact,
    CameraLidarEdgeControl,
    CameraLidarEdgeProvenance,
    camera_lidar_edge_json_schema,
    load_camera_lidar_edge,
)
from calibrex.data.ros2_camera_lidar import CameraLidarFrame
from calibrex.evaluation.camera_lidar_edge import (
    REP103_OPTICAL_FROM_BODY,
    EdgeAlignmentOptions,
    EdgeCamera,
    EdgeObjective,
    _profile_start,
    estimate_from_frames,
    image_edge_response,
    lidar_depth_edges,
    prepare_frames,
)

CAMERA = EdgeCamera(FOCAL, FOCAL, CX, CY)
FAST = EdgeAlignmentOptions(min_frames=8, blocks=4)


def _provenance() -> CameraLidarEdgeProvenance:
    return CameraLidarEdgeProvenance(
        generator="test",
        generator_version="0",
        dataset_family="synthetic",
        sequence_ids=["scene"],
        image_topic="/image",
        lidar_topic="/points",
        input_sha256="0" * 64,
        input_digest_scope="test",
        dataset_license="none",
    )


def _frames(truth: np.ndarray, count: int = 12) -> list[CameraLidarFrame]:
    frames = []
    for index in range(count):
        rects = make_scene(index)
        frames.append(
            CameraLidarFrame(
                time_s=float(index),
                image_time_s=float(index),
                image=render_image(rects),
                points=cast_points(rects, truth),
                cloud_frame_id="lidar",
                image_frame_id="camera",
            )
        )
    return frames


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.degrees(Rotation.from_matrix(first[:3, :3] @ second[:3, :3].T).as_rotvec())


@pytest.fixture(scope="module")
def truth() -> np.ndarray:
    return true_transform()


@pytest.fixture(scope="module")
def frames(truth: np.ndarray) -> list[CameraLidarFrame]:
    return _frames(truth)


@pytest.fixture(scope="module")
def recovered(truth: np.ndarray, frames: list[CameraLidarFrame]) -> CameraLidarEdgeArtifact:
    candidate = perturbed(truth, (0.8, -1.2, 0.0))
    return estimate_from_frames(frames, CAMERA, candidate, options=FAST, provenance=_provenance())


def test_lidar_depth_edges_find_the_rectangle_borders(truth: np.ndarray) -> None:
    rects = make_scene(3)
    points = cast_points(rects, truth)
    edge_points, weights, rings, on_beam = lidar_depth_edges(
        points, min_jump_m=0.3, exponent=0.5, max_points=4000, min_ring_fraction=0.5
    )
    assert 28 <= rings <= 36 and on_beam > 0.9
    assert 20 < len(edge_points) < len(points) // 4
    assert weights.shape == (len(edge_points),) and np.all(weights > 0.5)
    # every edge point sits on the near side: it belongs to a rectangle in front of a farther one
    ranges = np.linalg.norm(edge_points, axis=1)
    assert np.all(ranges < 29.0)


def test_unstructured_cloud_gives_no_edges() -> None:
    rng = np.random.default_rng(0)
    cloud = rng.uniform(-10.0, 10.0, (5000, 3)).astype(np.float32)
    edge_points, _weights, _rings, _fraction = lidar_depth_edges(
        cloud, min_jump_m=0.3, exponent=0.5, max_points=4000, min_ring_fraction=0.95
    )
    assert edge_points.shape == (0, 3)


def test_image_edge_response_is_one_on_edges_and_decays(truth: np.ndarray) -> None:
    image = render_image(make_scene(1))
    result = image_edge_response(image, max_width=960, gamma=0.98, radius=40)
    assert result is not None
    response, factor = result
    assert factor == 1 and response.shape == image.shape and response.dtype == np.float32
    assert float(response.min()) >= 0.0 and float(response.max()) <= 1.0 + 1e-6
    flat = np.full_like(image, 100)
    assert image_edge_response(flat, max_width=960, gamma=0.98, radius=40) is None


def test_camera_projection_models() -> None:
    points = np.array([[0.5, -0.25, 2.0], [0.0, 0.0, -1.0]])
    u, v, valid = EdgeCamera(100.0, 110.0, 50.0, 60.0).project(points)
    assert valid.tolist() == [True, False]
    assert (u[0], v[0]) == pytest.approx((75.0, 46.25))
    # equidistant: theta_d = theta (1 + k1 theta^2 ...); zero coefficients is the pinhole angle
    fisheye = EdgeCamera(100.0, 100.0, 0.0, 0.0, "equidistant", (0.0, 0.0, 0.0, 0.0))
    u2, _v2, _ = fisheye.project(np.array([[1.0, 0.0, 1.0]]))
    assert u2[0] == pytest.approx(100.0 * np.pi / 4)
    radtan = EdgeCamera(100.0, 100.0, 0.0, 0.0, "radtan", (0.1, 0.0, 0.0, 0.0))
    u3, _v3, _ = radtan.project(np.array([[0.5, 0.0, 1.0]]))
    assert u3[0] == pytest.approx(100.0 * 0.5 * (1.0 + 0.1 * 0.25))


def test_objective_peaks_at_the_true_extrinsic(
    truth: np.ndarray, frames: list[CameraLidarFrame]
) -> None:
    prepared, dropped = prepare_frames(frames, CAMERA, truth, FAST)
    assert not dropped and len(prepared) == len(frames)
    objective = EdgeObjective(prepared, CAMERA, truth)
    at_truth = objective.score(np.zeros(6))
    for axis in range(3):
        for sign in (-1.0, 1.0):
            theta = np.zeros(6)
            theta[axis] = sign * 1.0
            assert objective.score(theta) < at_truth


def test_recovers_a_rotation_offset(truth: np.ndarray, recovered: CameraLidarEdgeArtifact) -> None:
    assert recovered.transform is not None
    estimate = np.eye(4)
    estimate[:3, :3] = Rotation.from_quat(recovered.transform.rotation_quat_xyzw).as_matrix()
    estimate[:3, 3] = recovered.transform.translation_m
    error = np.abs(_rotation_error_deg(estimate, truth))
    assert error[0] < 0.3 and error[1] < 0.3 and error[2] < 0.6  # the optical axis is weakest
    # the start (candidate) is recorded; translation is left at the candidate's
    assert recovered.start_transform is not None
    assert recovered.transform.translation_m == pytest.approx(
        recovered.start_transform.translation_m
    )
    assert recovered.objective is not None and recovered.objective.gain > 0.0
    assert recovered.samples.frames == 12 and recovered.jackknife_fits == FAST.blocks


def test_rotation_axes_are_observed_with_detected_controls(
    recovered: CameraLidarEdgeArtifact,
) -> None:
    by_name = {record.name: record for record in recovered.dofs}
    rotation = [by_name[name] for name in ("roll", "pitch", "yaw")]
    assert all(record.known_bad_control is not None for record in by_name.values())
    assert [record.status for record in rotation].count("estimated") >= 2
    for record in rotation:
        control = record.known_bad_control
        assert control is not None and control.unit == "deg"
        if record.status == "estimated":
            assert control.detected and record.std_reported <= 0.5
    # translation is never estimated by default, whatever its control says
    for name in ("x", "y", "z"):
        assert by_name[name].status == "unobservable"
        assert by_name[name].delta_from_start == pytest.approx(0.0, abs=1e-12)
    assert recovered.policy_status in {"pass", "warn"}
    assert sorted(recovered.calibrated_dofs) == sorted(
        record.name for record in recovered.dofs if record.status == "estimated"
    )


def test_artifact_round_trips_and_matches_its_schema(
    recovered: CameraLidarEdgeArtifact, tmp_path: Path
) -> None:
    path = tmp_path / "edge.yaml"
    recovered.save(path)
    loaded = load_camera_lidar_edge(path)
    assert loaded == recovered
    assert loaded.schema_version == "slac.camera_lidar_edge/v0.1"
    assert loaded.provenance.generator and loaded.provenance.input_sha256
    schema = camera_lidar_edge_json_schema()
    assert "dofs" in schema["properties"] and "provenance" in schema["properties"]


def test_a_pass_cannot_claim_unobserved_rotation(recovered: CameraLidarEdgeArtifact) -> None:
    payload = recovered.model_dump(mode="json")
    payload["policy_status"] = "pass"
    payload["calibrated_dofs"] = ["roll"]
    for record in payload["dofs"]:
        record["status"] = "estimated" if record["name"] == "roll" else "unobservable"
    with pytest.raises(ValueError, match="rotation DoFs"):
        CameraLidarEdgeArtifact.model_validate(payload)


def test_too_few_frames_is_inconclusive_with_the_reason(
    truth: np.ndarray, frames: list[CameraLidarFrame]
) -> None:
    artifact = estimate_from_frames(
        frames[:3], CAMERA, truth, options=FAST, provenance=_provenance()
    )
    assert artifact.solver_status == "insufficient_samples"
    assert artifact.policy_status == "inconclusive"
    assert artifact.transform is None and artifact.calibrated_dofs == []
    assert "usable frame" in artifact.policy_reasons[0]


def test_candidate_that_sees_nothing_is_reported_not_crashed(
    truth: np.ndarray, frames: list[CameraLidarFrame]
) -> None:
    # a candidate that points the camera backwards puts every LiDAR point behind it
    backwards = perturbed(truth, (0.0, 180.0, 0.0))
    artifact = estimate_from_frames(
        frames, CAMERA, backwards, options=FAST, provenance=_provenance()
    )
    assert artifact.solver_status == "insufficient_samples"
    assert any("fall in the image under the candidate" in r for r in artifact.policy_reasons)
    assert not any("REP 103 body frame" in r for r in artifact.policy_reasons)


def test_image_in_the_camera_body_frame_is_named(
    truth: np.ndarray, frames: list[CameraLidarFrame]
) -> None:
    # the same extrinsic, given for the camera's body frame (x forward) as a tree with
    # the image published in the body frame would compose it
    body_from_optical = np.eye(4)
    body_from_optical[:3, :3] = REP103_OPTICAL_FROM_BODY.T
    artifact = estimate_from_frames(
        frames,
        CAMERA,
        body_from_optical @ truth,
        options=FAST,
        provenance=_provenance(),
        parent_frame="camera_link",
    )
    assert artifact.solver_status == "insufficient_samples"
    reasons = [r for r in artifact.policy_reasons if "REP 103 body frame" in r]
    assert len(reasons) == 1
    assert "'camera_link'" in reasons[0] and "12 of 12 frames" in reasons[0]
    assert "--frame-map" in reasons[0]


def test_options_validate() -> None:
    with pytest.raises(ValueError, match="search levels"):
        EdgeAlignmentOptions(rotation_levels_deg=(1.0,), translation_levels_m=(0.1, 0.2))
    with pytest.raises(ValueError, match="three blocks"):
        EdgeAlignmentOptions(blocks=2)


def test_solid_state_cloud_edges_are_found_among_angular_neighbours(truth: np.ndarray) -> None:
    points = cast_points(make_scene(3), truth, solid_state_directions(7))
    beams_only = lidar_depth_edges(
        points, min_jump_m=0.3, exponent=0.5, max_points=4000, min_ring_fraction=0.5
    )
    assert beams_only[0].shape == (0, 3)
    edge_points, weights, rings, _fraction = lidar_depth_edges(
        points,
        min_jump_m=0.3,
        exponent=0.5,
        max_points=4000,
        min_ring_fraction=0.5,
        unstructured=True,
    )
    assert rings == 0
    assert 100 < len(edge_points) < len(points) // 10
    assert weights.shape == (len(edge_points),)
    # Near-side points of the rectangles (5-16 m), not the 30 m wall seen at a grazing angle.
    assert float(np.linalg.norm(edge_points, axis=1).max()) < 17.0


def test_solid_state_frames_recover_the_extrinsic(truth: np.ndarray) -> None:
    frames = [
        CameraLidarFrame(
            time_s=float(index),
            image_time_s=float(index),
            image=render_image(make_scene(index)),
            points=cast_points(make_scene(index), truth, solid_state_directions(100 + index)),
            cloud_frame_id="lidar",
            image_frame_id="camera",
        )
        for index in range(12)
    ]
    candidate = perturbed(truth, (0.8, -1.2, 0.0))
    artifact = estimate_from_frames(
        frames, CAMERA, candidate, options=FAST, provenance=_provenance()
    )
    assert artifact.policy_status == "pass"
    assert artifact.samples.median_rings == 0
    assert artifact.transform is not None
    estimate = Rotation.from_quat(artifact.transform.rotation_quat_xyzw).as_matrix()
    assert np.all(np.abs(_rotation_error_deg(estimate, truth)) < 0.3)


class _TwoPeaks:
    """A pitch profile with a local maximum at +1 deg and the global one at -5 deg."""

    def score(self, theta: np.ndarray) -> float:
        pitch = float(theta[1])
        return float(
            0.6 * np.exp(-((pitch - 1.0) ** 2))
            + np.exp(-((pitch + 5.0) ** 2) / 2.0)
            - 0.01 * (theta[0] ** 2 + theta[2] ** 2)
        )


def test_profile_start_takes_each_axis_maximum() -> None:
    start = _profile_start(_TwoPeaks(), EdgeAlignmentOptions())  # type: ignore[arg-type]
    assert start is not None
    assert start[1] == pytest.approx(-5.0)
    assert start[0] == 0.0 and start[2] == 0.0 and not np.any(start[3:])
    off = EdgeAlignmentOptions(start_scan_deg=None)
    assert _profile_start(_TwoPeaks(), off) is None  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="start scan"):
        EdgeAlignmentOptions(start_scan_deg=20.0)


def test_each_profile_maximum_is_also_a_start_of_its_own() -> None:
    from calibrex.evaluation.camera_lidar_edge import _profile_starts

    scan = np.array([-3.0, 8.0, 0.0, 0.0, 0.0, 0.0])
    starts = _profile_starts(scan)
    # roll alone, pitch alone, then both: a camera turned about one axis is found from that
    # axis's maximum even when another axis's profile only rises towards the scan's end
    assert [list(start[:3]) for start in starts] == [
        [-3.0, 0.0, 0.0],
        [0.0, 8.0, 0.0],
        [-3.0, 8.0, 0.0],
    ]
    assert _profile_starts(np.array([0.0, -2.5, 0.0, 0.0, 0.0, 0.0]))[0][1] == -2.5
    assert len(_profile_starts(np.array([0.0, -2.5, 0.0, 0.0, 0.0, 0.0]))) == 1
    assert _profile_starts(np.zeros(6)) == []
    assert _profile_starts(None) == []


def test_a_far_optimum_replaces_the_current_one_only_with_more_axes_and_none_at_the_bound() -> None:
    from calibrex.evaluation.camera_lidar_edge import _replaces_current

    options = EdgeAlignmentOptions()
    inside = np.array([-3.0, 0.2, 0.1, 0.0, 0.0, 0.0])
    controls = [_control(4.0)] * 6
    assert _replaces_current((1,), (0, 1, 2), inside, controls, options)
    assert _replaces_current((), (0, 1), inside, controls, options)
    # as many DoFs observed: the objective no longer decides (KITTI 0029, yaw 9.6 deg off)
    assert not _replaces_current((0, 1), (0, 1), inside, controls, options)
    # a single DoF observed is not enough (a turned run of KITTI 0056, roll 6.6 deg off)
    assert not _replaces_current((), (0,), inside, controls, options)
    assert _replaces_current(
        (), (0,), inside, controls, EdgeAlignmentOptions(start_scan_min_observed=1)
    )
    # a rotation DoF at the search bound (KITTI 0056, yaw at -12 deg)
    at_bound = inside.copy()
    at_bound[2] = -options.rotation_bound_deg
    assert not _replaces_current((), (0, 1, 2), at_bound, controls, options)
    with pytest.raises(ValueError, match="start_scan_min_observed"):
        EdgeAlignmentOptions(start_scan_min_observed=4)


def _control(t: float | None) -> CameraLidarEdgeControl:
    return CameraLidarEdgeControl(
        amount=2.0,
        unit="deg",
        held_out_blocks=4,
        detected_blocks=4,
        mean_objective_drop=0.01,
        min_t_statistic=t,
        detected=t is not None and t >= 2.5,
    )


@pytest.mark.parametrize("t", [2.65, 3.499999, None, float("nan"), float("inf")])
def test_far_control_margin_checks_every_observed_axis(t: float | None) -> None:
    from calibrex.evaluation.camera_lidar_edge import _replaces_current

    options = EdgeAlignmentOptions(start_scan_min_control_t=3.5)
    theta = np.zeros(6)
    # The weak axis was already observed; the newly observed axis is strong.
    # Checking only newly observed axes would admit this optimum incorrectly.
    controls = [_control(3.5), _control(t)] + [_control(None)] * 4
    assert not _replaces_current((1,), (0, 1), theta, controls, options)


def test_far_control_margin_includes_boundary_and_ignores_unobserved_axes() -> None:
    from calibrex.evaluation.camera_lidar_edge import _replaces_current

    options = EdgeAlignmentOptions(start_scan_min_control_t=3.5)
    controls = [_control(3.5), _control(4.0)] + [_control(None)] * 4
    assert _replaces_current((1,), (0, 1), np.zeros(6), controls, options)
    assert options.as_dict()["start_scan_min_control_t"] == 3.5
    assert EdgeAlignmentOptions().as_dict()["start_scan_min_control_t"] is None


@pytest.mark.parametrize("margin", [2.49, -1.0, float("nan"), float("inf")])
def test_far_control_margin_cannot_weaken_the_ordinary_control(margin: float) -> None:
    with pytest.raises(ValueError, match="start_scan_min_control_t"):
        EdgeAlignmentOptions(start_scan_min_control_t=margin)
    with pytest.raises(ValueError, match="start_scan_min_control_t"):
        EdgeAlignmentOptions(min_control_t=4.0, start_scan_min_control_t=3.5)


@pytest.mark.parametrize(
    "margin, far_control_t, expected",
    [(None, 2.65, ["roll", "pitch"]), (3.5, 2.65, ["pitch"]), (3.5, 3.5, ["roll", "pitch"])],
)
def test_rejected_far_fit_retains_current_transform_and_ordinary_evidence(
    monkeypatch: pytest.MonkeyPatch,
    margin: float | None,
    far_control_t: float,
    expected: list[str],
) -> None:
    from calibrex.evaluation import camera_lidar_edge as edge

    current = np.array([0.0, 0.05, 0.0, 0.0, 0.0, 0.0])
    far = np.array([-1.5, -0.15, -5.3, 0.0, 0.0, 0.0])
    current_controls = [_control(0.4), _control(3.0), _control(1.0)] + [_control(None)] * 3
    far_controls = [_control(far_control_t), _control(4.0), _control(1.23)] + [_control(None)] * 3
    monkeypatch.setattr(edge, "_profile_start", lambda *_: np.ones(6))
    monkeypatch.setattr(edge, "_profile_starts", lambda *_: [np.ones(6)])
    monkeypatch.setattr(
        edge, "fit_extrinsic", lambda _, start, opts: (current if not start.any() else far, 1.0)
    )

    def evidence(*args: object) -> tuple[np.ndarray, np.ndarray, list[CameraLidarEdgeControl]]:
        theta = args[3]
        controls = current_controls if theta is current else far_controls
        return np.tile(theta, (4, 1)), np.full(6, 0.025), controls

    monkeypatch.setattr(edge, "_evidence", evidence)
    truth = true_transform()
    artifact = estimate_from_frames(
        _frames(truth, count=8),
        CAMERA,
        truth,
        options=EdgeAlignmentOptions(min_frames=8, blocks=4, start_scan_min_control_t=margin),
        provenance=_provenance(),
    )
    assert artifact.solver_status == "converged"
    assert artifact.calibrated_dofs == expected
    rejected = expected == ["pitch"]
    selected = current if rejected else far
    assert [d.delta_from_start for d in artifact.dofs[:3]] == list(selected[:3])
    assert artifact.dofs[1].known_bad_control == (
        current_controls[1] if rejected else far_controls[1]
    )
    assert artifact.options["start_scan_min_control_t"] == margin
    assert any("profile maxima" in reason for reason in artifact.policy_reasons) == (not rejected)


class _BlockDrops:
    """A held-out block whose objective drops by a set amount when an axis is shifted."""

    def __init__(self, drops: dict[int, float]) -> None:
        self.drops = drops

    def score(self, theta: np.ndarray) -> float:
        return -sum(drop for axis, drop in self.drops.items() if theta[axis] != 0.0)


def test_a_control_needs_t_of_2_5() -> None:
    from calibrex.evaluation.camera_lidar_edge import _controls

    assert EdgeAlignmentOptions().min_control_t == 2.5
    # per-block drops for roll and pitch whose t statistics are 2.33 and 2.62
    roll = [-0.5488, 0.4512, 1.4512, 2.4512, 0.9512, 0.9512]
    pitch = [-0.4304, 0.5696, 1.5696, 2.5696, 1.0696, 1.0696]
    blocks = [_BlockDrops({0: r, 1: p}) for r, p in zip(roll, pitch, strict=True)]
    fits = np.zeros((6, 6))
    t = [float(np.mean(d) / (np.std(d, ddof=1) / np.sqrt(6))) for d in (roll, pitch)]
    assert t[0] == pytest.approx(2.33, abs=0.01) and t[1] == pytest.approx(2.62, abs=0.01)
    controls = _controls(blocks, fits, EdgeAlignmentOptions())  # type: ignore[arg-type]
    assert [controls[0].detected, controls[1].detected] == [False, True]
    earlier = _controls(blocks, fits, EdgeAlignmentOptions(min_control_t=2.0))  # type: ignore[arg-type]
    assert earlier[0].detected  # the bound before KITTI's first audit judged it


def test_observed_rotations_counts_axes_by_the_record_rule() -> None:
    from calibrex.core.camera_lidar_edge import CameraLidarEdgeControl
    from calibrex.evaluation.camera_lidar_edge import _observed_rotation_axes

    def control(detected: bool) -> CameraLidarEdgeControl:
        return CameraLidarEdgeControl(
            amount=2.0,
            unit="deg",
            held_out_blocks=6,
            detected_blocks=6 if detected else 2,
            mean_objective_drop=0.01,
            min_t_statistic=3.0 if detected else 0.5,
            detected=detected,
        )

    options = EdgeAlignmentOptions()
    floor = np.full(6, 0.025)
    controls = [control(True), control(False), control(True)] + [control(False)] * 3
    jackknife = np.array([0.1, 0.1, 0.8, 0.01, 0.01, 0.01])
    theta = np.zeros(6)
    # roll: observed; pitch: control missed; yaw: std above the 0.5 deg bound.
    assert _observed_rotation_axes(theta, jackknife, controls, floor, options) == (0,)
    theta[0] = options.rotation_bound_deg
    assert _observed_rotation_axes(theta, jackknife, controls, floor, options) == ()


def test_background_subtraction_stops_the_pull_towards_texture() -> None:
    # A faint edge where the LiDAR's edge points belong, above a busy texture: the raw
    # response is higher anywhere in the texture than on the faint edge, so a search
    # would leave the calibration for the texture (KITTI highway and city drives did).
    rng = np.random.default_rng(3)
    image = np.full((300, 400), 100, dtype=np.uint8)
    image[:150, 200:] = 130
    image[150:] = rng.integers(0, 256, (150, 400), dtype=np.uint8)
    rows = np.arange(20, 131)

    def mean_response(response: np.ndarray, du: int, dv: int) -> float:
        return float(response[rows + dv, 200 + du].mean())

    raw = image_edge_response(image, max_width=960, gamma=0.98, radius=40)
    flat = image_edge_response(image, max_width=960, gamma=0.98, radius=40, background_sigma=20)
    assert raw is not None and flat is not None
    assert mean_response(raw[0], 0, 160) > mean_response(raw[0], 0, 0)
    at_edge = mean_response(flat[0], 0, 0)
    assert at_edge > mean_response(flat[0], 0, 160)
    assert at_edge > max(mean_response(flat[0], du, 0) for du in (-40, -20, 20, 40))


def test_the_background_width_is_an_option() -> None:
    assert EdgeAlignmentOptions().edge_background_sigma_px == 20.0
    assert EdgeAlignmentOptions(edge_background_sigma_px=None).edge_background_sigma_px is None


def _repeated(truth: np.ndarray, solid_state: bool) -> list[CameraLidarFrame]:
    """Frames 0, 0, 0, 1, 2: a rig standing at scene 0 for three frames, then moving."""

    frames = []
    for time, index in enumerate((0, 0, 0, 1, 2)):
        directions = solid_state_directions(100 + time) if solid_state else None
        frames.append(
            CameraLidarFrame(
                time_s=float(time),
                image_time_s=float(time),
                image=render_image(make_scene(index)),
                points=cast_points(make_scene(index), truth, directions),
                cloud_frame_id="lidar",
                image_frame_id="camera",
            )
        )
    return frames


def test_a_spinning_lidar_at_a_standing_rig_repeats_one_view(truth: np.ndarray) -> None:
    prepared, dropped = prepare_frames(_repeated(truth, False), CAMERA, truth, FAST)
    assert [frame.time_s for frame in prepared] == [0.0, 3.0, 4.0]
    assert len(dropped) == 2 and all("repeats the last kept view" in item for item in dropped)
    kept, none_dropped = prepare_frames(
        _repeated(truth, False), CAMERA, truth, EdgeAlignmentOptions(repeated_view_change=None)
    )
    assert len(kept) == 5 and not none_dropped


def test_a_solid_state_lidar_at_a_standing_rig_keeps_its_frames(truth: np.ndarray) -> None:
    prepared, dropped = prepare_frames(_repeated(truth, True), CAMERA, truth, FAST)
    assert len(prepared) == 5 and not dropped
