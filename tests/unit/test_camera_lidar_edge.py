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
    true_transform,
)

from calibrex.core.camera_lidar_edge import (
    CameraLidarEdgeArtifact,
    CameraLidarEdgeProvenance,
    camera_lidar_edge_json_schema,
    load_camera_lidar_edge,
)
from calibrex.data.ros2_camera_lidar import CameraLidarFrame
from calibrex.evaluation.camera_lidar_edge import (
    EdgeAlignmentOptions,
    EdgeCamera,
    EdgeObjective,
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


def test_options_validate() -> None:
    with pytest.raises(ValueError, match="search levels"):
        EdgeAlignmentOptions(rotation_levels_deg=(1.0,), translation_levels_m=(0.1, 0.2))
    with pytest.raises(ValueError, match="three blocks"):
        EdgeAlignmentOptions(blocks=2)
