"""camera-lidar in ``calibrex check`` / ``estimate`` / ``drift`` on a synthetic rosbag2.

The bag is written by ``Rosbag2Writer`` from the synthetic scene of ``camera_lidar_scene``:
twelve scans (one per second) with the image rendered at the same instant, the scene's
camera intrinsics as a ``CameraInfo``, and the true extrinsic (or a perturbed one) in
``/tf_static``.  The planner, the estimator adapter, the evidence artifact, the cache, the
verdict rule, ``estimate`` with its prior gate and the hints all run for real.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation
from tests.unit.camera_lidar_scene import (
    CX,
    CY,
    FOCAL,
    HEIGHT,
    WIDTH,
    cast_points,
    make_scene,
    perturbed,
    render_image,
    true_transform,
)

from calibrex.check.estimate import NEEDS_INITIAL_GUESS
from calibrex.check.hints import pair_hint
from calibrex.check.runner import CheckRunOptions, build_calibration_check
from calibrex.check.verdict import AxisEstimate, VerdictOptions, judge_pair
from calibrex.cli.main import main
from calibrex.core.calibration_check import CHECK_PAIR_NAMES, CalibrationCheckArtifact
from calibrex.core.camera_lidar_edge import load_camera_lidar_edge
from calibrex.core.validation import validate_file
from calibrex.data.ros_cdr_writer import (
    POINT_FIELD_FLOAT32,
    encode_camera_info,
    encode_image,
    encode_pointcloud2,
    encode_tf_message,
)
from calibrex.data.rosbag2_writer import Rosbag2Writer

CAMERA_FRAME = "camera_optical"
LIDAR_FRAME = "lidar"
FRAMES = 12
FIELDS = (
    ("x", 0, POINT_FIELD_FLOAT32, 1),
    ("y", 4, POINT_FIELD_FLOAT32, 1),
    ("z", 8, POINT_FIELD_FLOAT32, 1),
    ("intensity", 12, POINT_FIELD_FLOAT32, 1),
)
FAST_OPTIONS = {"max_duration_s": None}


def _tf_payload(t_camera_lidar: np.ndarray) -> bytes:
    quat = Rotation.from_matrix(t_camera_lidar[:3, :3]).as_quat()
    transform = (
        CAMERA_FRAME,
        LIDAR_FRAME,
        tuple(float(v) for v in t_camera_lidar[:3, 3]),
        tuple(float(v) for v in quat),
    )
    return encode_tf_message([transform])


def write_bag(path: Path, truth: np.ndarray, *, candidate: np.ndarray | None = None) -> Path:
    """A bag whose data follow ``truth`` and whose /tf_static says ``candidate`` (default truth)."""

    with Rosbag2Writer(path) as writer:
        writer.add_topic("/tf_static", "tf2_msgs/msg/TFMessage", latched=True)
        writer.add_topic("/camera/image_raw", "sensor_msgs/msg/Image")
        writer.add_topic("/camera/camera_info", "sensor_msgs/msg/CameraInfo")
        writer.add_topic("/lidar/points", "sensor_msgs/msg/PointCloud2")
        writer.write("/tf_static", 1, _tf_payload(truth if candidate is None else candidate))
        for index in range(FRAMES):
            stamp = 10_000_000_000 + index * 1_000_000_000
            rects = make_scene(index)
            image = render_image(rects)
            points = cast_points(rects, truth)
            intensity = np.full((points.shape[0], 1), 0.5, dtype=np.float32)
            packed = np.ascontiguousarray(np.hstack([points, intensity]), dtype=np.float32)
            writer.write(
                "/camera/camera_info",
                stamp,
                encode_camera_info(
                    frame_id=CAMERA_FRAME,
                    timestamp_ns=stamp,
                    height=HEIGHT,
                    width=WIDTH,
                    k=(FOCAL, 0.0, CX, 0.0, FOCAL, CY, 0.0, 0.0, 1.0),
                ),
            )
            writer.write(
                "/camera/image_raw",
                stamp + 5_000_000,
                encode_image(
                    frame_id=CAMERA_FRAME,
                    timestamp_ns=stamp + 5_000_000,
                    height=HEIGHT,
                    width=WIDTH,
                    encoding="mono8",
                    step=WIDTH,
                    data=image.tobytes(),
                ),
            )
            writer.write(
                "/lidar/points",
                stamp,
                encode_pointcloud2(
                    frame_id=LIDAR_FRAME,
                    timestamp_ns=stamp,
                    fields=FIELDS,
                    point_step=16,
                    data=packed.tobytes(),
                ),
            )
    return path


@pytest.fixture(scope="module")
def truth() -> np.ndarray:
    return true_transform()


@pytest.fixture(scope="module")
def good_bag(tmp_path_factory: pytest.TempPathFactory, truth: np.ndarray) -> Path:
    return write_bag(tmp_path_factory.mktemp("cl") / "good", truth)


@pytest.fixture(scope="module")
def bad_bag(tmp_path_factory: pytest.TempPathFactory, truth: np.ndarray) -> Path:
    return write_bag(
        tmp_path_factory.mktemp("cl") / "bad", truth, candidate=perturbed(truth, (0.0, 3.0, 0.0))
    )


def _run(bag: Path, tmp_path: Path, **kwargs: object) -> CalibrationCheckArtifact:
    run = CheckRunOptions(
        pairs=("camera-lidar",),
        evidence_dir=tmp_path / "evidence",
        base_dir=tmp_path,
        **kwargs,  # type: ignore[arg-type]
    )
    return build_calibration_check(bag, run=run)


# ------------------------------------------------------------------- planner and verdict


def test_pair_is_registered_and_planned(good_bag: Path) -> None:
    assert "camera-lidar" in CHECK_PAIR_NAMES
    plan = build_calibration_check(good_bag)
    (record,) = [p for p in plan.pairs if p.pair == "camera-lidar"]
    assert record.status == "planned"
    assert record.frames == [CAMERA_FRAME, LIDAR_FRAME]  # T_camera_lidar: parent camera
    assert record.topics == ["/camera/image_raw", "/lidar/points"]


def test_hint_for_missing_lidar_names_the_message_type() -> None:
    from calibrex.core.calibration_check import CheckPairRecord

    record = CheckPairRecord(
        pair="camera-lidar",
        status="skipped",
        reason_code="missing_topic",
        reason="no topic for: lidar",
    )
    hint = pair_hint(record)
    assert hint is not None and "PointCloud2" in hint


def test_rotation_axes_are_judged_and_translation_is_unchecked_by_the_verdict_rule() -> None:
    estimates = [
        AxisEstimate("roll", "deg", 0.1, 0.05, True),
        AxisEstimate("pitch", "deg", 0.9, 0.05, True),
        AxisEstimate("yaw", "deg", 0.0, 0.2, False, "control", "control_not_detected"),
        AxisEstimate("x", "m", 0.0, 0.0, False, "translation is not estimated", "unobservable"),
    ]
    judged = judge_pair(estimates, VerdictOptions())
    assert judged.verdict == "warn" and judged.coverage == "partial"
    assert [axis.name for axis in judged.axes] == ["roll", "pitch"]
    assert {axis.name: axis.status for axis in judged.axes} == {"roll": "pass", "pitch": "warn"}
    assert [(u.name, u.reason_code) for u in judged.unchecked] == [
        ("yaw", "control_not_detected"),
        ("x", "unobservable"),
    ]


# ------------------------------------------------------------------------ the full check


def test_check_passes_the_true_extrinsic_and_writes_valid_evidence(
    good_bag: Path, tmp_path: Path
) -> None:
    artifact = _run(good_bag, tmp_path)
    (record,) = [p for p in artifact.pairs if p.pair == "camera-lidar"]
    assert record.status == "pass" and record.verdict == "pass"
    assert record.estimator is not None and record.estimator.startswith("edge_alignment/v0.1")
    assert record.coverage == "partial"  # translation is never judged
    assert {axis.name for axis in record.axes} <= {"roll", "pitch", "yaw"} and record.axes
    assert all(axis.status == "pass" for axis in record.axes)
    unchecked = {item.name: item for item in record.unchecked_axes}
    assert {"x", "y", "z"} <= set(unchecked)
    assert all(unchecked[name].reason_code == "unobservable" for name in "xyz")
    assert record.compared_transform is not None
    assert (record.compared_transform.parent_frame, record.compared_transform.child_frame) == (
        CAMERA_FRAME,
        LIDAR_FRAME,
    )
    # evidence: schema-valid artifact with provenance, digest-bound to the check record
    (evidence,) = record.evidence
    assert evidence.schema_version == "slac.camera_lidar_edge/v0.1"
    path = tmp_path / evidence.path
    validate_file(path)
    edge = load_camera_lidar_edge(path)
    assert edge.provenance.input_sha256 == artifact.bag.input_sha256
    assert edge.provenance.image_topic == "/camera/image_raw"
    assert edge.samples.frames == FRAMES
    assert record.runtime_s is not None


def test_check_flags_a_pitch_error_of_three_degrees(bad_bag: Path, tmp_path: Path) -> None:
    artifact = _run(bad_bag, tmp_path)
    (record,) = [p for p in artifact.pairs if p.pair == "camera-lidar"]
    assert record.status in {"warn", "fail"}
    judged = {axis.name: axis for axis in record.axes}
    assert "pitch" in judged and judged["pitch"].status == "fail"
    assert abs(judged["pitch"].candidate_error) == pytest.approx(3.0, abs=0.6)
    assert artifact.overall_verdict in {"warn", "fail"}


def test_estimator_cache_is_keyed_on_the_candidate(
    good_bag: Path, bad_bag: Path, tmp_path: Path
) -> None:
    cache = tmp_path / "cache"
    first = _run(good_bag, tmp_path / "a", cache_dir=cache)
    again = _run(good_bag, tmp_path / "b", cache_dir=cache)
    (record_first,) = [p for p in first.pairs if p.pair == "camera-lidar"]
    (record_again,) = [p for p in again.pairs if p.pair == "camera-lidar"]
    assert record_first.evidence_from_cache is False
    assert record_again.evidence_from_cache is True
    assert [a.model_dump() for a in record_again.axes] == [
        a.model_dump() for a in record_first.axes
    ]
    # another candidate starts the search elsewhere: it must not reuse that entry
    other = _run(bad_bag, tmp_path / "c", cache_dir=cache)
    (record_other,) = [p for p in other.pairs if p.pair == "camera-lidar"]
    assert record_other.evidence_from_cache is False


def test_missing_intrinsics_is_skipped_with_a_reason(tmp_path: Path, truth: np.ndarray) -> None:
    bag = write_bag(tmp_path / "bag", truth)
    # a bag without the CameraInfo topic and without a camchain has no intrinsics
    from tests.unit.check_fixtures import CLOUD_TYPE, IMAGE_TYPE, Transform, write_check_bag

    transforms: list[Transform] = [
        (CAMERA_FRAME, LIDAR_FRAME, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    ]
    stub = write_check_bag(
        tmp_path / "stub.db3",
        tf_messages=[transforms],
        sensor_frames={
            "/camera/image_raw": (IMAGE_TYPE, CAMERA_FRAME),
            "/lidar/points": (CLOUD_TYPE, LIDAR_FRAME),
        },
    )
    del bag
    artifact = _run(stub, tmp_path)
    (record,) = [p for p in artifact.pairs if p.pair == "camera-lidar"]
    assert record.status == "skipped" and record.reason_code == "missing_intrinsics"


# --------------------------------------------------------------- estimate and the prior gate


def test_estimate_needs_a_prior_and_exports_the_estimated_rotation(
    good_bag: Path, truth: np.ndarray, tmp_path: Path
) -> None:
    assert "camera-lidar" in NEEDS_INITIAL_GUESS
    # no tf in the bag-estimate of a bag without /tf_static would be a placeholder; here the bag's
    # own /tf_static is the (correct) prior, so the pair runs and its frame is exported
    out = tmp_path / "out"
    code = main(
        ["estimate", str(good_bag), "--output", str(out), "--pairs", "camera-lidar", "--no-cache"]
    )
    assert code == 0
    artifact = json.loads((out / "bag_estimate.json").read_text())
    (pair,) = [p for p in artifact["pairs"] if p["pair"] == "camera-lidar"]
    assert pair["status"] in {"estimated", "partial"}
    observed = {a["name"] for a in pair["axes"] if a["status"] == "observed"}
    assert observed and observed <= {"roll", "pitch", "yaw"}
    translation = {a["name"]: a["status"] for a in pair["axes"] if a["name"] in {"x", "y", "z"}}
    assert set(translation.values()) == {"unobservable"}
    frames = yaml.safe_load((out / "frames.yaml").read_text())
    (entry,) = [f for f in frames["frames"] if f["name"] == LIDAR_FRAME]
    exported = np.asarray(entry["translation_m"])
    # translation comes from the prior (marked NOT MEASURED), rotation from the estimate
    np.testing.assert_allclose(exported, truth[:3, 3], atol=1e-9)
    rotation = Rotation.from_quat(entry["rotation_quat_xyzw"]).as_matrix()
    error = Rotation.from_matrix(rotation @ truth[:3, :3].T).magnitude()
    assert np.degrees(error) < 0.8
    assert "NOT MEASURED" in (out / "frames.yaml").read_text()
    validate_file(out / "bag_estimate.json")


def test_estimate_without_a_prior_skips_camera_lidar(tmp_path: Path, truth: np.ndarray) -> None:
    from tests.unit.check_fixtures import CLOUD_TYPE, IMAGE_TYPE, write_check_bag

    bag = write_check_bag(
        tmp_path / "bag.db3",
        tf_messages=[],
        sensor_frames={
            "/camera/image_raw": (IMAGE_TYPE, CAMERA_FRAME),
            "/lidar/points": (CLOUD_TYPE, LIDAR_FRAME),
        },
    )
    del truth
    out = tmp_path / "out"
    code = main(
        ["estimate", str(bag), "--output", str(out), "--pairs", "camera-lidar", "--no-cache"]
    )
    assert code == 0
    artifact = json.loads((out / "bag_estimate.json").read_text())
    (pair,) = [p for p in artifact["pairs"] if p["pair"] == "camera-lidar"]
    assert pair["status"] == "skipped" and pair["reason_code"] == "no_candidate_calibration"
    assert any("camera-lidar" in step for step in artifact["next_steps"])


# ------------------------------------------------------------------------------------ CLI


def test_check_cli_runs_camera_lidar_and_writes_json(good_bag: Path, tmp_path: Path) -> None:
    output = tmp_path / "check.json"
    code = main(
        [
            "check",
            str(good_bag),
            "--pairs",
            "camera-lidar",
            "--output",
            str(output),
            "--evidence-dir",
            str(tmp_path / "ev"),
            "--no-cache",
        ]
    )
    assert code == 0
    payload = json.loads(output.read_text())
    (record,) = [p for p in payload["pairs"] if p["pair"] == "camera-lidar"]
    assert record["status"] == "pass"


# ---------------------------------------------------------------------------------- drift


def test_drift_flags_a_remounted_camera_and_ignores_translation(
    tmp_path: Path, truth: np.ndarray
) -> None:
    from calibrex.check.drift import build_calibration_drift

    remounted = perturbed(truth, (0.0, 2.5, 0.0))
    bags = [
        write_bag(tmp_path / "rec1", truth),
        write_bag(tmp_path / "rec2", truth),
        write_bag(tmp_path / "rec3", remounted),
    ]
    artifact = build_calibration_drift(
        bags, output_dir=tmp_path / "drift", run=CheckRunOptions(pairs=("camera-lidar",))
    )
    (pair,) = [p for p in artifact.pairs if p.pair == "camera-lidar"]
    assert artifact.overall_verdict == "drift" and pair.verdict == "drift"
    assert pair.deviating_bags == ["rec3"]
    axes = {axis.name: axis for axis in pair.axes}
    assert axes["pitch"].status == "drift"
    assert not any(name in axes and axes[name].status == "drift" for name in ("x", "y", "z"))
    assert axes["pitch"].max_abs_difference == pytest.approx(2.5, abs=0.7)
