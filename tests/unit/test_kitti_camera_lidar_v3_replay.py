"""Exact spent-input capture, replay parity, and retention of failures."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tests.unit.camera_lidar_scene import true_transform
from tests.unit.test_camera_lidar_edge import CAMERA, _frames, _provenance
from tests.unit.test_kitti_raw_to_rosbag2 import _write_sweep_stamps, add_camera, write_drive
from tests.unit.test_score_hilti_camera_camera import _load

from calibrex.check.tf_sources import load_bag_tf_static
from calibrex.data.kitti_ins_lidar import se3_matrix
from calibrex.data.kitti_raw_to_rosbag2 import convert_kitti_raw_to_rosbag2
from calibrex.data.ros2_camera_lidar import CameraLidarFrame
from calibrex.evaluation import camera_lidar_edge as edge

replay = _load("kitti_camera_lidar_v3_replay")
DRIVE = "2011_09_26_0005"


def _snapshot(path: Path, *, point_times: bool = True) -> CameraLidarFrame:
    frame = CameraLidarFrame(
        time_s=1.1,
        image_time_s=1.12,
        image=np.arange(96, dtype=np.uint8).reshape(8, 12),
        points=np.arange(30, dtype=np.float32).reshape(10, 3),
        cloud_frame_id="velo_link",
        image_frame_id="cam2_optical",
        point_time_offsets_s=np.linspace(-0.05, 0.05, 10) if point_times else None,
    )
    replay.save_snapshot(
        path,
        [frame],
        CAMERA,
        np.eye(4),
        drive=DRIVE,
        run="baseline",
        conversion_sha256="1" * 64,
        options=edge.EdgeAlignmentOptions(),
        provenance=_provenance(),
    )
    return frame


@pytest.mark.parametrize("point_times", [True, False])
def test_snapshot_preserves_array_bytes_types_and_timing(tmp_path: Path, point_times: bool) -> None:
    source = _snapshot(tmp_path / "snapshot", point_times=point_times)
    metadata, frames, camera, candidate, options = replay.load_snapshot(tmp_path / "snapshot")
    (frame,) = frames
    assert metadata.drive == DRIVE
    assert options.as_dict() == edge.EdgeAlignmentOptions().as_dict()
    assert camera == CAMERA
    assert frame.time_s == source.time_s and frame.image_time_s == source.image_time_s
    for name in ("image", "points", "point_time_offsets_s"):
        left, right = getattr(source, name), getattr(frame, name)
        if left is None:
            assert right is None
        else:
            assert left.dtype == right.dtype
            assert left.tobytes() == right.tobytes()
            assert not right.flags.writeable
    assert not candidate.flags.writeable
    with pytest.raises(FileExistsError):
        _snapshot(tmp_path / "snapshot")


def test_corrupt_payload_is_rejected_before_numpy_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "snapshot"
    _snapshot(path)
    (path / "arrays.npz").write_bytes(b"corrupted")
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("unverified arrays loaded"))
    with pytest.raises(ValueError, match="digest mismatch"):
        replay.load_snapshot(path)


def test_changed_option_schema_requires_explicit_migration(tmp_path: Path) -> None:
    path = tmp_path / "snapshot"
    _snapshot(path)
    metadata = json.loads((path / "snapshot.json").read_text())
    del metadata["options"]["min_control_t"]
    (path / "snapshot.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="explicit snapshot migration"):
        replay.load_snapshot(path)


def test_capture_rejects_a_fresh_id_before_opening_any_bag(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a spent KITTI raw"):
        replay.capture_drive(tmp_path / "absent", "2013_05_28_0004", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def _bag(tmp_path: Path) -> Path:
    source = write_drive(tmp_path / "synthetic", number=5)
    add_camera(source)
    _write_sweep_stamps(source)
    renamed = source.with_name("2011_09_26_drive_0005_sync")
    source.rename(renamed)
    bag = tmp_path / "bag"
    convert_kitti_raw_to_rosbag2(renamed, bag, camera="image_02", velodyne_time="azimuth")
    return bag


def test_rounded_calibration_reference_matches_serialized_tf(tmp_path: Path) -> None:
    source = write_drive(tmp_path / "synthetic", number=5)
    add_camera(source)
    _write_sweep_stamps(source)
    source = source.rename(source.with_name("2011_09_26_drive_0005_sync"))
    # Simulate published decimal precision, rather than exact-rigid test matrices.
    for name in ("calib_imu_to_velo.txt", "calib_velo_to_cam.txt", "calib_cam_to_cam.txt"):
        path = source.parent / name
        rows = []
        for row in path.read_text().splitlines():
            key, values = row.split(":", 1)
            if key in {"R", "R_rect_00"}:
                values = " " + " ".join(f"{float(v):.6e}" for v in values.split())
            rows.append(f"{key}:{values}")
        path.write_text("\n".join(rows) + "\n")
    result = convert_kitti_raw_to_rosbag2(
        source, tmp_path / "bag", camera="image_02", velodyne_time="azimuth",
    )
    loaded = load_bag_tf_static(result.bag)
    assert loaded is not None
    edges = {edge.child: edge.transform for edge in loaded.edges}
    candidate = se3_matrix(edges["cam2_optical"].inverse().compose(edges["velo_link"]))
    reference = replay._reference(result.provenance, DRIVE)
    np.testing.assert_allclose(reference, candidate, rtol=0, atol=1e-14)
    raw = np.asarray(result.provenance["camera"]["t_cam_velo"])
    assert np.max(np.abs(raw - candidate)) > 1e-8
    for run, injection in replay.RUNS.items():
        turned = candidate.copy()
        if injection is not None:
            axis, degrees = injection.split(":")
            turn = Rotation.from_rotvec(np.radians(float(degrees)) * np.eye(3)["xyz".index(axis)])
            turned[:3] = turn.as_matrix() @ turned[:3]
        replay._check_candidate(turned, reference, run)
        wrong = turned.copy()
        wrong[0, 3] += 1e-5
        with pytest.raises(ValueError, match="candidate differs"):
            replay._check_candidate(wrong, reference, run)
    legacy = result.provenance.copy()
    legacy["camera"] = {k: v for k, v in legacy["camera"].items() if k != "t_imu_velo"}
    with pytest.raises(ValueError, match="regenerate the bag"):
        replay._reference(legacy, DRIVE)


@pytest.fixture
def captured(tmp_path: Path) -> tuple[Path, Path]:
    bag = _bag(tmp_path)
    captures = tmp_path / "captures"
    assert replay.capture_drive(bag, DRIVE, captures / DRIVE) == 0
    return bag, captures


def test_real_check_capture_and_replay_work_after_source_bag_is_deleted(
    tmp_path: Path,
    captured: tuple[Path, Path],
) -> None:
    bag, captures = captured
    manifest = json.loads((captures / DRIVE / "capture_manifest.json").read_text())
    assert len(manifest["runs"]) == 7
    for run in replay.RUNS:
        metadata, frames, _, _, _ = replay.load_snapshot(captures / DRIVE / run)
        assert metadata.run == run and frames
        assert metadata.motion_compensation is not None
        assert metadata.motion_compensation.per_point_time
        assert all(frame.point_time_offsets_s is not None for frame in frames)
    shutil.rmtree(bag)
    report = replay.compare_drives(captures, tmp_path / "paired", drives=[DRIVE])
    assert report["errors"] == {}
    assert report["status"] == "incomplete"  # a one-drive subset is not the full 55-drive study
    for metrics in report["metrics"].values():
        assert metrics["inconclusive_drives"] == [DRIVE]  # the tiny synthetic bag has <8 frames
        assert metrics["failed_drives"] == []
    assert report["coverage_change_percentage_points"] is None
    assert report["audit_thresholds"] is None
    assert "verdict" not in report
    current = json.loads((tmp_path / "paired" / DRIVE / "current/baseline.json").read_text())
    proposed = json.loads((tmp_path / "paired" / DRIVE / "proposed/baseline.json").read_text())
    assert current["provenance"]["input_sha256"] == proposed["provenance"]["input_sha256"]
    assert current["options"]["start_scan_min_control_t"] is None
    assert proposed["options"]["start_scan_min_control_t"] == 3.5


@pytest.mark.parametrize("mismatch", ["edge", "check"])
def test_replay_parity_failure_is_retained_in_both_rule_summaries(
    tmp_path: Path,
    captured: tuple[Path, Path],
    mismatch: str,
) -> None:
    _, captures = captured
    root = captures / DRIVE
    relative = "baseline/captured_edge.json" if mismatch == "edge" else "baseline.json"
    target = root / relative
    payload = json.loads(target.read_text())
    if mismatch == "edge":
        payload["policy_reasons"].append("changed after capture")
    else:
        (pair,) = [item for item in payload["pairs"] if item["pair"] == "camera-lidar"]
        pair["verdict"] = "fail"
    target.write_text(json.dumps(payload))
    manifest = json.loads((root / "capture_manifest.json").read_text())
    manifest["files_sha256"][relative] = hashlib.sha256(target.read_bytes()).hexdigest()
    (root / "capture_manifest.json").write_text(json.dumps(manifest))
    report = replay.compare_drives(captures, tmp_path / "paired", drives=[DRIVE])
    assert "differs from" in report["errors"][DRIVE]
    assert report["status"] == "incomplete"
    assert all(m["failed_drives"] == [DRIVE] for m in report["metrics"].values())


def test_absent_captures_count_as_failures_instead_of_disappearing(tmp_path: Path) -> None:
    report = replay.compare_drives(tmp_path / "missing", tmp_path / "paired")
    assert report["status"] == "incomplete"
    assert len(report["errors"]) == 55
    assert all(len(m["failed_drives"]) == 55 for m in report["metrics"].values())
    assert report["coverage_change_percentage_points"] is None


def test_solver_failure_preserves_inputs_and_capture_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bag = _bag(tmp_path)

    def crash(*args: object, **kwargs: object) -> None:
        raise RuntimeError("synthetic solver crash")

    monkeypatch.setattr(edge, "estimate_from_frames", crash)
    output = tmp_path / "captures" / DRIVE
    assert replay.capture_drive(bag, DRIVE, output) == 1
    manifest = json.loads((output / "capture_manifest.json").read_text())
    assert not manifest["complete"]
    assert len(manifest["runs"]) == 7
    assert all(not record["ok"] for record in manifest["runs"].values())
    for run in replay.RUNS:
        assert (output / run / "snapshot.json").is_file()
        assert "synthetic solver crash" in manifest["runs"][run]["error"]


def test_capture_refuses_a_bag_whose_tf_disagrees_with_its_calibration(tmp_path: Path) -> None:
    bag = _bag(tmp_path)
    path = bag / "calibrex_conversion.json"
    conversion = json.loads(path.read_text())
    conversion["camera"]["t_cam_velo"][0][3] += 1.0
    path.write_text(json.dumps(conversion))
    output = tmp_path / "captures" / DRIVE
    assert replay.capture_drive(bag, DRIVE, output) == 1
    manifest = json.loads((output / "capture_manifest.json").read_text())
    assert not manifest["complete"]
    assert all("candidate differs" in entry["error"] for entry in manifest["runs"].values())


def test_converged_replay_scores_real_injections_on_the_same_saved_frames(tmp_path: Path) -> None:
    truth = true_transform()
    frames = _frames(truth)
    root = tmp_path / "captures" / DRIVE
    root.mkdir(parents=True)
    conversion = {
        "source": {
            "dataset_family": "kitti_raw",
            "drives": [
                {"drive": "2011_09_26_drive_0005_sync"},
            ],
        },
        "camera": {"directory": "image_02", "t_cam_velo": truth.tolist()},
        "velodyne_time": "azimuth",
    }
    (root / "conversion.json").write_text(json.dumps(conversion))
    conversion_sha = replay._sha(root / "conversion.json")
    for run, spec in replay.RUNS.items():
        candidate = truth.copy()
        if spec is not None:
            axis, degrees = spec.split(":")
            rotation = Rotation.from_rotvec(
                np.radians(float(degrees)) * np.eye(3)["xyz".index(axis)]
            ).as_matrix()
            candidate[:3, :3] = rotation @ truth[:3, :3]
            candidate[:3, 3] = rotation @ truth[:3, 3]
        options = edge.EdgeAlignmentOptions()
        replay.save_snapshot(
            root / run,
            frames,
            CAMERA,
            candidate,
            drive=DRIVE,
            run=run,
            conversion_sha256=conversion_sha,
            options=options,
            provenance=_provenance(),
        )
        artifact = edge.estimate_from_frames(
            frames,
            CAMERA,
            candidate,
            options=options,
            provenance=_provenance(),
        )
        assert artifact.solver_status == "converged"
        artifact.save(root / run / "captured_edge.json")
        judged = replay._run_input(artifact, candidate)
        (root / f"{run}.json").write_text(
            json.dumps(
                {
                    "pairs": [
                        {
                            "pair": "camera-lidar",
                            "verdict": judged.verdict,
                            "axes": [
                                {"name": name, "status": status}
                                for name, status in judged.judged.items()
                            ],
                        }
                    ]
                }
            )
        )
    files = {p.relative_to(root).as_posix(): replay._sha(p) for p in root.rglob("*") if p.is_file()}
    (root / "capture_manifest.json").write_text(
        json.dumps(
            {
                "drive": DRIVE,
                "complete": True,
                "conversion_sha256": conversion_sha,
                "runs": {name: {"ok": True} for name in replay.RUNS},
                "files_sha256": files,
            }
        )
    )
    report = replay.compare_drives(root.parent, tmp_path / "paired", drives=[DRIVE])
    assert report["errors"] == {}
    assert report["status"] == "incomplete"  # only a synthetic subset, never a real study
    for rule in ("current", "proposed"):
        metrics = report["metrics"][rule]
        assert metrics["scored_drives"] == 1
        assert metrics["judged_axes"] >= 2
        assert metrics["flagged_baselines"] == []
        assert metrics["trials"]["3deg"]["counted"] >= 2
        assert metrics["trials"]["3deg"]["flagged_fraction"] == 1.0
        assert metrics["max_turned_judged_error_deg"] < 0.5
    assert report["coverage_change_percentage_points"] == 0.0
