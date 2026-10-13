"""Importing a sensor_calib_tas result as a ``calibrex check --tf`` frame file.

sensor_calib_tas cannot run here, so its two outputs are synthesised as it writes them
(``extrinsic_sensor_calib/core/sensor_calib.cpp``, ``ros2/sensor_calib_ros.cpp``): static
transforms ``reference -> <sensor>_calibrated`` next to the vehicle's own tree, and the
latched ``EstimatedPoses`` on ``~/sensor_poses`` as ``ros2 topic echo`` prints it. The
last test runs the camera-LiDAR check on a bag whose deployed tf_static is 3 degrees off,
with and without the imported target calibration.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from scipy.spatial.transform import Rotation
from tests.unit.camera_lidar_scene import perturbed, true_transform
from tests.unit.test_check_camera_lidar import CAMERA_FRAME, LIDAR_FRAME, write_bag

from calibrex.check.runner import CheckRunOptions, build_calibration_check
from calibrex.check.tf_sources import load_tf_file
from calibrex.cli.main import main
from calibrex.core.validation import validate_file
from calibrex.data.ros_cdr_writer import encode_tf_message
from calibrex.data.rosbag2_writer import Rosbag2Writer
from calibrex.importers.sensor_calib_tas import (
    read_sensor_calib_tas_poses,
    sensor_calib_tas_frames,
)

REFERENCE = "sensor/lidar/vls128_roof"
FRONT = "sensor/camera/surround/front"
REAR = "sensor/radar/rear"
DEPLOYED_FRONT = ((0.1, 0.0, -0.2), (0.5, -0.5, 0.5, -0.5))
CALIBRATED_FRONT = ((0.12, -0.01, -0.21), (0.4999, -0.5, 0.5001, -0.5))
CALIBRATED_REAR = ((-3.9, 0.0, -1.1), (0.0, 0.0, 1.0, 0.0))


def _tas_bag(path: Path, *, calibrated: bool = True, preview: bool = False) -> Path:
    """The vehicle's tf_static and, once solved, what sensor_calib_tas adds to it."""

    vehicle = [
        ("base_link", REFERENCE, (1.2, 0.0, 1.9), (0.0, 0.0, 0.0, 1.0)),
        (REFERENCE, FRONT, *DEPLOYED_FRONT),
        ("base_link", REAR, (-2.7, 0.0, 0.8), (0.0, 0.0, 1.0, 0.0)),
    ]
    with Rosbag2Writer(path) as writer:
        writer.add_topic("/tf_static", "tf2_msgs/msg/TFMessage", latched=True)
        writer.write("/tf_static", 1, encode_tf_message(vehicle))  # type: ignore[arg-type]
        if calibrated:
            # an earlier solve, then the final one: the latest value is the result
            stale = [(REFERENCE, f"{FRONT}_calibrated", (9.0, 9.0, 9.0), (0.0, 0.0, 0.0, 1.0))]
            writer.write("/tf_static", 2, encode_tf_message(stale))  # type: ignore[arg-type]
            solved = [
                (REFERENCE, f"/{FRONT}_calibrated", *CALIBRATED_FRONT),
                (REFERENCE, f"{REAR}_calibrated", *CALIBRATED_REAR),
            ]
            if preview:  # the preview republishes the poses under the sensors' own names
                solved.append((REFERENCE, REAR, *CALIBRATED_REAR))
            writer.write("/tf_static", 3, encode_tf_message(solved))  # type: ignore[arg-type]
    return path


def _transform(parent: str, child: str, pose: tuple[tuple[float, ...], ...]) -> dict[str, object]:
    translation, rotation = pose
    return {
        "header": {"stamp": {"sec": 5, "nanosec": 0}, "frame_id": parent},
        "child_frame_id": child,
        "transform": {
            "translation": dict(zip("xyz", translation, strict=True)),
            "rotation": dict(zip("xyzw", rotation, strict=True)),
        },
    }


def _echo(path: Path) -> Path:
    """``ros2 topic echo /sensor_calib/sensor_poses``: two messages, the last is final."""

    first = {"poses": [_transform(REFERENCE, FRONT, DEPLOYED_FRONT)]}
    final = {
        "poses": [
            _transform(f"/{REFERENCE}", REAR, CALIBRATED_REAR),
            _transform(REFERENCE, FRONT, CALIBRATED_FRONT),
        ],
        # another convention (rotated into vehicle axes): never imported
        "poses_in_vehicle_coordinate_system": [
            _transform(REFERENCE, FRONT, ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)))
        ],
    }
    path.write_text(yaml.safe_dump_all([first, final]), encoding="utf-8")
    return path


def _expected() -> dict[str, tuple[str, tuple[float, ...], tuple[float, ...]]]:
    return {
        FRONT: (REFERENCE, *CALIBRATED_FRONT),
        REAR: (REFERENCE, *CALIBRATED_REAR),
    }


@pytest.mark.parametrize("preview", [False, True])
def test_a_bag_gives_the_latest_calibrated_pose_of_each_sensor(
    tmp_path: Path, preview: bool
) -> None:
    poses, kind = read_sensor_calib_tas_poses(_tas_bag(tmp_path / "bag", preview=preview))
    assert kind == "bag_tf_static"
    read = {
        pose.sensor_frame: (pose.reference_frame, pose.translation_m, pose.rotation_quat_xyzw)
        for pose in poses
    }
    assert read == _expected()


def test_an_echoed_sensor_poses_message_gives_the_same_poses(tmp_path: Path) -> None:
    poses, kind = read_sensor_calib_tas_poses(_echo(tmp_path / "poses.yaml"))
    assert kind == "estimated_poses_yaml"
    read = {
        pose.sensor_frame: (pose.reference_frame, pose.translation_m, pose.rotation_quat_xyzw)
        for pose in poses
    }
    assert read == _expected()


def test_a_bag_recorded_before_solving_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="_calibrated"):
        read_sensor_calib_tas_poses(_tas_bag(tmp_path / "bag", calibrated=False))
    empty = tmp_path / "empty.yaml"
    empty.write_text("poses: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="EstimatedPoses"):
        read_sensor_calib_tas_poses(empty)


def test_the_frame_file_is_a_valid_candidate_with_provenance(tmp_path: Path) -> None:
    frames = sensor_calib_tas_frames(_tas_bag(tmp_path / "bag"), tool_version="1.0.0")
    assert frames.provenance is not None
    assert frames.provenance.tool == "sensor_calib_tas"
    assert frames.provenance.tool_license == "BSD-3-Clause"
    assert frames.provenance.source_kind == "bag_tf_static"
    assert len(frames.provenance.source_sha256) == 64
    assert any("1.0.0" in note for note in frames.provenance.notes)

    output = tmp_path / "tas_frames.yaml"
    assert main(["external-run", "import-sensor-calib-tas", str(tmp_path / "bag"),
                 "--output", str(output), "--json"]) == 0  # fmt: skip
    assert validate_file(output).valid
    loaded = load_tf_file(output)
    edges = {edge.child: edge.parent for edge in loaded.edges}
    assert edges == {FRONT: REFERENCE, REAR: REFERENCE}


def test_the_cli_reports_a_source_without_a_result(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bag = _tas_bag(tmp_path / "bag", calibrated=False)
    code = main(
        ["external-run", "import-sensor-calib-tas", str(bag), "--output", str(tmp_path / "o")]
    )
    assert code == 2
    assert "_calibrated" in capsys.readouterr().err
    assert not (tmp_path / "o").exists()


def test_the_imported_calibration_overrides_a_wrong_deployed_one(tmp_path: Path) -> None:
    """A driving bag whose tf_static is 3 deg off fails; the TAS result passes on it."""

    truth = true_transform()  # T_camera_lidar
    driving = write_bag(tmp_path / "drive", truth, candidate=perturbed(truth, (0.0, 3.0, 0.0)))
    # the TAS session: the camera is its reference, the LiDAR calibrated against it
    quat = Rotation.from_matrix(truth[:3, :3]).as_quat()
    session = tmp_path / "tas_session"
    with Rosbag2Writer(session) as writer:
        writer.add_topic("/tf_static", "tf2_msgs/msg/TFMessage", latched=True)
        edge = (
            CAMERA_FRAME,
            f"{LIDAR_FRAME}_calibrated",
            tuple(float(v) for v in truth[:3, 3]),
            tuple(float(v) for v in quat),
        )
        writer.write("/tf_static", 1, encode_tf_message([edge]))  # type: ignore[list-item]
    frames_file = tmp_path / "tas_frames.yaml"
    assert main(["external-run", "import-sensor-calib-tas", str(session),
                 "--output", str(frames_file), "--json"]) == 0  # fmt: skip

    def check(tf_files: list[Path]) -> tuple[str, list[str]]:
        run = CheckRunOptions(
            pairs=("camera-lidar",),
            evidence_dir=tmp_path / "evidence",
            base_dir=tmp_path,
            max_duration_s=None,
        )
        artifact = build_calibration_check(driving, tf_files=tf_files, run=run)
        (record,) = [p for p in artifact.pairs if p.pair == "camera-lidar"]
        return record.status, artifact.frame_tree.overrides

    deployed_status, no_overrides = check([])
    imported_status, overrides = check([frames_file])
    assert (deployed_status, no_overrides) == ("fail", [])
    assert imported_status == "pass"
    assert len(overrides) == 1 and f"frame '{LIDAR_FRAME}'" in overrides[0]


def test_the_cli_prints_how_to_use_the_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "tas.yaml"
    assert main(["external-run", "import-sensor-calib-tas", str(_echo(tmp_path / "p.yaml")),
                 "--output", str(output), "--json"]) == 0  # fmt: skip
    payload = json.loads(capsys.readouterr().out)
    assert payload["source_kind"] == "estimated_poses_yaml"
    assert payload["usage"] == f"calibrex check <driving bag> --tf {output}"
    assert sorted(payload["frames"]) == [f"{REFERENCE} -> {FRONT}", f"{REFERENCE} -> {REAR}"]
