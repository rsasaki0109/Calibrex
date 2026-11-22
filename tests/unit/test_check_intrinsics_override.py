from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from calibrex.check import estimators
from calibrex.check.intrinsics_override import IntrinsicsOverride, match_intrinsics

CAM0 = {
    "camera_model": "pinhole",
    "intrinsics": [420.0, 418.0, 243.0, 176.0],
    "distortion_model": "radtan",
    "distortion_coeffs": [-0.12, 0.05, 0.0008, -0.0005],
    "resolution": [480, 360],
}
CAM1 = {
    **CAM0,
    "intrinsics": [235.0, 234.0, 240.0, 180.0],
    "distortion_model": "equidistant",
    "distortion_coeffs": [0.04, -0.02, 0.006, -0.001],
}


def _chain(path: Path, chain: dict) -> Path:
    path.write_text(yaml.safe_dump(chain))
    return path


def test_parse_reads_topic_file_and_camera(tmp_path: Path) -> None:
    path = _chain(tmp_path / "stereo.yaml", {"cam0": CAM0, "cam1": CAM1})

    assert IntrinsicsOverride.parse(str(path)) == IntrinsicsOverride(path)
    assert IntrinsicsOverride.parse(f"/right/image_raw={path}:cam1") == IntrinsicsOverride(
        path, "/right/image_raw", "cam1"
    )
    assert IntrinsicsOverride.parse(f"{path}:cam0").key == "cam0"
    # A relative path with '=' is a file, not a topic.
    assert IntrinsicsOverride.parse("a=b.yaml").topic is None


def test_matching_by_topic_by_rostopic_and_for_the_only_camera(tmp_path: Path) -> None:
    stereo = _chain(tmp_path / "stereo.yaml", {"cam0": CAM0, "cam1": CAM1})
    tagged = _chain(
        tmp_path / "tagged.yaml",
        {"cam0": {**CAM0, "rostopic": "/left"}, "cam1": {**CAM1, "rostopic": "/right"}},
    )
    single = _chain(tmp_path / "single.yaml", {"cam0": CAM0})
    cameras = ("/left", "/right")

    by_topic = (IntrinsicsOverride(stereo, "/right", "cam1"),)
    assert match_intrinsics(by_topic, "/right", cameras)[1] == "cam1"
    assert match_intrinsics(by_topic, "/left", cameras) is None

    by_rostopic = (IntrinsicsOverride(tagged),)
    assert match_intrinsics(by_rostopic, "/right", cameras)[1] == "cam1"
    assert match_intrinsics(by_rostopic, "/left", cameras)[1] == "cam0"

    # One camera in the bag: the only override applies whatever the topic.
    only = (IntrinsicsOverride(single),)
    assert match_intrinsics(only, "/camera/image_raw", ("/camera/image_raw",))[1] == "cam0"
    # Two cameras: an untargeted single-camera file is ambiguous, so it is not used.
    assert match_intrinsics(only, "/left", cameras) is None
    # Two cameras in the file and one in the bag: also ambiguous without :camN.
    assert match_intrinsics((IntrinsicsOverride(stereo),), "/cam", ("/cam",)) is None
    assert (
        match_intrinsics((IntrinsicsOverride(stereo, None, "cam1"),), "/cam", ("/cam",))[1]
        == "cam1"
    )


def _ctx(intrinsics: tuple[IntrinsicsOverride, ...], pair: str = "camera-imu"):
    topic = "/camera/image_raw"
    return estimators.PairContext(
        bag=Path("unused"),
        pair=SimpleNamespace(pair=pair),  # type: ignore[arg-type]
        topic_types={topic: "sensor_msgs/msg/Image"},
        sensor_topics={"camera": (topic,)},
        candidate=np.eye(4),
        sources=[],
        controls=estimators.RunControls(intrinsics=intrinsics),
    )


def test_the_override_comes_before_the_other_intrinsics_sources(tmp_path: Path) -> None:
    path = _chain(tmp_path / "intrinsics_camchain.yaml", {"cam0": CAM1})

    model, description, key = estimators.resolve_camera_intrinsics(
        _ctx((IntrinsicsOverride(path),)), "/camera/image_raw", "camera_optical"
    )

    assert description == "intrinsics_camchain.yaml cam0 (--intrinsics)" and key == "cam0"
    assert model.distortion_model == "equidistant"
    assert (model.fx, model.cy) == (235.0, 180.0)


def test_a_bad_override_is_a_skip_naming_it(tmp_path: Path) -> None:
    path = _chain(tmp_path / "broken.yaml", {"cam0": {**CAM0, "camera_model": "omni"}})

    with pytest.raises(estimators.CheckSkipError, match=r"broken.yaml cam0 \(--intrinsics\)"):
        estimators.resolve_camera_intrinsics(
            _ctx((IntrinsicsOverride(path),)), "/camera/image_raw", "camera_optical"
        )


def test_cli_rejects_a_missing_camchain(tmp_path: Path) -> None:
    from calibrex.cli.main import main

    with pytest.raises(SystemExit):
        main(["check", str(tmp_path), "--intrinsics", str(tmp_path / "nope.yaml")])


def test_check_hands_the_overrides_to_the_camera_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.unit.check_fixtures import default_tf, write_check_bag

    from calibrex.check.runner import CheckRunOptions, build_calibration_check

    bag = write_check_bag(tmp_path / "bag.db3", tf_messages=[default_tf()])
    path = _chain(tmp_path / "intrinsics_camchain.yaml", {"cam0": CAM0})
    seen: list[str] = []

    def camera_imu(ctx: estimators.PairContext) -> estimators.EstimatorRun:
        topic = ctx.sensor_topics["camera"][0]
        _, description, _ = estimators.resolve_camera_intrinsics(ctx, topic, ctx.pair.frames[0])
        seen.append(description)
        raise estimators.CheckSkipError("missing_dependency", "stub")

    monkeypatch.setattr(estimators, "ESTIMATORS", {"camera-imu": camera_imu})

    build_calibration_check(
        bag,
        run=CheckRunOptions(
            pairs=("camera-imu",),
            intrinsics=(IntrinsicsOverride(path),),
            evidence_dir=tmp_path / "evidence",
            base_dir=tmp_path,
        ),
    )

    assert seen == ["intrinsics_camchain.yaml cam0 (--intrinsics)"]


def test_estimate_exports_a_camchain_from_the_override(tmp_path: Path) -> None:
    from calibrex.check.estimate import camchain_export
    from calibrex.core.bag_estimate import EstimateFrameEntry, EstimateFrames
    from calibrex.core.calibration_check import CheckTopicRecord, CheckTransform

    path = _chain(tmp_path / "intrinsics_camchain.yaml", {"cam0": CAM1})
    frames = EstimateFrames(
        root="imu_link",
        entries=[
            EstimateFrameEntry(
                frame="camera_optical",
                parent="imu_link",
                pair="camera-imu",
                transform=CheckTransform(
                    parent_frame="imu_link",
                    child_frame="camera_optical",
                    translation_m=[0.1, 0.0, 0.0],
                    rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
                ),
            )
        ],
    )
    topics = [
        CheckTopicRecord(
            topic="/camera/image_raw",
            message_type="sensor_msgs/msg/Image",
            role="camera",
            mapped_frame="camera_optical",
        )
    ]

    text = camchain_export(
        [], frames, bag_name="bag", intrinsics=(IntrinsicsOverride(path),), topics=topics
    )

    assert text is not None and "Intrinsics are copied from intrinsics_camchain.yaml" in text
    entry = yaml.safe_load(text)["cam0"]
    assert entry["distortion_model"] == "equidistant"
    assert entry["rostopic"] == "/camera/image_raw"
    # T_cam_imu inverts T_imu_camera.
    np.testing.assert_allclose(np.asarray(entry["T_cam_imu"])[:3, 3], [-0.1, 0.0, 0.0])
