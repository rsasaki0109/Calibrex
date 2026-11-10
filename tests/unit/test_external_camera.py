"""Cameras outside the bag: image folders, videos and the camera-IMU clock search."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibrex.core.exceptions import DatasetError
from calibrex.data.external_camera import (
    folder_frames,
    frame_interval_s,
    iter_external_gray,
    parse_camera_source,
)
from calibrex.evaluation.clock_alignment import align_camera_clock

cv2 = pytest.importorskip("cv2")


def _write_images(folder: Path, names: list[str], depth: int = 8) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(names):
        value = 40 * index + 10
        image = np.full((8, 10), value << 8 if depth == 16 else value, dtype=f"uint{depth}")
        cv2.imwrite(str(folder / name), image)


def test_parse_a_folder_a_video_and_a_given_offset(tmp_path: Path) -> None:
    (tmp_path / "imgs").mkdir()
    video = tmp_path / "cam.mp4"
    video.write_bytes(b"x")

    folder = parse_camera_source(f"cam0={tmp_path / 'imgs'}")
    assert (folder.name, folder.kind, folder.clock_offset_s) == ("cam0", "images", None)
    assert folder.topic == "/external/cam0"
    given = parse_camera_source(f"front={video}@-12.5")
    assert (given.kind, given.clock_offset_s, given.path) == ("video", -12.5, video)
    for bad in ("cam0", f"=x{tmp_path}", f"cam 0={video}", f"cam0={tmp_path / 'none'}"):
        with pytest.raises(ValueError, match="--camera-source"):
            parse_camera_source(bad)
    (tmp_path / "notes.txt").write_text("x")
    with pytest.raises(ValueError, match="not a video"):
        parse_camera_source(f"cam0={tmp_path / 'notes.txt'}")


def test_folder_times_from_a_euroc_data_csv_in_the_parent(tmp_path: Path) -> None:
    _write_images(tmp_path / "cam0" / "data", ["a.png", "b.png", "c.png"])
    (tmp_path / "cam0" / "data.csv").write_text(
        "#timestamp [ns],filename\n1520530308300000000,c.png\n1520530308200000000,b.png\n"
        "1520530308100000000,a.png\n"
    )

    frames = folder_frames(tmp_path / "cam0" / "data")

    assert [path.name for _t, path in frames] == ["a.png", "b.png", "c.png"]
    assert frames[1][0] == pytest.approx(1520530308.2)


def test_folder_times_from_file_names_in_ns_or_seconds(tmp_path: Path) -> None:
    _write_images(tmp_path / "ns", ["1520530308100000000.png", "1520530308150000000.png"])
    _write_images(tmp_path / "s", ["12.50.png", "12.55.png"])

    assert [t for t, _p in folder_frames(tmp_path / "ns")] == pytest.approx(
        [1520530308.1, 1520530308.15]
    )
    assert [t for t, _p in folder_frames(tmp_path / "s")] == pytest.approx([12.5, 12.55])
    _write_images(tmp_path / "bad", ["frame_a.png"])
    with pytest.raises(DatasetError, match=r"no capture time for frame_a\.png"):
        folder_frames(tmp_path / "bad")


def test_images_are_read_gray_strided_and_shifted_and_16_bit_keeps_the_high_byte(
    tmp_path: Path,
) -> None:
    names = [f"{index}.0.png" for index in range(5)]
    _write_images(tmp_path / "f", names, depth=16)
    source = parse_camera_source(f"cam0={tmp_path / 'f'}")

    frames = list(iter_external_gray(source, stride=2, time_shift_s=100.0))

    assert [t for t, _g in frames] == pytest.approx([100.0, 102.0, 104.0])
    assert all(g.dtype == np.uint8 and g.ndim == 2 for _t, g in frames)
    assert [int(g[0, 0]) for _t, g in frames] == [10, 90, 170]
    assert frame_interval_s(source) == pytest.approx(1.0)


def test_video_frames_carry_their_presentation_times(tmp_path: Path) -> None:
    video = tmp_path / "cam.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 20.0, (32, 24))
    for index in range(30):
        writer.write(np.full((24, 32, 3), 8 * index, dtype=np.uint8))
    writer.release()
    source = parse_camera_source(f"cam0={video}")

    frames = list(iter_external_gray(source, stride=10))

    assert [t for t, _g in frames] == pytest.approx([0.0, 0.5, 1.0], abs=1e-3)
    assert frames[0][1].shape == (24, 32)
    assert frame_interval_s(source) == pytest.approx(0.05)


def _rig(seconds: float = 120.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, seconds, 0.005)
    kernel = np.ones(200) / 10.0
    rates = np.stack([np.convolve(rng.normal(size=t.size), kernel, "same") for _ in range(3)], 1)
    return t, rates


def _camera(t: np.ndarray, rates: np.ndarray, start: float, end: float, mount: Rotation):  # type: ignore[no-untyped-def]
    times = np.arange(start, end, 0.05)
    orientations = [np.eye(3)]
    for a, b in itertools.pairwise(times):
        inside = (t >= a) & (t < b)
        step = mount.apply((rates[inside] * 0.005).sum(axis=0))
        orientations.append(orientations[-1] @ Rotation.from_rotvec(step).as_matrix())
    return times, np.array(orientations), np.zeros(len(times), dtype=np.int64)


def test_the_clock_offset_is_found_whatever_the_mounting() -> None:
    t, rates = _rig()
    times, orientations, segment = _camera(t, rates, 30.0, 90.0, Rotation.random(random_state=3))

    found = align_camera_clock(times - 996.3, orientations, segment, t, rates)

    assert found.accepted
    assert found.offset_s == pytest.approx(996.3, abs=0.02)
    assert found.correlation > 0.9 and found.runner_up < 0.5
    assert "camera clock + 996.3" in found.reason


def test_a_camera_from_another_recording_is_refused() -> None:
    t, rates = _rig(seed=0)
    other_t, other_rates = _rig(seed=7)
    times, orientations, segment = _camera(other_t, other_rates, 30.0, 90.0, Rotation.identity())

    found = align_camera_clock(times, orientations, segment, t, rates)

    assert not found.accepted and found.offset_s is None
    assert "correlate at most" in found.reason


def test_a_motion_that_repeats_is_refused_rather_than_guessed() -> None:
    t = np.arange(0.0, 120.0, 0.005)
    speed = 1.0 + 0.8 * np.sin(2.0 * np.pi * t / 4.0)  # the same swing every 4 s
    rates = np.column_stack([speed, np.zeros_like(t), np.zeros_like(t)])
    times, orientations, segment = _camera(t, rates, 30.0, 90.0, Rotation.identity())

    found = align_camera_clock(times + 3.0, orientations, segment, t, rates)

    assert not found.accepted
    assert "motion repeats" in found.reason and "@SECONDS" in found.reason


def test_the_atan_camera_inverts_kontikis_projection() -> None:
    from calibrex.evaluation.visual_rotation import CameraModel

    # the LiU GoPro Hero 3+ (camera_parameters.h5)
    camera = CameraModel.from_kalibr(
        {
            "camera_model": "pinhole",
            "intrinsics": [868.10, 881.58, 987.87, 529.78],
            "distortion_model": "atan",
            "distortion_coeffs": [0.8924, 0.000386, -0.003408],
        }
    )
    gamma, wc = 0.8924, np.array([0.000386, -0.003408])
    rays = np.array([[0.0, 0.0], [0.4, -0.2], [-0.9, 0.5], [1.1, 0.6]])
    # Kontiki AtanView::EvaluateProjection: L = a - wc, f = atan(r gamma) / gamma
    offset = rays - wc
    radius = np.linalg.norm(offset, axis=1)
    distorted = (
        wc + offset * (np.arctan(radius * gamma) / gamma / np.maximum(radius, 1e-12))[:, None]
    )
    pixels = np.column_stack([868.10 * distorted[:, 0] + 987.87, 881.58 * distorted[:, 1] + 529.78])

    assert camera.normalize(pixels) == pytest.approx(rays, abs=1e-9)
    with pytest.raises(ValueError, match="gamma"):
        CameraModel.from_kalibr(
            {"intrinsics": [1, 1, 0, 0], "distortion_model": "atan", "distortion_coeffs": [0.9]}
        )


def test_an_atan_camera_is_read_only_by_the_pairs_that_unproject(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from calibrex.check import estimators

    def ctx(pair: str, path: Path) -> SimpleNamespace:
        return SimpleNamespace(
            pair=SimpleNamespace(pair=pair),
            sources=[SimpleNamespace(path=path, kind="kalibr_camchain")],
        )

    chain = tmp_path / "camchain.yaml"
    chain.write_text(
        "cam0:\n  camera_model: pinhole\n  intrinsics: [868.1, 881.6, 987.9, 529.8]\n"
        "  distortion_model: atan\n  distortion_coeffs: [0.8924, 0.0004, -0.0034]\n"
        "  T_cam_imu: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]\n"
    )
    model, _source, _key = estimators.resolve_camera_intrinsics(
        ctx("camera-imu", chain), "/external/cam0", "cam0"
    )
    assert model.distortion_model == "atan"
    with pytest.raises(estimators.CheckSkipError, match="atan"):
        estimators.resolve_camera_intrinsics(ctx("camera-lidar", chain), "/c", "cam0")
