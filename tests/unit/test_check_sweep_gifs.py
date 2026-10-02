"""Tests for the calibrex-check yaw-sweep / rig README GIF generator."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import check_sweep_gifs as gifs  # noqa: E402
import check_yaw_sweep_run as sweep_run  # noqa: E402

KITTI_DRIVE_CANDIDATES = (
    Path.home() / "data/public/kitti_raw/2011_09_26/2011_09_26_drive_0005_sync",
)
VERDICTS = {"pass", "warn", "fail", "inconclusive"}


def test_default_angles_cover_the_sweep() -> None:
    angles = sweep_run.default_angles()
    assert angles[0] == -3.0
    assert angles[-1] == 3.0
    assert len(angles) == 25
    assert 0.0 in angles
    assert sweep_run.step_name(-0.25) == "step_-0.25"
    assert sweep_run.step_name(3.0) == "step_+3.00"


def test_loop_order_is_seamless() -> None:
    order = gifs.loop_order(25)
    assert order[0] == 0
    assert order[24] == 24
    assert order[-1] == 1  # the next frame of the loop is index 0 again
    assert len(order) == 48
    assert len(gifs.frame_durations_ms(48)) == 48
    durations = gifs.frame_durations_ms(48)
    assert durations[0] > durations[1]
    assert durations[24] > durations[23]


def test_committed_step_summaries_are_consistent() -> None:
    steps = gifs.load_steps()
    manifest = json.loads((gifs.SWEEP_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert [s["yaw_injected_deg"] for s in steps] == manifest["angles_deg"]
    assert len(steps) == 25
    assert "non-commercial" in manifest["dataset"]
    for step in steps:
        assert step["schema"] == sweep_run.SWEEP_SCHEMA
        assert step["overall_verdict"] in VERDICTS
        assert step["bag_input_sha256"] == manifest["bag_input_sha256"]
        assert "--tf" in step["command"]
        for record in step["pairs"].values():
            assert record["status"] in VERDICTS
            for axis in record["axes"].values():
                assert axis["tolerance_deg"] > 0
    lidar_vehicle = [gifs.pair_yaw_axis(step, "lidar-vehicle") for step in steps]
    assert all(axis is not None for axis in lidar_vehicle)


def test_perturbation_is_rotation_about_parent_z() -> None:
    vendor = gifs.se3(gifs.rotation_z(10.0), [1.0, 2.0, 3.0])
    turned = gifs.perturbed_velo_tf(vendor, 2.0)
    np.testing.assert_allclose(turned[:3, 3], vendor[:3, 3])
    np.testing.assert_allclose(turned[:3, :3], gifs.rotation_z(12.0), atol=1e-12)
    np.testing.assert_allclose(gifs.invert(vendor) @ vendor, np.eye(4), atol=1e-12)


def _write_calib(directory: Path) -> None:
    (directory / "calib_velo_to_cam.txt").write_text(
        "R: 7.533745e-03 -9.999714e-01 -6.166020e-04 1.480249e-02 7.280733e-04 -9.998902e-01 "
        "9.998621e-01 7.523790e-03 1.480755e-02\nT: -4.069766e-03 -7.631618e-02 -2.717806e-01\n",
        encoding="utf-8",
    )
    (directory / "calib_cam_to_cam.txt").write_text(
        "R_rect_00: 9.999239e-01 9.837760e-03 -7.445048e-03 -9.869795e-03 9.999421e-01 "
        "-4.278459e-03 7.402527e-03 4.351614e-03 9.999631e-01\n"
        "S_rect_02: 1.242000e+03 3.750000e+02\n"
        "P_rect_02: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 0.000000e+00 "
        "7.215377e+02 1.728540e+02 2.163791e-01 0.000000e+00 0.000000e+00 1.000000e+00 "
        "2.745884e-03\n",
        encoding="utf-8",
    )
    (directory / "calib_imu_to_velo.txt").write_text(
        "R: 9.999976e-01 7.553071e-04 -2.035826e-03 -7.854027e-04 9.998898e-01 -1.482298e-02 "
        "2.024406e-03 1.482454e-02 9.998881e-01\nT: -8.086759e-01 3.195559e-01 -7.997231e-01\n",
        encoding="utf-8",
    )


def test_zero_yaw_projection_is_the_kitti_calibration(tmp_path: Path) -> None:
    _write_calib(tmp_path)
    calib = gifs.KittiCalib(tmp_path)
    vendor = gifs.invert(calib.t_velo_imu)
    points = np.array([[10.0, 1.0, 0.5], [20.0, -3.0, 1.0]])
    uv0, depth0 = calib.project(points, vendor, vendor)
    homogeneous = np.hstack([points, np.ones((2, 1))])
    cam = (calib.r_rect @ calib.t_cam_velo @ homogeneous.T).T
    expected = (calib.p_rect2 @ cam.T).T
    np.testing.assert_allclose(uv0, expected[:, :2] / expected[:, 2:3], atol=1e-6)
    np.testing.assert_allclose(depth0, cam[:, 2], atol=1e-9)
    uv3, _ = calib.project(points, gifs.perturbed_velo_tf(vendor, 3.0), vendor)
    assert np.all(np.abs(uv3[:, 0] - uv0[:, 0]) > 5.0)  # 3 deg is many pixels at 10-20 m


def test_rig_gif_from_committed_summaries(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    steps = gifs.load_steps()
    frames, durations = gifs.build_rig_frames(steps, None)
    assert len(frames) == len(durations) == len(gifs.loop_order(len(steps)))
    assert frames[0].size == (gifs.WIDTH, gifs.HEIGHT)
    output = tmp_path / "rig.gif"
    gifs.save_gif(frames, durations, output)
    from PIL import Image

    with Image.open(output) as image:
        assert image.size == (gifs.WIDTH, gifs.HEIGHT)
        assert image.n_frames == len(frames)
        assert image.info["loop"] == 0
    assert output.stat().st_size < gifs.MAX_BYTES


def test_yaw_sweep_gif_when_kitti_is_available(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    drive = next((p for p in KITTI_DRIVE_CANDIDATES if p.exists()), None)
    if drive is None:
        pytest.skip("KITTI raw drive 0005 is not available")
    steps = gifs.load_steps()
    frames, durations = gifs.build_yaw_sweep_frames(steps, drive, gifs.DEFAULT_IMAGE_INDEX)
    assert len(frames) == len(durations) == len(gifs.loop_order(len(steps)))
    assert frames[0].size == (gifs.WIDTH, gifs.HEIGHT)
    output = tmp_path / "sweep.gif"
    gifs.save_gif(frames, durations, output)
    assert output.stat().st_size < gifs.MAX_BYTES


def test_committed_json_validates_against_schemas() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    step_schema = json.loads((gifs.SWEEP_DIR / "step.schema.json").read_text(encoding="utf-8"))
    steps = gifs.load_steps()
    for step in steps:
        jsonschema.validate(step, step_schema)
    manifest = json.loads(gifs.GIF_MANIFEST.read_text(encoding="utf-8"))
    schema = json.loads((gifs.SWEEP_DIR / "gifs.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(manifest, schema)
    assert manifest["steps"]["count"] == len(steps)
    for name in manifest["gifs"]:
        assert (ROOT / name).exists()
