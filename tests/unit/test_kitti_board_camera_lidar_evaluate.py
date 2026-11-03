"""The KITTI calibration-recording board audit harness on synthetic recordings.

The pipeline runs end to end on a zip laid out like a KITTI ``_extract.zip`` (grey stereo
images of a single shot of five boards, Velodyne text scans, OXTS records with a still
stretch, timestamp files whose frame numbers do not start at zero): the still interval,
the stereo square sizes, the board poses and the calibration against the truth. The pass
rule and the parsers are checked on their own.
"""

from __future__ import annotations

import io
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tests.unit.test_board_camera_lidar_multi import CAMERA, SHOT, SIZE, TRUE, _paint, _pose, _scan
from tools.kitti_board_camera_lidar_evaluate import (
    CALIBRATION_DRIVES,
    estimate_day,
    parse_calibration_text,
    parse_timestamps,
    score_day,
    stereo_square_size,
    still_interval,
    verdicts,
)

from calibrex.evaluation.board_camera_lidar import Checkerboard, find_boards

cv2 = pytest.importorskip("cv2")

BASELINE = np.array([-0.54, 0.0, 0.0])  # x_cam01 = x_cam00 + BASELINE, as KITTI's T_01
CAM_TO_CAM = f"""calib_time: 09-Jan-2012 13:57:47
corner_dist: 9.950000e-02
S_00: 1.392000e+03 5.120000e+02
K_00: {CAMERA.fx} 0 {CAMERA.cx} 0 {CAMERA.fy} {CAMERA.cy} 0 0 1
D_00: 0 0 0 0 0
R_00: 1 0 0 0 1 0 0 0 1
T_00: 0 0 0
K_01: {CAMERA.fx} 0 {CAMERA.cx} 0 {CAMERA.fy} {CAMERA.cy} 0 0 1
D_01: 0 0 0 0 0
R_01: 1 0 0 0 1 0 0 0 1
T_01: {BASELINE[0]} {BASELINE[1]} {BASELINE[2]}
"""


def _stamp_lines(start: datetime, count: int, step_s: float) -> str:
    lines = []
    for i in range(count):
        t = start + timedelta(seconds=i * step_s)
        lines.append(t.strftime("%Y-%m-%d %H:%M:%S.") + f"{t.microsecond:06d}000")
    return "\n".join(lines) + "\n"


def _render(truths, offset: np.ndarray) -> np.ndarray:
    image = np.full((SIZE[1], SIZE[0]), 110, np.uint8)
    for board, rotation, translation in truths:
        _paint(image, board, rotation, translation + offset)
    return image


def _extract_zip(path: Path, right_delay_s: float = 0.0) -> Path:
    """A KITTI-like calibration recording: still from OXTS record 5 to 25 (of 30)."""

    truths = []
    for columns, rows, square, centre, yaw, pitch in SHOT:
        board = Checkerboard(columns, rows, square)
        truths.append((board, *_pose(board, centre, yaw, pitch)))
    left, right = _render(truths, np.zeros(3)), _render(truths, BASELINE)
    rng = np.random.default_rng(11)
    root = "2011_09_26/2011_09_26_drive_0119_extract"
    start = datetime(2011, 9, 26, 13, 0, 0)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{root}/oxts/timestamps.txt", _stamp_lines(start, 30, 0.1))
        for i in range(30):
            speed = 0.0 if 5 <= i <= 25 else 2.0
            fields = [49.0, 8.4, 110.0, 0, 0, 0, 0, 0, speed, 0.0, 0.0] + [0.0] * 19
            archive.writestr(f"{root}/oxts/data/{i:010d}.txt", " ".join(map(str, fields)))
        # images and scans numbered from 3: the files' numbers index the timestamp lines
        for folder, image in (("image_00", left), ("image_01", right)):
            delay = timedelta(seconds=right_delay_s if folder == "image_01" else 0.0)
            archive.writestr(
                f"{root}/{folder}/timestamps.txt", _stamp_lines(start + delay, 12, 0.25)
            )
            for i in range(3, 12):
                ok, png = cv2.imencode(".png", image)
                assert ok
                archive.writestr(f"{root}/{folder}/data/{i:010d}.png", png.tobytes())
        archive.writestr(f"{root}/velodyne_points/timestamps.txt", _stamp_lines(start, 25, 0.1))
        for i in range(3, 25):
            cloud = _scan(truths, rng)
            buffer = io.StringIO()
            np.savetxt(buffer, np.column_stack([cloud, np.zeros(len(cloud))]), fmt="%.4f")
            archive.writestr(f"{root}/velodyne_points/data/{i:010d}.txt", buffer.getvalue())
    return path


def test_the_pipeline_recovers_a_synthetic_single_shot(tmp_path: Path) -> None:
    extract = _extract_zip(tmp_path / "extract.zip")

    record = estimate_day("2011_09_26", extract, parse_calibration_text(CAM_TO_CAM))

    assert record["status"] == "converged", record.get("reason")
    assert record["left_boards"] == len(SHOT) and record["right_boards"] == len(SHOT)
    assert record["still_interval_s"] == pytest.approx([0.5, 2.5], abs=1e-6)
    # stereo depth over a 0.54 m baseline: square sizes within 2 % at up to 7 m
    squares = sorted(b["square_m"] for b in record["boards"])
    assert squares == pytest.approx(sorted(square for _c, _r, square, *_ in SHOT), rel=0.02)
    assert record["poses_used"] == len(SHOT)
    reference = TRUE
    scored = score_day(record, reference)
    assert scored["converged"] and scored["accurate"], scored
    # five boards give translation stds of about 2 cm: the observed axes within the check rule
    assert scored["check_fail_axes"] == []


def test_stereo_square_size_finds_the_right_board_and_corner_order() -> None:
    board = Checkerboard(7, 5, 0.09)
    rotation, translation = _pose(board, (0.5, 0.2, 5.0), 20.0, -10.0)
    left = _render([(board, rotation, translation)], np.zeros(3))
    right = _render([(board, rotation, translation)], BASELINE)
    (left_board,) = find_boards(left)
    (right_board,) = find_boards(right)
    # whatever order OpenCV numbered the right corners in, the same board is found
    turned = type(right_board)(
        right_board.columns, right_board.rows, right_board.grid[::-1, ::-1].reshape(-1, 2)
    )

    for candidate in (right_board, turned):
        stereo = stereo_square_size(left_board, [candidate], CAMERA, CAMERA, np.eye(3), BASELINE)
        assert stereo is not None
        assert stereo.square_m == pytest.approx(0.09, abs=0.001)
        assert stereo.error_px < 0.5
    other = Checkerboard(6, 4, 0.09)
    elsewhere = _render([(other, *_pose(other, (-1.5, 0.0, 6.0), 0.0, 0.0))], BASELINE)
    (wrong,) = find_boards(elsewhere)
    assert stereo_square_size(left_board, [wrong], CAMERA, CAMERA, np.eye(3), BASELINE) is None


def test_parsers_and_the_still_interval() -> None:
    values = parse_calibration_text(CAM_TO_CAM)
    assert "calib_time" not in values and values["T_01"][0] == pytest.approx(-0.54)
    stamps = parse_timestamps("2011-09-26 13:02:25.964389445\n2011-09-26 13:02:26.064389445\n")
    assert stamps[1] - stamps[0] == pytest.approx(0.1, abs=1e-6)
    times = np.arange(10.0)
    speeds = np.array([1, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=float)
    assert still_interval(times, speeds) == (4.0, 7.0)
    assert still_interval(times, np.ones(10)) is None


def _record(rotation_deg: float, shift_m: float, std_deg: float = 0.1, std_m: float = 0.01):
    estimate = TRUE.copy()
    estimate[:3, :3] = (
        Rotation.from_rotvec(np.radians([rotation_deg, 0, 0])).as_matrix() @ TRUE[:3, :3]
    )
    estimate[:3, 3] = TRUE[:3, 3] + np.array([shift_m, 0, 0])
    axes = [
        {"name": n, "unit": "deg", "value": 0.0, "std": std_deg, "observed": True}
        for n in ("roll", "pitch", "yaw")
    ] + [
        {"name": n, "unit": "m", "value": 0.0, "std": std_m, "observed": True}
        for n in ("x", "y", "z")
    ]
    return {"status": "converged", "transform_camera_lidar": estimate.tolist(), "axes": axes}


def test_the_pass_rule() -> None:
    good = score_day(_record(0.3, 0.05), TRUE)
    assert good["accurate"] and good["check_fail_axes"] == []
    assert not score_day(_record(0.6, 0.0), TRUE)["accurate"]  # rotation 0.5 deg or more
    assert not score_day(_record(0.0, 0.25), TRUE)["accurate"]  # position 0.20 m or more
    # an observed axis 1.2 deg off with a 0.1 deg std fails the check rule (2 x 0.5 deg)
    assert score_day(_record(1.2, 0.0), TRUE)["check_fail_axes"] == ["roll"]
    unsolved = score_day({"status": "insufficient_poses", "transform_camera_lidar": None}, TRUE)
    assert not unsolved["converged"] and not unsolved["accurate"]

    days = sorted(CALIBRATION_DRIVES)
    all_good = dict.fromkeys(days, good)
    assert verdicts(all_good)["accuracy"]["verdict"] == "supported"
    assert verdicts(all_good)["check_honesty"]["verdict"] == "supported"
    one_bad = {**all_good, days[0]: unsolved}
    assert verdicts(one_bad)["accuracy"]["verdict"] == "refuted"
    assert verdicts(one_bad)["check_honesty"]["verdict"] == "supported"


def test_cameras_half_a_frame_apart_need_the_still_stereo_variant(tmp_path: Path) -> None:
    """KITTI's 2011_09_28-30 recordings: the right camera runs ~0.1 s after the left one."""

    extract = _extract_zip(tmp_path / "extract.zip", right_delay_s=0.1)
    calibration = parse_calibration_text(CAM_TO_CAM)

    frozen = estimate_day("2011_09_26", extract, calibration)
    variant = estimate_day("2011_09_26", extract, calibration, still_stereo=True)

    assert frozen["status"] == "no_stereo"
    assert variant["status"] == "converged", variant.get("reason")
    assert variant["right_gap_s"] == pytest.approx(0.1, abs=1e-3)
    assert all(v["stereo"] == len(SHOT) for v in variant["boards_per_frame"].values())
