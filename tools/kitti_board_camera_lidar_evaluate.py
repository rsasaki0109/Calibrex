#!/usr/bin/env python3
"""Board camera-LiDAR audit on KITTI's calibration recordings (pre-registered, run once).

An evaluation harness, not part of the Calibrex package. KITTI's raw data hold one
calibration recording per day (``CALIBRATION_DRIVES``): the vehicle stands still in front
of a dozen checkerboards of different sizes, all in view at once. For each day this tool

1. downloads the recording's ``_extract.zip`` with ``If-Match`` on the pinned ETag (S3
   answers 412 to any other object) and checks its size, and the day's
   ``_calib.zip`` against its pinned SHA-256;
2. takes the longest run of OXTS records with a ground speed under ``STILL_SPEED_MPS``
   as the still interval;
3. finds every board (``find_boards``) in each left grey image (``image_00``) of the
   interval and keeps the frame with the most boards (the earliest on a tie);
4. finds the boards in the right grey image (``image_01``) of the same instant and, for
   each left board, the right board and corner order that triangulate with the smallest
   reprojection error (under ``MAX_STEREO_ERROR_PX``), using only the cameras'
   intrinsics and their stereo extrinsic from ``calib_cam_to_cam.txt``; the board's
   square size is the median distance between neighbouring triangulated corners (the
   boards' sizes are not documented);
5. accumulates the ``MAX_SCANS`` Velodyne scans of the interval nearest the chosen frame;
6. runs ``calibrate_board`` with a board per pose, started from the axis permutation
   only (``START``: the Velodyne's x forward, y left, z up into the camera's optical
   axes, zero translation);
7. only then reads ``calib_velo_to_cam.txt`` and scores the estimate against it.

The reference camera-LiDAR transform is never an input of steps 1-6. The pass rule is
the pre-registration's (``docs/benchmarks/kitti_board_camera_lidar_preregistration.yaml``)
and is fixed here as module constants::

    python tools/kitti_board_camera_lidar_evaluate.py --work-dir WORK --output-dir RUNS
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from calibrex import __version__
from calibrex.evaluation.board_camera_lidar import (
    BoardCorners,
    BoardPose,
    calibrate_board,
    find_boards,
    plane_residuals,
)
from calibrex.evaluation.board_camera_lidar import (
    board_view as pnp_board_view,
)
from calibrex.evaluation.visual_rotation import CameraModel

FloatArray = NDArray[np.float64]
S3 = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
# day -> (drive, extract archive size, extract archive ETag)
CALIBRATION_DRIVES: dict[str, tuple[str, int, str]] = {
    "2011_09_26": ("0119", 142562751, "ad2c1f34039ae3ba760fb1db6b5bc4d6-17"),
    "2011_09_28": ("0225", 161910585, "ba54b08433f3bb7bb923c139a40f3878-20"),
    "2011_09_29": ("0108", 167792021, "b88f50b4f248380c6e6def107804c0eb-21"),
    "2011_09_30": ("0072", 96236911, "52bcd08619a10ebbdf502678dca0955a-12"),
    "2011_10_03": ("0058", 308389763, "a0b3b1067d83b893e352dc5ac9bc298a-37"),
}
# pinned by the KITTI camera-LiDAR pre-registrations v1 (2011_09_26) and v2 (the others)
CALIBRATION_SHA256: dict[str, str] = {
    "2011_09_26": "e0108cfd000cf802c14ed94fba38601185c792e5640abba015a34b7a85b812e0",
    "2011_09_28": "510b643c00adb7bc6b39616f1ed9b83cd0d689620d52ef5bc43ec019da6030d1",
    "2011_09_29": "260fb836916ac3cbbab7104e253531239a0191b87c411fa60b99178abbb8c40d",
    "2011_09_30": "2ed01563b616baec14f25659671bd9a77810bd87752d709d184fe304ff7258e3",
    "2011_10_03": "67f1f4dcfff4370f8ad1ccaf4fb013fca3f02abd650115f318ce9b1d0991a58c",
}
STILL_SPEED_MPS = 0.1
MIN_PATTERN = (4, 3)
MAX_BOARDS = 30
MAX_STEREO_ERROR_PX = 1.0
MAX_STEREO_GAP_S = 0.005
MAX_SCANS = 20
START = np.array(
    [[0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)
# the user's acceptance rule for a calibration (not loosened after the results)
MAX_ROTATION_ERROR_DEG = 0.5
MAX_POSITION_ERROR_M = 0.20
# the calibrex check rule for the second claim: an observed axis fails beyond twice
# max(3 std, floor)
SIGMA_K = 3.0
ROTATION_FLOOR_DEG = 0.5
TRANSLATION_FLOOR_M = 0.02
FAIL_FACTOR = 2.0
MIN_CONVERGED_DAYS = 3


# ------------------------------------------------------------------------ acquisition


def _curl(*args: str) -> bytes:
    return subprocess.run(
        ["curl", "-sSfL", "-m", "1800", *args], check=True, capture_output=True
    ).stdout


def fetch_extract(day: str, work: Path) -> Path:
    """The day's calibration recording, pinned by size and ETag."""

    drive, size, etag = CALIBRATION_DRIVES[day]
    name = f"{day}_drive_{drive}"
    path = work / f"{name}_extract.zip"
    if not path.exists() or path.stat().st_size != size:
        url = f"{S3}/{name}/{name}_extract.zip"
        subprocess.run(
            ["curl", "-sSf", "-m", "3600", "-H", f'If-Match: "{etag}"', "-o", str(path), url],
            check=True,
        )
    if path.stat().st_size != size:
        raise RuntimeError(f"{path.name}: {path.stat().st_size} bytes, pinned {size}")
    return path


def fetch_calibration(day: str, work: Path) -> Path:
    path = work / f"{day}_calib.zip"
    if not path.exists():
        path.write_bytes(_curl(f"{S3}/{day}_calib.zip"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != CALIBRATION_SHA256[day]:
        raise RuntimeError(f"{path.name}: SHA-256 {digest}, pinned {CALIBRATION_SHA256[day]}")
    return path


# ---------------------------------------------------------------------------- parsing


def parse_calibration_text(text: str) -> dict[str, FloatArray]:
    """``KEY: v1 v2 ...`` lines of a KITTI calibration file (non-numeric values skipped)."""

    values: dict[str, FloatArray] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        try:
            values[key.strip()] = np.array([float(v) for v in rest.split()], dtype=np.float64)
        except ValueError:
            continue
    return values


def camera_model(calibration: dict[str, FloatArray], index: str) -> CameraModel:
    """The unrectified camera ``index`` (``00``, ``01``) of ``calib_cam_to_cam.txt``."""

    k = calibration[f"K_{index}"].reshape(3, 3)
    distortion = tuple(float(v) for v in calibration[f"D_{index}"])
    return CameraModel(k[0, 0], k[1, 1], k[0, 2], k[1, 2], "radtan", distortion)


def stereo_extrinsic(calibration: dict[str, FloatArray]) -> tuple[FloatArray, FloatArray]:
    """``(R, t)`` with ``x_cam01 = R x_cam00 + t`` (``R_01``, ``T_01``)."""

    return calibration["R_01"].reshape(3, 3), calibration["T_01"].reshape(3)


def reference_transform(velo_to_cam: dict[str, FloatArray]) -> FloatArray:
    """``T_cam00_velo`` from ``calib_velo_to_cam.txt`` (the unrectified left grey camera)."""

    matrix = np.eye(4)
    matrix[:3, :3] = velo_to_cam["R"].reshape(3, 3)
    matrix[:3, 3] = velo_to_cam["T"].reshape(3)
    return matrix


def parse_timestamps(text: str) -> FloatArray:
    """Seconds since the epoch of KITTI's ``YYYY-MM-DD HH:MM:SS.fffffffff`` lines."""

    stamps = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        whole, _, fraction = line.partition(".")
        base = datetime.strptime(whole, "%Y-%m-%d %H:%M:%S").timestamp()
        stamps.append(base + (float("0." + fraction) if fraction else 0.0))
    return np.asarray(stamps, dtype=np.float64)


def still_interval(stamps: FloatArray, speeds: FloatArray) -> tuple[float, float] | None:
    """The longest run of OXTS records with ground speed below ``STILL_SPEED_MPS``."""

    best: tuple[float, float] | None = None
    start: int | None = None
    for index, speed in enumerate([*speeds, math.inf]):
        if speed < STILL_SPEED_MPS:
            start = index if start is None else start
            continue
        if start is not None:
            run = (float(stamps[start]), float(stamps[index - 1]))
            if best is None or run[1] - run[0] > best[1] - best[0]:
                best = run
            start = None
    return best


# ----------------------------------------------------------------------------- stereo


def _symmetries(grid: FloatArray) -> list[FloatArray]:
    """The eight orderings of a corner grid (turns and flips)."""

    out = []
    for flipped in (grid, grid[:, ::-1]):
        for turns in range(4):
            out.append(np.rot90(flipped, turns))
    return out


def _triangulate(
    left: FloatArray, right: FloatArray, rotation: FloatArray, translation: FloatArray
) -> FloatArray:
    """3D points in the left camera from normalized coordinates (midpoint method)."""

    points = []
    for a, b in zip(left, right, strict=True):
        ray_l = np.array([a[0], a[1], 1.0])
        ray_r = rotation.T @ np.array([b[0], b[1], 1.0])
        origin_r = -rotation.T @ translation
        matrix = np.column_stack([ray_l, -ray_r])
        depths, *_ = np.linalg.lstsq(matrix, origin_r, rcond=None)
        points.append(0.5 * (depths[0] * ray_l + origin_r + depths[1] * ray_r))
    return np.asarray(points)


@dataclass(frozen=True)
class StereoBoard:
    left: BoardCorners
    square_m: float
    error_px: float
    corners_camera: FloatArray


def stereo_square_size(
    left: BoardCorners,
    rights: list[BoardCorners],
    left_camera: CameraModel,
    right_camera: CameraModel,
    rotation: FloatArray,
    translation: FloatArray,
) -> StereoBoard | None:
    """The left board's square size from the right board and order that triangulate best."""

    left_norm = left_camera.normalize(left.pixels)
    best: StereoBoard | None = None
    for right in rights:
        if right.rows * right.columns != left.rows * left.columns:
            continue
        for grid in _symmetries(right.grid):
            if grid.shape[:2] != (left.rows, left.columns):
                continue
            right_norm = right_camera.normalize(grid.reshape(-1, 2))
            points = _triangulate(left_norm, right_norm, rotation, translation)
            if np.any(points[:, 2] <= 0.0):
                continue
            in_right = points @ rotation.T + translation
            error_l = np.linalg.norm(points[:, :2] / points[:, 2:3] - left_norm, axis=1)
            error_r = np.linalg.norm(in_right[:, :2] / in_right[:, 2:3] - right_norm, axis=1)
            error = float(
                np.mean(np.concatenate([error_l * left_camera.fx, error_r * right_camera.fx]))
            )
            if best is not None and error >= best.error_px:
                continue
            cube = points.reshape(left.rows, left.columns, 3)
            spacing = np.concatenate(
                [
                    np.linalg.norm(np.diff(cube, axis=1), axis=2).ravel(),
                    np.linalg.norm(np.diff(cube, axis=0), axis=2).ravel(),
                ]
            )
            best = StereoBoard(left, float(np.median(spacing)), error, points)
    if best is None or best.error_px > MAX_STEREO_ERROR_PX:
        return None
    return best


# -------------------------------------------------------------------------- the run


def _png(archive: zipfile.ZipFile, name: str) -> NDArray[np.uint8]:
    import cv2

    data = np.frombuffer(archive.read(name), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"{name}: not an image")
    return np.asarray(image, dtype=np.uint8)


def frames(archive: zipfile.ZipFile, folder: str) -> list[tuple[float, str]]:
    """``(time, member)`` of a stream's files: a file's frame number is its timestamp line."""

    name = next(n for n in archive.namelist() if n.endswith(f"/{folder}/timestamps.txt"))
    stamps = parse_timestamps(archive.read(name).decode())
    out = []
    for member in archive.namelist():
        if f"/{folder}/data/" not in member or member.endswith("/"):
            continue
        number = int(Path(member).stem)
        if number < len(stamps):
            out.append((float(stamps[number]), member))
    return sorted(out)


def calibration_member(archive: zipfile.ZipFile, file_name: str) -> dict[str, FloatArray]:
    name = next(n for n in archive.namelist() if n.endswith(file_name))
    return parse_calibration_text(archive.read(name).decode())


def estimate_day(
    day: str,
    extract: Path,
    cam_to_cam: dict[str, FloatArray],
    *,
    still_stereo: bool = False,
) -> dict[str, Any]:
    """Steps 2-6 for one day: no reference camera-LiDAR transform is read here.

    The defaults are the pre-registered harness. ``still_stereo`` is the development
    variant written after the audit (on the then spent recordings): the right image is the
    nearest one inside the still interval, with no time bound (the cameras of these
    recordings run half a frame apart), and the frame is the one with the most boards that
    triangulate with it, not the most detections (fragments of boards do not triangulate).
    """

    left_camera, right_camera = camera_model(cam_to_cam, "00"), camera_model(cam_to_cam, "01")
    rotation, translation = stereo_extrinsic(cam_to_cam)
    record: dict[str, Any] = {"day": day, "drive": CALIBRATION_DRIVES[day][0]}
    with zipfile.ZipFile(extract) as archive:
        oxts = frames(archive, "oxts")
        speeds = []
        for _time, name in oxts:
            fields = [float(v) for v in archive.read(name).decode().split()]
            speeds.append(math.hypot(fields[8], fields[9]))  # vf, vl
        oxts_stamps = np.asarray([time for time, _name in oxts])
        interval = still_interval(oxts_stamps, np.asarray(speeds))
        record["still_interval_s"] = (
            None
            if interval is None
            else [interval[0] - oxts_stamps[0], interval[1] - oxts_stamps[0]]
        )
        if interval is None:
            record.update(status="not_still", reason="no OXTS record under the still speed")
            return record
        lefts, rights = frames(archive, "image_00"), frames(archive, "image_01")
        candidates = [i for i, (t, _name) in enumerate(lefts) if interval[0] <= t <= interval[1]]
        still_rights = [item for item in rights if interval[0] <= item[0] <= interval[1]]

        def detect(name: str) -> list[BoardCorners]:
            return find_boards(_png(archive, name), min_pattern=MIN_PATTERN, max_boards=MAX_BOARDS)

        def right_of(time: float) -> tuple[float, str] | None:
            pool = still_rights if still_stereo else rights
            if not pool:
                return None
            nearest = min(pool, key=lambda item: abs(item[0] - time))
            if not still_stereo and abs(nearest[0] - time) > MAX_STEREO_GAP_S:
                return None
            return nearest

        best: tuple[int, list[BoardCorners]] | None = None
        best_score = -1
        counts: dict[str, Any] = {}
        right_cache: dict[str, list[BoardCorners]] = {}
        for index in candidates:
            boards = detect(lefts[index][1])
            score = len(boards)
            if still_stereo:
                right = right_of(lefts[index][0])
                if right is not None and right[1] not in right_cache:
                    right_cache[right[1]] = detect(right[1])
                partners = right_cache.get(right[1], []) if right is not None else []
                score = sum(
                    stereo_square_size(
                        b, partners, left_camera, right_camera, rotation, translation
                    )
                    is not None
                    for b in boards
                )
                counts[Path(lefts[index][1]).stem] = {"detections": len(boards), "stereo": score}
            else:
                counts[Path(lefts[index][1]).stem] = len(boards)
            if score > best_score:
                best, best_score = (index, boards), score
        record["boards_per_frame"] = counts
        if best is None or not best[1]:
            record.update(status="no_boards", reason="no board in any still left image")
            return record
        index, left_boards = best
        frame_time, frame_name = lefts[index]
        record.update(frame=Path(frame_name).stem, left_boards=len(left_boards))
        right = right_of(frame_time)
        if right is None:
            record.update(status="no_stereo", reason="no right image of the chosen instant")
            return record
        right_boards = right_cache.get(right[1]) or detect(right[1])
        record.update(right_boards=len(right_boards), right_gap_s=abs(right[0] - frame_time))
        scans = [
            (t, n) for t, n in frames(archive, "velodyne_points") if interval[0] <= t <= interval[1]
        ]
        scans.sort(key=lambda item: abs(item[0] - frame_time))
        clouds = [
            np.loadtxt(io.BytesIO(archive.read(name)), dtype=np.float64).reshape(-1, 4)[:, :3]
            for _time, name in sorted(scans[:MAX_SCANS])
        ]
    record["scans"] = len(clouds)
    if not clouds:
        record.update(status="no_scans", reason="no Velodyne scan in the still interval")
        return record
    cloud = np.concatenate(clouds)
    boards: list[dict[str, Any]] = []
    poses: list[BoardPose] = []
    for number, corners in enumerate(left_boards):
        entry: dict[str, Any] = {"board": number, "pattern": [corners.columns, corners.rows]}
        stereo = stereo_square_size(
            corners, right_boards, left_camera, right_camera, rotation, translation
        )
        if stereo is None:
            entry["reason"] = "no right board triangulates within the stereo error bound"
            boards.append(entry)
            continue
        board = corners.board(stereo.square_m)
        view = pnp_board_view(corners.pixels, board, left_camera)
        entry.update(square_m=stereo.square_m, stereo_error_px=stereo.error_px)
        if view is None:
            entry["reason"] = "PnP failed"
            boards.append(entry)
            continue
        entry.update(
            distance_m=float(np.linalg.norm(view.translation)), reprojection_px=view.reprojection_px
        )
        boards.append(entry)
        poses.append(BoardPose(f"board{number}", view, cloud, board=board))
    result = calibrate_board(poses, None, START)
    rejected = dict(result.poses_rejected)
    for entry in boards:
        pose_id = f"board{entry['board']}"
        entry["used"] = pose_id in result.poses_used
        if pose_id in rejected:
            entry["reason"] = rejected[pose_id]
    if result.transform_camera_lidar is not None:
        for pose in poses:
            found = plane_residuals(pose, None, result.transform_camera_lidar)
            entry = next(e for e in boards if f"board{e['board']}" == pose.pose_id)
            if found is not None:
                entry.update(normal_residual_deg=found[1], offset_residual_m=found[2])
    record.update(
        status=result.status,
        reason=result.reason,
        boards=boards,
        poses_used=len(result.poses_used),
        normal_singular_values=result.normal_singular_values,
        unobserved_directions=[list(d) for d in result.unobserved_directions],
        axes=[axis.__dict__ for axis in result.axes],
        transform_camera_lidar=None
        if result.transform_camera_lidar is None
        else result.transform_camera_lidar.tolist(),
    )
    return record


# -------------------------------------------------------------------------- scoring


def score_day(record: dict[str, Any], reference: FloatArray) -> dict[str, Any]:
    """The pre-registered verdicts of one day (step 7: the reference is read only here)."""

    from scipy.spatial.transform import Rotation

    scored: dict[str, Any] = {"converged": record.get("status") == "converged"}
    if record.get("transform_camera_lidar") is None:
        scored.update(accurate=False, check_fail_axes=None)
        return scored
    estimate = np.asarray(record["transform_camera_lidar"])
    relative = Rotation.from_matrix(reference[:3, :3] @ estimate[:3, :3].T)
    rotation_error = float(np.degrees(relative.magnitude()))
    position_error = float(np.linalg.norm(estimate[:3, 3] - reference[:3, 3]))
    per_axis = [*np.degrees(relative.as_rotvec()), *(reference[:3, 3] - estimate[:3, 3])]
    failing = []
    axes = {axis["name"]: axis for axis in record["axes"]}
    for index, name in enumerate(("roll", "pitch", "yaw", "x", "y", "z")):
        axis = axes[name]
        if not axis["observed"]:
            continue
        floor = ROTATION_FLOOR_DEG if index < 3 else TRANSLATION_FLOOR_M
        tolerance = max(SIGMA_K * axis["std"], floor)
        if abs(per_axis[index]) > FAIL_FACTOR * tolerance:
            failing.append(name)
    scored.update(
        rotation_error_deg=rotation_error,
        position_error_m=position_error,
        axis_errors=dict(
            zip(("roll", "pitch", "yaw", "x", "y", "z"), map(float, per_axis), strict=True)
        ),
        accurate=bool(
            scored["converged"]
            and rotation_error < MAX_ROTATION_ERROR_DEG
            and position_error < MAX_POSITION_ERROR_M
        ),
        check_fail_axes=failing,
    )
    return scored


def verdicts(scores: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Claim 1: every day accurate. Claim 2: no observed axis fails, on 3+ converged days."""

    converged = [day for day, s in scores.items() if s["converged"]]
    accurate = [day for day, s in scores.items() if s["accurate"]]
    failing = [day for day, s in scores.items() if s.get("check_fail_axes")]
    return {
        "accuracy": {
            "accurate_days": accurate,
            "verdict": "supported" if len(accurate) == len(CALIBRATION_DRIVES) else "refuted",
        },
        "check_honesty": {
            "converged_days": converged,
            "failing_days": failing,
            "verdict": "supported"
            if len(converged) >= MIN_CONVERGED_DAYS and not failing
            else "refuted",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--days", nargs="*", default=sorted(CALIBRATION_DRIVES))
    parser.add_argument("--keep-archives", action="store_true")
    parser.add_argument(
        "--still-stereo",
        action="store_true",
        help="development variant (not the pre-registered harness): pair the stereo images "
        "over the still interval and choose the frame by boards that triangulate",
    )
    args = parser.parse_args(argv)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores: dict[str, dict[str, Any]] = {}
    for day in args.days:
        calibration = fetch_calibration(day, args.work_dir)
        with zipfile.ZipFile(calibration) as archive:
            cam_to_cam = calibration_member(archive, "calib_cam_to_cam.txt")
        extract = fetch_extract(day, args.work_dir)
        record = estimate_day(day, extract, cam_to_cam, still_stereo=args.still_stereo)
        record["variant"] = "still_stereo (development)" if args.still_stereo else "pre-registered"
        record["calibrex_version"] = __version__
        (args.output_dir / f"{day}_estimate.json").write_text(json.dumps(record, indent=1) + "\n")
        with zipfile.ZipFile(calibration) as archive:
            reference = reference_transform(calibration_member(archive, "calib_velo_to_cam.txt"))
        scores[day] = score_day(record, reference)
        (args.output_dir / f"{day}_score.json").write_text(json.dumps(scores[day], indent=1) + "\n")
        print(day, record.get("status"), json.dumps(scores[day]))
        if not args.keep_archives:
            extract.unlink()
    summary = {
        "scores": scores,
        "verdicts": verdicts(scores) if len(scores) == len(CALIBRATION_DRIVES) else None,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["verdicts"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
