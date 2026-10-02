"""Extract the Hilti 2022 exp21 camera-rate / gyro series for the camera-imu-lock GIF.

Runs calibrex's own visual rotation tracking (the one ``calibrex camera-imu rotation``
uses, with the default options the committed development artifact records) on the
exp21 development recording only, and writes a small ``series.npz`` window plus a
provenance JSON.  Never point this at the held-out exp01-exp04 recordings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from calibrex import __version__
from calibrex.core.io import read_mapping
from calibrex.data.ros2_camera_imu import iter_gray_images, load_ros2_imu
from calibrex.evaluation.camera_imu_rotation import CameraImuRunOptions
from calibrex.evaluation.visual_rotation import CameraModel, track_camera_rotations

ALLOWED_BAG = "exp21_ros2"
WINDOW_S = 8.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rotvec(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    import cv2

    return np.asarray(cv2.Rodrigues(matrix)[0], dtype=np.float64).reshape(3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--camchain", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--bag-sha256-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets/camera_imu_lock"))
    parser.add_argument("--camera", default="cam0")
    parser.add_argument("--image-topic", default="/alphasense/cam0/image_raw")
    parser.add_argument("--imu-topic", default="/alphasense/imu")
    args = parser.parse_args()
    if args.bag.name != ALLOWED_BAG:
        raise SystemExit(f"only the development recording {ALLOWED_BAG} may be used")

    chain = read_mapping(args.camchain)
    entry = chain[args.camera]
    options = CameraImuRunOptions()  # defaults equal the artifact's recorded options
    track = track_camera_rotations(
        iter_gray_images(args.bag, args.image_topic, stride=options.frame_stride),
        CameraModel.from_kalibr(entry),
        options.visual,
    )
    imu = load_ros2_imu(args.bag, args.imu_topic, acceleration_unit="mps2")

    t = track.times_s
    rel = np.array(
        [_rotvec(track.orientations[i].T @ track.orientations[i + 1]) for i in range(len(t) - 1)]
    )
    dts = np.diff(t)
    ok = (track.segment[1:] == track.segment[:-1]) & (dts > 0)
    rate = rel / dts[:, None]  # camera-frame angular rate over [t_i, t_i+1]
    mid = 0.5 * (t[:-1] + t[1:])
    imu_lo, imu_hi = float(imu.times_s[0]), float(imu.times_s[-1])
    best, start = -1.0, float(t[0])
    for candidate in mid[::2]:
        if candidate < imu_lo + 1.0 or candidate + WINDOW_S > imu_hi - 1.0:
            continue
        sel = ok & (mid >= candidate) & (mid < candidate + WINDOW_S)
        if sel.sum() < 0.95 * WINDOW_S / np.median(dts):
            continue
        score = float(np.sum(np.sum(rate[sel] ** 2, axis=1)))
        if score > best:
            best, start = score, float(candidate)
    pad = 0.3
    sel = ok & (mid >= start) & (mid < start + WINDOW_S)
    gsel = (imu.times_s >= start - pad) & (imu.times_s <= start + WINDOW_S + pad)
    artifact = read_mapping(args.artifact)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "series.npz",
        t0=np.float64(start),
        cam_start_s=t[:-1][sel] - start,
        cam_end_s=t[1:][sel] - start,
        cam_rate_rps=rate[sel],
        imu_t_s=imu.times_s[gsel] - start,
        gyro_rps=imu.gyro_rps[gsel],
        est_quat_xyzw=np.asarray(artifact["rotation_quat_xyzw"], dtype=np.float64),
        est_time_offset_s=np.float64(artifact["time_offset_s"]),
        est_gyro_bias_rps=np.asarray(artifact["gyro_bias_rps"], dtype=np.float64),
        kalibr_T_cam_imu=np.asarray(entry["T_cam_imu"], dtype=np.float64),
        kalibr_time_offset_s=np.float64(entry["timeshift_cam_imu"]),
    )
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    manifest = {
        "schema": "slac.gif_series_provenance/v0.1",
        "output": "docs/assets/camera-imu-lock.gif",
        "series": "docs/assets/camera_imu_lock/series.npz",
        "calibrex_version": __version__,
        "calibrex_git_commit": commit,
        "commands": [
            "python tools/extract_camera_imu_lock_series.py --bag <hilti2022>/exp21_ros2 "
            "--camchain <hilti2022>/calibration_files/calib_3_cam0-1-camchain-imucam.yaml "
            "--artifact docs/assets/hilti2022_camera_imu/cam0_exp21.yaml "
            "--bag-sha256-file <hilti2022>/exp21_raw_bag.sha256",
            "python tools/generate_camera_imu_lock_gif.py",
        ],
        "dataset": {
            "family": "hilti2022",
            "sequence": ALLOWED_BAG,
            "role": "development recording (held-out exp01-exp04 never touched)",
            "license": "Hilti SLAM Challenge 2022 terms",
            "image_topic": args.image_topic,
            "imu_topic": args.imu_topic,
            "camera": args.camera,
        },
        "inputs_sha256": {
            "camchain": _sha256(args.camchain),
            "estimate_artifact": _sha256(args.artifact),
            "bag_metadata_yaml": _sha256(args.bag / "metadata.yaml"),
            "bag_sha256_file_contents": (
                args.bag_sha256_file.read_text().split()[0] if args.bag_sha256_file else None
            ),
        },
        "estimate_artifact_generator_version": artifact["provenance"]["generator_version"],
        "estimate_artifact_git_commit": artifact["provenance"]["git_commit"],
        "tracking_options": "VisualRotationOptions() defaults, frame_stride=4 (as the artifact)",
        "window": {
            "start_s_in_bag": start,
            "duration_s": WINDOW_S,
            "camera_intervals": int(sel.sum()),
            "imu_samples": int(gsel.sum()),
        },
        "visual_rotation_stats": {
            "pairs": track.stats.pairs,
            "essential": track.stats.essential,
            "pure_rotation": track.stats.pure_rotation,
            "failed": track.stats.failed,
        },
    }
    (args.output_dir / "series.provenance.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest["window"]))


if __name__ == "__main__":
    main()
