"""Score IMU-LiDAR rotation candidates on shared held-out spans of a Livox bag.

Two steps, because collecting a candidate's odometry takes about 25 minutes
per recording and the candidates can run in parallel::

    # 1. Deskew with each candidate's own rotation, clock offset, and gyro bias
    #    and pickle its odometry windows.
    python tools/score_mid360_imu_lidar_candidates.py collect BAG PROFILE LABEL \\
        QUAT_XYZW_JSON DT_S BIAS_JSON OUT.pkl

    # 2. Hold out the design reference's held-out spans for every candidate
    #    and write one score JSON per candidate (windows_X.pkl -> score_X.json).
    python tools/score_mid360_imu_lidar_candidates.py score REFERENCE.pkl CANDIDATE.pkl...

The score JSONs are the input of ``tools/build_mid360_imu_lidar_audit.py``.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from calibrex.evaluation.imu_lidar_rotation import (
    ImuLidarRunOptions,
    collect_livox_windows,
    score_imu_lidar_candidate,
)
from calibrex.solvers.imu_lidar_rotation_solver import gyro_rotation_model


def collect(bag: str, profile: str, label: str, quat: str, dt: str, bias: str, out: str) -> None:
    quat_xyzw = json.loads(quat)
    rotation = Rotation.from_quat(quat_xyzw).as_matrix()
    options = ImuLidarRunOptions()
    start = time.time()
    gyro, _ = collect_livox_windows([bag], profile, options, max_scans=1)
    model = gyro_rotation_model(gyro, rotation, np.array(json.loads(bias)), float(dt))
    gyro, windows = collect_livox_windows([bag], profile, options, rotation_model=model)
    payload = {
        "label": label,
        "quat": quat_xyzw,
        "dt": float(dt),
        "gyro": gyro,
        "windows": windows,
        "seconds": time.time() - start,
    }
    with open(out, "wb") as stream:
        pickle.dump(payload, stream)
    print(label, len(windows), round(time.time() - start))


def _load(path: str) -> dict[str, Any]:
    with open(path, "rb") as stream:
        return dict(pickle.load(stream))


def score(reference_path: str, candidate_paths: list[str]) -> None:
    options = ImuLidarRunOptions()
    reference = _load(reference_path)
    spans = score_imu_lidar_candidate(
        reference["gyro"],
        reference["windows"],
        Rotation.from_quat(reference["quat"]).as_matrix(),
        reference["dt"],
        options,
    ).holdout_spans_s
    for path in candidate_paths:
        candidate = _load(path)
        rotation = Rotation.from_quat(candidate["quat"]).as_matrix()
        args = (candidate["gyro"], candidate["windows"], rotation, candidate["dt"], options)
        shared = score_imu_lidar_candidate(*args, holdout_spans_s=spans)
        own = score_imu_lidar_candidate(*args)
        rpy_deg = Rotation.from_matrix(rotation).as_euler("xyz", degrees=True)
        result = {
            "label": candidate["label"],
            "rotation_rpy_deg": rpy_deg.tolist(),
            "time_offset_s": candidate["dt"],
            "odometry_windows": len(candidate["windows"]),
            "holdout_reference": reference["label"],
            "holdout_median_rate_residual_rps": shared.holdout_median_rate_residual_rps,
            "holdout_rms_rate_residual_rps": shared.holdout_rms_rate_residual_rps,
            "holdout_windows": shared.holdout_windows,
            "holdout_intervals": shared.holdout_intervals,
            "refit_gyro_bias_rps": shared.refit_gyro_bias_rps.tolist(),
            "holdout_window_medians_rps": {
                str(key): value for key, value in shared.holdout_window_medians_rps.items()
            },
            "own_segmentation_holdout_median_rate_residual_rps": (
                own.holdout_median_rate_residual_rps
            ),
            "seconds": candidate["seconds"],
        }
        target = Path(path).with_suffix(".json")
        target = target.with_name(target.name.replace("windows_", "score_", 1))
        target.write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(
            candidate["label"],
            f"{shared.holdout_windows} spans",
            f"median {shared.holdout_median_rate_residual_rps:.4f} rad/s",
            f"(own segmentation {own.holdout_median_rate_residual_rps:.4f})",
        )


def main() -> None:
    command, arguments = sys.argv[1], sys.argv[2:]
    if command == "collect":
        collect(*arguments)
    elif command == "score":
        score(arguments[0], arguments[1:])
    else:
        raise SystemExit(f"unknown command {command!r}; use collect or score")


if __name__ == "__main__":
    main()
