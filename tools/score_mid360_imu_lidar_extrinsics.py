"""Score complete IMU-LiDAR extrinsics on one common odometry (pre-registered procedure).

See ``docs/benchmarks/mid360_imu_lidar_translation_preregistration.yaml``.
The common odometry is collected once per recording with the design
reference deskew::

    python tools/score_mid360_imu_lidar_candidates.py collect BAG rtk-slam \\
        design_reference '[0, 0, 0, 1]' 0.0 '[0, 0, 0]' COMMON.pkl

and every candidate (rotation quaternion of ``T_lidar_imu``, clock offset,
lever arm) is then scored on it::

    python tools/score_mid360_imu_lidar_extrinsics.py COMMON.pkl BAG CANDIDATES.json OUT_DIR

``CANDIDATES.json`` maps a method id to ``{"quat_xyzw": [...],
"time_offset_s": ..., "translation_m": [...]}``.  One ``score_<method>.json``
is written per candidate.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from calibrex.data.livox_ros2 import LIVOX_PROFILES, load_livox_imu
from calibrex.evaluation.imu_lidar_rotation import score_imu_lidar_candidate
from calibrex.evaluation.imu_lidar_translation import score_imu_lidar_translation_candidate
from calibrex.solvers.imu_lidar_translation_solver import ImuPreintegrator, TranslationOptions

# The pre-registered scoring nuisances: gravity and accelerometer bias per span piece,
# one velocity per 2-second segment.  Fixed here so that changing the lever-arm
# solver's defaults never changes how candidates are scored.
SCORING_OPTIONS = TranslationOptions(
    segment_duration_s=2.0, min_segment_scans=4, gravity_per_segment=False
)


def main() -> None:
    common_path, bag, candidates_path, output = sys.argv[1:5]
    with open(common_path, "rb") as stream:
        common = pickle.load(stream)
    gyro, windows = common["gyro"], common["windows"]
    imu = load_livox_imu(bag, LIVOX_PROFILES["rtk-slam"])
    spans = score_imu_lidar_candidate(gyro, windows, np.eye(3), 0.0).holdout_spans_s
    candidates = json.loads(Path(candidates_path).read_text(encoding="utf-8"))
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    for method, candidate in candidates.items():
        rotation = Rotation.from_quat(candidate["quat_xyzw"]).as_matrix()
        time_offset = float(candidate["time_offset_s"])
        translation = np.asarray(candidate["translation_m"], dtype=np.float64)
        # Gyro bias refit on the common train windows with R and dt fixed.
        bias = score_imu_lidar_candidate(gyro, windows, rotation, time_offset).refit_gyro_bias_rps
        preintegrator = ImuPreintegrator(imu.times_s, imu.gyro_rps, imu.accel_mps2, bias)
        score = score_imu_lidar_translation_candidate(
            preintegrator, windows, rotation, time_offset, translation, spans, SCORING_OPTIONS
        )
        values = list(score.holdout_span_rms_m.values())
        result = {
            "method_id": method,
            "candidate": candidate,
            "refit_gyro_bias_rps": bias.tolist(),
            "holdout_spans": len(spans),
            "holdout_rows": score.holdout_rows,
            "mean_span_rms_m": float(np.mean(values)),
            "holdout_span_rms_m": {
                str(key): value for key, value in score.holdout_span_rms_m.items()
            },
            "holdout_span_median_m": {
                str(key): value for key, value in score.holdout_span_median_m.items()
            },
        }
        target = out_dir / f"score_{method}.json"
        target.write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(method, f"{len(values)} spans", f"mean span RMS {1000 * np.mean(values):.2f} mm")


if __name__ == "__main__":
    main()
