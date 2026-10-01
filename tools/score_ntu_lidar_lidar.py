"""Score LiDAR-LiDAR extrinsics on held-out blocks of NTU VIRAL recordings.

Two steps, because the reference odometry is the slow part::

    # 1. Reference odometry, map scans, and target samples per recording (cached).
    python tools/score_ntu_lidar_lidar.py collect BAG DESIGN_DIR --cache CACHE.pkl

    # 2. Held-out scores and cross-recording consistency.
    python tools/score_ntu_lidar_lidar.py score CACHE.pkl... OUT_DIR

``DESIGN_DIR`` holds the dataset's ``lidar_horz.yaml`` and ``lidar_vert.yaml``.
Collection follows ``collect_map_samples`` with the default
``LidarLidarMapOptions``, but keeps every reference scan so that maps of any
half window can be rebuilt.

Scoring, for the pooled recordings with every third 10-second block held out
(keys are recording and block, sorted):

* ``calibrex_map``: the map-registration extrinsic (maps of +-5 reference
  scans) fitted on the train blocks, started from the design value;
* ``scan_to_scan``: the same solver with a map of only the nearest reference
  scan, i.e. registering concurrent scans, fitted on the same train blocks
  from the same start;
* ``ntu_design``: the dataset's design value ``inv(T_body_horz) T_body_vert``.

The per-block metric is the mean over target points of ``min((r / 0.1 m)^2,
25)``, with ``r`` the point-to-plane residual against a map of +-10 reference
scans (lost points count 25), so neither fitted method is scored on its own
map.  Separately, per recording, Calibrex fitted on that recording's train
blocks is compared with Calibrex fitted on the other recordings' train
blocks; the metric is ``max(rotation / 0.3 deg, translation / 3 cm)``.

When only the development recording is scored, the cross-recording
consistency step is undefined (there is no other recording); the result marks
it ``nan`` and the caller ignores it.
"""

from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from calibrex.data.livox_ros2 import LivoxStreamProfile, bag_input_digest, iter_livox_points
from calibrex.evaluation.lidar_lidar_map import LidarLidarMapOptions, read_transform
from calibrex.solvers.lidar_lidar_map_solver import MapSample, solve_map_extrinsic
from calibrex.solvers.scan_to_scan_odometry import IncrementalScanOdometry, preprocess_scan

REFERENCE_TOPIC = "/os1_cloud_node1/points"
TARGET_TOPIC = "/os1_cloud_node2/points"
POINT_TIME_FIELD = "t"
OPTIONS = LidarLidarMapOptions()
FIT_HALF_WINDOW = OPTIONS.map_half_window_scans  # 5
SCORE_HALF_WINDOW = 10
SCORE_SIGMA_M = 0.1
CLIP = 25.0
ROTATION_TOLERANCE_DEG = 0.3
TRANSLATION_TOLERANCE_M = 0.03


def design_transform(directory: str) -> np.ndarray:
    horz = read_transform(Path(directory) / "lidar_horz.yaml")
    vert = read_transform(Path(directory) / "lidar_vert.yaml")
    return np.linalg.inv(horz) @ vert


def collect(bag: str, design_dir: str, cache: str) -> None:
    opts = OPTIONS

    def profile(topic: str) -> LivoxStreamProfile:
        return LivoxStreamProfile("lidar", topic, "", POINT_TIME_FIELD, "offset_s", "mps2")

    odometry = IncrementalScanOdometry(opts.odometry)
    scans: list[np.ndarray] = []
    times: list[float] = []
    first: float | None = None
    for time_s, points, offsets in iter_livox_points(bag, profile(REFERENCE_TOPIC)):
        first = time_s if first is None else first
        finite = np.isfinite(points).all(axis=1)
        odometry.add(points[finite], time_s, None if offsets is None else offsets[finite])
        pose = odometry.poses[-1]
        local = preprocess_scan(points[finite], opts.preprocess)
        scans.append((local @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32))
        times.append(time_s)
        if len(times) % 500 == 0:
            print("reference scans", len(times), flush=True)
    assert first is not None
    poses = [np.asarray(pose, dtype=np.float64) for pose in odometry.poses]
    reference_times = np.asarray(times)
    half = SCORE_HALF_WINDOW  # the widest window any map needs
    samples = []
    for index, (time_s, points, _) in enumerate(iter_livox_points(bag, profile(TARGET_TOPIC))):
        if index % opts.sample_every_scans:
            continue
        nearest = int(np.argmin(np.abs(reference_times - time_s)))
        if abs(reference_times[nearest] - time_s) > opts.max_time_gap_s:
            continue
        if nearest < half or nearest >= len(scans) - half:
            continue
        target = preprocess_scan(points[np.isfinite(points).all(axis=1)], opts.preprocess)
        if len(target) < opts.min_target_points:
            continue
        samples.append(
            {
                "time_s": float(time_s),
                "block": int((time_s - first) // opts.block_duration_s),
                "nearest": nearest,
                "target_points": target.astype(np.float32),
            }
        )
    digest, scope = bag_input_digest([bag])
    data = {
        "name": Path(bag).name,
        "scans": scans,
        "poses": poses,
        "samples": samples,
        "design": design_transform(design_dir),
        "input_digest": digest,
        "input_digest_scope": scope,
    }
    with open(cache, "wb") as stream:
        pickle.dump(data, stream)
    print(Path(bag).name, "reference scans", len(scans), "samples", len(samples), flush=True)


def map_samples(data: dict, half: int, key_prefix: str) -> list[tuple[str, MapSample]]:
    """MapSamples of one recording with a map of +-``half`` reference scans."""

    scans, poses = data["scans"], data["poses"]
    out = []
    for item in data["samples"]:
        nearest = item["nearest"]
        out.append(
            (
                f"{key_prefix}/b{item['block']:03d}",
                MapSample(
                    time_s=item["time_s"],
                    block=item["block"],
                    reference_pose=poses[nearest],
                    target_points=item["target_points"].astype(np.float64),
                    map_points=np.concatenate(scans[nearest - half : nearest + half + 1]).astype(
                        np.float64
                    ),
                ),
            )
        )
    return out


def block_score(samples: list[MapSample], extrinsic: np.ndarray) -> float:
    total = 0.0
    points = 0
    for sample in samples:
        residual, _, _ = sample.residuals(extrinsic, OPTIONS.solver)
        lost = len(sample.target_points) - len(residual)
        total += float(np.sum(np.minimum((residual / SCORE_SIGMA_M) ** 2, CLIP))) + CLIP * lost
        points += len(sample.target_points)
    return total / points if points else math.nan


def disagreement(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    rotation = math.degrees(
        float(np.linalg.norm(Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).as_rotvec()))
    )
    translation = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    return (
        max(rotation / ROTATION_TOLERANCE_DEG, translation / TRANSLATION_TOLERANCE_M),
        rotation,
        translation,
    )


def _fit(samples: list[MapSample], initial: np.ndarray) -> np.ndarray | None:
    return solve_map_extrinsic(samples, initial, OPTIONS.solver).transform


def _describe(extrinsic: np.ndarray | None, design: np.ndarray) -> dict | None:
    if extrinsic is None:
        return None
    local = np.degrees(Rotation.from_matrix(extrinsic[:3, :3] @ design[:3, :3].T).as_rotvec())
    return {
        "matrix": extrinsic.tolist(),
        "rotation_minus_design_deg": local.tolist(),
        "translation_minus_design_m": (extrinsic[:3, 3] - design[:3, 3]).tolist(),
    }


def score(caches: list[str], output: str) -> None:
    datasets = []
    for cache in caches:
        with open(cache, "rb") as stream:
            datasets.append(pickle.load(stream))
    datasets.sort(key=lambda data: data["name"])
    design = datasets[0]["design"]
    for data in datasets:
        assert np.allclose(data["design"], design), "recordings disagree on the design value"
    fit_sets = {data["name"]: map_samples(data, FIT_HALF_WINDOW, data["name"]) for data in datasets}
    single_sets = {data["name"]: map_samples(data, 0, data["name"]) for data in datasets}
    keys = sorted({key for tagged in fit_sets.values() for key, _ in tagged})
    holdout_keys = {key for index, key in enumerate(keys) if index % OPTIONS.holdout_every == 1}

    def train(sets: dict, names: list[str]) -> list[MapSample]:
        return [s for name in names for key, s in sets[name] if key not in holdout_keys]

    names = [data["name"] for data in datasets]
    calibrex = _fit(train(fit_sets, names), design)
    scan_to_scan = _fit(train(single_sets, names), design)
    methods = {"calibrex_map": calibrex, "scan_to_scan": scan_to_scan, "ntu_design": design}
    del single_sets
    scores: dict[str, dict[str, float]] = {method: {} for method in methods}
    for data in datasets:
        scored = map_samples(data, SCORE_HALF_WINDOW, data["name"])
        for key in sorted({key for key, _ in scored} & holdout_keys):
            block = [sample for tag, sample in scored if tag == key]
            for method, extrinsic in methods.items():
                scores[method][key] = (
                    math.nan if extrinsic is None else block_score(block, extrinsic)
                )
        del scored
    consistency = {}
    for name in names:
        others = [other for other in names if other != name]
        own = _fit(train(fit_sets, [name]), design)
        rest = _fit(train(fit_sets, others), design) if others else None
        if own is None or rest is None:
            consistency[name] = {"normalized": math.nan}
            continue
        normalized, rotation, translation = disagreement(own, rest)
        consistency[name] = {
            "normalized": normalized,
            "rotation_deg": rotation,
            "translation_m": translation,
            "own_translation_m": own[:3, 3].tolist(),
            "own_rotvec_deg": np.degrees(Rotation.from_matrix(own[:3, :3]).as_rotvec()).tolist(),
        }

    result = {
        "recordings": names,
        "estimates": {
            method: _describe(extrinsic, design) for method, extrinsic in methods.items()
        },
        "holdout_block_scores": scores,
        "consistency": consistency,
        "design": {
            "matrix": design.tolist(),
            "rotation_rpy_deg": np.degrees(
                Rotation.from_matrix(design[:3, :3]).as_euler("xyz")
            ).tolist(),
            "translation_m": design[:3, 3].tolist(),
        },
        "input_digests": {data["name"]: data["input_digest"] for data in datasets},
        "input_digest_scopes": {data["name"]: data["input_digest_scope"] for data in datasets},
    }
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scores.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    table = {method: np.array(list(values.values())) for method, values in scores.items()}
    print("held-out blocks", len(table["calibrex_map"]))
    for method, values in table.items():
        print(" ", method, "mean", round(float(np.nanmean(values)), 4))
    for other in ("scan_to_scan", "ntu_design"):
        better = int(np.sum(table["calibrex_map"] < table[other]))
        print(f"  calibrex better than {other} in {better}")
    for method, extrinsic in methods.items():
        described = _describe(extrinsic, design)
        print(
            " ",
            method,
            None
            if described is None
            else {k: np.round(v, 4).tolist() for k, v in described.items() if k != "matrix"},
        )
    print("consistency", {name: round(v["normalized"], 3) for name, v in consistency.items()})


def split_halves(cache: str, output_dir: str) -> None:
    """Development only: split one recording's cache into two halves by time."""

    with open(cache, "rb") as stream:
        data = pickle.load(stream)
    blocks = sorted({item["block"] for item in data["samples"]})
    middle = blocks[len(blocks) // 2]
    for suffix, keep in (("a", lambda b: b < middle), ("b", lambda b: b >= middle)):
        part = dict(data, name=f"{data['name']}_{suffix}")
        part["samples"] = [item for item in data["samples"] if keep(item["block"])]
        with open(Path(output_dir) / f"{part['name']}.pkl", "wb") as stream:
            pickle.dump(part, stream)


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        cache_index = sys.argv.index("--cache")
        collect(sys.argv[2], sys.argv[3], sys.argv[cache_index + 1])
    elif sys.argv[1] == "split-halves":
        split_halves(sys.argv[2], sys.argv[3])
    else:
        score(sys.argv[2:-1], sys.argv[-1])
