"""Evaluate a fixed 1 px stereo round-trip mask without changing optimizer settings."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict
from tools.diagnose_stereo_cycle import round_trip_error
from tools.koide_shared_bundle_pandey import read_koide_ply
from tools.run_camera_lidar_development_matrix import pose_metrics
from tools.run_stereo_translation_refinement import digest
from tools.stereo_rerectification import restore_left_depth

from calibrex.data.kitti import read_calibration_file
from calibrex.solvers.stereo_translation_refiner import (
    StereoSurfaceObservation,
    StereoTranslationOptions,
    _surface,
    refine_stereo_translation,
)


class FilterComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    scope: str = "spent-data 1 px cycle mask versus archived unmasked re-rectification"
    source_commit: str
    source_sha256: dict[str, str]
    input_sha256: dict[str, str]
    frames: list[dict[str, Any]]
    cases: list[dict[str, Any]]


def mask_depth(depth: np.ndarray, cycle: np.ndarray) -> np.ndarray:
    result = depth.copy()
    result[~(np.isfinite(cycle) & (cycle <= 1.0))] = np.nan
    return result


def prepare(frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    meta = json.loads(Path(frame["source_metadata"]).read_text())
    images = [Path(p) for p in meta["source_sha256"] if p.endswith(".png")]
    cam_path = next(Path(p) for p in meta["source_sha256"] if p.endswith("calib_cam_to_cam.txt"))
    cam = read_calibration_file(cam_path)
    projections = [np.array(cam["P_rect_" + c]).reshape(3, 4) for c in ["02", "03"]]
    g = frame["geometry"]
    if cv2.__version__ != g["opencv_version"]:
        raise ValueError("OpenCV version changed")
    rotations = [np.array(g[k]) for k in ["r1", "r2"]]
    new_projections = [np.array(g[k]) for k in ["p1", "p2"]]
    ims = []
    masks = []
    for path, p, r, new in zip(images, projections, rotations, new_projections, strict=True):
        im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        h, w = im.shape
        x, y = cv2.initUndistortRectifyMap(
            p[:, :3], np.zeros(5), r, new[:, :3], (w, h), cv2.CV_32FC1
        )
        ims.append(cv2.remap(im, x, y, cv2.INTER_LINEAR))
        masks.append((x >= 0) & (x < w - 1) & (y >= 0) & (y < h - 1))
    forward = cv2.StereoSGBM_create(**frame["settings"]).compute(*ims).astype(float) / 16
    # Horizontal flip converts right-to-left negative disparity to the same positive range.
    reverse = (
        cv2.StereoSGBM_create(**frame["settings"])
        .compute(np.ascontiguousarray(ims[1][:, ::-1]), np.ascontiguousarray(ims[0][:, ::-1]))
        .astype(float)[:, ::-1]
        / 16
    )
    v, u = np.indices(forward.shape)
    ur = np.rint(u - forward).astype(int)
    valid = (forward > 0) & masks[0] & (ur >= 0) & (ur < w)
    valid &= masks[1][v, np.clip(ur, 0, w - 1)]
    base = new_projections[0][0, 3] - new_projections[1][0, 3]
    depth = np.full(forward.shape, np.nan)
    depth[valid] = base / forward[valid]
    depth[(depth < 0.1) | (depth > 80)] = np.nan
    restored = restore_left_depth(
        depth, rotations[0], projections[0][:, :3], new_projections[0][:, :3]
    )
    saved_path = next(Path(p) for p in frame["depth_sha256"] if p.endswith("-rerectified.npy"))
    if digest(saved_path) != frame["depth_sha256"][str(saved_path)]:
        raise ValueError("saved depth changed")
    np.testing.assert_array_equal(restored, np.load(saved_path, allow_pickle=False))
    # Reject reverse endpoints outside the image remap support too.
    ul = np.rint(u + reverse).astype(int)
    reverse[~(masks[1] & (ul >= 0) & (ul < w) & masks[0][v, np.clip(ul, 0, w - 1)])] = np.nan
    cycle = round_trip_error(forward, reverse)

    filtered = restore_left_depth(
        mask_depth(depth, cycle), rotations[0], projections[0][:, :3], new_projections[0][:, :3]
    )
    keep = np.isfinite(filtered)
    np.testing.assert_array_equal(filtered[keep], restored[keep])
    return restored, filtered


def run(previous_path: Path, workspace: Path, output: Path) -> FilterComparison:
    import cv2

    cv2.setNumThreads(1)
    prior = json.loads(previous_path.read_text())
    for path, expected in {**prior["source_sha256"], **prior["input_sha256"]}.items():
        if digest(Path(path)) != expected:
            raise ValueError(f"changed frozen source/input: {path}")
    workspace.mkdir(parents=True, exist_ok=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("do not overwrite comparison")
    sources = {
        **prior["source_sha256"],
        str(Path(__file__)): digest(Path(__file__)),
        "tools/diagnose_stereo_cycle.py": digest(Path("tools/diagnose_stereo_cycle.py")),
    }
    report = FilterComparison(
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256=sources,
        input_sha256={**prior["input_sha256"], str(previous_path): digest(previous_path)},
        frames=[],
        cases=[],
    )
    plan = json.loads(Path("docs/assets/stereo_priority_20261029/plan.json").read_text())
    options = StereoTranslationOptions(sampling="point_priority")
    cache: dict[str, StereoSurfaceObservation] = {}
    for frame in prior["frames"]:
        original, filtered = prepare(frame)
        bundle = Path(next(c["bundle"] for c in plan["cases"] if c["case_id"] == frame["case_id"]))
        name = frame["frame_id"]
        config = json.loads((bundle / "calib-frontend.json").read_text())
        meta_path = Path(frame["source_metadata"])
        meta = json.loads(meta_path.read_text())
        points = read_koide_ply(bundle / (name + ".ply"))[:, :3]
        obs = StereoSurfaceObservation(
            name,
            filtered,
            points,
            tuple(config["camera"]["intrinsics"]),
            meta["stereo"]["baseline_m"],
        )
        before = StereoSurfaceObservation(name, original, points, obs.intrinsics, obs.baseline_m)
        cache[str(meta_path)] = obs
        destination = workspace / (frame["case_id"] + "-" + name + ".npy")
        np.save(destination, filtered)
        for path, expected in frame["depth_sha256"].items():
            if path.endswith("-rerectified.npy"):
                report.input_sha256[path] = expected
        report.frames.append(
            {
                "case_id": frame["case_id"],
                "frame_id": name,
                "depth_before": int(np.isfinite(original).sum()),
                "depth_after": int(np.isfinite(filtered).sum()),
                "surface_before": int(_surface(before, options).valid.sum()),
                "surface_after": int(_surface(obs, options).valid.sum()),
                "output": str(destination),
                "output_sha256": digest(destination),
            }
        )
    for case in plan["cases"]:
        bundle = Path(case["bundle"])
        mapping = json.loads((bundle / "frontend.json").read_text())
        names = json.loads((bundle / "calib-frontend.json").read_text())["meta"]["bag_names"]
        observations = [
            cache[str(Path(mapping[name + ".ply"]["source"]).parent / "frontend.json")]
            for name in names
        ]
        control = next(
            c
            for c in prior["cases"]
            if c["case_id"] == case["case_id"] and c["mode"] == "rerectified"
        )
        initial = np.array(control["initial_T_camera_lidar"])
        fit = refine_stereo_translation(observations, initial, options)
        # Scoring reference is opened only after the fit finishes.
        reference = np.array(
            json.loads((bundle / "seed.json").read_text())["reference_T_camera_lidar"]
        )
        candidate = np.array(fit.candidate_T_camera_lidar)
        np.testing.assert_array_equal(candidate[:3, :3], initial[:3, :3])
        if not fit.accepted:
            np.testing.assert_array_equal(fit.output_T_camera_lidar, initial)
        record = {
            "case_id": case["case_id"],
            "result": fit.as_dict(),
            "initial_T_camera_lidar": initial.tolist(),
            "baseline": control["baseline"],
            "unmasked_raw_candidate": control["raw_candidate"],
            "unmasked_status": control["result"]["status"],
            "raw_candidate": pose_metrics(candidate, reference, True),
            "returned": pose_metrics(np.array(fit.output_T_camera_lidar), reference, True),
        }
        report.cases.append(record)
        output.write_text(report.model_dump_json(indent=2) + "\n")
        print(case["case_id"], fit.status, record["raw_candidate"]["translation_m"], flush=True)
    output.with_suffix(".schema.json").write_text(
        json.dumps(FilterComparison.model_json_schema(), indent=2) + "\n"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous", type=Path)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.previous, args.workspace, args.output)
