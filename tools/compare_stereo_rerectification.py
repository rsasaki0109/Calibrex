"""Frozen spent-data comparison: corrected projection versus stereo re-rectification."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict
from tools.diagnose_stereo_reference_controls import surface_check
from tools.kitti_aligned_stereo_bundle import depth_from_disparity
from tools.koide_shared_bundle_pandey import read_koide_ply
from tools.run_camera_lidar_development_matrix import pose_metrics
from tools.run_stereo_translation_refinement import digest
from tools.stereo_rerectification import compute_depth

from calibrex.data.kitti import read_calibration_file
from calibrex.solvers.stereo_translation_refiner import (
    StereoSurfaceObservation,
    StereoTranslationOptions,
    refine_stereo_translation,
)


class Comparison(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    scope: str = "spent-data stereo frontend comparison; saved D2D initialization, translation only"
    source_commit: str
    source_sha256: dict[str, str]
    input_sha256: dict[str, str]
    frames: list[dict[str, Any]]
    cases: list[dict[str, Any]]


def run(archive: Path, workspace: Path, output: Path) -> Comparison:
    import cv2

    cv2.setNumThreads(1)
    workspace.mkdir(parents=True, exist_ok=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("do not overwrite prior results")
    plan_path = archive / "plan.json"
    plan = json.loads(plan_path.read_text())
    sources = [
        Path(__file__),
        Path("tools/stereo_rerectification.py"),
        Path("tools/kitti_aligned_stereo_bundle.py"),
        Path("tools/diagnose_stereo_reference_controls.py"),
        Path("tools/diagnose_stereo_translation.py"),
        Path("tools/run_camera_lidar_development_matrix.py"),
        Path("tools/koide_shared_bundle_pandey.py"),
        Path("src/calibrex/solvers/stereo_translation_refiner.py"),
    ]
    report = Comparison(
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256={str(p): digest(p) for p in sources},
        input_sha256={str(plan_path): digest(plan_path)},
        frames=[],
        cases=[],
    )
    cache: dict[str, dict[str, StereoSurfaceObservation]] = {}
    options = StereoTranslationOptions(sampling="point_priority")
    for case in plan["cases"]:
        bundle = Path(case["bundle"])
        saved_path = archive / case["case_id"] / "refinement.json"
        saved = json.loads(saved_path.read_text())
        for filename, expected in saved["input_sha256"].items():
            if digest(Path(filename)) != expected:
                raise ValueError(f"changed input {filename}")
        report.input_sha256.update(saved["input_sha256"])
        report.input_sha256[str(saved_path)] = digest(saved_path)
        config = json.loads((bundle / "calib-frontend.json").read_text())
        mapping = json.loads((bundle / "frontend.json").read_text())
        observations: dict[str, list[StereoSurfaceObservation]] = {
            "projection_only": [],
            "rerectified": [],
        }
        for name in config["meta"]["bag_names"]:
            meta_path = Path(mapping[name + ".ply"]["source"]).parent / "frontend.json"
            key = str(meta_path)
            if key not in cache:
                meta = json.loads(meta_path.read_text())
                cam_path = next(
                    Path(p) for p in meta["source_sha256"] if p.endswith("calib_cam_to_cam.txt")
                )
                images = [Path(p) for p in meta["source_sha256"] if p.endswith(".png")]
                if len(images) != 2:
                    raise ValueError("two original grayscale inputs required")
                for p in [cam_path, *images]:
                    if digest(p) != meta["source_sha256"][str(p)]:
                        raise ValueError(f"changed frontend source {p}")
                    report.input_sha256[str(p)] = digest(p)
                if cv2.__version__ != meta["stereo"]["opencv_version"]:
                    raise ValueError("OpenCV differs from control frontend")
                cam = read_calibration_file(cam_path)
                pl, pr = [np.array(cam["P_rect_" + c]).reshape(3, 4) for c in ["02", "03"]]
                old = np.load(bundle / (name + ".depth.npy"), allow_pickle=False)
                disparity = pl[0, 0] * meta["stereo"]["baseline_m"] / old
                control = depth_from_disparity(disparity, pl, pr)
                new, geometry = compute_depth(
                    *(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in images),
                    pl,
                    pr,
                    meta["stereo"]["settings"],
                )
                points = read_koide_ply(bundle / (name + ".ply"))[:, :3]
                cache[key] = {}
                outputs = {}
                for mode, depth in [("projection_only", control), ("rerectified", new)]:
                    cache[key][mode] = StereoSurfaceObservation(
                        name,
                        depth,
                        points,
                        tuple(config["camera"]["intrinsics"]),
                        meta["stereo"]["baseline_m"],
                    )
                    dest = workspace / (
                        case["case_id"].replace("-positive", "") + "-" + name + "-" + mode + ".npy"
                    )
                    np.save(dest, depth)
                    outputs[str(dest)] = digest(dest)
                report.frames.append(
                    {
                        "source_metadata": key,
                        "case_id": case["case_id"],
                        "frame_id": name,
                        "geometry": geometry,
                        "settings": meta["stereo"]["settings"],
                        "depth_sha256": outputs,
                        "control_valid_pixels": int(np.isfinite(control).sum()),
                    }
                )
            for mode in observations:
                observations[mode].append(cache[key][mode])
        # Neither frontend nor fit has access to the reference transform.
        initial = np.array(saved["initial_T_camera_lidar"])
        fits = {
            mode: refine_stereo_translation(obs, initial, options)
            for mode, obs in observations.items()
        }
        seed = bundle / "seed.json"
        reference = np.array(json.loads(seed.read_text())["reference_T_camera_lidar"])
        report.input_sha256[str(seed)] = digest(seed)
        for mode, fit in fits.items():
            candidate = np.array(fit.candidate_T_camera_lidar)
            np.testing.assert_array_equal(candidate[:3, :3], initial[:3, :3])
            if not fit.accepted:
                np.testing.assert_array_equal(fit.output_T_camera_lidar, initial)
            report.cases.append(
                {
                    "case_id": case["case_id"],
                    "mode": mode,
                    "result": fit.as_dict(),
                    "initial_T_camera_lidar": initial.tolist(),
                    "baseline": pose_metrics(initial, reference, True),
                    "raw_candidate": pose_metrics(candidate, reference, True),
                    "returned": pose_metrics(np.array(fit.output_T_camera_lidar), reference, True),
                    "surface_checks": [
                        surface_check(o, reference, options).model_dump()
                        for o in observations[mode]
                    ],
                }
            )
            print(
                case["case_id"],
                mode,
                fit.status,
                report.cases[-1]["raw_candidate"]["translation_m"],
                flush=True,
            )
        output.write_text(report.model_dump_json(indent=2) + "\n")
    output.with_suffix(".schema.json").write_text(
        json.dumps(Comparison.model_json_schema(), indent=2) + "\n"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.archive, args.workspace, args.output)
