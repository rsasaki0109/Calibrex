"""Post-hoc projection audit on immutable spent-data bundles, without refitting."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict
from tools.diagnose_stereo_translation import identified_correspondences
from tools.kitti_aligned_stereo_bundle import depth_from_disparity, frame_timestamp
from tools.koide_shared_bundle_pandey import read_koide_ply
from tools.run_stereo_translation_refinement import digest

from calibrex.data.kitti import read_calibration_file
from calibrex.solvers.stereo_translation_refiner import (
    StereoSurfaceObservation,
    StereoTranslationOptions,
    _surface,
)


class Audit(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    scope: str = "post-hoc projection/timestamp consistency; no refitting or verdict changes"
    source_commit: str
    source_sha256: dict[str, str]
    input_sha256: dict[str, str]
    frames: list[dict[str, Any]]


def run(archive: Path, output: Path) -> Audit:
    plan_path = archive / "plan.json"
    plan = json.loads(plan_path.read_text())
    report = Audit(
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256={
            str(p): digest(p)
            for p in [
                Path(__file__),
                Path("tools/kitti_aligned_stereo_bundle.py"),
                Path("tools/diagnose_stereo_translation.py"),
                Path("src/calibrex/solvers/stereo_translation_refiner.py"),
            ]
        },
        input_sha256={str(plan_path): digest(plan_path)},
        frames=[],
    )
    for case in plan["cases"]:
        if not case["case_id"].endswith("positive"):
            continue  # Both seeds have the same sensor arrays; audit each frame once.
        bundle = Path(case["bundle"])
        saved_path = archive / case["case_id"] / "refinement.json"
        saved = json.loads(saved_path.read_text())
        for filename, expected in saved["input_sha256"].items():
            if digest(Path(filename)) != expected:
                raise ValueError(f"changed input: {filename}")
        report.input_sha256.update(saved["input_sha256"])
        report.input_sha256[str(saved_path)] = digest(saved_path)
        seed = bundle / "seed.json"
        report.input_sha256[str(seed)] = digest(seed)
        reference = np.array(json.loads(seed.read_text())["reference_T_camera_lidar"])
        mapping = json.loads((bundle / "frontend.json").read_text())
        config = json.loads((bundle / "calib-frontend.json").read_text())
        options = StereoTranslationOptions(**saved["result"]["options"])
        for name in config["meta"]["bag_names"]:
            meta_path = Path(mapping[name + ".ply"]["source"]).parent / "frontend.json"
            meta = json.loads(meta_path.read_text())
            cam_path = next(
                Path(p) for p in meta["source_sha256"] if p.endswith("calib_cam_to_cam.txt")
            )
            if digest(cam_path) != meta["source_sha256"][str(cam_path)]:
                raise ValueError("camera calibration changed")
            report.input_sha256[str(cam_path)] = digest(cam_path)
            velo_path = cam_path.parent / "calib_velo_to_cam.txt"
            report.input_sha256[str(velo_path)] = digest(velo_path)
            cam = read_calibration_file(cam_path)
            velo = read_calibration_file(velo_path)
            pl, pr = [np.array(cam["P_rect_" + c]).reshape(3, 4) for c in ("02", "03")]
            k = pl[:, :3]
            tl, tr = np.linalg.solve(k, pl[:, 3]), np.linalg.solve(pr[:, :3], pr[:, 3])
            expected = np.eye(4)
            rect = np.array(cam["R_rect_00"]).reshape(3, 3)
            expected[:3, :3] = rect @ np.array(velo["R"]).reshape(3, 3)
            expected[:3, 3] = rect @ np.array(velo["T"]) + tl
            np.testing.assert_allclose(expected, reference, atol=1e-7, rtol=0)
            old = np.load(bundle / (name + ".depth.npy"), allow_pickle=False)
            legacy_base = (pl[0, 3] - pr[0, 3]) / k[0, 0]
            np.testing.assert_allclose(legacy_base, meta["stereo"]["baseline_m"])
            disparity = k[0, 0] * legacy_base / old
            new = depth_from_disparity(disparity, pl, pr)
            obs = StereoSurfaceObservation(
                name,
                old,
                read_koide_ply(bundle / (name + ".ply"))[:, :3],
                tuple(config["camera"]["intrinsics"]),
                legacy_base,
            )
            surface = _surface(obs, options)
            _, _, _, ids = identified_correspondences(
                surface, reference[:3, :3], reference[:3, 3], options
            )
            h, w = old.shape
            indices, pixels = ids // (h * w), ids % (h * w)
            v, u = pixels // w, pixels % w
            camera = obs.lidar_points[indices] @ reference[:3, :3].T + reference[:3, 3]
            good = np.isfinite(new[v, u])
            old_residual = camera[:, 2] - old[v, u]
            new_residual = camera[good, 2] - new[v[good], u[good]]
            delta = tr - tl
            # Predicted right-minus-left row at corrected stereo depths.
            z = new[v[good], u[good]]
            row_shift = (k[1, 1] * delta[1] - (v[good] - k[1, 2]) * delta[2]) / (z + delta[2])
            timestamp_matches = []
            for filename, sha in meta["source_sha256"].items():
                if filename.endswith("timestamps.txt"):
                    path = Path(filename)
                    if digest(path) != sha:
                        raise ValueError("timestamp source changed")
                    report.input_sha256[str(path)] = sha
                    timestamp_matches.append(frame_timestamp(path, int(name.split("-")[-1])))
            if sorted(timestamp_matches) != sorted(
                [meta["timing"]["image_header_ns"], meta["timing"]["right_image_header_ns"]]
            ):
                raise ValueError("stereo timestamp identity mismatch")
            report.frames.append(
                {
                    "case_id": case["case_id"],
                    "frame_id": name,
                    "right_from_left_translation_m": delta.tolist(),
                    "reference_matrix_max_difference": float(np.max(np.abs(expected - reference))),
                    "selected_pairs": len(ids),
                    "corrected_valid_pairs": int(good.sum()),
                    "depth_correction_median_m": float(
                        np.median(new[v[good], u[good]] - old[v[good], u[good]])
                    ),
                    "old_signed_z_median_m": float(np.median(old_residual[good])),
                    "corrected_signed_z_median_m": float(np.median(new_residual)),
                    "predicted_right_row_shift_median_px": float(np.median(row_shift)),
                    "timing": meta["timing"],
                    "motion": meta["motion"],
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("do not overwrite audit results")
    output.write_text(report.model_dump_json(indent=2) + "\n")
    output.with_suffix(".schema.json").write_text(
        json.dumps(Audit.model_json_schema(), indent=2) + "\n"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.archive, args.output)
