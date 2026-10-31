"""Independent stereo round-trip and disparity-band diagnostics on spent frames."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict
from tools.koide_shared_bundle_pandey import read_koide_ply
from tools.run_stereo_translation_refinement import digest
from tools.stereo_rerectification import restore_left_depth

from calibrex.data.kitti import read_calibration_file


class CycleReport(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    scope: str = "post-hoc spent-data diagnostic; reference scoring only; no fitting"
    source_commit: str
    source_sha256: dict[str, str]
    input_sha256: dict[str, str]
    disparity_edges_px: list[int] = [0, 8, 16, 32, 64, 128]
    round_trip_threshold_px: float = 1.0
    frames: list[dict[str, Any]]


def round_trip_error(left: np.ndarray, right_positive: np.ndarray) -> np.ndarray:
    """Positive disparities on both domains; missing reverse values remain NaN."""
    v, u = np.indices(left.shape)
    valid = np.isfinite(left) & (left > 0)
    ur = np.rint(u - np.where(valid, left, 0)).astype(int)
    valid &= (ur >= 0) & (ur < left.shape[1])
    reverse = right_positive[v, np.clip(ur, 0, left.shape[1] - 1)]
    valid &= np.isfinite(reverse) & (reverse > 0)
    result = np.full(left.shape, np.nan)
    result[valid] = np.abs(left[valid] - reverse[valid])
    return result


def stats(disparity_error: np.ndarray, z_error: np.ndarray) -> dict[str, Any]:
    return {
        "count": len(z_error),
        "median_disparity_error_px": float(np.median(disparity_error)) if len(z_error) else None,
        "p90_absolute_disparity_error_px": float(np.percentile(np.abs(disparity_error), 90))
        if len(z_error)
        else None,
        "median_z_error_m": float(np.median(z_error)) if len(z_error) else None,
        "p90_absolute_z_error_m": float(np.percentile(np.abs(z_error), 90))
        if len(z_error)
        else None,
    }


def run(previous_path: Path, output: Path) -> CycleReport:
    import cv2

    cv2.setNumThreads(1)
    prior = json.loads(previous_path.read_text())
    for path, expected in {**prior["source_sha256"], **prior["input_sha256"]}.items():
        if digest(Path(path)) != expected:
            raise ValueError(f"changed frozen source/input: {path}")
    report = CycleReport(
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256={**prior["source_sha256"], str(Path(__file__)): digest(Path(__file__))},
        input_sha256={**prior["input_sha256"], str(previous_path): digest(previous_path)},
        frames=[],
    )
    plan_path = Path("docs/assets/stereo_priority_20261029/plan.json")
    plan = json.loads(plan_path.read_text())
    for frame in prior["frames"]:
        meta = json.loads(Path(frame["source_metadata"]).read_text())
        images = [Path(p) for p in meta["source_sha256"] if p.endswith(".png")]
        cam_path = next(
            Path(p) for p in meta["source_sha256"] if p.endswith("calib_cam_to_cam.txt")
        )
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
        report.input_sha256[str(saved_path)] = digest(saved_path)
        # Reject reverse endpoints outside the image remap support too.
        ul = np.rint(u + reverse).astype(int)
        reverse[~(masks[1] & (ul >= 0) & (ul < w) & masks[0][v, np.clip(ul, 0, w - 1)])] = np.nan
        cycle = round_trip_error(forward, reverse)
        bundle = Path(next(c["bundle"] for c in plan["cases"] if c["case_id"] == frame["case_id"]))
        reference = np.array(
            json.loads((bundle / "seed.json").read_text())["reference_T_camera_lidar"]
        )
        points = read_koide_ply(bundle / (frame["frame_id"] + ".ply"))[:, :3]
        camera = (points @ reference[:3, :3].T + reference[:3, 3]) @ rotations[0].T
        camera = camera[camera[:, 2] > 0.1]
        q = camera @ new_projections[0][:, :3].T
        pu, pv = np.rint(q[:, 0] / q[:, 2]).astype(int), np.rint(q[:, 1] / q[:, 2]).astype(int)
        inside = (pu >= 0) & (pu < w) & (pv >= 0) & (pv < h)
        camera, pu, pv = camera[inside], pu[inside], pv[inside]
        order = np.argsort(camera[:, 2], kind="stable")
        _, first = np.unique((pv * w + pu)[order], return_index=True)
        selected = order[first]
        camera, pu, pv = camera[selected], pu[selected], pv[selected]
        supported = np.isfinite(depth[pv, pu])
        camera, pu, pv = camera[supported], pu[supported], pv[supported]
        d = forward[pv, pu]
        de = d - base / camera[:, 2]
        ze = camera[:, 2] - depth[pv, pu]
        ce = cycle[pv, pu]
        groups = {
            "all": np.ones(len(d), dtype=bool),
            "consistent": ce <= 1,
            "inconsistent": ce > 1,
            "reverse_missing": ~np.isfinite(ce),
        }
        bins = []
        for lo, hi in zip(
            report.disparity_edges_px[:-1], report.disparity_edges_px[1:], strict=True
        ):
            band = (d >= lo) & (d < hi)
            bins.append(
                {
                    "lower_px": lo,
                    "upper_px": hi,
                    "groups": {
                        label: stats(de[band & mask], ze[band & mask])
                        for label, mask in groups.items()
                    },
                }
            )
        report.frames.append(
            {
                "case_id": frame["case_id"],
                "frame_id": frame["frame_id"],
                "projected_visible_pixels": len(selected),
                "stereo_supported_pixels": len(d),
                "groups": {label: stats(de[mask], ze[mask]) for label, mask in groups.items()},
                "bins": bins,
            }
        )
        print(
            frame["case_id"], frame["frame_id"], len(d), int(groups["consistent"].sum()), flush=True
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("do not overwrite diagnostic")
    output.write_text(report.model_dump_json(indent=2) + "\n")
    output.with_suffix(".schema.json").write_text(
        json.dumps(CycleReport.model_json_schema(), indent=2) + "\n"
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("previous", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    run(a.previous, a.output)
