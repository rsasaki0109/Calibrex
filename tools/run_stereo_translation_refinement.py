"""Refine a saved D2D translation using stereo geometry, without reading a reference."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict
from tools.koide_shared_bundle_pandey import read_koide_ply

from calibrex.solvers.stereo_translation_refiner import (
    StereoSurfaceObservation,
    StereoTranslationOptions,
    StereoTranslationResult,
    refine_stereo_translation,
)


class RefinementRun(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    scope: str
    source_commit: str
    driver_sha256: str
    solver_sha256: str
    input_sha256: dict[str, str]
    initial_T_camera_lidar: list[list[float]]
    elapsed_seconds: float
    result: StereoTranslationResult


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(
    bundle: Path, output: Path, options: StereoTranslationOptions | None = None
) -> RefinementRun:
    """Only depth, cloud, camera intrinsics/baseline and the saved D2D output enter fit."""
    config_path = bundle / "calib-frontend.json"
    native_path = bundle / "d2d.json"
    mapping_path = bundle / "frontend.json"
    config = json.loads(config_path.read_text())
    native = json.loads(native_path.read_text())
    if native["result"]["status"] != "converged":
        raise ValueError("saved D2D initialization did not converge")
    initial = np.array(native["output_T_camera_lidar"])
    mapping = json.loads(mapping_path.read_text())
    camera = config["camera"]
    if camera["camera_model"] != "plumb_bob" or any(camera["distortion_coeffs"]):
        raise ValueError("rectified pinhole depth required")
    names = config["meta"]["bag_names"]
    if len(names) != 3 or len(set(names)) != 3:
        raise ValueError("three distinct frames required")
    files = [config_path, native_path, mapping_path]
    observations = []
    for name in names:
        if Path(name).name != name:
            raise ValueError("local frame basename required")
        depth_path, cloud_path = bundle / (name + ".depth.npy"), bundle / (name + ".ply")
        for p in [depth_path, cloud_path]:
            if digest(p) != mapping[p.name]["sha256"]:
                raise ValueError("frozen frontend array changed")
        metadata = Path(mapping[cloud_path.name]["source"]).parent / "frontend.json"
        baseline = json.loads(metadata.read_text())["stereo"]["baseline_m"]
        observations.append(
            StereoSurfaceObservation(
                name,
                np.load(depth_path, allow_pickle=False),
                read_koide_ply(cloud_path)[:, :3],
                tuple(camera["intrinsics"]),
                baseline,
            )
        )
        files.extend([depth_path, cloud_path, metadata])
    hashes = {str(p): digest(p) for p in files}
    started = time.perf_counter()
    result = refine_stereo_translation(observations, initial, options)
    elapsed = time.perf_counter() - started
    np.testing.assert_array_equal(np.array(result.output_T_camera_lidar)[:3, :3], initial[:3, :3])
    report = RefinementRun(
        scope="experimental stereo point-to-plane translation refinement; no reference input",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        driver_sha256=digest(Path(__file__)),
        solver_sha256=digest(Path("src/calibrex/solvers/stereo_translation_refiner.py")),
        input_sha256=hashes,
        initial_T_camera_lidar=initial.tolist(),
        elapsed_seconds=elapsed,
        result=result,
    )
    output.write_text(report.model_dump_json(indent=2) + "\n")
    output.with_suffix(".schema.json").write_text(
        json.dumps(RefinementRun.model_json_schema(), indent=2) + "\n"
    )
    print(result.status, elapsed, flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--sampling", choices=["uniform", "point_priority"], default="uniform")
    a = parser.parse_args()
    run(a.bundle, a.output, StereoTranslationOptions(sampling=a.sampling))
