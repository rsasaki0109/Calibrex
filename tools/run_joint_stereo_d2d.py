"""Run the existing six-DoF D2D solver jointly on multiple motion-aligned stereo frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from tools.koide_shared_bundle_pandey import SharedBundleRun, read_koide_ply

from calibrex.core.geometry import SE3
from calibrex.solvers.borer_depth_to_depth_solver import (
    DepthToDepthCameraModel,
    DepthToDepthObservation,
)
from calibrex.solvers.borer_six_dof_solver import BorerSixDofOptions, BorerSixDofSolver


def run(bundle: Path, seed: Path, output: Path) -> SharedBundleRun:
    """Freeze default search settings and use the same float32 PLY as Koide."""
    initial = np.array(json.loads(seed.read_text())["initial_T_camera_lidar"])
    transform = SE3(tuple(initial[:3, 3]), tuple(Rotation.from_matrix(initial[:3, :3]).as_quat()))
    config_path = bundle / "calib.json"
    config = json.loads(config_path.read_text())
    names = config["meta"]["bag_names"]
    if len(names) != 3 or len(set(names)) != 3:
        raise ValueError("the frozen joint experiment requires three distinct frames")
    camera = config["camera"]
    files = [seed, config_path, bundle / "frontend.json"]
    # Capture input identity before execution; Koide later updates calib.json.
    observations = []
    for name in names:
        if Path(name).name != name:
            raise ValueError("frame names must be local basenames")
        depth_path, cloud_path = bundle / (name + ".depth.npy"), bundle / (name + ".ply")
        depth = np.load(depth_path, allow_pickle=False)
        points = read_koide_ply(cloud_path)[:, :3]
        model = DepthToDepthCameraModel(
            width=depth.shape[1],
            height=depth.shape[0],
            **dict(zip(("fx", "fy", "cx", "cy"), camera["intrinsics"], strict=True)),
        )
        observations.append(DepthToDepthObservation(name, depth, points, model))
        files.extend([depth_path, cloud_path])
    input_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    options = BorerSixDofOptions()
    started = time.perf_counter()
    result = BorerSixDofSolver().solve(observations, transform, options)
    elapsed = time.perf_counter() - started
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(
        result.transform_camera_lidar.rotation_quat_xyzw
    ).as_matrix()
    matrix[:3, 3] = result.transform_camera_lidar.translation_m
    settings = asdict(options)
    settings.pop("projector")
    report = SharedBundleRun(
        scope="stereo-assisted three-frame joint development; not monocular comparison",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256=input_hashes,
        options=settings,
        initial_T_camera_lidar=initial.tolist(),
        output_T_camera_lidar=matrix.tolist(),
        elapsed_seconds=elapsed,
        result=result.as_dict(),
    )
    output.write_text(report.model_dump_json(indent=2) + "\n")
    output.with_suffix(".schema.json").write_text(
        json.dumps(SharedBundleRun.model_json_schema(), indent=2) + "\n"
    )
    print(result.status, elapsed, flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["bundle", "seed", "output"]:
        parser.add_argument(name, type=Path)
    a = parser.parse_args()
    run(a.bundle, a.seed, a.output)


if __name__ == "__main__":
    main()
