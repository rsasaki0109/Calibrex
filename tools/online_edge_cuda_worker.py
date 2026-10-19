"""Frozen-source worker for 48 real public image/cloud pairs; performance only."""

# Imports follow the frozen source path selection.
# ruff: noqa: E402

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

VARIANT = sys.argv[1]
assert VARIANT in {"numpy", "fused", "numba", "cuda"}
OUT = Path(sys.argv[2]).resolve()
ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(sys.argv[3]).resolve()
os.environ.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
sys.path[:0] = [str(SOURCE / "src"), str(ROOT), str(ROOT / "tools")]
import numpy as np
from tools.benchmark_bspline_lidar_imu import _verify_sources

from calibrex.core.geometry import SE3, quaternion_xyzw_from_rotation_matrix
from calibrex.data.kitti import read_calibration_file, read_png_luminance
from calibrex.evaluation.camera_lidar_edge import lidar_depth_edges
from calibrex.solvers import levinson_thrun_online_solver as online
from calibrex.solvers.levinson_online_preparation import (
    LevinsonOnlineImage,
    prepare_levinson_online_frames,
)
from calibrex.solvers.native_levinson_thrun_online_solver import _block_mean


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    expected = _verify_sources(SOURCE)
    protocol = json.loads((Path(sys.argv[4]).resolve().parent / "protocol.json").read_bytes())
    for name, path in (
        ("tools/online_edge_cuda_worker.py", Path(__file__)),
        (
            "tools/benchmark_bspline_lidar_imu.py",
            Path(sys.modules[_verify_sources.__module__].__file__),
        ),
    ):
        if sha(path) != protocol["toolchain_sha256"][name]:
            raise ValueError("executed benchmark tool differs from frozen source")
    cache = Path(os.environ["CUPY_CACHE_DIR"])
    cache_empty = not cache.exists() or not any(cache.iterdir())
    if VARIANT == "cuda" and not cache_empty:
        raise ValueError("CUDA worker compilation cache must start empty")
    receipt_path = Path(sys.argv[6]).resolve()
    receipt = json.loads(receipt_path.read_bytes())
    if receipt["complete"] is not True or len(receipt["frame_indices"]) != 48:
        raise ValueError("requires complete 48 distinct pairs")
    result_folder = Path(sys.argv[4]).resolve()
    result_folder.mkdir()
    folder = OUT
    settings = online.LevinsonThrunOnlineOptions()
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": protocol["source_commit"],
        "base_git_commit": protocol["base_git_commit"],
        "source_kind": protocol["source_kind"],
        "variant": VARIANT,
        "protocol_sha256": sha(result_folder.parent / "protocol.json"),
        "source_manifest_sha256": sha(SOURCE / "manifest.json"),
        "acquisition_receipt_sha256": sha(receipt_path),
        "launcher_sha256": sha(Path(__file__)),
        "scope": "Performance diagnostic on already-spent data, no new accuracy or SOTA evaluation",
    }
    config = {
        "options": asdict(settings),
        "image_downsample": 4,
        "edge_max_radius": 32,
        "edge_gamma": 0.98**4,
        "discontinuity_min_jump_m": 0.3,
        "discontinuity_exponent": 0.5,
        "max_points_per_frame": 2000,
        "minimum_ring_fraction": 0.5,
        "frame_indices": receipt["frame_indices"],
        "extraction": (
            "Existing beam-recovered KITTI near-side discontinuities, not the A2D2 adapter"
        ),
        "initial_transform": (
            "Official rectified camera-02/Velodyne calibration; performance control only"
        ),
        "provenance": provenance,
    }
    config["preparation_workers"] = 3
    config["edge_evaluation_backend"] = VARIANT if VARIANT in {"numba", "cuda"} else "numpy"
    config["grid_evaluation_backend"] = config["edge_evaluation_backend"]
    (result_folder / "config.json").write_text(
        json.dumps(config, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    begin = time.perf_counter()
    for name, info in receipt["files"].items():
        if sha(folder / name) != info["sha256"]:
            raise ValueError("fixture changed")
    calibration = read_calibration_file(folder / "calib_cam_to_cam.txt")
    extrinsic = read_calibration_file(folder / "calib_velo_to_cam.txt")
    projection = np.asarray(calibration["P_rect_02"]).reshape(3, 4)
    intrinsic = projection[:, :3]
    rect, velo, shift = np.eye(4), np.eye(4), np.eye(4)
    rect[:3, :3] = np.asarray(calibration["R_rect_00"]).reshape(3, 3)
    velo[:3, :3] = np.asarray(extrinsic["R"]).reshape(3, 3)
    velo[:3, 3] = extrinsic["T"]
    shift[:3, 3] = np.linalg.solve(intrinsic, projection[:, 3])
    matrix = shift @ rect @ velo
    transform = SE3(
        tuple(float(v) for v in matrix[:3, 3]),
        quaternion_xyzw_from_rotation_matrix(matrix[:3, :3].ravel()),
    )
    frame_counts = {}
    count_lock = Lock()

    def load(index: int) -> LevinsonOnlineImage:
        decoded = read_png_luminance(folder / f"image_02/data/{index:010d}.png")
        if decoded is None:
            raise ValueError("unsupported PNG")
        image = _block_mean(np.asarray(decoded.rows, dtype=float), 4)
        points, weights, rings, fraction = lidar_depth_edges(
            np.fromfile(folder / f"velodyne_points/data/{index:010d}.bin", dtype="<f4").reshape(
                -1, 4
            )[:, :3],
            min_jump_m=0.3,
            exponent=0.5,
            max_points=2000,
            min_ring_fraction=0.5,
        )
        camera = online.LevinsonCameraModel(
            width=image.shape[1],
            height=image.shape[0],
            fx=float(intrinsic[0, 0]) / 4,
            fy=float(intrinsic[1, 1]) / 4,
            cx=float(intrinsic[0, 2]) / 4,
            cy=float(intrinsic[1, 2]) / 4,
        )
        frame_id = f"kitti-{index:010d}"
        with count_lock:
            frame_counts[frame_id] = {
                "frame_id": frame_id,
                "points": len(points),
                "rings": rings,
                "fraction_on_ring": fraction,
            }
        return LevinsonOnlineImage(
            frame_id,
            image,
            np.asarray(points, dtype=float),
            np.asarray(weights, dtype=float),
            camera,
        )

    frames = prepare_levinson_online_frames(
        receipt["frame_indices"],
        load,
        workers=3,
        evaluation_backend=config["edge_evaluation_backend"],
        gamma=0.98**4,
        max_radius=32,
    )
    counts = [frame_counts[frame.frame_id] for frame in frames]

    prepared = time.perf_counter()
    calls, point_rows = 0, 0
    native = online.evaluate_levinson_objective

    def evaluate(items, pose):
        nonlocal calls, point_rows
        calls += 1
        point_rows += sum(len(frame.lidar_points) for frame in items)
        return native(items, pose)

    if VARIANT == "fused":
        from online_edge_fused_control import NativeRawGridControl, install

        if (
            sha(Path(sys.modules[install.__module__].__file__))
            != protocol["toolchain_sha256"]["tools/online_edge_fused_control.py"]
        ):
            raise ValueError("compiled CPU control source differs")
        install(online)
    online.evaluate_levinson_objective = evaluate
    solver = online.LevinsonThrunOnlineSolver(evaluation_backend=config["grid_evaluation_backend"])
    raw_grid_evaluations = 0
    if VARIANT == "fused":
        control = NativeRawGridControl(online)
        result = solver._solve(frames, transform, settings, control)
        calls += control.pose_evaluations
        point_rows += control.point_rows
        raw_grid_evaluations = control.pose_evaluations
    else:
        result = solver.solve(frames, transform, settings)
    if VARIANT == "numba":
        execution = result.execution
        assert isinstance(execution, online.OnlineCpuExecution)
        calls, point_rows = execution.objective_evaluations, execution.point_rows
    (result_folder / "result.json").write_text(
        json.dumps(
            {"result": result.as_dict(), "provenance": provenance}, indent=2, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    elapsed = time.perf_counter() - begin
    entries = []
    record = {
        "scope": (
            "Complete fresh-worker record; compare parent time, including imports and validation"
        ),
        "elapsed_seconds": elapsed,
        "preparation_seconds": prepared - begin,
        "solve_and_save_seconds": elapsed - (prepared - begin),
        "frame_points": counts,
        "evaluations": calls,
        "native_raw_grid_pose_evaluations": raw_grid_evaluations,
        "point_rows": point_rows,
        "timeline_steps": len(result.timeline),
        "functions": entries,
        "provenance": provenance,
    }
    capture = hashlib.sha256()
    for frame in frames:
        capture.update(repr((frame.frame_id, frame.camera)).encode("utf-8"))
        for array in (frame.lidar_points, frame.discontinuity_weights, frame.edge_response):
            capture.update(repr((array.shape, array.dtype.str)).encode())
            capture.update(array.tobytes(order="C"))
    record["prepared_capture_sha256"] = capture.hexdigest()
    record["software"] = {"python": sys.version, "numpy": np.__version__}
    record["cuda_cache_empty"] = cache_empty if VARIANT == "cuda" else None
    record["cuda_driver_cache_disabled"] = (
        os.environ["CUDA_CACHE_DISABLE"] == "1" if VARIANT == "cuda" else None
    )
    if VARIANT in {"fused", "numba"}:
        import numba

        record["software"]["numba"] = numba.__version__
    record["result_sha256"] = sha(result_folder / "result.json")
    record["config_sha256"] = sha(result_folder / "config.json")
    record["imported_source_sha256"] = {}
    for name, module in list(sys.modules.items()):
        if (name == "calibrex" or name.startswith("calibrex.")) and getattr(
            module, "__file__", None
        ):
            path = Path(module.__file__).resolve()
            assert path.is_relative_to(SOURCE / "src")
            relative = path.relative_to(SOURCE / "src").as_posix()
            assert sha(path) == expected[relative]
            record["imported_source_sha256"][relative] = sha(path)
    record["image_kernel_sha256"] = None
    if VARIANT == "cuda":
        from calibrex.solvers.cuda_online_edge_adapter import IMAGE_KERNEL_SHA256

        record["image_kernel_sha256"] = IMAGE_KERNEL_SHA256
    elif VARIANT == "numba":
        from calibrex.solvers.numba_online_edge_adapter import IMAGE_KERNEL_SHA256

        record["image_kernel_sha256"] = IMAGE_KERNEL_SHA256
    import jsonschema

    for artifact in ("result", "config"):
        jsonschema.validate(
            json.loads((result_folder / (artifact + ".json")).read_bytes()),
            json.loads(
                (result_folder.parent / "schemas" / (artifact + ".schema.json")).read_bytes()
            ),
        )
    (result_folder / "measurement.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    _verify_sources(SOURCE)
    print(
        json.dumps(
            {
                k: v
                for k, v in record.items()
                if k not in ("functions", "frame_points", "provenance")
            }
        )
    )
    print(json.dumps(entries[:15]))


if __name__ == "__main__":
    main()
