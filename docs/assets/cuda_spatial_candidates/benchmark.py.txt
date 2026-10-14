"""Compare exact GPU query candidates on a captured public GLIM registration.

This measures warmed query kernels, not complete calibration acceleration.
Runtime adapters are unchanged; candidate shaders and inputs remain in output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import tarfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from calibrex.solvers.scipy_spatial_snapshot import capture_spatial_tree

FloatArray = NDArray[np.float64]


def _heap_source(source: str) -> str:
    """Replace sorted insertion by an exact max heap and final ascending sort."""
    insertion = """                if(d<best[k-1] || (d==best[k-1] && position<chosen[k-1])){
                    int slot=k-1;
                    while(slot>0 && (d<best[slot-1] ||
                         (d==best[slot-1] && position<chosen[slot-1]))){
                        best[slot]=best[slot-1];chosen[slot]=chosen[slot-1];--slot;
                    }
                    best[slot]=d;chosen[slot]=position;
                }"""
    replacement = """                if(d<best[0] || (d==best[0] && position<chosen[0])){
                    int slot=0;
                    while(2*slot+1<k){
                        int child=2*slot+1;
                        if(child+1<k && (best[child+1]>best[child] ||
                           (best[child+1]==best[child] && chosen[child+1]>chosen[child]))) ++child;
                        if(best[child]<d || (best[child]==d && chosen[child]<=position)) break;
                        best[slot]=best[child];chosen[slot]=chosen[child];slot=child;
                    }
                    best[slot]=d;chosen[slot]=position;
                }"""
    output = "    for(int j=0;j<k;j++){distance[row*k+j]=sqrt(best[j]);index[row*k+j]=chosen[j];}"
    sort = """    for(int end=k-1;end>0;--end){
        double d=best[end];long long position=chosen[end];
        best[end]=best[0];chosen[end]=chosen[0];
        int slot=0;
        while(2*slot+1<end){
            int child=2*slot+1;
            if(child+1<end && (best[child+1]>best[child] ||
               (best[child+1]==best[child] && chosen[child+1]>chosen[child]))) ++child;
            if(best[child]<d || (best[child]==d && chosen[child]<=position)) break;
            best[slot]=best[child];chosen[slot]=chosen[child];slot=child;
        }
        best[slot]=d;chosen[slot]=position;
    }
"""
    if source.count(insertion) != 1 or source.count(output) != 1:
        raise ValueError("query shader changed; candidate requires review")
    return source.replace(insertion, replacement).replace("best[k-1]", "best[0]").replace(
        output, sort + output,
    )


def _capture(bag: Path, output: Path) -> tuple[FloatArray, FloatArray, FloatArray]:
    from calibrex.data.livox_ros2 import LivoxStreamProfile
    from calibrex.evaluation import lidar_imu_bag_calibration as runner
    from calibrex.solvers.cuda_scan_odometry_adapter import CudaOdometryTarget

    class Captured(BaseException):
        pass

    calls = 0
    captured: tuple[FloatArray, FloatArray, FloatArray] | None = None
    original = CudaOdometryTarget.register

    def capture(target: Any, source: FloatArray, initial: FloatArray) -> Any:
        nonlocal calls, captured
        calls += 1
        if calls == 6:
            captured = (target.host.copy(), source.copy(), initial.copy())
            raise Captured()
        return original(target, source, initial)

    options = runner.LidarImuBagOptions(max_rounds=4)
    window = options.rotation.windowing
    options = replace(options, rotation=replace(options.rotation, windowing=replace(
        window, odometry_backend="cuda", odometry=replace(
            window.odometry, voxel_size_m=.03, local_map_scans=3, geometry_quantum_m=1e-9,
            motion_translation_quantum_m=1e-6, motion_rotation_quantum_rad=1e-7,
        ),
    )))
    CudaOdometryTarget.register = capture
    try:
        runner.calibrate_lidar_imu_bag(
            bag, LivoxStreamProfile("public-glim", "/livox/points", "/livox/imu", "t",
                                    "offset_s", "g"),
            options, dataset_family="glim_versatile",
            dataset_license="upstream AIST GLIM dataset terms", max_seconds=6,
            command=["python", "-m", "tools.benchmark_cuda_spatial_candidates"],
        )
    except Captured:
        pass
    finally:
        CudaOdometryTarget.register = original
    if captured is None:
        raise ValueError("public workload did not reach its sixth registration")
    target, source, initial = captured
    np.savez(output / "inputs.npz", target=target, source=source, initial=initial)
    return captured


def _measure(
    target: FloatArray, moving: FloatArray, shaders: dict[str, str], leafsizes: list[int],
    batches: int, launches: int,
) -> list[dict[str, Any]]:
    import cupy as cp

    reference = cp.asarray(target)
    queries = {2: cp.asarray(moving), 11: reference, 17: reference}
    expected: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    records = []
    for leafsize in leafsizes:
        tree = cKDTree(target, leafsize=leafsize, copy_data=True)
        (axes, splits, children, spans), boxes, _ = capture_spatial_tree(tree)
        arrays = (cp.asarray(tree.indices, dtype=cp.int64), cp.asarray(axes), cp.asarray(splits),
                  cp.asarray(children), cp.asarray(spans), cp.asarray(boxes))
        for k, points in queries.items():
            for method, shader in shaders.items():
                kernel = cp.RawKernel(shader, "query3", options=("--fmad=false",))
                distances = cp.empty((len(points), k), dtype=cp.float64)
                indexes = cp.empty((len(points), k), dtype=cp.int64)
                args = (points, reference, *arrays, np.int32(len(points)), np.int32(len(target)),
                        np.int32(k), distances, indexes)
                grid = ((len(points) + 127) // 128,)
                kernel(grid, (128,), args)
                cp.cuda.get_current_stream().synchronize()
                actual = cp.asnumpy(distances), cp.asnumpy(indexes)
                if k not in expected:
                    expected[k] = actual
                for values, baseline in zip(actual, expected[k], strict=True):
                    np.testing.assert_array_equal(values, baseline)
                seconds = []
                for _ in range(batches):
                    start, end = cp.cuda.Event(), cp.cuda.Event()
                    start.record()
                    for _ in range(launches):
                        kernel(grid, (128,), args)
                    end.record()
                    end.synchronize()
                    seconds.append(float(cp.cuda.get_elapsed_time(start, end)) / (1000 * launches))
                record = {"leafsize": leafsize, "neighbors": k, "method": method,
                          "query_rows": len(points), "batch_seconds_per_launch": seconds,
                          "kernel_median_seconds": statistics.median(seconds),
                          "exact_distance_and_index_parity": True}
                records.append(record)
                print(json.dumps(record), flush=True)
    return records


def validate_measurement(record: dict[str, Any]) -> None:
    """Validate the tool schema, full candidate coverage and reported medians."""
    import jsonschema

    schema = Path(__file__).with_name("schemas") / "cuda_spatial_candidates.schema.json"
    jsonschema.validate(record, json.loads(schema.read_bytes()))
    keys = [(r["leafsize"], r["neighbors"], r["method"]) for r in record["records"]]
    expected = {(leaf, k, method) for leaf in (16, 8, 4) for k in (2, 11, 17)
                for method in ("sorted", "heap")}
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError("candidate coverage is incomplete or duplicated")
    rows: dict[int, int] = {}
    for item in record["records"]:
        samples = item["batch_seconds_per_launch"]
        if (len(samples) != record["workload"]["batches"]
                or statistics.median(samples) != item["kernel_median_seconds"]):
            raise ValueError("candidate timing median or batch count differs")
        count = item["query_rows"]
        if count != rows.setdefault(item["neighbors"], count):
            raise ValueError("candidate query counts differ")
    if rows[11] != rows[17]:
        raise ValueError("normal query counts differ")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, default=Path("data/public/glim_versatile/livox.bag"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--launches", type=int, default=10)
    args = parser.parse_args()
    if args.batches < 1 or args.launches < 1:
        parser.error("batches and launches must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    generator_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source_root = Path(__file__).resolve().parents[1] / "src"
    manifest = {p.relative_to(source_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (source_root / "calibrex").rglob("*.py")}
    from calibrex.solvers.cuda_lidar_map_adapter import _SOURCE
    from calibrex.solvers.cuda_scan_odometry_adapter import execution_identity

    target, source, initial = _capture(args.bag, args.output)
    shaders = {"sorted": _SOURCE, "heap": _heap_source(_SOURCE)}
    for name, shader in shaders.items():
        (args.output / f"{name}.cu.txt").write_bytes(shader.encode())
    moving = source @ initial[:3, :3].T + initial[:3, 3]
    records = _measure(target, moving, shaders, [16, 8, 4], args.batches, args.launches)
    executed = {}
    for name, module in list(sys.modules.items()):
        if not (name.startswith("calibrex.") and getattr(module, "__file__", None)
                and Path(module.__file__).suffix == ".py"):
            continue
        path = Path(module.__file__).resolve()
        relative = path.relative_to(source_root).as_posix()
        executed[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        if executed[relative] != manifest[relative]:
            raise RuntimeError("executed source changed during measurement")
    if generator_digest != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise RuntimeError("benchmark generator changed during measurement")
    result = {
        "schema_version": "slac.cuda_spatial_candidates/v0.1", "records": records,
        "scope": "warmed GPU query kernels; not complete calibration",
        "workload": {"capture_registration": 6, "max_seconds": 6, "voxel_size_m": .03,
                     "local_map_scans": 3, "geometry_quantum_m": 1e-9,
                     "motion_translation_quantum_m": 1e-6, "motion_rotation_quantum_rad": 1e-7,
                     "batches": args.batches, "launches_per_batch": args.launches},
        "provenance": {
            "generator_sha256": generator_digest,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "dataset_family": "glim_versatile",
            "dataset_license": "upstream AIST GLIM dataset terms",
            "bag_sha256": hashlib.sha256(args.bag.read_bytes()).hexdigest(),
            "inputs_sha256": hashlib.sha256((args.output / "inputs.npz").read_bytes()).hexdigest(),
            "executed_sources": executed, "baseline_execution_identity": execution_identity(),
            "shader_sha256": {name: hashlib.sha256(shader.encode()).hexdigest()
                              for name, shader in shaders.items()},
        },
    }
    validate_measurement(result)
    (args.output / "benchmark.py.txt").write_bytes(Path(__file__).read_bytes())
    schema = Path(__file__).with_name("schemas") / "cuda_spatial_candidates.schema.json"
    (args.output / "measurement.schema.json").write_bytes(schema.read_bytes())
    imported = {relative: manifest[relative] for relative in sorted(executed)}
    (args.output / "sources-manifest.json").write_text(json.dumps(imported, indent=2) + "\n",
                                                       encoding="utf-8")
    with tarfile.open(args.output / "sources.tar.gz", "w:gz") as bundle:
        for relative in imported:
            path = source_root / relative
            if hashlib.sha256(path.read_bytes()).hexdigest() != imported[relative]:
                raise RuntimeError("source changed before archival")
            bundle.add(path, arcname="src/" + relative, recursive=False)
        bundle.add(source_root.parent / "LICENSE", arcname="LICENSE", recursive=False)
    (args.output / "measurement.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8",
    )


if __name__ == "__main__":
    main()
