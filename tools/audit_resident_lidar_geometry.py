"""Recompute the resident LiDAR geometry publication using CPU-only evidence."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import statistics
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

import jsonschema
from tools.benchmark_bspline_lidar_imu import (
    _compare,
    _same_float_bits,
    _validate_backend_execution,
    _validate_plane_execution,
    _validate_plane_fit_pair,
    _validate_resident_execution,
    _validate_resident_pair,
)

from calibrex.core.lidar_imu_bag_calibration import LidarImuBagCalibrationArtifact

_VARIANTS = ("cpu", "native", "resident")
_COMMITS = {
    "baseline": "ae5a55822485b4ee9daf2b0195aa35bd4c5b7484",
    "current": "5fda3a1f29a33289828798a04cd4236af9d5b4b7",
}
_BAG_SHA256 = "f57a2e85339316d4235ab693ab2876aac5e113625c7f8b0f7873b62f92e09643"
_ORDER = [
    ["cpu", "native", "resident"],
    ["resident", "cpu", "native"],
    ["native", "resident", "cpu"],
]
_OPTIONS = {
    "max_seconds": 6.0,
    "max_rounds": 4,
    "voxel_size_m": 0.03,
    "local_map_scans": 3,
    "geometry_quantum_m": 1e-9,
    "motion_translation_quantum_m": 1e-6,
    "motion_rotation_quantum_rad": 1e-7,
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read(root: Path, name: str) -> Any:
    return json.loads((root / name).read_bytes())


def _sources(root: Path, prefix: str) -> dict[str, bytes]:
    expected = _read(root, prefix + ".sources-manifest.json")
    sources = {}
    total = 0
    with tarfile.open(root / (prefix + ".sources.tar.gz"), "r:gz") as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            total += member.size
            if (
                not member.isfile()
                or name.is_absolute()
                or ".." in name.parts
                or not member.name.startswith("src/calibrex/")
                or name.suffix != ".py"
                or member.name.removeprefix("src/") in sources
                or total > 32 * 1024**2
            ):
                raise ValueError("unexpected source capsule member")
            stream = archive.extractfile(member)
            assert stream is not None
            sources[member.name.removeprefix("src/")] = stream.read()
    if {key: _sha(value) for key, value in sources.items()} != expected:
        raise ValueError("source capsule differs from its complete manifest")
    return sources


def _constants(raw: bytes) -> dict[str, str]:
    values = {}
    for node in ast.parse(raw).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value.value, str):
                    values[target.id] = node.value.value
    return values


def _kernel_hashes(sources: dict[str, bytes], *, resident: bool) -> dict[str, str]:
    system = _constants(sources["calibrex/solvers/cuda_scan_odometry_adapter.py"])
    if resident:
        extra = _constants(sources["calibrex/solvers/_cuda_resident_kernels.py"])
        sources_to_hash = {
            "query_sha256": extra["_GRID_SOURCE"],
            "system_sha256": system["_SYSTEM_SOURCE"]
            + extra["_FUSED_GRID_SOURCE"]
            + system["_COVARIANCE_SOURCE"]
            + system["_NORMAL_SOURCE"],
            "geometry_sha256": extra["_VOXEL_SOURCE"]
            + extra["_GYRO_SOURCE"]
            + extra["_GEOMETRY_SOURCE"],
        }
    else:
        query = _constants(sources["calibrex/solvers/cuda_lidar_map_adapter.py"])["_SOURCE"]
        sources_to_hash = {
            "query_sha256": query,
            "system_sha256": "".join(
                system[name]
                for name in (
                    "_SYSTEM_SOURCE",
                    "_FUSED_SOURCE",
                    "_COVARIANCE_SOURCE",
                    "_NORMAL_SOURCE",
                )
            ),
        }
    return {key: _sha(value.encode()) for key, value in sources_to_hash.items()}


def _bind_worker(
    root: Path,
    name: str,
    sources: dict[str, bytes],
    protocol: dict[str, Any],
    benchmark_schema: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], float, list[dict[str, Any]]]:
    result = _read(root, name + "/result.json")
    LidarImuBagCalibrationArtifact.model_validate(result)
    measurement = _read(root, name + "/measurement.json")
    jsonschema.validate(measurement, benchmark_schema)
    provenance = measurement["provenance"]
    if provenance["result_sha256"] != _sha((root / name / "result.json").read_bytes()):
        raise ValueError("worker result binding differs")
    variant = name.split("-", 1)[1]
    source_kind = "current" if variant == "resident" else "baseline"
    if (
        provenance["source_manifest_sha256"]
        != protocol["provenance"][source_kind + "_manifest_sha256"]
        or provenance["launcher_sha256"] != protocol["provenance"]["benchmark_sha256"]
        or provenance["bag_sha256"] != protocol["bag"]["sha256"]
        or not provenance["executed_sources"]
        or any(_sha(sources[key]) != value for key, value in provenance["executed_sources"].items())
        or provenance["blas_threads"]
        != {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
        or provenance.get("cuda_cache_initially_empty") is not True
        or result["provenance"]["git_commit"] != protocol["commits"][source_kind]
    ):
        raise ValueError("worker source/input/thread/cache provenance differs")
    backend = "cpu" if variant == "cpu" else "cuda"
    _validate_backend_execution(result, backend)
    for key in _OPTIONS:
        if key in ("max_seconds", "max_rounds"):
            if result["options"][key] != _OPTIONS[key]:
                raise ValueError("worker calibration options differ")
        elif result["options"]["odometry_options"][key] != _OPTIONS[key]:
            raise ValueError("worker odometry density/geometry options differ")
    if (
        len(result["options"]["initial_odometry_execution"]) != 7
        or len(result["options"]["segment_odometry_execution"]) != 1
        or len(result["rounds"]) != 4
        or len(measurement["rounds"]) != 4
        or len(result["controls"]) != 32
        or measurement["control_evaluation_calls"] != 1
        or measurement["control_score_calls"] != 34
    ):
        raise ValueError("worker omitted odometry passes, complete fits or controls")
    for fitted, reported in zip(measurement["rounds"], result["rounds"], strict=True):
        if (
            fitted["status"] != reported["fit_status"]
            or fitted["iterations"] != reported["iterations"]
        ):
            raise ValueError("complete fit and saved bag evidence disagree")
    plane = "numpy" if variant == "cpu" else "cuda"
    plane_hash = None
    if plane == "cuda":
        plane_hash = _sha(
            _constants(sources["calibrex/solvers/cuda_bspline_plane_adapter.py"])[
                "_SOURCE"
            ].encode()
        )
    _validate_plane_execution(result, measurement["rounds"], plane, plane_hash)
    _validate_resident_execution(
        result,
        variant == "resident",
        _kernel_hashes(sources, resident=True) if variant == "resident" else None,
    )
    whole = _read(root, name + "/whole.json")
    if whole["exit_code"] != 0 or whole["seconds"] <= 0:
        raise ValueError("worker did not complete a positive whole-process measurement")
    if whole["provenance"]["protocol_sha256"] != _sha((root / "protocol.json").read_bytes()):
        raise ValueError("worker protocol binding differs")
    for stream in ("stdout", "stderr"):
        if whole["provenance"][stream + "_sha256"] != _sha(
            (root / name / (stream + ".log")).read_bytes()
        ):
            raise ValueError("worker log binding differs")
    placement = _read(root, name + "/placement.json")
    if (
        placement["provenance"]["launcher_sha256"] != protocol["provenance"]["launcher_sha256"]
        or placement["provenance"]["source_manifest_sha256"] != provenance["source_manifest_sha256"]
    ):
        raise ValueError("point placement binding differs")
    return result, measurement, float(whole["seconds"]), placement["calls"]


def audit_archive(root: Path) -> dict[str, Any]:
    """Validate every bound worker and independently recompute the reported ratios."""
    root = root.resolve()
    index_raw = (root / "archive-index.json").read_bytes()
    index = json.loads(index_raw)
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix() not in ("archive-index.json", "audit.json")
    }
    if set(index["files"]) != actual_files:
        raise ValueError("archive index does not cover every published payload")
    for name, digest in index["files"].items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or ":" in name:
            raise ValueError("archive index path escapes the publication")
        if _sha((root / name).read_bytes()) != digest:
            raise ValueError("published asset checksum differs: " + name)
    protocol_raw = (root / "protocol.json").read_bytes()
    protocol = json.loads(protocol_raw)
    if (
        protocol["commits"] != _COMMITS
        or protocol["bag"]["sha256"] != _BAG_SHA256
        or protocol["bag"]["bytes"] != 613350674
        or index["schema_version"] != "calibrex.resident_geometry_archive/v0.1"
        or index["provenance"]
        != {
            "protocol_sha256": _sha(protocol_raw),
            "commits": _COMMITS,
        }
    ):
        raise ValueError("publication source commits or public input binding differs")
    if protocol["launch_order"] != _ORDER or protocol["options"] != _OPTIONS:
        raise ValueError("publication did not use the declared fresh balanced cohort")
    if protocol["gates"] != {
        "all_original_physical_evidence_and_decisions": True,
        "all_full_fits_rtol": 1e-8,
        "all_full_fits_atol": 1e-9,
        "native_resident_all_float_bits_equal_after_declared_execution_metadata": True,
        "native_resident_all_placement_bits_equal": True,
        "median_cpu_to_resident_minimum": 2.0,
        "median_native_to_resident_minimum_exclusive": 1.0,
    }:
        raise ValueError("predeclared gates differ")
    for name, record in (
        ("launcher.py.txt", "launcher_sha256"),
        ("benchmark.py.txt", "benchmark_sha256"),
        ("benchmark.schema.json", "schema_sha256"),
    ):
        if _sha((root / name).read_bytes()) != protocol["provenance"][record]:
            raise ValueError("frozen launcher/schema binding differs")
    sources = {key: _sources(root, key) for key in ("baseline", "current", "correctness")}
    for key in ("baseline", "current"):
        if (
            _sha((root / (key + ".sources-manifest.json")).read_bytes())
            != protocol["provenance"][key + "_manifest_sha256"]
        ):
            raise ValueError("source manifest is not bound to the executed protocol")

    def normalize(values: dict[str, bytes]) -> dict[str, bytes]:
        return {key: value.replace(b"\r\n", b"\n") for key, value in values.items()}

    if normalize(sources["correctness"]) != normalize(sources["current"]):
        raise ValueError("live correctness runtime differs from measured runtime")
    for kind in ("gpu", "host"):
        base = "correctness/" + kind
        summary = _read(root, base + "/summary.json")
        provenance_raw = (root / base / "provenance.json").read_bytes()
        provenance = json.loads(provenance_raw)
        test_log = (root / base / "pytest.log").read_text(encoding="utf-8")
        if ("47 passed" if kind == "gpu" else "92 passed") not in test_log:
            raise ValueError("live regression suite did not complete the expected tests")
        expected_sources = {
            "src/" + key: _sha(value) for key, value in sources["correctness"].items()
        }
        if (
            summary["exit_code"] != 0
            or provenance["sources"] != expected_sources
            or summary["provenance_sha256"] != _sha(provenance_raw)
            or summary["pytest_log_sha256"] != _sha((root / base / "pytest.log").read_bytes())
            or provenance["launcher_sha256"] != _sha((root / base / "launcher.py.txt").read_bytes())
        ):
            raise ValueError("live regression tests do not bind the measured runtime")
        for name, digest in provenance["tests"].items():
            if _sha((root / base / (Path(name).name + ".txt")).read_bytes()) != digest:
                raise ValueError("live regression test snapshot differs")
    benchmark_schema = _read(root, "benchmark.schema.json")
    rows = []
    placement_rows = 0
    for repeat in range(3):
        workers = {
            variant: _bind_worker(
                root,
                f"{repeat:02d}-{variant}",
                sources["current" if variant == "resident" else "baseline"],
                protocol,
                benchmark_schema,
            )
            for variant in _VARIANTS
        }
        for variant in ("native", "resident"):
            if not _compare(workers["cpu"][0], workers[variant][0])["passed"]:
                raise ValueError("CPU/GPU complete physics or evidence comparison failed")
            _validate_plane_fit_pair(workers["cpu"][1]["rounds"], workers[variant][1]["rounds"])
        hashes = _kernel_hashes(sources["baseline"], resident=False)
        _validate_resident_pair(
            workers["native"][0],
            workers["resident"][0],
            workers["native"][1]["rounds"],
            workers["resident"][1]["rounds"],
            (hashes["query_sha256"], hashes["system_sha256"]),
            _kernel_hashes(sources["current"], resident=True),
        )
        placements = [
            [
                {key: value for key, value in row.items() if key != "seconds"}
                for row in workers[v][3]
            ]
            for v in ("native", "resident")
        ]
        if not _same_float_bits(*placements):
            raise ValueError("complete GPU point placement bits differ")
        placement_rows += sum(row["rows"] for row in placements[0])
        row = {variant: workers[variant][2] for variant in _VARIANTS}
        declared = _read(root, f"triad-{repeat:02d}.json")
        if (
            declared["seconds"] != row
            or not declared["quality_passed"]
            or declared["errors"]
            or not declared["gpu_complete_result_and_full_fit_bits_equal"]
            or not declared["gpu_placement_bits_equal"]
        ):
            raise ValueError("declared triad differs from complete worker evidence")
        expected_records = {
            f"{repeat:02d}-{variant}/{path.name}": _sha(path.read_bytes())
            for variant in _VARIANTS
            for path in sorted((root / f"{repeat:02d}-{variant}").glob("*.json"))
        }
        declared_records = declared["provenance"]["worker_records"]
        portable_records = {
            name.replace("\\", "/"): digest for name, digest in declared_records.items()
        }
        if (
            len(portable_records) != len(declared_records)
            or declared["provenance"]["protocol_sha256"] != _sha(protocol_raw)
            or portable_records != expected_records
        ):
            raise ValueError("triad worker provenance differs")
        rows.append(row)
    medians = {variant: statistics.median(row[variant] for row in rows) for variant in _VARIANTS}
    ratios = (medians["cpu"] / medians["resident"], medians["native"] / medians["resident"])
    declared = _read(root, "summary.json")
    if declared["provenance"] != {
        "protocol_sha256": _sha(protocol_raw),
        "triads": {
            f"triad-{repeat:02d}.json": _sha((root / f"triad-{repeat:02d}.json").read_bytes())
            for repeat in range(3)
        },
    }:
        raise ValueError("summary cohort provenance differs")
    if (
        declared["medians"] != medians
        or declared["cpu_to_resident"] != ratios[0]
        or declared["native_to_resident"] != ratios[1]
        or not declared["quality_passed"]
        or not declared["production_runtime_measured"]
        or declared["advance"] != (ratios[0] >= 2 and ratios[1] > 1)
    ):
        raise ValueError("declared cumulative/incremental ratios or gates differ")
    if "cupy" in sys.modules:
        raise ValueError("archive audit unexpectedly imported a GPU runtime")
    report = {
        "schema_version": "calibrex.resident_geometry_audit/v0.1",
        "status": "confirmed" if declared["advance"] else "not_confirmed",
        "fresh_triads": 3,
        "complete_workers": 9,
        "complete_fits": 36,
        "cpu_gpu_fit_comparisons": 24,
        "gpu_bit_exact_fit_comparisons": 12,
        "gpu_bit_exact_point_rows": placement_rows,
        "medians": medians,
        "measurements_seconds": rows,
        "ranges_seconds": {
            variant: [min(row[variant] for row in rows), max(row[variant] for row in rows)]
            for variant in _VARIANTS
        },
        "cpu_improved_runs": sum(row["cpu"] > row["resident"] for row in rows),
        "native_improved_runs": sum(row["native"] > row["resident"] for row in rows),
        "cpu_to_resident": ratios[0],
        "native_to_resident": ratios[1],
        "quality_passed": True,
        "gpu_runtime_imported": False,
        "provenance": {
            "archive_index_sha256": _sha(index_raw),
            "protocol_sha256": _sha(protocol_raw),
            "auditor_sha256": _sha(Path(__file__).read_bytes()),
            "commits": protocol["commits"],
        },
    }
    schema = Path(__file__).with_name("schemas") / "resident_lidar_geometry_audit.schema.json"
    jsonschema.validate(report, json.loads(schema.read_bytes()))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--check", action="store_true", help="also verify the saved audit report")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report = audit_archive(args.archive)
    if args.check and report != _read(args.archive, "audit.json"):
        raise ValueError("saved audit report differs from recomputed evidence")
    if args.write_report:
        (args.archive / "audit.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
