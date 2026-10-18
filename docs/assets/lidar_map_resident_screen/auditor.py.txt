"""CPU-only audit of the rejected LiDAR-map resident-odometry integration."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
import sys
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

import jsonschema
from tools.audit_resident_lidar_geometry import _kernel_hashes, _sources
from tools.benchmark_bspline_lidar_imu import _same_float_bits, _without_run_metadata
from tools.benchmark_cuda_lidar_map import _compare

from calibrex.core.lidar_lidar_extrinsic import LidarLidarExtrinsicArtifact

_COMMITS = {
    "prior": "0a53989f9275795f6ca16381731a5aee1cbfa3e3",
    "current": "fe37dd0c37d6dd82b70087fc0dd21382b310b70e",
}
_VARIANTS = ("cpu", "native", "resident")
_ORDERS = [
    ["cpu", "native", "resident"],
    ["resident", "cpu", "native"],
    ["native", "resident", "cpu"],
]
_CHANGED = {
    "calibrex/cli/main.py",
    "calibrex/evaluation/lidar_lidar_map.py",
    "calibrex/solvers/scan_to_scan_odometry.py",
}
_WORKLOAD = {
    "seconds": 6.0,
    "map_registration_backend": "cpu",
    "sample_every_scans": 5,
    "map_half_window_scans": 1,
    "block_duration_s": 1.0,
    "jackknife_groups": 4,
    "odometry_voxel_m": 0.03,
    "odometry_local_map_scans": 3,
    "geometry_quantum_m": 1e-9,
    "translation_quantum_m": 1e-6,
    "rotation_quantum_rad": 1e-7,
    "initial_translation_m": [0.08, -0.07, 0.05],
    "copied_topic": "/livox/points",
    "target_noise_sigma_m": 0.01,
    "target_noise_seed": 20261018,
}
_GATES = {
    "all_original_map_tolerances": True,
    "all_gpu_fit_result_sample_bits": True,
    "all_five_fits": True,
    "all_four_jackknives_and_six_controls": True,
    "median_cpu_to_resident_minimum": 2.0,
    "median_native_to_resident_minimum_exclusive": 1.0,
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read(root: Path, name: str) -> Any:
    return json.loads((root / name).read_bytes())


def _source_trees(root: Path, index: dict[str, Any]) -> dict[str, dict[str, bytes]]:
    dependency = root.parent / "resident_scan_validation"
    if set(index["provenance"]["source_dependency"]) != {
        "current.sources-manifest.json",
        "current.sources.tar.gz",
    }:
        raise ValueError("source dependency coverage differs")
    for name, digest in index["provenance"]["source_dependency"].items():
        if name not in ("current.sources-manifest.json", "current.sources.tar.gz"):
            raise ValueError("unexpected source dependency")
        if _sha((dependency / name).read_bytes()) != digest:
            raise ValueError("already published source dependency differs")
    prior = _sources(dependency, "current")
    current = dict(prior)
    changed = set()
    total = 0
    with tarfile.open(root / "candidate.source-delta.tar.gz", "r:gz") as bundle:
        for member in bundle.getmembers():
            path = PurePosixPath(member.name)
            total += member.size
            key = member.name.removeprefix("src/")
            if (
                not member.isfile()
                or path.is_absolute()
                or ".." in path.parts
                or key not in _CHANGED
                or key in changed
                or total > 2 * 1024**2
            ):
                raise ValueError("unexpected candidate source delta member")
            stream = bundle.extractfile(member)
            assert stream is not None
            current[key] = stream.read()
            changed.add(key)
    if changed != _CHANGED:
        raise ValueError("candidate source delta is incomplete")
    expected = _read(root, "candidate.sources-manifest.json")
    if {k: _sha(v) for k, v in current.items()} != expected:
        raise ValueError("candidate complete source manifest differs")
    return {"prior": prior, "current": current}


def audit_archive(root: Path) -> dict[str, Any]:
    """Recompute the failed speed gate while retaining complete quality evidence."""
    root = root.resolve()
    index_raw = (root / "archive-index.json").read_bytes()
    index = json.loads(index_raw)
    files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if set(index["files"]) != files - {"archive-index.json", "audit.json"}:
        raise ValueError("archive index does not cover every payload")
    for name, digest in index["files"].items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or ":" in name:
            raise ValueError("archive index path escapes publication")
        if _sha((root / name).read_bytes()) != digest:
            raise ValueError("published asset checksum differs: " + name)
    raw = (root / "protocol.json").read_bytes()
    protocol_sha = _sha(raw)
    protocol = json.loads(raw)
    if (
        index["schema_version"] != "calibrex.lidar_map_resident_screen_archive/v0.1"
        or protocol["commits"] != _COMMITS
        or protocol["orders"] != _ORDERS
        or protocol["workload"] != _WORKLOAD
        or protocol["gates"] != _GATES
        or protocol["bag"]["sha256"]
        != "f57a2e85339316d4235ab693ab2876aac5e113625c7f8b0f7873b62f92e09643"
        or protocol["bag"]["bytes"] != 613350674
        or index["provenance"]["protocol_sha256"] != protocol_sha
    ):
        raise ValueError("predeclared sources, input, workload or gates differ")
    if _sha((root / "launcher.py.txt").read_bytes()) != protocol["provenance"]["launcher_sha256"]:
        raise ValueError("frozen launcher differs")
    for name, digest in protocol["provenance"]["helpers"].items():
        if _sha((root / (Path(name).name + ".txt")).read_bytes()) != digest:
            raise ValueError("frozen comparison helper differs")
    sources = _source_trees(root, index)
    source_manifests = {
        "prior": index["provenance"]["source_dependency"]["current.sources-manifest.json"],
        "current": _sha((root / "candidate.sources-manifest.json").read_bytes()),
    }
    if protocol["provenance"]["source_manifests"] != {
        "cpu": source_manifests["prior"],
        "resident": source_manifests["current"],
    }:
        raise ValueError("executed complete source manifests differ")
    workers: dict[str, Any] = {}
    options = None
    for repeat, order in enumerate(_ORDERS):
        for variant in order:
            name = f"{repeat:02d}-{variant}"
            result = _read(root, name + "/result.json")
            LidarLidarExtrinsicArtifact.model_validate(result)
            measurement = _read(root, name + "/measurement.json")
            prov = measurement["provenance"]
            kind = "current" if variant == "resident" else "prior"
            if (
                measurement["variant"] != variant
                or prov["source_commit"] != _COMMITS[kind]
                or result["provenance"]["git_commit"] != _COMMITS[kind]
                or prov["source_manifest_sha256"] != source_manifests[kind]
                or prov["protocol_sha256"] != protocol_sha
                or prov["bag_sha256"] != protocol["bag"]["sha256"]
                or prov["launcher_sha256"] != protocol["provenance"]["launcher_sha256"]
                or prov["result_sha256"] != _sha((root / name / "result.json").read_bytes())
                or not prov["executed_sources"]
                or any(_sha(sources[kind][k]) != v for k, v in prov["executed_sources"].items())
                or prov["blas_threads"]
                != {"OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
                or prov["cuda_cache_initially_empty"] is not True
            ):
                raise ValueError("worker source/input/cache/thread binding differs")
            if options is None:
                options = measurement["options"]
            if measurement["options"] != options:
                raise ValueError("worker effective calibration options differ")
            if any(
                options[k] != _WORKLOAD[k]
                for k in (
                    "sample_every_scans",
                    "map_half_window_scans",
                    "block_duration_s",
                    "jackknife_groups",
                )
            ) or any(
                options["odometry"][k] != value
                for k, value in (
                    ("voxel_size_m", 0.03),
                    ("local_map_scans", 3),
                    ("geometry_quantum_m", 1e-9),
                    ("motion_translation_quantum_m", 1e-6),
                    ("motion_rotation_quantum_rad", 1e-7),
                )
            ):
                raise ValueError("worker did not execute the predeclared settings")
            if (
                len(measurement["fits"]) != 5
                or result["jackknife_fits"] != 4
                or len(result["dofs"]) != 6
                or any(d["known_bad_control"] is None for d in result["dofs"])
                or len(measurement["prepared_samples"]) != result["samples"]["samples"]
                or result["samples"]["holdout_samples"] <= 0
            ):
                raise ValueError("worker omitted fits, jackknives, holdout or controls")
            for fit in measurement["fits"]:
                if (
                    fit["execution_backend"] != "numpy_scipy"
                    or fit["transform"] is None
                    or fit["covariance"] is None
                    or fit["correspondences"] < 30
                ):
                    raise ValueError("complete native map fit is missing")
            saved = result["options"]
            if variant == "cpu":
                if "odometry_backend" in saved or "resident_odometry" in saved:
                    raise ValueError("CPU arm requested a CUDA backend")
            else:
                if (
                    saved["odometry_backend"] != "cuda"
                    or saved["odometry_options"] != options["odometry"]
                ):
                    raise ValueError("CUDA odometry options differ")
                identity = dict(
                    part.split("=", 1) for part in saved["odometry_execution"].split(";")[1:]
                )
                expected = {
                    "fp64": "true",
                    "query_fma": "false",
                    **_kernel_hashes(sources[kind], resident=variant == "resident"),
                }
                if any(identity.get(k) != v for k, v in expected.items()):
                    raise ValueError("executed odometry kernels differ from frozen sources")
            if variant == "resident":
                work = saved["resident_geometry_execution"]
                stats = saved["odometry_statistics"]
                if (
                    saved["resident_odometry"] is not True
                    or work["source_captures"] != result["samples"]["reference_scans"]
                    or work["source_cache_hits"] != 0
                    or work["gyro_rows"] != 0
                    or not 0 < work["source_cache_bytes"] <= 512 * 1024**2
                    or stats["gpu_grid_captures"] + stats["gpu_grid_native_fallbacks"]
                    != work["source_captures"] - 1
                    or any(type(v) is not int or v < 0 for v in work.values())
                ):
                    raise ValueError("resident geometry work did not execute")
            whole = _read(root, name + "/whole.json")
            if (
                whole["exit_code"] != 0
                or not math.isfinite(whole["seconds"])
                or whole["seconds"] <= 0
            ):
                raise ValueError("worker did not complete a positive whole-process time")
            if whole["provenance"]["protocol_sha256"] != protocol_sha:
                raise ValueError("whole-process protocol differs")
            for stream in ("stdout", "stderr"):
                if whole["provenance"][stream + "_sha256"] != _sha(
                    (root / name / f"{stream}.log").read_bytes()
                ):
                    raise ValueError("worker log binding differs")
            normalized = copy.deepcopy(_without_run_metadata(result))
            for field in (
                "odometry_backend",
                "odometry_execution",
                "odometry_options",
                "resident_odometry",
                "resident_geometry_execution",
                "odometry_statistics",
            ):
                normalized["options"].pop(field, None)
            workers[name] = (result, measurement, whole["seconds"], normalized)
    for repeat in range(3):
        data = {v: workers[f"{repeat:02d}-{v}"] for v in _VARIANTS}
        errors = {v: _compare(data["cpu"][3], data[v][3]) for v in ("native", "resident")}
        fit_errors = {}
        for variant in ("native", "resident"):
            found = []
            for i, (a, b) in enumerate(
                zip(data["cpu"][1]["fits"], data[variant][1]["fits"], strict=True)
            ):
                a, b = dict(a), dict(b)
                if a.pop("execution_backend") != b.pop("execution_backend"):
                    raise ValueError("map fit backend differs")
                found.extend(
                    f"fit-{i}:" + error for error in _compare({"result": a}, {"result": b})
                )
            fit_errors[variant] = found
        exact = _same_float_bits(data["native"][3], data["resident"][3])
        fit_exact = _same_float_bits(data["native"][1]["fits"], data["resident"][1]["fits"])
        sample_exact = _same_float_bits(
            data["native"][1]["prepared_samples"], data["resident"][1]["prepared_samples"]
        )
        if (
            any(errors.values())
            or any(fit_errors.values())
            or not (exact and fit_exact and sample_exact)
        ):
            raise ValueError("complete calibration, full fits or sample quality differs")
        seconds = {v: data[v][2] for v in _VARIANTS}
        expected = {
            "repeat": repeat,
            "seconds": seconds,
            "quality_passed": True,
            "cpu_to_resident": seconds["cpu"] / seconds["resident"],
            "native_to_resident": seconds["native"] / seconds["resident"],
            "errors": errors,
            "fit_errors": fit_errors,
            "gpu_result_bits_equal": True,
            "gpu_fit_bits_equal": True,
            "gpu_sample_bits_equal": True,
        }
        triad = _read(root, f"triad-{repeat:02d}.json")
        if {k: v for k, v in triad.items() if k != "provenance"} != expected:
            raise ValueError("saved triad claims differ from raw workers")
        records = {
            p.relative_to(root).as_posix(): _sha(p.read_bytes())
            for v in _VARIANTS
            for p in sorted((root / f"{repeat:02d}-{v}").glob("*.json"))
        }
        if triad["provenance"] != {"protocol_sha256": protocol_sha, "worker_records": records}:
            raise ValueError("triad worker bindings differ")
    medians = {
        v: statistics.median(workers[f"{i:02d}-{v}"][2] for i in range(3)) for v in _VARIANTS
    }
    cpu_ratio, native_ratio = (
        medians["cpu"] / medians["resident"],
        medians["native"] / medians["resident"],
    )
    advance = cpu_ratio >= 2.0 and native_ratio > 1.0
    expected_summary = {
        "medians_seconds": medians,
        "cpu_to_resident": cpu_ratio,
        "native_to_resident": native_ratio,
        "quality_passed": True,
        "advance": advance,
        "provenance": {
            "protocol_sha256": protocol_sha,
            "triads": {
                f"triad-{i:02d}.json": _sha((root / f"triad-{i:02d}.json").read_bytes())
                for i in range(3)
            },
        },
    }
    if _read(root, "summary.json") != expected_summary:
        raise ValueError("reported ratios or advance gate differ from raw timings")
    qualification = _read(root, "correctness/provenance.json")
    if set(qualification["sources"]) != {"src/" + k for k in sources["current"]}:
        raise ValueError("correctness source coverage differs")
    for key, digest in qualification["sources"].items():
        raw = sources["current"][key.removeprefix("src/")]
        lf = raw.replace(b"\r\n", b"\n")
        if digest not in {_sha(raw), _sha(lf), _sha(lf.replace(b"\n", b"\r\n"))}:
            raise ValueError("qualified source differs from measured source")
    for name, digest in qualification["tests"].items():
        if _sha((root / "correctness" / (Path(name).name + ".txt")).read_bytes()) != digest:
            raise ValueError("qualified test source differs")
    receipt = _read(root, "correctness/summary.json")
    if (
        receipt["exit_code"] != 0
        or receipt["provenance_sha256"] != _sha((root / "correctness/provenance.json").read_bytes())
        or receipt["pytest_log_sha256"] != _sha((root / "correctness/pytest.log").read_bytes())
        or qualification["launcher_sha256"]
        != _sha((root / "correctness/launcher.py.txt").read_bytes())
        or b"100 passed" not in (root / "correctness/pytest.log").read_bytes()
    ):
        raise ValueError("candidate qualification receipt differs")
    report = {
        "schema_version": "calibrex.lidar_map_resident_screen_audit/v0.1",
        "decision": "would_advance" if advance else "not_adopted",
        "production_changed": False,
        "complete_workers": 9,
        "complete_fits": 45,
        "cpu_gpu_fit_comparisons": 30,
        "gpu_fit_bit_comparisons": 15,
        "quality_passed": True,
        "medians_seconds": medians,
        "cpu_to_resident": cpu_ratio,
        "native_to_resident": native_ratio,
        "candidate_slowdown_fraction": medians["resident"] / medians["native"] - 1,
        "candidate_faster_pairs": sum(
            workers[f"{i:02d}-resident"][2] < workers[f"{i:02d}-native"][2] for i in range(3)
        ),
        "gpu_runtime_imported": any(n == "cupy" or n.startswith("cupy.") for n in sys.modules),
        "provenance": {
            "protocol_sha256": protocol_sha,
            "archive_index_sha256": _sha(index_raw),
            "auditor_sha256": _sha(Path(__file__).read_bytes()),
            "commits": _COMMITS,
        },
    }
    jsonschema.validate(report, _read(root, "audit.schema.json"))
    return report


def main() -> None:
    """Check a saved screen and optionally save the recomputed audit report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    report = audit_archive(args.archive)
    if args.check and _read(args.archive, "audit.json") != report:
        raise ValueError("saved audit report differs")
    raw = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.write:
        (args.archive / "audit.json").write_text(raw, encoding="utf-8")
    print(raw, end="")


if __name__ == "__main__":
    main()
