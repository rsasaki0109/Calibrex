"""CPU-only replay of the complete resident scan-validation comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import jsonschema
from tools.audit_resident_lidar_geometry import _constants, _kernel_hashes, _sources
from tools.benchmark_bspline_lidar_imu import (
    _compare,
    _same_float_bits,
    _validate_backend_execution,
    _validate_plane_execution,
    _validate_resident_execution,
    _without_run_metadata,
)

from calibrex.core.lidar_imu_bag_calibration import LidarImuBagCalibrationArtifact

_COMMITS = {
    "prior": "3dbd181830a744c7635aefc4b2fc11f2fd74fa32",
    "current": "3fd317259d32406a20be38c5ff9ad8425177e2a7",
}
_NEW_FIELDS = ("source_validation_rows", "gyro_cached_validations")
_OPTIONS = {
    "max_seconds": 6.0,
    "max_rounds": 4,
    "voxel_size_m": 0.03,
    "local_map_scans": 3,
    "geometry_quantum_m": 1e-9,
    "motion_translation_quantum_m": 1e-6,
    "motion_rotation_quantum_rad": 1e-7,
}
_ORDERS = [["prior", "current"], ["current", "prior"], ["prior", "current"]]
_GATES = {
    "all_result_fit_placement_float_bits": True,
    "only_new_work_fields_removed": list(_NEW_FIELDS),
    "median_prior_to_current_minimum": 1.10,
    "faster_pairs_minimum": 2,
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read(root: Path, name: str) -> Any:
    return json.loads((root / name).read_bytes())


def audit_archive(root: Path) -> dict[str, Any]:
    """Bind all six workers and recompute every quality gate and timing ratio."""
    root = root.resolve()
    index_raw = (root / "archive-index.json").read_bytes()
    index = json.loads(index_raw)
    files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if set(index["files"]) != files - {"archive-index.json", "audit.json"}:
        raise ValueError("archive index does not cover every published payload")
    for name, digest in index["files"].items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or ":" in name:
            raise ValueError("archive index path escapes the publication")
        if _sha((root / name).read_bytes()) != digest:
            raise ValueError("published asset checksum differs: " + name)
    protocol_raw = (root / "protocol.json").read_bytes()
    protocol_sha = _sha(protocol_raw)
    protocol = json.loads(protocol_raw)
    if (
        index["schema_version"] != "calibrex.resident_scan_validation_archive/v0.1"
        or index["provenance"] != {"protocol_sha256": protocol_sha, "commits": _COMMITS}
        or protocol["commits"] != _COMMITS
        or protocol["bag"]["sha256"]
        != "f57a2e85339316d4235ab693ab2876aac5e113625c7f8b0f7873b62f92e09643"
        or protocol["bag"]["bytes"] != 613350674
        or protocol["options"] != _OPTIONS
        or protocol["orders"] != _ORDERS
        or protocol["gates"] != _GATES
    ):
        raise ValueError("declared sources, input, options or gates differ")
    for name, key in (
        ("launcher.py.txt", "launcher_sha256"),
        ("benchmark.py.txt", "benchmark_sha256"),
    ):
        if _sha((root / name).read_bytes()) != protocol["provenance"][key]:
            raise ValueError("frozen launcher binding differs")
    sources = {v: _sources(root, v) for v in _COMMITS}
    for v in _COMMITS:
        if (
            _sha((root / f"{v}.sources-manifest.json").read_bytes())
            != protocol["provenance"]["source_manifests"][v]
        ):
            raise ValueError("executed source manifest binding differs")
    changed = {k for k in sources["prior"] if sources["prior"][k] != sources["current"].get(k)}
    if set(sources["prior"]) != set(sources["current"]) or changed != {
        "calibrex/solvers/cuda_resident_odometry_adapter.py",
    }:
        raise ValueError("comparison changes more than resident scan validation")
    schema = _read(root, "benchmark.schema.json")
    workers: dict[str, Any] = {}
    for repeat, order in enumerate(_ORDERS):
        for variant in order:
            name = f"{repeat:02d}-{variant}"
            result = _read(root, name + "/result.json")
            LidarImuBagCalibrationArtifact.model_validate(result)
            measurement = _read(root, name + "/measurement.json")
            jsonschema.validate(measurement, schema)
            prov = measurement["provenance"]
            if (
                prov["source_manifest_sha256"]
                != protocol["provenance"]["source_manifests"][variant]
                or prov["launcher_sha256"] != protocol["provenance"]["benchmark_sha256"]
                or prov["bag_sha256"] != protocol["bag"]["sha256"]
                or prov["result_sha256"] != _sha((root / name / "result.json").read_bytes())
                or not prov["executed_sources"]
                or any(_sha(sources[variant][k]) != v for k, v in prov["executed_sources"].items())
                or prov["blas_threads"]
                != {
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                }
                or prov["cuda_cache_initially_empty"] is not True
                or prov["odometry_backend"] != "cuda"
                or prov["fit_plane_backend"] != "cuda"
                or prov["resident_odometry"] is not True
                or result["provenance"]["git_commit"] != _COMMITS[variant]
            ):
                raise ValueError("worker source/input/cache/thread binding differs")
            _validate_backend_execution(result, "cuda")
            _validate_resident_execution(
                result, True, _kernel_hashes(sources[variant], resident=True)
            )
            plane = _constants(sources[variant]["calibrex/solvers/cuda_bspline_plane_adapter.py"])
            _validate_plane_execution(
                result, measurement["rounds"], "cuda", _sha(plane["_SOURCE"].encode())
            )
            for key, value in _OPTIONS.items():
                options = (
                    result["options"]
                    if key in ("max_seconds", "max_rounds")
                    else result["options"]["odometry_options"]
                )
                if options[key] != value:
                    raise ValueError("worker options differ")
            if (
                len(result["options"]["initial_odometry_execution"]) != 7
                or len(result["options"]["segment_odometry_execution"]) != 1
                or len(result["rounds"]) != 4
                or len(measurement["rounds"]) != 4
                or len(result["controls"]) != 32
                or measurement["control_evaluation_calls"] != 1
                or measurement["control_score_calls"] != 34
            ):
                raise ValueError("worker omitted passes, full fits or controls")
            for fit, reported in zip(measurement["rounds"], result["rounds"], strict=True):
                if (
                    fit["status"] != reported["fit_status"]
                    or fit["iterations"] != reported["iterations"]
                ):
                    raise ValueError("full fit and bag evidence disagree")
            whole = _read(root, name + "/whole.json")
            if (
                whole["exit_code"] != 0
                or not math.isfinite(whole["seconds"])
                or whole["seconds"] <= 0
            ):
                raise ValueError("worker did not finish a positive whole-process measurement")
            if whole["provenance"]["protocol_sha256"] != protocol_sha:
                raise ValueError("worker protocol binding differs")
            for stream in ("stdout", "stderr"):
                if whole["provenance"][stream + "_sha256"] != _sha(
                    (root / name / f"{stream}.log").read_bytes()
                ):
                    raise ValueError("worker log binding differs")
            placement = _read(root, name + "/placement.json")
            if placement["provenance"] != {
                "launcher_sha256": protocol["provenance"]["launcher_sha256"],
                "source_manifest_sha256": prov["source_manifest_sha256"],
            }:
                raise ValueError("placement binding differs")
            workers[name] = (result, measurement, whole["seconds"], placement["calls"])
    ratios, rows = [], 0
    for repeat in range(3):
        prior, current = (workers[f"{repeat:02d}-{v}"] for v in _COMMITS)
        old_work, work = (r[0]["options"]["resident_geometry_execution"] for r in (prior, current))
        if any(k in old_work for k in _NEW_FIELDS) or any(
            type(work.get(k)) is not int or work[k] <= 0 for k in _NEW_FIELDS
        ):
            raise ValueError("new capture validation did not execute")
        normalized = [_without_run_metadata(r[0]) for r in (prior, current)]
        for key in _NEW_FIELDS:
            normalized[1]["options"]["resident_geometry_execution"].pop(key)
        fits = [
            [{k: v for k, v in fit.items() if k != "seconds"} for fit in r[1]["rounds"]]
            for r in (prior, current)
        ]
        if (
            not _same_float_bits(*normalized)
            or not _same_float_bits(*fits)
            or not _same_float_bits(prior[3], current[3])
        ):
            raise ValueError("complete result, fit or placement float bits differ")
        quality = _compare(prior[0], current[0])
        if not quality["passed"]:
            raise ValueError("physical and evidence gates differ")
        pair = _read(root, f"pair-{repeat:02d}.json")
        seconds = {v: workers[f"{repeat:02d}-{v}"][2] for v in _COMMITS}
        ratio = seconds["prior"] / seconds["current"]
        placed = sum(r["rows"] for r in prior[3])
        expected = {
            "repeat": repeat,
            "seconds": seconds,
            "prior_to_current_ratio": ratio,
            "quality_passed": True,
            "complete_result_float_bits_equal": True,
            "all_full_fit_float_bits_equal": True,
            "all_placement_bits_equal": True,
            "placed_rows_per_variant": placed,
            "full_fits_per_variant": 4,
            "quality": quality,
            "new_validation_work": {k: work[k] for k in _NEW_FIELDS},
        }
        if {k: v for k, v in pair.items() if k != "provenance"} != expected:
            raise ValueError("saved pair claims differ from all raw workers")
        records = {
            p.relative_to(root).as_posix(): _sha(p.read_bytes())
            for v in _COMMITS
            for p in sorted((root / f"{repeat:02d}-{v}").glob("*.json"))
        }
        if pair["provenance"] != {"protocol_sha256": protocol_sha, "worker_records": records}:
            raise ValueError("pair worker binding differs")
        ratios.append(ratio)
        rows += placed
    medians = {v: statistics.median(workers[f"{r:02d}-{v}"][2] for r in range(3)) for v in _COMMITS}
    ratio = medians["prior"] / medians["current"]
    faster = sum(r > 1 for r in ratios)
    advance = ratio >= 1.10 and faster >= 2
    expected_summary = {
        "medians_seconds": medians,
        "prior_to_current_ratio": ratio,
        "faster_pairs": faster,
        "quality_passed": True,
        "advance": advance,
        "provenance": {
            "protocol_sha256": protocol_sha,
            "pairs": {
                f"pair-{i:02d}.json": _sha((root / f"pair-{i:02d}.json").read_bytes())
                for i in range(3)
            },
        },
    }
    if _read(root, "summary.json") != expected_summary:
        raise ValueError("reported ratios or gates differ from complete raw timings")
    qualification = _read(root, "correctness/provenance.json")
    if set(qualification["sources"]) != {"src/" + key for key in sources["current"]}:
        raise ValueError("correctness did not cover the complete runtime source tree")
    for key, digest in qualification["sources"].items():
        raw = sources["current"][key.removeprefix("src/")]
        lf = raw.replace(b"\r\n", b"\n")
        # Git contains both historical CRLF and LF files. The receipt must
        # match an exact reconstructed byte form, with no other normalization.
        if digest not in {_sha(raw), _sha(lf), _sha(lf.replace(b"\n", b"\r\n"))}:
            raise ValueError("correctness sources differ from measured runtime")
    for name, digest in qualification["tests"].items():
        if _sha((root / "correctness" / (Path(name).name + ".txt")).read_bytes()) != digest:
            raise ValueError("correctness test binding differs")
    correctness = _read(root, "correctness/summary.json")
    if (
        correctness["exit_code"] != 0
        or correctness["provenance_sha256"]
        != _sha((root / "correctness/provenance.json").read_bytes())
        or correctness["pytest_log_sha256"] != _sha((root / "correctness/pytest.log").read_bytes())
        or qualification["launcher_sha256"]
        != _sha((root / "correctness/launcher.py.txt").read_bytes())
        or b"67 passed" not in (root / "correctness/pytest.log").read_bytes()
    ):
        raise ValueError("actual-device correctness qualification differs")
    report = {
        "schema_version": "calibrex.resident_scan_validation_audit/v0.1",
        "status": "confirmed" if advance else "speed_gate_not_met",
        "complete_workers": 6,
        "complete_fit_bit_comparisons": 12,
        "point_rows_bit_compared": rows,
        "medians_seconds": medians,
        "prior_to_current_ratio": ratio,
        "faster_pairs": faster,
        "pair_ratios": ratios,
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
    """Audit a saved archive and optionally compare or write its report."""
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
