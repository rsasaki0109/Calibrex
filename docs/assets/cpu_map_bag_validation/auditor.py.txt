"""Audit whole-bag CPU map comparisons without importing CUDA or raw bags."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import statistics
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import jsonschema
from tools import benchmark_cpu_map_bag as benchmark
from tools.audit_cpu_map_query_cache import _delta
from tools.audit_resident_lidar_geometry import _kernel_hashes, _sources
from tools.benchmark_bspline_lidar_imu import _same_float_bits

from calibrex.core.lidar_lidar_extrinsic import LidarLidarExtrinsicArtifact

_COMMITS = {
    "prior": "96871f64588a122e33e01c713e6710f0f0f36d71",
    "current": "917748b10d5eff9bb077b332a018eafdb33e9fcd",
}
_VARIANTS = ("prior", "lazy", "parallel")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read(root: Path, name: str) -> Any:
    return json.loads((root / name).read_bytes())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _definitions(raw: bytes, names: set[str]) -> dict[str, str]:
    return {
        node.name: ast.dump(node, include_attributes=False)
        for node in ast.parse(raw).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    }


def audit_archive(root: Path) -> dict[str, Any]:
    """Bind all complete workers, original summaries, sources and stage scopes."""
    root = root.resolve()
    index = _read(root, "archive-index.json")
    _require(
        index["schema_version"] == "calibrex.cpu_map_bag_archive/v0.1", "archive version differs"
    )
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    _require(
        set(index["files"]) == actual - {"archive-index.json", "audit.json"},
        "archive payload coverage differs",
    )
    for name, digest in index["files"].items():
        path = PurePosixPath(name)
        _require(
            not path.is_absolute() and ".." not in path.parts and ":" not in name,
            "archive path escapes publication",
        )
        _require(_sha((root / name).read_bytes()) == digest, "payload checksum differs: " + name)
    dependencies = {
        "resident_scan_validation": {"current.sources.tar.gz", "current.sources-manifest.json"},
        "cpu_map_query_cache": {"lazy.source-delta.tar.gz", "lazy.sources-manifest.json"},
    }
    _require(set(index["source_dependencies"]) == set(dependencies), "source dependencies differ")
    for directory, names in dependencies.items():
        _require(
            set(index["source_dependencies"][directory]) == names, "source dependency files differ"
        )
        for name, digest in index["source_dependencies"][directory].items():
            _require(
                _sha((root.parent / directory / name).read_bytes()) == digest,
                "shared source dependency hash differs",
            )
    prior = _sources(root.parent / "resident_scan_validation", "current")
    lazy = _delta(
        root.parent / "cpu_map_query_cache",
        "lazy",
        prior,
        {
            "calibrex/cli/main.py",
            "calibrex/solvers/lidar_lidar_map_solver.py",
        },
    )
    _require(
        {name: _sha(raw) for name, raw in lazy.items()}
        == _read(root.parent / "cpu_map_query_cache", "lazy.sources-manifest.json"),
        "shared lazy source manifest differs",
    )
    current = _delta(root, "current", lazy, {"calibrex/evaluation/lidar_lidar_map.py"})
    sources = {"prior": prior, "current": current}
    for kind, tree in sources.items():
        _require(
            len(tree) == 396
            and {name: _sha(raw) for name, raw in tree.items()}
            == _read(root, kind + ".sources-manifest.json"),
            "complete source manifest differs",
        )
    driver = (root / "benchmark_cpu_map_bag.py.txt").read_bytes()
    definitions = {"_physical", "compare_workers", "summarize"}
    _require(
        _definitions(driver, definitions)
        == _definitions(
            Path(benchmark.__file__).read_bytes(),
            definitions,
        )
        and len(_definitions(driver, definitions)) == 3,
        "comparison helper definitions differ",
    )
    common = Path(__file__).with_name("benchmark_bspline_lidar_imu.py").read_bytes()
    _require(
        _definitions(common, {"_same_float_bits"})
        == _definitions(
            (root / "benchmark_bspline_lidar_imu.py.txt").read_bytes(),
            {"_same_float_bits"},
        ),
        "bit comparator definition differs",
    )
    benchmark_schema = _read(root, "cpu_map_bag_comparison.schema.json.txt")

    def validate(value: Any, kind: str) -> None:
        schema = dict(benchmark_schema, **{"$ref": "#/$defs/" + kind})
        jsonschema.validate(value, schema)

    protocol = _read(root, "protocol.json")
    validate(protocol, "protocol")
    _require(
        protocol["commits"] == _COMMITS
        and protocol["repeats"] == 3
        and protocol["scenarios"]
        == {"map-000": {"map_voxel_size_m": 0.2}, "map-001": {"map_voxel_size_m": 0.03}}
        and protocol["query_workers"] == {"prior": 1, "lazy": 1, "parallel": 8}
        and protocol["orders"] == [list(_VARIANTS[i:] + _VARIANTS[:i]) for i in range(3)]
        and protocol["settings"]["odometry_backend"] == "cuda"
        and protocol["settings"]["max_seconds"] == 6.0
        and protocol["settings"]["topic"] == "/livox/points"
        and protocol["settings"]["point_time_field"] == "t"
        and protocol["settings"]["dataset_family"] == "glim_versatile"
        and protocol["settings"]["dataset_license"] == "upstream AIST GLIM dataset terms"
        and protocol["bag"]["bytes"] == 613350674
        and protocol["bag"]["sha256"]
        == "f57a2e85339316d4235ab693ab2876aac5e113625c7f8b0f7873b62f92e09643",
        "frozen source/input/workload differs",
    )
    _require(set(protocol["helpers_sha256"]) == set(benchmark._HELPERS), "helper coverage differs")
    for name, digest in protocol["helpers_sha256"].items():
        _require(
            _sha((root / (Path(name).name + ".txt")).read_bytes()) == digest,
            "frozen helper bytes differ",
        )
    for kind in sources:
        _require(
            protocol["source_manifests_sha256"][kind]
            == _sha((root / (kind + ".sources-manifest.json")).read_bytes()),
            "protocol source manifest binding differs",
        )
    original_summary = _read(root, "summary.json")
    validate(original_summary, "summary")
    summaries = {}
    pairs = 0
    triad_hashes = {}
    for scenario, settings in protocol["scenarios"].items():
        rows = []
        data: dict[str, list[dict[str, Any]]] = {variant: [] for variant in _VARIANTS}
        for repeat in range(3):
            measurements = {}
            seconds = {}
            for variant in _VARIANTS:
                folder = root / scenario / f"{repeat:02d}-{variant}"
                record = _read(folder, "measurement.json")
                validate(record, "worker")
                whole = _read(folder, "whole.json")
                validate(whole, "whole")
                artifact = _read(folder, "result.json")
                LidarLidarExtrinsicArtifact.model_validate(artifact)
                _require(_same_float_bits(artifact, record["artifact"]), "saved artifact differs")
                kind = "prior" if variant == "prior" else "current"
                prov = record["provenance"]
                _require(
                    record["variant"] == variant
                    and record["scenario"] == scenario
                    and record["repeat"] == repeat
                    and prov["source_commit"] == _COMMITS[kind]
                    and artifact["provenance"]["git_commit"] == _COMMITS[kind]
                    and prov["bag_sha256"] == protocol["bag"]["sha256"]
                    and prov["protocol_sha256"] == _sha((root / "protocol.json").read_bytes())
                    and prov["helpers_sha256"] == protocol["helpers_sha256"]
                    and prov["source_manifest_sha256"] == protocol["source_manifests_sha256"][kind]
                    and prov["result_sha256"] == _sha((folder / "result.json").read_bytes())
                    and prov["query_workers"] == protocol["query_workers"][variant]
                    and prov["cuda_cache_initially_empty"] is True
                    and prov["odometry_backend"] == "cuda",
                    "worker source/input/cache/execution binding differs",
                )
                _require(
                    prov["executed_sources"]
                    and all(
                        name in sources[kind] and digest == _sha(sources[kind][name])
                        for name, digest in prov["executed_sources"].items()
                    ),
                    "executed source bytes differ",
                )
                identity = artifact["options"]["odometry_execution"]
                _require(
                    artifact["options"]["odometry_backend"] == "cuda"
                    and identity.startswith(
                        "calibrex.cuda_scan_odometry/v0.1;fp64=true;query_fma=false;"
                    )
                    and all(
                        f"{name}={digest}" in artifact["options"]["odometry_execution"].split(";")
                        for name, digest in _kernel_hashes(sources[kind], resident=False).items()
                    ),
                    "common odometry kernel identity differs",
                )
                _require(
                    record["options"]["preprocess"]["voxel_size_m"] == settings["map_voxel_size_m"]
                    and all(
                        record["options"]["odometry"][key] == value
                        for key, value in protocol["settings"]["odometry_options"].items()
                    )
                    and record["options"]["map_half_window_scans"] == 5,
                    "executed map/odometry options differ",
                )
                _require(
                    artifact["jackknife_fits"] == 4
                    and len(artifact["dofs"]) == 6
                    and all(row.get("known_bad_control") is not None for row in artifact["dofs"])
                    and artifact["samples"]["reference_scans"] == 60
                    and len(record["fits"]) == 5
                    and all(
                        fit["transform"] is not None
                        and fit["covariance"] is not None
                        and fit["sigma_m"] is not None
                        for fit in record["fits"]
                    ),
                    "complete fit/evidence coverage differs",
                )
                work = record["work"]
                _require(
                    work["map_rows"]
                    == sum(
                        sample["map_points"]["shape"][0] for sample in record["prepared_samples"]
                    )
                    and work["target_rows"]
                    == sum(
                        sample["target_points"]["shape"][0] for sample in record["prepared_samples"]
                    )
                    and 0 < work["normal_rows"] <= work["map_rows"],
                    "map work coverage differs",
                )
                _require(
                    whole["returncode"] == 0
                    and whole["protocol_sha256"] == prov["protocol_sha256"]
                    and whole["helpers_sha256"] == protocol["helpers_sha256"]
                    and record["api_seconds"] <= whole["seconds"]
                    and sum(record["stages"].values()) <= record["api_seconds"]
                    and sum(record["fit_seconds"]) <= record["stages"]["evaluate_seconds"],
                    "whole/stage timing scope differs",
                )
                _require(
                    all(not (folder / name).read_bytes() for name in ("stdout.log", "stderr.log")),
                    "worker diagnostic log is not empty",
                )
                measurements[variant] = record
                seconds[variant] = whole["seconds"]
                data[variant].append(record)
            errors = {
                variant: benchmark.compare_workers(measurements["prior"], measurements[variant])
                for variant in _VARIANTS[1:]
            }
            _require(not any(errors.values()), "physical fit/evidence bits differ: " + str(errors))
            pairs += 10
            name = f"{scenario}-triad-{repeat:02d}.json"
            triad = _read(root, name)
            validate(triad, "triad")
            expected = {
                "scenario": scenario,
                "repeat": repeat,
                "seconds": seconds,
                "errors": errors,
                "measurement_sha256": {
                    variant: _sha(
                        (root / scenario / f"{repeat:02d}-{variant}/measurement.json").read_bytes(),
                    )
                    for variant in _VARIANTS
                },
            }
            _require(_same_float_bits(triad, expected), "original triad binding differs")
            triad_hashes[name] = _sha((root / name).read_bytes())
            rows.append(triad)
        summary = benchmark.summarize(rows)
        _require(
            _same_float_bits(original_summary["scenarios"][scenario], summary),
            "original whole-process summary differs",
        )
        stages = {
            variant: {
                key: statistics.median(record["stages"][key] for record in records)
                for key in ("collect_seconds", "evaluate_seconds")
            }
            for variant, records in data.items()
        }
        summaries[scenario] = {
            "map_voxel_size_m": settings["map_voxel_size_m"],
            "whole": summary,
            "median_stages_seconds": stages,
            "normal_rows": {
                variant: [r["work"]["normal_rows"] for r in records]
                for variant, records in data.items()
            },
        }
    _require(
        original_summary["triads_sha256"] == triad_hashes
        and original_summary["protocol_sha256"] == _sha((root / "protocol.json").read_bytes())
        and set(original_summary["scenarios"]) == set(protocol["scenarios"]),
        "summary transitive binding differs",
    )
    report = {
        "schema_version": "calibrex.cpu_map_bag_audit/v0.1",
        "complete_workers": 18,
        "complete_fits": 90,
        "fit_bit_comparisons": pairs,
        "quality_passed": True,
        "scenarios": summaries,
        "gpu_runtime_imported": "cupy" in sys.modules,
        "provenance": {
            "archive_index_sha256": _sha((root / "archive-index.json").read_bytes()),
            "protocol_sha256": _sha((root / "protocol.json").read_bytes()),
            "auditor_sha256": _sha(Path(__file__).read_bytes()),
            "commits": _COMMITS,
        },
    }
    schema = Path(__file__).with_name("schemas") / "cpu_map_bag_audit.schema.json"
    jsonschema.validate(report, json.loads(schema.read_bytes()))
    _require(
        (root / "auditor.py.txt").read_bytes() == Path(__file__).read_bytes(),
        "frozen auditor differs",
    )
    _require(
        (root / "audit.schema.json").read_bytes() == schema.read_bytes(),
        "frozen audit schema differs",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    report = audit_archive(args.archive)
    if args.check and report != _read(args.archive, "audit.json"):
        raise ValueError("saved audit differs")
    if args.write:
        (args.archive / "audit.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
