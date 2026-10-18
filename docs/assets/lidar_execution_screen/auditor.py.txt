"""Replay rejected LiDAR execution screens without CUDA or scan coordinates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import jsonschema
from tools.audit_cpu_map_query_cache import _delta
from tools.audit_resident_lidar_geometry import _kernel_hashes, _sources
from tools.benchmark_bspline_lidar_imu import _same_float_bits

from calibrex.core.lidar_lidar_extrinsic import LidarLidarExtrinsicArtifact

_COMMIT = "917748b10d5eff9bb077b332a018eafdb33e9fcd"
_PROTOCOLS = {
    "sample-parallel": "4215bdbb4833f1dbb6b6c1dcb2724af9c7eaf1c0c6939374450e909ea5379d89",
    "system-transfer": "b0a002413bff86ee76c43272f6b7421fcf313e469a6cee1f343746fc0d1ddbc3",
    "normal-fusion": "4bf812962dba800d4d6821e03fc91945ad1d72431f3ebe31e9167993fe251796",
    "certified-search": "1bb2527e9678ed5dc8febc99a2cabd0caa099c270eaae1e785acbb5c636489b1",
}
_GPU = {
    "system-transfer": (("prior", "pinned", "mapped"), 4, "candidate-register.py.txt"),
    "normal-fusion": (("prior", "fused"), 6, "fused-normal-query.cu.txt"),
    "certified-search": (("prior", "fused"), 6, "certified-query.cu.txt"),
}
_DEPENDENCIES = {
    "resident_scan_validation": ("current.sources.tar.gz", "current.sources-manifest.json"),
    "cpu_map_query_cache": ("lazy.source-delta.tar.gz", "lazy.sources-manifest.json"),
    "cpu_map_bag_validation": ("current.source-delta.tar.gz", "current.sources-manifest.json"),
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read(root: Path, name: str) -> Any:
    return json.loads((root / name).read_bytes())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _seconds(value: object) -> float:
    _require(type(value) is float and math.isfinite(value) and value > 0, "invalid timing")
    return float(value)  # type: ignore[arg-type]


def _source_tree(root: Path, index: dict[str, Any]) -> dict[str, bytes]:
    _require(
        set(index["source_dependencies"]) == set(_DEPENDENCIES), "source dependency set differs"
    )
    for directory, names in _DEPENDENCIES.items():
        expected = index["source_dependencies"][directory]
        _require(set(expected) == set(names), "source dependency files differ")
        for name, digest in expected.items():
            _require(
                _sha((root.parent / directory / name).read_bytes()) == digest,
                "source dependency checksum differs",
            )
    prior = _sources(root.parent / "resident_scan_validation", "current")
    lazy = _delta(
        root.parent / "cpu_map_query_cache",
        "lazy",
        prior,
        {"calibrex/cli/main.py", "calibrex/solvers/lidar_lidar_map_solver.py"},
    )
    current = _delta(
        root.parent / "cpu_map_bag_validation",
        "current",
        lazy,
        {"calibrex/evaluation/lidar_lidar_map.py"},
    )
    _require(
        len(current) == 396
        and {name: _sha(raw) for name, raw in current.items()}
        == _read(root.parent / "cpu_map_bag_validation", "current.sources-manifest.json"),
        "source manifest differs",
    )
    return current


def audit_archive(root: Path) -> dict[str, Any]:
    """Check original receipts, complete fit/registration bits and separate medians."""
    root = root.resolve()
    index = _read(root, "archive-index.json")
    _require(
        index["schema_version"] == "calibrex.lidar_execution_screen_archive/v0.1",
        "archive version differs",
    )
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    _require(
        set(index["files"]) == actual - {"archive-index.json", "audit.json"},
        "archive coverage differs",
    )
    for name, digest in index["files"].items():
        path = PurePosixPath(name)
        _require(
            not path.is_absolute() and ".." not in path.parts and ":" not in name,
            "archive path escapes publication",
        )
        _require(_sha((root / name).read_bytes()) == digest, "payload checksum differs: " + name)
    sources = _source_tree(root, index)
    source_hashes = {name: _sha(raw) for name, raw in sources.items()}
    manifest_hash = _sha(
        (root.parent / "cpu_map_bag_validation/current.sources-manifest.json").read_bytes()
    )
    kernels = _kernel_hashes(sources, resident=False)
    protocols = {}
    for name, digest in _PROTOCOLS.items():
        folder = root / name
        _require(
            _sha((folder / "protocol.json").read_bytes()) == digest,
            "original protocol differs: " + name,
        )
        protocol = _read(folder, "protocol.json")
        _require(
            protocol["source_commit"] == _COMMIT
            and protocol["source_manifest_sha256"] == manifest_hash
            and protocol["launcher_sha256"] == _sha((folder / "launcher.py.txt").read_bytes())
            and protocol["blas_threads"] == 1,
            "source/launcher/thread binding differs",
        )
        protocols[name] = protocol
    cpu = root / "sample-parallel"
    cpu_protocol = protocols["sample-parallel"]
    _require(
        cpu_protocol["query_budget"] == 8
        and cpu_protocol["order"] == [["queries", "samples"], ["samples", "queries"]],
        "CPU order or worker budget differs",
    )
    pairs = []
    cpu_progress = []
    for repeat, order in enumerate(cpu_protocol["order"]):
        physical, seconds = {}, {}
        for variant in order:
            name = f"{repeat:02d}-{variant}"
            measurement = _read(cpu, name + "/measurement.json")
            whole = _read(cpu, name + "/whole.json")
            seconds[variant] = _seconds(whole["seconds"])
            cpu_progress.append({"repeat": repeat, "variant": variant, "seconds": seconds[variant]})
            _require(
                whole["returncode"] == 0
                and whole["measurement_sha256"]
                == _sha((cpu / name / "measurement.json").read_bytes()),
                "original worker receipt differs",
            )
            _require(
                _seconds(measurement["api_seconds"]) <= seconds[variant]
                and len(measurement["full_fits"]) == 5
                and measurement["artifact"]["jackknife_fits"] == 4
                and len(measurement["artifact"]["dofs"]) == 6,
                "complete fit/evidence scope differs",
            )
            LidarLidarExtrinsicArtifact.model_validate(measurement["artifact"])
            saved = _read(cpu, name + "/result.json")
            _require(
                _same_float_bits(saved, measurement["artifact"])
                and measurement["result_sha256"] == _sha((cpu / name / "result.json").read_bytes()),
                "saved artifact differs",
            )
            _require(
                measurement["launcher_sha256"] == cpu_protocol["launcher_sha256"]
                and measurement["protocol_sha256"] == _PROTOCOLS["sample-parallel"]
                and measurement["artifact"]["provenance"]["git_commit"] == _COMMIT
                and measurement["artifact"]["options"]["map_options"]["query_workers"] == 8,
                "CPU worker provenance differs",
            )
            _require(
                measurement["executed_sources"]
                and all(
                    source_hashes[key] == value
                    for key, value in measurement["executed_sources"].items()
                ),
                "executed source binding differs",
            )
            for key, value in measurement["provenance"]["source_sha256"].items():
                raw = (
                    sources[key.removeprefix("src/")]
                    if key.startswith("src/")
                    else (root / "helpers" / (Path(key).name + ".txt")).read_bytes()
                )
                _require(_sha(raw) == value, "CPU benchmark helper differs")
            value = copy.deepcopy(measurement)
            for key in ("api_seconds", "worker_cpu_seconds", "result_sha256"):
                value.pop(key)
            value["artifact"]["provenance"]["created_at"] = "<RUN_TIME>"
            physical[variant] = value
            for log in ("stdout.log", "stderr.log"):
                _require((cpu / name / log).read_bytes() == b"", "CPU worker log is not empty")
        _require(
            _same_float_bits(physical["queries"], physical["samples"]),
            "CPU physical fit/evidence bits differ",
        )
        pair = {"repeat": repeat, "seconds": seconds, "errors": []}
        _require(
            _same_float_bits(pair, _read(cpu, f"pair-{repeat:02d}.json")),
            "CPU original pair differs",
        )
        pairs.append(pair)
        cpu_progress.append(pair)
    medians = {v: statistics.median(p["seconds"][v] for p in pairs) for v in ("queries", "samples")}
    cpu_summary = {
        "pairs": pairs,
        "medians_seconds": medians,
        "ratio": medians["queries"] / medians["samples"],
        "all_float_bits_equal": True,
        "protocol_sha256": _PROTOCOLS["sample-parallel"],
    }
    _require(_same_float_bits(cpu_summary, _read(cpu, "summary.json")), "CPU summary differs")
    cpu_progress.append(cpu_summary)
    _require(
        _same_float_bits(
            cpu_progress,
            [
                json.loads(line)
                for line in (cpu / "supervisor.stdout.log").read_text(encoding="utf-8").splitlines()
            ],
        )
        and (cpu / "supervisor.stderr.log").read_bytes() == b"",
        "CPU original supervisor differs",
    )
    cohorts = {
        "sample-parallel": {
            "observations": 4,
            "medians_seconds": medians,
            "ratios": {"samples": cpu_summary["ratio"]},
        }
    }
    registration_comparisons = 0
    for name, (variants, repeats, candidate) in _GPU.items():
        folder = root / name
        protocol = protocols[name]
        candidate_key = (
            "candidate_register_sha256" if name == "system-transfer" else "candidate_kernel_sha256"
        )
        _require(
            protocol[candidate_key] == _sha((folder / candidate).read_bytes())
            and type(protocol["cuda_cache_initially_empty"]) is bool,
            "candidate code/cache binding differs",
        )
        for filename, digest in protocol["fixtures_sha256"].items():
            if filename.endswith(".json"):
                raw = (root / "fixtures" / filename).read_bytes()
                metadata = json.loads(raw)
                _require(
                    _sha(raw) == digest
                    and metadata["fixture_sha256"]
                    == protocol["fixtures_sha256"][filename[:-5] + ".npz"],
                    "coordinate input binding differs",
                )
        runs = _read(folder, "runs.json")
        order = [
            (i, v, c)
            for i in range(repeats)
            for v in variants[i % len(variants) :] + variants[: i % len(variants)]
            for c in range(8)
        ]
        _require(
            [(r["repeat"], r["variant"], r["case"]) for r in runs] == order,
            "GPU observations missing, duplicated or reordered",
        )
        references = {r["case"]: r for r in runs if r["repeat"] == 0 and r["variant"] == "prior"}
        warmups = _read(folder, "warmups.json")
        _require(
            [(r["repeat"], r["variant"], r["case"]) for r in warmups]
            == [(-1, v, 0) for v in variants],
            "warmup coverage differs",
        )
        for row in [*warmups, *runs]:
            _require(
                math.isclose(
                    _seconds(row["target_seconds"]) + _seconds(row["registration_seconds"]),
                    _seconds(row["seconds"]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ),
                "component timing scope differs",
            )
            reference = references[row["case"]]
            _require(
                _same_float_bits(row["result"], reference["result"])
                and row["statistics"] == reference["statistics"],
                "registration physical bits/work differ",
            )
            identity = dict(
                part.split("=", 1) for part in row["result"]["execution_backend"].split(";")[1:]
            )
            _require(
                all(identity.get(key) == value for key, value in kernels.items())
                and identity.get("normal_schedule") == "async-v1",
                "ordinary kernel identity differs",
            )
        registration_comparisons += len(runs) - 8 + len(warmups)
        medians = {
            v: statistics.median(
                sum(r["seconds"] for r in runs if r["variant"] == v and r["repeat"] == i)
                for i in range(repeats)
            )
            for v in variants
        }
        ratios = {v: medians["prior"] / medians[v] for v in variants}
        summary = {
            "observations": len(runs),
            "all_registration_float_bits_equal": True,
            "medians_seconds": medians,
            "protocol_sha256": _PROTOCOLS[name],
            "runs_sha256": _sha((folder / "runs.json").read_bytes()),
        }
        summary.update(
            {"ratios": ratios} if name == "system-transfer" else {"ratio": ratios["fused"]}
        )
        _require(
            _same_float_bits(summary, _read(folder, "summary.json")), "GPU original summary differs"
        )
        progress = [
            {
                "repeat": i,
                "variant": v,
                "seconds": sum(
                    r["seconds"] for r in runs if r["repeat"] == i and r["variant"] == v
                ),
            }
            for i in range(repeats)
            for v in variants[i % len(variants) :] + variants[: i % len(variants)]
        ]
        _require(
            _same_float_bits(
                [*progress, summary],
                [
                    json.loads(line)
                    for line in (folder / "supervisor.stdout.log")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ],
            )
            and (folder / "supervisor.stderr.log").read_bytes() == b"",
            "GPU original supervisor differs",
        )
        cohorts[name] = {"observations": len(runs), "medians_seconds": medians, "ratios": ratios}
    report = {
        "schema_version": "calibrex.lidar_execution_screen_audit/v0.1",
        "decision": "no_candidate_adopted",
        "complete_map_workers": 4,
        "complete_map_fits": 20,
        "gpu_timed_registrations": 288,
        "retained_gpu_warmups": 7,
        "registration_bit_comparisons": registration_comparisons,
        "physical_bits_equal": True,
        "gpu_runtime_imported": "cupy" in sys.modules,
        "cohorts": cohorts,
        "provenance": {
            "source_commit": _COMMIT,
            "source_manifest_sha256": manifest_hash,
            "archive_index_sha256": _sha((root / "archive-index.json").read_bytes()),
            "auditor_sha256": _sha((root / "auditor.py.txt").read_bytes()),
        },
    }
    jsonschema.validate(report, _read(root, "audit.schema.json"))
    return report


def main(argv: list[str] | None = None) -> int:
    """Replay an archive and optionally save or compare its audit report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    result = audit_archive(args.archive)
    if args.check:
        _require(_same_float_bits(result, _read(args.archive, "audit.json")), "saved audit differs")
    if args.write:
        (args.archive / "audit.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
