"""CPU replay of broad-search, cooperative ICP and complete-pipeline follow-ups."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
from tools.audit_lidar_execution_screen import _read, _require, _seconds, _sha, _source_tree
from tools.audit_resident_lidar_geometry import _kernel_hashes
from tools.benchmark_bspline_lidar_imu import _same_float_bits

from calibrex.core.lidar_imu_bag_calibration import LidarImuBagCalibrationArtifact

_PROTOCOLS = {
    "captured-tree": "7a6f36b523c08dac75ccdff9f2b52fc3614712edfab50ceda8c95bb60a1b0d9b",
    "radix-tree": "f1b65a3284635005f44da6601ba998a5569042c226ef69d23dabc4665c46812f",
    "resident-tree": "1c86e5f759539ee413a8bee884c30626c9b28eb3dcc40de6a4c8d319df8f6eb0",
    "block-tree": "f57d24cbaaa8516f947c9641764f318928809c95073329f9f098f28918d29ac4",
    "persistent": "29af0841316425f1b0a95d3deb00701da1f15bb564d1961bc88fe38728bce43d",
    "whole": "8817dae5796f04b79e7a0c6e9a33a2c36fbc4be2db506983974138200a44edd8",
}
_COMMIT = "917748b10d5eff9bb077b332a018eafdb33e9fcd"
_BAG = "f57a2e85339316d4235ab693ab2876aac5e113625c7f8b0f7873b62f92e09643"
_SPATIAL = {
    "gpu_grid_captures",
    "gpu_grid_native_fallbacks",
    "gpu_grid_brute_rows",
    "native_tree_captures",
    "gpu_parallel_fallback_dispatches",
    "broad_normal_fallback_rows",
    "broad_registration_fallback_rows",
    "device_tree_build_kernel_calls",
    "native_tree_builds",
    "serialized_tree_captures",
    "public_node_tree_captures",
}


def _close(left: dict[str, Any], right: dict[str, Any]) -> None:
    for key in ("converged", "iterations", "correspondences", "execution_backend"):
        _require(left[key] == right[key], "persistent registration decisions differ")
    for key in ("transform", "rmse_m", "hessian_min_eigenvalue"):
        _require(
            bool(np.allclose(left[key], right[key], rtol=1e-6, atol=1e-9)),
            "persistent registration quality differs",
        )


def audit_archive(root: Path) -> dict[str, Any]:
    """Verify all observations, fixed quality gates, whole fits and diagnostic scope."""
    root = root.resolve()
    index = _read(root, "archive-index.json")
    _require(
        index["schema_version"] == "calibrex.lidar_pipeline_followup_archive/v0.1",
        "archive version differs",
    )
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    _require(
        actual - {"archive-index.json", "audit.json"} == set(index["files"]),
        "archive coverage differs",
    )
    for name, digest in index["files"].items():
        path = Path(name)
        _require(
            not path.is_absolute() and ".." not in path.parts and ":" not in name,
            "archive path escapes publication",
        )
        _require(_sha((root / name).read_bytes()) == digest, "payload checksum differs")
    sources = _source_tree(root, index)
    manifest = _sha(
        (root.parent / "cpu_map_bag_validation/current.sources-manifest.json").read_bytes()
    )
    kernels = _kernel_hashes(sources, resident=False)
    protocols = {}
    for name, digest in _PROTOCOLS.items():
        folder = root / name
        _require(
            _sha((folder / "protocol.json").read_bytes()) == digest, "original protocol differs"
        )
        protocol = _read(folder, "protocol.json")
        _require(
            protocol["source_commit"] == _COMMIT
            and protocol["source_manifest_sha256"] == manifest
            and protocol["launcher_sha256"] == _sha((folder / "launcher.py.txt").read_bytes())
            and protocol["blas_threads"] == 1,
            "source/launcher/thread binding differs",
        )
        protocols[name] = protocol
    cohorts = {}
    exact_comparisons = tolerant_comparisons = 0
    for name in (key for key in _PROTOCOLS if key != "whole"):
        folder = root / name
        protocol = protocols[name]
        variant = "persistent" if name == "persistent" else "broad"
        variants = ("prior", variant)
        if name == "persistent":
            for field, file in {
                "candidate_kernel_sha256": "persistent.cu.txt",
                "candidate_helper_sha256": "candidate.py.txt",
                "solver_helper_sha256": "solver-dependency.py.txt",
            }.items():
                _require(
                    protocol[field] == _sha((folder / file).read_bytes()),
                    "persistent implementation binding differs",
                )
            _require(
                protocol["quality"]
                == {"rtol": 1e-6, "atol": 1e-9, "exact_decisions_iterations_correspondences": True},
                "quality gate changed",
            )
        else:
            for file, digest in protocol["candidate_sources_sha256"].items():
                _require(
                    _sha((folder / file).read_bytes()) == digest, "candidate code binding differs"
                )
        for file, digest in protocol["fixtures_sha256"].items():
            if file.endswith(".json"):
                raw = (root.parent / "lidar_execution_screen/fixtures" / file).read_bytes()
                _require(
                    _sha(raw) == digest
                    and json.loads(raw)["fixture_sha256"]
                    == protocol["fixtures_sha256"][file[:-5] + ".npz"],
                    "coordinate input binding differs",
                )
        runs = _read(folder, "runs.json")
        order = [
            (i, v, c)
            for i in range(6)
            for v in variants[i % 2 :] + variants[: i % 2]
            for c in range(8)
        ]
        _require(
            [(r["repeat"], r["variant"], r["case"]) for r in runs] == order,
            "observations missing, duplicated or reordered",
        )
        references = {r["case"]: r for r in runs if r["repeat"] == 0 and r["variant"] == "prior"}
        warmups = _read(folder, "warmups.json")
        _require(
            [(r["repeat"], r["variant"], r["case"]) for r in warmups]
            == [(-1, v, 0) for v in variants],
            "warmup coverage differs",
        )
        for row in [*runs, *warmups]:
            _require(
                math.isclose(
                    _seconds(row["target_seconds"]) + _seconds(row["registration_seconds"]),
                    _seconds(row["seconds"]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ),
                "timing scope differs",
            )
            left, right = references[row["case"]]["result"], row["result"]
            if name == "persistent":
                _close(left, right)
            else:
                _require(_same_float_bits(left, right), "registration float bits differ")
                original_work = {
                    k: v
                    for k, v in references[row["case"]]["statistics"].items()
                    if k not in _SPATIAL
                }
                work = {k: v for k, v in row["statistics"].items() if k not in _SPATIAL}
                _require(original_work == work, "registration logical work differs")
            _require(
                all(type(value) is int and value >= 0 for value in row["statistics"].values()),
                "registration work is invalid",
            )
            if name == "persistent" and row["variant"] == variant:
                _require(
                    row["statistics"]["persistent_calls"] == 1
                    and row["statistics"]["persistent_fallbacks"] == 0
                    and row["statistics"]["persistent_iterations"] == right["iterations"],
                    "persistent pilot work/fallback scope differs",
                )
            identity = dict(
                part.split("=", 1) for part in right["execution_backend"].split(";")[1:]
            )
            _require(
                all(identity.get(key) == value for key, value in kernels.items()),
                "base kernel identity differs",
            )
        comparisons = len(runs) + len(warmups) - 8
        if name == "persistent":
            tolerant_comparisons += comparisons
        else:
            exact_comparisons += comparisons
        medians = {
            v: statistics.median(
                sum(r["seconds"] for r in runs if r["variant"] == v and r["repeat"] == i)
                for i in range(6)
            )
            for v in variants
        }
        summary = {
            "observations": 96,
            (
                "existing_cpu_cuda_quality_gate_passed"
                if name == "persistent"
                else "all_registration_float_bits_equal"
            ): True,
            "medians_seconds": medians,
            "ratio": medians["prior"] / medians[variant],
            "protocol_sha256": _PROTOCOLS[name],
            "runs_sha256": _sha((folder / "runs.json").read_bytes()),
        }
        _require(
            _same_float_bits(summary, _read(folder, "summary.json")), "original summary differs"
        )
        progress = [
            {
                "repeat": i,
                "variant": v,
                "seconds": sum(
                    r["seconds"] for r in runs if r["repeat"] == i and r["variant"] == v
                ),
            }
            for i in range(6)
            for v in variants[i % 2 :] + variants[: i % 2]
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
            "original supervisor differs",
        )
        cohorts[name] = {"observations": 96, "ratio": summary["ratio"], "medians_seconds": medians}
    whole = root / "whole"
    protocol = protocols["whole"]
    _require(
        protocol["input_bag_sha256"] == _BAG
        and protocol["order"] == ["prior", "current"]
        and protocol["max_seconds"] == 6.0
        and protocol["max_rounds"] == 4,
        "whole pipeline input/scope differs",
    )
    for file, digest in protocol["helpers_sha256"].items():
        _require(_sha((whole / "helpers" / file).read_bytes()) == digest, "whole helper differs")
    physical, fits, seconds, digests = {}, {}, {}, {}
    for variant in ("prior", "current"):
        folder = whole / variant
        result = _read(folder, "result.json")
        LidarImuBagCalibrationArtifact.model_validate(result)
        row = _read(folder, "measurement.json")
        schema_payload = dict(row)
        schema_payload.pop("prototype_provenance")
        jsonschema.validate(schema_payload, _read(whole, "worker.schema.json"))
        receipt = _read(folder, "whole.json")
        digests[variant] = _sha((folder / "measurement.json").read_bytes())
        _require(
            receipt["returncode"] == 0
            and receipt["measurement_sha256"] == digests[variant]
            and row["provenance"]["result_sha256"] == _sha((folder / "result.json").read_bytes())
            and row["provenance"]["bag_sha256"] == _BAG
            and row["provenance"]["source_manifest_sha256"] == manifest,
            "whole worker provenance differs",
        )
        _require(
            len(row["rounds"]) == 4
            and len(result["controls"]) == 32
            and len(result["options"]["initial_odometry_execution"]) == 7
            and len(result["options"]["segment_odometry_execution"]) == 1
            and row["control_score_calls"] == 34,
            "whole fit/evidence scope differs",
        )
        _require(
            row["prototype_provenance"]["protocol_sha256"] == _PROTOCOLS["whole"]
            and row["prototype_provenance"]["helpers_sha256"] == protocol["helpers_sha256"]
            and row["provenance"]["cuda_cache_initially_empty"] is True,
            "whole prototype/cache binding differs",
        )
        _require(
            row["provenance"]["launcher_sha256"] == protocol["helpers_sha256"]["benchmark.py.txt"]
            and result["provenance"]["git_commit"] == _COMMIT
            and result["options"]["odometry_options"]["voxel_size_m"] == 0.03
            and result["options"]["odometry_options"]["local_map_scans"] == 3,
            "whole runtime/options binding differs",
        )
        for file, digest in row["prototype_provenance"]["generated_code_sha256"].items():
            _require(_sha((folder / file).read_bytes()) == digest, "whole generated code differs")
        _require(
            all(
                _sha(sources[key]) == value
                for key, value in row["provenance"]["executed_sources"].items()
            ),
            "whole executed source differs",
        )
        seconds[variant] = _seconds(receipt["seconds"])
        _require(_seconds(row["pipeline_seconds"]) <= seconds[variant], "whole time omits pipeline")
        value = copy.deepcopy(result)
        value["provenance"]["created_at"] = "<RUN_TIME>"
        command = value["provenance"]["command"]
        position = command.index("--variant") + 1
        _require(command[position] == variant, "whole worker command variant differs")
        command[position] = "<DECLARED_VARIANT>"
        for key in ("initial_odometry_execution", "segment_odometry_execution"):
            for report in value["options"][key]:
                for field in _SPATIAL:
                    report["statistics"].pop(field, None)
        physical[variant] = value
        fits[variant] = [{k: v for k, v in r.items() if k != "seconds"} for r in row["rounds"]]
    _require(
        _same_float_bits(physical["prior"], physical["current"]),
        "whole physical/evidence bits differ",
    )
    _require(_same_float_bits(fits["prior"], fits["current"]), "whole full-fit bits differ")
    expected = {
        "seconds": seconds,
        "whole_ratio": seconds["prior"] / seconds["current"],
        "physical_bits_equal": True,
        "all_four_full_fit_bits_equal": True,
        "protocol_sha256": _PROTOCOLS["whole"],
        "measurements_sha256": digests,
    }
    _require(
        _same_float_bits(expected, _read(whole, "summary.json")), "whole original summary differs"
    )
    progress = [{"variant": v, "whole_seconds": seconds[v]} for v in ("prior", "current")]
    _require(
        _same_float_bits(
            [*progress, expected],
            [
                json.loads(line)
                for line in (whole / "supervisor.stdout.log")
                .read_text(encoding="utf-8")
                .splitlines()
            ],
        )
        and (whole / "supervisor.stderr.log").read_bytes() == b"",
        "whole supervisor differs",
    )
    # The candidate validator changes only the required spatial-build counter.
    original = (whole / "helpers/benchmark.py.txt").read_text(encoding="utf-8")
    node = next(
        n
        for n in ast.parse(original).body
        if isinstance(n, ast.FunctionDef) and n.name == "_validate_resident_execution"
    )
    validator = ast.get_source_segment(original, node)
    assert validator is not None
    validator = validator.replace(
        'statistics.get("gpu_grid_captures", 0) + statistics.get(\n'
        '                "gpu_grid_native_fallbacks", 0,\n            )',
        'statistics.get("device_tree_build_kernel_calls", 0)',
    )
    _require(
        ast.dump(ast.parse(validator))
        == ast.dump(
            ast.parse((whole / "current/candidate-resident-validator.py.txt").read_bytes())
        ),
        "candidate validator changed beyond spatial counter",
    )
    diagnostic = root / "diagnostic"
    profile = _read(diagnostic, "summary.json")
    provenance = _read(diagnostic, "provenance.json")
    _require(
        provenance["source_commit"] == _COMMIT
        and provenance["source_manifest_sha256"] == manifest
        and provenance["launcher_sha256"] == _sha((diagnostic / "launcher.py.txt").read_bytes())
        and profile["provenance_sha256"] == _sha((diagnostic / "provenance.json").read_bytes())
        and profile["measurement_sha256"] == _sha((diagnostic / "measurement.json").read_bytes())
        and profile["result_sha256"] == _sha((diagnostic / "result.json").read_bytes())
        and profile["calls"]["odometry_add"] == 456,
        "diagnostic provenance/scope differs",
    )
    LidarImuBagCalibrationArtifact.model_validate(_read(diagnostic, "result.json"))
    _require(
        len(_read(diagnostic, "measurement.json")["rounds"]) == 4,
        "diagnostic did not complete full fits",
    )
    report = {
        "schema_version": "calibrex.lidar_pipeline_followup_audit/v0.1",
        "decision": "no_candidate_adopted",
        "timed_registrations": 480,
        "retained_warmups": 10,
        "registration_bit_comparisons": exact_comparisons,
        "registration_tolerance_comparisons": tolerant_comparisons,
        "complete_workers": 2,
        "complete_fits": 8,
        "diagnostic_fits": 4,
        "whole": expected,
        "cohorts": cohorts,
        "gpu_runtime_imported": "cupy" in sys.modules,
        "provenance": {
            "source_commit": _COMMIT,
            "source_manifest_sha256": manifest,
            "archive_index_sha256": _sha((root / "archive-index.json").read_bytes()),
            "auditor_sha256": _sha((root / "auditor.py.txt").read_bytes()),
        },
    }
    jsonschema.validate(report, _read(root, "audit.schema.json"))
    return report


def main(argv: list[str] | None = None) -> int:
    """Audit a frozen follow-up and optionally save/check its report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    report = audit_archive(args.archive)
    if args.check:
        _require(_same_float_bits(report, _read(args.archive, "audit.json")), "saved audit differs")
    if args.write:
        (args.archive / "audit.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
