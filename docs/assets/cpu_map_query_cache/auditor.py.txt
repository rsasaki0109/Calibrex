"""Replay CPU map-query evidence without importing CUDA or reading raw scans."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import sys
import tarfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

import jsonschema
from tools.audit_resident_lidar_geometry import _sources
from tools.benchmark_bspline_lidar_imu import _same_float_bits

from calibrex.core.lidar_lidar_extrinsic import LidarLidarExtrinsicArtifact

_COMMITS = {
    "prior": "96871f64588a122e33e01c713e6710f0f0f36d71",
    "lazy": "09f27193c9418ec9b6584caea459661e047fe875",
    "current": "eeb6fddcb4fbdcee069f569df56c762d1815afce",
}
_COHORTS = {
    "lazy-v1": {"prior": ("prior", 1), "lazy": ("lazy", 1), "parallel": ("lazy", 8)},
    "certified-v2": {"prior": ("prior", 1), "lazy": ("lazy", 8), "certified": ("current", 8)},
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read(root: Path, name: str) -> Any:
    return json.loads((root / name).read_bytes())


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _delta(root: Path, prefix: str, prior: dict[str, bytes], changed: set[str]) -> dict[str, bytes]:
    current = dict(prior)
    seen = set()
    total = 0
    with tarfile.open(root / (prefix + ".source-delta.tar.gz"), "r:gz") as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            key = member.name.removeprefix("src/")
            total += member.size
            _require(
                member.isfile()
                and not path.is_absolute()
                and ".." not in path.parts
                and key in changed
                and key not in seen
                and total <= 2 * 1024**2,
                "unexpected source delta member",
            )
            stream = archive.extractfile(member)
            assert stream is not None
            current[key] = stream.read()
            seen.add(key)
    _require(seen == changed, "incomplete source delta")
    return current


def _physical(record: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(record["artifact"])
    for key in ("created_at", "git_commit"):
        value["provenance"].pop(key)
    value["options"]["map_options"].pop("query_workers", None)
    return {
        "artifact": value,
        "evaluation": record["evaluation"],
        "full_fits": record["full_fits"],
        **{
            key: record[key]
            for key in (
                "prepared_sha256",
                "partition_sha256",
                "map_points",
                "target_points",
            )
        },
    }


def audit_archive(root: Path) -> dict[str, Any]:
    """Check source/input/work bindings and recompute both complete cohorts."""
    root = root.resolve()
    index = _read(root, "archive-index.json")
    files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    _require(
        set(index["files"]) == files - {"archive-index.json", "audit.json"},
        "archive payload coverage differs",
    )
    for name, digest in index["files"].items():
        path = PurePosixPath(name)
        _require(
            not path.is_absolute() and ".." not in path.parts and ":" not in name,
            "archive path escapes publication",
        )
        _require(_sha((root / name).read_bytes()) == digest, "payload checksum differs: " + name)
    _require(
        index["schema_version"] == "calibrex.cpu_map_query_cache_archive/v0.1",
        "archive version differs",
    )
    dependency = root.parent / "resident_scan_validation"
    dependency_names = {"current.sources.tar.gz", "current.sources-manifest.json"}
    _require(set(index["source_dependency"]) == dependency_names, "source dependency differs")
    for name, digest in index["source_dependency"].items():
        _require(_sha((dependency / name).read_bytes()) == digest, "source dependency hash differs")
    sources = {"prior": _sources(dependency, "current")}
    sources["lazy"] = _delta(
        root,
        "lazy",
        sources["prior"],
        {
            "calibrex/cli/main.py",
            "calibrex/solvers/lidar_lidar_map_solver.py",
        },
    )
    sources["current"] = _delta(
        root,
        "current",
        sources["lazy"],
        {
            "calibrex/evaluation/lidar_lidar_map.py",
            "calibrex/solvers/lidar_lidar_map_solver.py",
        },
    )
    for kind, tree in sources.items():
        _require(len(tree) == 396, "source module coverage differs")
        _require(
            {name: _sha(raw) for name, raw in tree.items()}
            == _read(root, kind + ".sources-manifest.json"),
            "complete source manifest differs",
        )
    summaries = {}
    inputs = None
    fit_comparisons = 0
    for cohort, variants in _COHORTS.items():
        base = root / cohort
        protocol = _read(base, "protocol.json")
        names = tuple(variants)
        order = [list(names[i:] + names[:i]) for i in range(3)]
        expected_commits = {
            "prior_commit": _COMMITS["prior"],
            "current_commit": _COMMITS["lazy" if cohort == "lazy-v1" else "current"],
        }
        if cohort == "certified-v2":
            expected_commits["lazy_commit"] = _COMMITS["lazy"]
        _require(
            all(protocol[key] == value for key, value in expected_commits.items()),
            "predeclared commits differ",
        )
        _require(
            protocol["order"] == order
            and protocol["repeats"] == 3
            and protocol["variants"] == list(names)
            and protocol["query_workers"] == {v: workers for v, (_, workers) in variants.items()}
            and protocol["half_window"] == 1
            and protocol["fits_per_worker"] == 5
            and protocol["jackknife_groups"] == 4
            and protocol["target_noise_seed"] == 20261012
            and protocol["target_noise_std_m"] == 0.01,
            "predeclared workload differs",
        )
        gates = {"all_physical_and_evidence_float_bits_equal": True}
        if cohort == "lazy-v1":
            gates["prior_to_parallel_median_at_least"] = 2.0
        else:
            gates.update(prior_to_certified_median_at_least=2.0, lazy_to_certified_median_above=1.0)
            _require(
                protocol["query_cache"]
                == {
                    "min_rows": 4096,
                    "max_retained_bytes_per_sample": 8388608,
                    "bytes_per_row": 40,
                    "margin_machine_eps": 256,
                },
                "query certificate bounds differ",
            )
        _require(protocol["adoption_gate"] == gates, "predeclared gates differ")
        _require(
            _sha((base / "launcher.py.txt").read_bytes()) == protocol["launcher_sha256"],
            "frozen launcher differs",
        )
        for helper, digest in protocol["helpers_sha256"].items():
            _require(
                _sha((root / (Path(helper).name + ".txt")).read_bytes()) == digest,
                "frozen helper differs",
            )
        if inputs is None:
            inputs = protocol["input_sha256"]
            _require(
                len(inputs) == 9 and all("2011_09_30_drive_0018_sync/" in key for key in inputs),
                "public input coverage differs",
            )
        _require(protocol["input_sha256"] == inputs, "cohort input identities differ")
        walls: dict[str, list[float]] = {variant: [] for variant in variants}
        for repeat in range(3):
            records = {}
            for variant, (kind, budget) in variants.items():
                folder = base / f"{repeat:02d}-{variant}"
                record = _read(folder, "measurement.json")
                result = _read(folder, "result.json")
                LidarLidarExtrinsicArtifact.model_validate(result)
                _require(_same_float_bits(result, record["artifact"]), "saved artifact differs")
                _require(
                    record["result_sha256"] == _sha((folder / "result.json").read_bytes()),
                    "saved result hash differs",
                )
                artifact = record["artifact"]
                _require(
                    artifact["provenance"]["git_commit"] == _COMMITS[kind],
                    "artifact commit differs",
                )
                _require(
                    datetime.fromisoformat(
                        artifact["provenance"]["created_at"].replace("Z", "+00:00"),
                    ).tzinfo
                    is not None,
                    "creation time is not timezone aware",
                )
                _require(
                    artifact["options"]["map_options"].get("query_workers", 1) == budget,
                    "serialized worker budget differs",
                )
                _require(
                    len(record["full_fits"]) == 5
                    and artifact["jackknife_fits"] == 4
                    and len(artifact["dofs"]) == 6
                    and all(row.get("known_bad_control") is not None for row in artifact["dofs"])
                    and artifact["samples"]["train_samples"] == 4
                    and artifact["samples"]["holdout_samples"] == 2,
                    "complete fit/evidence coverage differs",
                )
                prov = record["provenance"]
                _require(
                    prov["source_commit"] == _COMMITS[kind]
                    and prov["source_manifest_sha256"]
                    == _sha((root / (kind + ".sources-manifest.json")).read_bytes())
                    and prov["source_manifest_sha256"]
                    == protocol["source_manifests_sha256"][variant]
                    and prov["launcher_sha256"] == protocol["launcher_sha256"]
                    and prov["protocol_sha256"] == _sha((base / "protocol.json").read_bytes())
                    and prov["helpers_sha256"] == protocol["helpers_sha256"]
                    and set(prov["threads"].values()) == {"1"},
                    "worker provenance differs",
                )
                _require(
                    prov["executed_sources_sha256"]
                    and all(
                        name in sources[kind] and _sha(sources[kind][name]) == digest
                        for name, digest in prov["executed_sources_sha256"].items()
                    ),
                    "executed source differs",
                )
                for path, digest in prov["source_sha256"].items():
                    if path.startswith("src/"):
                        _require(
                            path[4:] in sources[kind] and _sha(sources[kind][path[4:]]) == digest,
                            "workload source hash differs",
                        )
                    else:
                        _require(
                            protocol["helpers_sha256"].get(path) == digest,
                            "workload helper hash differs",
                        )
                _require(
                    {key.replace("\\", "/"): value for key, value in prov["input_sha256"].items()}
                    == inputs,
                    "worker inputs differ",
                )
                work = record["work"]
                _require(
                    work["gpu_runtime_imported"] is False
                    and work["requested_workers"] == budget
                    and {row["workers"] for row in work["queries"]} == {budget},
                    "worker execution budget differs",
                )
                for key, label in (
                    ("normal_rows", "normal"),
                    ("correspondence_rows", "correspondence"),
                ):
                    _require(
                        work[key]
                        == sum(row["rows"] for row in work["queries"] if row["kind"] == label),
                        "work counters differ",
                    )
                _require(
                    work["normal_rows"] == (2286890 if kind == "prior" else 1247324),
                    "normal work coverage differs",
                )
                if cohort == "certified-v2":
                    _require(work["projected_rows"] == 9782926, "projection work differs")
                    if variant == "certified":
                        _require(
                            work["certified_rows"] > 0
                            and work["cached_query_rows"] == work["correspondence_rows"]
                            and work["certified_rows"]
                            + work["cached_query_rows"]
                            - work["native_tie_rows"]
                            == work["projected_rows"],
                            "certified query coverage differs",
                        )
                whole = _read(folder, "whole.json")
                _require(
                    whole["returncode"] == 0
                    and whole["wall_seconds"] > 0
                    and whole["launcher_sha256"] == protocol["launcher_sha256"]
                    and whole["protocol_sha256"] == prov["protocol_sha256"],
                    "whole-process receipt differs",
                )
                _require(
                    all(not (folder / name).read_bytes() for name in ("stdout.log", "stderr.log")),
                    "worker diagnostics are not empty",
                )
                walls[variant].append(whole["wall_seconds"])
                records[variant] = record
            for variant in names[1:]:
                _require(
                    _same_float_bits(_physical(records["prior"]), _physical(records[variant])),
                    "physical result, covariance or evidence bits differ",
                )
                fit_comparisons += 5
            triad = _read(base, f"triad-{repeat:02d}.json")
            expected_errors = {
                v: ([".artifact.provenance.created_at"] if cohort == "lazy-v1" else [])
                for v in names[1:]
            }
            _require(
                triad["repeat"] == repeat
                and triad["errors"] == expected_errors
                and triad["whole_seconds"] == {v: walls[v][-1] for v in names}
                and triad["measurement_sha256"]
                == {
                    v: _sha((base / f"{repeat:02d}-{v}/measurement.json").read_bytes())
                    for v in names
                },
                "original triad record differs",
            )
        medians = {variant: statistics.median(values) for variant, values in walls.items()}
        last = names[-1]
        speedup = medians["prior"] / medians[last]
        incremental = medians["lazy"] / medians[last]
        original = _read(base, "summary.json")
        _require(
            original["median_seconds"] == medians
            and original["complete_workers"] == 9
            and original["complete_fits"] == 45
            and original["protocol_sha256"] == _sha((base / "protocol.json").read_bytes())
            and original["triads_sha256"]
            == [_sha((base / f"triad-{i:02d}.json").read_bytes()) for i in range(3)]
            and original["prior_to_lazy"] == medians["prior"] / medians["lazy"],
            "original summary differs",
        )
        advance = speedup >= 2.0 and (cohort == "lazy-v1" or incremental > 1.0)
        _require(
            original["prior_to_parallel" if cohort == "lazy-v1" else "prior_to_certified"]
            == speedup
            and original["quality_passed"] == (cohort != "lazy-v1")
            and original["advance"] == (advance and cohort != "lazy-v1"),
            "original decision differs",
        )
        if cohort == "certified-v2":
            _require(original["lazy_to_certified"] == incremental, "incremental ratio differs")
        summaries[cohort] = {
            "median_seconds": medians,
            "prior_to_candidate": speedup,
            "lazy_to_candidate": incremental,
            "physical_bit_parity": True,
            "original_quality_passed": original["quality_passed"],
            "metadata_only_comparator_correction": cohort == "lazy-v1",
            "advance": advance,
        }
    report = {
        "schema_version": "calibrex.cpu_map_query_cache_audit/v0.1",
        "complete_workers": 18,
        "complete_fits": 90,
        "fit_bit_comparisons": fit_comparisons,
        "cohorts": summaries,
        "gpu_runtime_imported": "cupy" in sys.modules,
        "provenance": {
            "archive_index_sha256": _sha((root / "archive-index.json").read_bytes()),
            "auditor_sha256": _sha(Path(__file__).read_bytes()),
            "commits": _COMMITS,
        },
    }
    schema = Path(__file__).with_name("schemas") / "cpu_map_query_cache_audit.schema.json"
    jsonschema.validate(report, json.loads(schema.read_bytes()))
    _require(
        (root / "auditor.py.txt").read_bytes() == Path(__file__).read_bytes(),
        "frozen auditor differs",
    )
    _require(
        (root / "audit.schema.json").read_bytes() == schema.read_bytes(), "audit schema differs"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write", action="store_true")
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
