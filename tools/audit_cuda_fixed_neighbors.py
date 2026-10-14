"""CPU-only audit of frozen complete public map-evaluation comparisons."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import pstats
import re
import statistics
import tarfile
from pathlib import Path
from typing import Any

import jsonschema
from tools.benchmark_cuda_fixed_neighbors import _quality

from calibrex.core.lidar_imu_bag_calibration import LidarImuBagCalibrationArtifact
from calibrex.core.lidar_lidar_extrinsic import LidarLidarExtrinsicArtifact


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root: Path) -> dict[str, Any]:
    """Recompute complete quality, timings, actual work and source bindings."""
    index = json.loads((root / "publication-index.json").read_bytes())
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != set(index["files"]) | {"publication-index.json"}:
        raise ValueError("publication does not bind every file")
    if any(_sha(root / name) != digest for name, digest in index["files"].items()):
        raise ValueError("publication file digest differs")
    schema = json.loads((root / "comparison.schema.json").read_bytes())
    jsonschema.validate(index, schema)
    protocol = json.loads((root / "protocol.json").read_bytes())
    summary = json.loads((root / "summary.json").read_bytes())
    for record in (protocol, summary):
        jsonschema.validate(record, schema)
    if (
        index["provenance"]["protocol_sha256"] != _sha(root / "protocol.json")
        or summary["provenance"]["protocol_sha256"] != _sha(root / "protocol.json")
        or summary["provenance"]["attempts_sha256"] != _sha(root / "attempts.json")
        or protocol["launcher_sha256"] != _sha(root / "benchmark.py.txt")
        or protocol["map_launcher_sha256"] != _sha(root / "map-benchmark.py.txt")
    ):
        raise ValueError("protocol or launcher binding differs")
    manifests, shaders, raw_sources, features, compacted = {}, {}, {}, {}, {}
    for variant in ("prior", "current"):
        manifest = json.loads((root / f"sources-{variant}-manifest.json").read_bytes())
        archived = {}
        with tarfile.open(root / f"sources-{variant}.tar.gz") as bundle:
            for member in bundle.getmembers():
                if not member.isfile() or not member.name.endswith(".py"):
                    continue
                if not member.name.startswith("src/calibrex/") or ".." in Path(member.name).parts:
                    raise ValueError("unsafe source archive member")
                stream = bundle.extractfile(member)
                assert stream is not None
                source = stream.read()
                name = member.name.removeprefix("src/")
                archived[name] = hashlib.sha256(source).hexdigest()
                if variant == "current":
                    raw_sources[name] = source
                if name == "calibrex/solvers/cuda_lidar_map_adapter.py":
                    parsed = ast.parse(source)
                    constants = {
                        target.id: node.value.value
                        for node in parsed.body
                        if isinstance(node, ast.Assign)
                        and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, str)
                        for target in node.targets
                        if isinstance(target, ast.Name)
                    }
                    shader = constants["_SOURCE"] + constants.get("_MAP_SOURCE", "")
                    compacted[variant] = "_MAP_SOURCE" in constants
                    features[variant] = {
                        node.value
                        for node in ast.walk(parsed)
                        if isinstance(node, ast.Constant) and isinstance(node.value, str)
                    }
                    shaders[variant] = hashlib.sha256(shader.encode()).hexdigest()
        if archived != manifest:
            raise ValueError("frozen source archive differs from manifest")
        manifests[variant] = manifest
    attempts = json.loads((root / "attempts.json").read_bytes())
    expected = {(repeat, variant) for repeat in range(protocol["repeats"]) for variant in manifests}
    if (
        len(attempts) != len(expected)
        or {(row["repeat"], row["variant"]) for row in attempts} != expected
    ):
        raise ValueError("missing or repeated complete worker")
    measurements = {}
    for attempt in attempts:
        if attempt["exit_code"] != 0:
            raise ValueError("failed worker retained")
        folder, variant = root / attempt["worker"], attempt["variant"]
        row = json.loads((folder / "measurement.json").read_bytes())
        jsonschema.validate(row, schema)
        artifact = LidarLidarExtrinsicArtifact.model_validate_json(
            (folder / "result.json").read_bytes(),
        )
        if (
            row["result_sha256"] != _sha(folder / "result.json")
            or row["artifact"] != artifact.model_dump(mode="json", exclude_none=True)
            or artifact.provenance.git_commit != protocol["commits"][variant]
        ):
            raise ValueError("complete artifact or measured commit differs")
        binding = row["frozen_provenance"]
        if (
            binding["source_commit"] != protocol["commits"][variant]
            or binding["source_manifest_sha256"] != _sha(root / f"sources-{variant}-manifest.json")
            or binding["launcher_sha256"] != protocol["launcher_sha256"]
            or binding["map_launcher_sha256"] != protocol["map_launcher_sha256"]
            or not binding["cuda_cache_initially_empty"]
            or set(binding["threads"].values()) != {"1"}
            or set(binding["threads"])
            != {
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            }
        ):
            raise ValueError("frozen execution policy or source binding differs")
        imported = binding["executed_sources"]
        if (
            "calibrex/solvers/cuda_lidar_map_adapter.py" not in imported
            or any(manifests[variant].get(name) != digest for name, digest in imported.items())
            or any(
                manifests[variant].get(name.removeprefix("src/")) != digest
                for name, digest in row["provenance"]["source_sha256"].items()
                if name.startswith("src/")
            )
            or row["provenance"]["source_sha256"].get("tools/benchmark_cuda_lidar_map.py")
            != protocol["map_launcher_sha256"]
            or row["provenance"]["input_sha256"] != protocol["input_sha256"]
        ):
            raise ValueError("actual imports or input bindings differ")
        identity = row["evaluation"]["result"]["execution_backend"]
        if (
            re.fullmatch(
                r"calibrex\.cuda_lidar_map/v0\.1;fp64=true;query_fma=false;"
                r"cupy=[^;]+;cuda=\d+;driver=\d+;cc=\d+\.\d+;" + "query_sha256=" + shaders[variant],
                identity,
            )
            is None
        ):
            raise ValueError("executed query shader differs")
        counters = row["statistics"]
        if (
            counters["prepared_samples"] != 6
            or counters["gpu_tree_builds"] != 6
            or counters["gpu_tree_build_kernel_calls"] != (30 if compacted[variant] else 18)
            or counters["gpu_query_calls"] <= 0
            or counters["gpu_query_rows"] < sum(row["map_points"])
            or row["map_points"] != [127158, 127158, 127303, 127303, 126598, 126598]
            or row["target_points"] != row["map_points"]
        ):
            raise ValueError("complete public query work differs")
        if "gpu_fixed_query_calls" in features[variant] and (
            counters.get("gpu_fixed_query_calls") != counters["gpu_query_calls"]
            or counters.get("gpu_fixed_query_rows") != counters["gpu_query_rows"]
        ):
            raise ValueError("specialized query accounting differs")
        if "gpu_map_fused_kernel_calls" in features[variant] and (
            counters.get("gpu_map_fused_kernel_calls", 0) <= 0
            or counters.get("gpu_map_fused_residual_rows", 0)
            != counters["gpu_query_rows"] - sum(row["map_points"])
            or not 0
            < counters.get("gpu_map_fused_system_rows", 0)
            <= counters["gpu_map_fused_residual_rows"]
        ):
            raise ValueError("fused map execution accounting differs")
        if "nvrtc_compilation" in binding:
            compilation = binding["nvrtc_compilation"]
            if compilation["calls"] <= 0 or not 0 < compilation["seconds"] <= row["api_seconds"]:
                raise ValueError("compilation timing scope differs")
        if not 0 < row["api_seconds"] <= attempt["whole_seconds"]:
            raise ValueError("timing scope differs")
        measurements[(attempt["repeat"], variant)] = row
    quality = [
        _quality(measurements[(repeat, "prior")], measurements[(repeat, "current")])
        for repeat in range(protocol["repeats"])
    ]
    if not all(quality) or quality != summary["quality_exact"] or not summary["quality_passed"]:
        raise ValueError("complete evaluation or controls differ")
    for scope in ("whole", "api"):
        medians = {
            variant: statistics.median(
                attempt["whole_seconds"]
                if scope == "whole"
                else measurements[(attempt["repeat"], variant)]["api_seconds"]
                for attempt in attempts
                if attempt["variant"] == variant
            )
            for variant in manifests
        }
        if (
            medians != summary[scope + "_median_seconds"]
            or medians["prior"] / medians["current"] != summary[scope + "_speedup"]
        ):
            raise ValueError("median or speed ratio differs")
    observed = json.loads((root / "observation.json").read_bytes())
    jsonschema.validate(observed, json.loads((root / "observation.schema.json").read_bytes()))
    binding = observed["provenance"]
    command = binding["benchmark_command"]
    if (
        binding["observer_sha256"] != _sha(root / "observer.py.txt")
        or binding["schema_sha256"] != _sha(root / "observation.schema.json")
        or binding["launcher_sha256"] != protocol["launcher_sha256"]
        or observed["exit_code"] != 0
        or binding["terminated_on_interference"]
        or command[command.index("--baseline-commit") + 1] != protocol["commits"]["prior"]
        or int(command[command.index("--repeats") + 1]) != protocol["repeats"]
        or binding["sampling_interval_seconds"] != 10
    ):
        raise ValueError("GPU observer binding differs")
    samples = observed["samples"]
    if [row["offset_seconds"] for row in samples] != sorted(
        row["offset_seconds"] for row in samples
    ) or any(row["foreign"] or row["unknown"] for row in samples):
        raise ValueError("GPU interference was observed")
    profile = root / "cold-profile"
    if profile.exists():
        report = json.loads((profile / "summary.json").read_bytes())
        jsonschema.validate(report, json.loads((profile / "summary.schema.json").read_bytes()))
        measurement = json.loads((profile / "measurement.json").read_bytes())
        binding = report["provenance"]
        if (
            binding["profile_sha256"] != _sha(profile / "profile.pstats")
            or binding["measurement_sha256"] != _sha(profile / "measurement.json")
            or binding["helper_sha256"] != _sha(profile / "helper.py.txt")
            or measurement["provenance"]["helper_sha256"] != binding["helper_sha256"]
            or measurement["provenance"]["pca_shader_sha256"]
            != hashlib.sha256(
                (profile / "pca.cu.txt").read_text(encoding="utf-8").encode(),
            ).hexdigest()
            or measurement["mode"] != "prior"
        ):
            raise ValueError("cold profile binding differs")
        for module, digest in measurement["provenance"]["executed_sources"].items():
            prefix = module.replace(".", "/")
            name = prefix + ".py" if prefix + ".py" in raw_sources else prefix + "/__init__.py"
            override = profile / "sources" / name
            contents = override.read_bytes() if override.exists() else raw_sources[name]
            if hashlib.sha256(contents).hexdigest() != digest:
                raise ValueError("cold profile imported source differs")
            if contents.replace(b"\r\n", b"\n") != raw_sources[name].replace(b"\r\n", b"\n"):
                raise ValueError("cold profile changed measured runtime code")
        stats = pstats.Stats(str(profile / "profile.pstats"))
        key = next(key for key in stats.stats if "nvrtc.compileProgram" in key[2])
        entry = stats.stats[key]
        if (
            report["compiled_programs"] != entry[1]
            or report["nvrtc_seconds"] != entry[2]
            or report["profile_total_seconds"] != stats.total_tt
        ):
            raise ValueError("cold compilation totals differ")
    extra = {}
    if all("nvrtc_compilation" in row["frozen_provenance"] for row in measurements.values()):
        extra["compilation_median_seconds"] = {
            variant: statistics.median(
                measurements[(repeat, variant)]["frozen_provenance"]["nvrtc_compilation"]["seconds"]
                for repeat in range(protocol["repeats"])
            )
            for variant in manifests
        }
        extra["compiled_programs"] = {
            variant: [
                measurements[(repeat, variant)]["frozen_provenance"]["nvrtc_compilation"]["calls"]
                for repeat in range(protocol["repeats"])
            ]
            for variant in manifests
        }
    warm = root / "warm"
    if warm.exists():
        warm_summary = json.loads((warm / "summary.json").read_bytes())
        jsonschema.validate(warm_summary, json.loads((warm / "summary.schema.json").read_bytes()))
        if warm_summary["provenance"]["helper_sha256"] != _sha(warm / "helper.py.txt"):
            raise ValueError("warm launcher binding differs")
        warm_records = {}
        medians = {}
        for variant in manifests:
            report = json.loads((warm / variant / "measurement.json").read_bytes())
            binding = report["provenance"]
            if (
                binding["helper_sha256"] != _sha(warm / "helper.py.txt")
                or binding["source_commit"] != protocol["commits"][variant]
                or binding["source_manifest_sha256"]
                != _sha(root / f"sources-{variant}-manifest.json")
            ):
                raise ValueError("warm source binding differs")
            if (
                len(report["rows"]) != 4
                or [row["repeat"] for row in report["rows"]] != list(range(4))
                or any(row["nvrtc_calls"] != 0 for row in report["rows"][1:])
            ):
                raise ValueError("warm cache execution differs")
            for repeat, row in enumerate(report["rows"]):
                folder = warm / variant / f"{repeat:02d}"
                record = json.loads((folder / "measurement.json").read_bytes())
                model = LidarLidarExtrinsicArtifact.model_validate_json(
                    (folder / "result.json").read_bytes(),
                )
                if (
                    record["result_sha256"] != _sha(folder / "result.json")
                    or record["artifact"] != model.model_dump(mode="json", exclude_none=True)
                    or model.provenance.git_commit != protocol["commits"][variant]
                    or row["api_seconds"] != record["api_seconds"]
                    or row["api_seconds"] > row["whole_call_seconds"]
                    or record["provenance"]["input_sha256"] != protocol["input_sha256"]
                ):
                    raise ValueError("warm complete result or timing differs")
                if (
                    any(
                        manifests[variant].get(name.removeprefix("src/")) != digest
                        for name, digest in record["provenance"]["source_sha256"].items()
                        if name.startswith("src/")
                    )
                    or record["provenance"]["source_sha256"].get(
                        "tools/benchmark_cuda_lidar_map.py"
                    )
                    != protocol["map_launcher_sha256"]
                    or record["evaluation"]["result"]["execution_backend"].split("query_sha256=")[
                        -1
                    ]
                    != shaders[variant]
                ):
                    raise ValueError("warm executed source or shader differs")
                warm_records[(repeat, variant)] = record
            medians[variant] = statistics.median(row["api_seconds"] for row in report["rows"][1:])
            if medians[variant] != report["warm_api_median_seconds"]:
                raise ValueError("warm median differs")
        quality = [
            _quality(warm_records[(repeat, "prior")], warm_records[(repeat, "current")])
            for repeat in range(4)
        ]
        if (
            not all(quality)
            or quality != warm_summary["quality_exact"]
            or medians != warm_summary["warm_api_median_seconds"]
            or medians["prior"] / medians["current"] != warm_summary["warm_api_speedup"]
        ):
            raise ValueError("warm full quality or ratio differs")
        extra["warm_api_speedup"] = warm_summary["warm_api_speedup"]
    joint = root / "joint-validation"
    if joint.exists():
        report = json.loads((joint / "summary.json").read_bytes())
        jsonschema.validate(report, json.loads((joint / "summary.schema.json").read_bytes()))
        if (
            report["provenance"]["launcher_sha256"] != _sha(joint / "benchmark.py.txt")
            or report["provenance"]["current_full_commit"] != protocol["commits"]["current"]
        ):
            raise ValueError("shared-path launcher binding differs")
        records, models = {}, {}
        for variant in manifests:
            folder = joint / variant
            records[variant] = json.loads((folder / "measurement.json").read_bytes())
            models[variant] = LidarImuBagCalibrationArtifact.model_validate_json(
                (folder / "result.json").read_bytes(),
            ).model_dump(mode="json")
            source_manifest = (
                joint / "sources-prior-manifest.json"
                if variant == "prior"
                else root / "sources-current-manifest.json"
            )
            source_hashes = json.loads(source_manifest.read_bytes())
            if variant == "prior":
                archived = {}
                with tarfile.open(joint / "sources-prior.tar.gz") as bundle:
                    for member in bundle.getmembers():
                        if not member.isfile() or not member.name.endswith(".py"):
                            continue
                        if (
                            not member.name.startswith("src/calibrex/")
                            or ".." in Path(member.name).parts
                        ):
                            raise ValueError("shared-path archive member differs")
                        stream = bundle.extractfile(member)
                        assert stream is not None
                        archived[member.name.removeprefix("src/")] = hashlib.sha256(
                            stream.read(),
                        ).hexdigest()
                if archived != source_hashes:
                    raise ValueError("shared-path source archive differs")
            binding = records[variant]["provenance"]
            if (
                binding["result_sha256"] != _sha(folder / "result.json")
                or binding["source_manifest_sha256"] != _sha(source_manifest)
                or any(
                    source_hashes.get(name) != digest
                    for name, digest in binding["executed_sources"].items()
                )
            ):
                raise ValueError("shared-path executed source or result differs")
            if variant == "current" and (
                binding["launcher_sha256"] != _sha(joint / "benchmark.py.txt")
                or not protocol["commits"][variant].startswith(
                    models[variant]["provenance"]["git_commit"]
                )
            ):
                raise ValueError("shared-path measured commit or launcher differs")
        if (
            report["quality_exact"] is not True
            or models["prior"]["controls"] != models["current"]["controls"]
            or len(models["current"]["controls"]) != 32
            or models["prior"]["holdout_whitened_rms"] != models["current"]["holdout_whitened_rms"]
            or len(records["current"]["rounds"]) != 4
        ):
            raise ValueError("shared-path complete evidence differs")
        for a, b in zip(records["prior"]["rounds"], records["current"]["rounds"], strict=True):
            if {key: value for key, value in a.items() if key != "seconds"} != {
                key: value for key, value in b.items() if key != "seconds"
            }:
                raise ValueError("shared-path full fit differs")
        extra["joint_fit_rounds_exact"] = 4
        extra["joint_signed_controls_exact"] = 32
    return {
        "workers": len(attempts),
        "quality_exact": True,
        "whole_speedup": summary["whole_speedup"],
        "api_speedup": summary["api_speedup"],
        "observations": len(samples),
        "foreign_samples": 0,
        "unknown_samples": 0,
        **extra,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    print(json.dumps(audit(parser.parse_args().archive), indent=2))


if __name__ == "__main__":
    main()
