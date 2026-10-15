"""CPU-only audit of the complete public GLIM B-spline acceleration archive."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import statistics
import tarfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import jsonschema
import numpy as np

from calibrex.core.lidar_imu_bag_calibration import LidarImuBagCalibrationArtifact


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _cuda_kernel_hashes(sources: dict[str, bytes]) -> tuple[str, str]:
    """Read bound CUDA string constants without importing an optional GPU runtime."""
    def constants(name: str) -> dict[str, str]:
        result = {}
        for node in ast.parse(sources[name]).body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if isinstance(target, ast.Name) and isinstance(node.value.value, str):
                        result[target.id] = node.value.value
        return result

    map_sources = constants("calibrex/solvers/cuda_lidar_map_adapter.py")
    query = map_sources["_SOURCE"] + map_sources.get("_MAP_SOURCE", "")
    system = constants("calibrex/solvers/cuda_scan_odometry_adapter.py")
    return _sha(query.encode()), _sha(
        (system["_SYSTEM_SOURCE"] + system["_FUSED_SOURCE"] + system["_COVARIANCE_SOURCE"]
         + system.get("_NORMAL_SOURCE", "") + system.get("_PREPROCESS_SOURCE", "")).encode()
    )


def _audit_rejected_pair(root: Path, compare: Callable[[dict, dict], dict]) -> None:
    """Check the retained exploratory failure and its exact imported module bytes."""
    folder = root / "rejected-no-motion-grid"
    if not folder.exists():
        return
    schema = json.loads((folder / "measurement.schema.json").read_bytes())
    results = {}
    helper_hash = _sha((folder / "pilot.py.txt").read_bytes())
    for backend in ("cpu", "cuda"):
        raw = (folder / f"{backend}.result.json").read_bytes()
        result = json.loads(raw)
        LidarImuBagCalibrationArtifact.model_validate(result)
        record = json.loads((folder / f"{backend}.measurement.json").read_bytes())
        jsonschema.validate(record, schema)
        provenance = record["provenance"]
        if (provenance["helper_sha256"] != helper_hash
                or provenance["result_sha256"] != _sha(raw)):
            raise ValueError("rejected pilot helper/result binding differs")
        expected = json.loads((folder / f"{backend}.sources-manifest.json").read_bytes())
        archived = {}
        with tarfile.open(folder / f"{backend}.sources.tar.gz") as bundle:
            for member in bundle.getmembers():
                if not member.isfile() or not member.name.endswith(".py"):
                    continue
                if not member.name.startswith("src/calibrex/") or ".." in Path(member.name).parts:
                    raise ValueError("unexpected rejected source archive path")
                stream = bundle.extractfile(member)
                assert stream is not None
                archived[member.name.removeprefix("src/")] = _sha(stream.read())
        if archived != expected:
            raise ValueError("rejected imported source bytes differ")
        for name, digest in provenance["executed_sources"].items():
            path = name.replace(".", "/")
            key = path + ".py" if path + ".py" in expected else path + "/__init__.py"
            if expected.get(key) != digest:
                raise ValueError("rejected actual imports differ from archived sources")
        for fitted, reported in zip(record["rounds"], result["rounds"], strict=True):
            if (fitted["status"] != reported["fit_status"]
                    or fitted["iterations"] != reported["iterations"]):
                raise ValueError("rejected per-round record differs")
        settings = result["options"]["odometry_options"]
        if any(settings.get(key, 0) != 0 for key in
               ("motion_translation_quantum_m", "motion_rotation_quantum_rad")):
            raise ValueError("rejected pair did not retain uncanonicalized motion")
        results[backend] = result
    actual = compare(results["cpu"], results["cuda"])
    declared = json.loads((folder / "comparison.json").read_bytes())
    bindings = declared.pop("provenance")
    for backend in results:
        if bindings[f"{backend}_result_sha256"] != _sha(
            (folder / f"{backend}.result.json").read_bytes()
        ):
            raise ValueError("rejected comparison result binding differs")
    actual_rotation = actual["errors"].pop("rotation_deg")
    declared_rotation = declared["errors"].pop("rotation_deg")
    if (actual["passed"] or actual != declared
            or not np.isclose(actual_rotation, declared_rotation, rtol=1e-12, atol=1e-12)):
        raise ValueError("retained exploratory rejection changed")


def audit(root: Path) -> dict[str, object]:
    """Validate digests, complete models, frozen imports, quality and speed ratios."""
    index = json.loads((root / "publication-index.json").read_bytes())
    index_schema = Path(__file__).with_name("schemas") / "bspline_lidar_imu_publication.schema.json"
    jsonschema.validate(index, json.loads(index_schema.read_bytes()))
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != set(index["files"]) | {"publication-index.json"}:
        raise ValueError("publication index does not bind every archive file")
    for relative, digest in index["files"].items():
        if _sha((root / relative).read_bytes()) != digest:
            raise ValueError(f"archive digest differs: {relative}")
    schema = json.loads((root / "comparison.schema.json").read_bytes())
    protocol = json.loads((root / "protocol.json").read_bytes())
    if index["provenance"]["input_protocol_sha256"] != _sha((root / "protocol.json").read_bytes()):
        raise ValueError("publication input binding differs")
    summary = json.loads((root / "summary.json").read_bytes())
    for record in (protocol, summary):
        jsonschema.validate(record, schema)
    launcher = root / "benchmark.py.txt"
    if _sha(launcher.read_bytes()) != protocol["launcher_sha256"]:
        raise ValueError("frozen comparison launcher differs")
    # Execute the bound comparison definition without creating cache files in
    # the immutable evidence archive.
    module = ModuleType("_frozen_bspline_comparison")
    module.__file__ = str(launcher)
    exec(compile(launcher.read_bytes(), str(launcher), "exec"), module.__dict__)
    fit_compare = None
    fit_options = protocol.get("fit_options")
    if fit_options and fit_options["prior_backend"] != fit_options["current_backend"]:
        fit_helper = root / "fit-quality.py.txt"
        if _sha(fit_helper.read_bytes()) != fit_options["quality_helper_sha256"]:
            raise ValueError("fitted-array comparison helper binding differs")
        fit_module = ModuleType("_frozen_fit_quality")
        fit_module.__file__ = str(fit_helper)
        exec(compile(fit_helper.read_bytes(), str(fit_helper), "exec"), fit_module.__dict__)
        fit_compare = fit_module._compare
    sources, source_bytes = {}, {}
    for variant in ("prior", "current"):
        expected = json.loads((root / f"sources-{variant}-manifest.json").read_bytes())
        with tarfile.open(root / f"sources-{variant}.tar.gz") as bundle:
            archived = {}
            raw_sources = {}
            for member in bundle.getmembers():
                if not member.isfile() or not member.name.endswith(".py"):
                    continue
                if not member.name.startswith("src/calibrex/") or ".." in Path(member.name).parts:
                    raise ValueError("unexpected frozen source archive path")
                stream = bundle.extractfile(member)
                assert stream is not None
                contents = stream.read()
                name = member.name.removeprefix("src/")
                archived[name] = _sha(contents)
                raw_sources[name] = contents
        if archived != expected:
            raise ValueError(f"frozen {variant} source tree differs")
        sources[variant] = expected
        source_bytes[variant] = raw_sources
    gpu_comparison = protocol.get("comparison") in ("cpu-cuda", "cuda-cuda")
    if (protocol.get("comparison") == "cpu-cuda"
            and protocol["baseline_commit"] != protocol["current_commit"]):
        raise ValueError("CPU/CUDA workers must use the same declared source revision")
    attempts = json.loads((root / "attempts.json").read_bytes())
    expected_workers = {(r, v) for r in range(protocol["repeats"]) for v in sources}
    if (
        len(attempts) != len(expected_workers)
        or {(a["repeat"], a["variant"]) for a in attempts} != expected_workers
    ):
        raise ValueError("comparison is missing or duplicating a complete worker")
    results, measurements = {}, {}
    for attempt in attempts:
        if attempt["exit_code"] != 0:
            raise ValueError("comparison contains a failed worker")
        directory = root / attempt["worker"]
        record = json.loads((directory / "measurement.json").read_bytes())
        jsonschema.validate(record, schema)
        variant = attempt["variant"]
        provenance = record["provenance"]
        raw_result = (directory / "result.json").read_bytes()
        result = json.loads(raw_result)
        LidarImuBagCalibrationArtifact.model_validate(result)
        expected_commit = protocol["baseline_commit" if variant == "prior" else "current_commit"]
        if result["provenance"]["git_commit"] != expected_commit:
            raise ValueError("result does not name the measured source commit")
        if (
            record["variant"] != variant
            or provenance["result_sha256"] != _sha(raw_result)
            or provenance["bag_sha256"] != protocol["bag_sha256"]
            or provenance["launcher_sha256"] != protocol["launcher_sha256"]
            or provenance["source_manifest_sha256"]
            != _sha((root / f"sources-{variant}-manifest.json").read_bytes())
        ):
            raise ValueError("worker source/input/result binding differs")
        if set(provenance["blas_threads"].values()) != {"1"}:
            raise ValueError("worker did not use the declared BLAS thread budget")
        executed = provenance["executed_sources"]
        if "calibrex/core/continuous_time_bspline_fit.py" not in executed:
            raise ValueError("worker did not bind the executed fitter")
        if any(sources[variant].get(name) != digest for name, digest in executed.items()):
            raise ValueError("actual worker imports differ from frozen sources")
        if fit_options:
            requested = fit_options[variant + "_backend"]
            module._validate_fit_execution(result, record["rounds"], requested)
            if (provenance.get("fit_backend") != requested
                or (requested == "numba_cpu" and (
                    provenance.get("numba_cache_initially_empty") is not True
                    or provenance.get("fit_workers") != fit_options["workers"]
                    or not provenance.get("numba")
                    or "calibrex/solvers/numba_bspline_plane_adapter.py" not in executed))):
                raise ValueError("plane fitting backend/cache/import binding differs")
        if gpu_comparison:
            backend = ("cuda" if protocol["comparison"] == "cuda-cuda" or variant == "current"
                       else "cpu")
            module._validate_backend_execution(result, backend)
            if (provenance.get("odometry_backend") != backend
                    or provenance.get("cuda_cache_initially_empty") is not True):
                raise ValueError("worker backend or isolated cold compilation cache differs")
            if any(result["options"]["odometry_options"].get(key) != value
                   for key, value in protocol["odometry_options"].items()):
                raise ValueError("worker did not use the declared common geometry settings")
            adapter = "calibrex/solvers/cuda_scan_odometry_adapter.py"
            if backend == "cpu":
                if adapter in executed or provenance.get("cupy") is not None:
                    raise ValueError("CPU worker imported the optional GPU runtime")
            else:
                if adapter not in executed:
                    raise ValueError("GPU worker did not bind its executed adapter")
                query_hash, system_hash = _cuda_kernel_hashes(source_bytes[variant])
                normal_counter_expected = any(
                    isinstance(node, ast.Constant) and node.value == "gpu_normal_kernel_calls"
                    for node in ast.walk(ast.parse(source_bytes[variant][adapter]))
                )
                adapter_constants = {node.value for node in ast.walk(
                    ast.parse(source_bytes[variant][adapter])) if isinstance(node, ast.Constant)
                    and isinstance(node.value, str)}
                identity = re.compile(
                    r"calibrex\.cuda_scan_odometry/v0\.1;fp64=true;query_fma=false;"
                    r"cupy=([^;]+);cuda=\d+;driver=\d+;cc=\d+\.\d+;"
                    rf"query_sha256={query_hash};system_sha256={system_hash}"
                )
                reports = (result["options"]["initial_odometry_execution"]
                           + result["options"]["segment_odometry_execution"])
                for report in reports:
                    match = identity.fullmatch(report["executed_backends"][0])
                    if match is None or match[1] != provenance.get("cupy"):
                        raise ValueError("executed GPU identity differs from its bound kernels")
                    counters = report["statistics"]
                    if "gpu_fixed_query_calls" in adapter_constants:
                        calls = counters.get("gpu_fixed_query_calls", -1)
                        rows = counters.get("gpu_fixed_query_rows", -1)
                        if (not 0 < calls <= counters.get("gpu_query_calls", 0)
                                or not calls <= rows <= counters.get("gpu_query_rows", 0)
                                or rows < counters.get("gpu_normal_rows", 0)):
                            raise ValueError("GPU fixed-query accounting is inconsistent")
                    if normal_counter_expected and "gpu_normal_kernel_calls" not in counters:
                        raise ValueError("GPU normal kernel accounting is missing")
                    if ("gpu_normal_kernel_calls" in counters
                            and counters.get("gpu_normal_rows", 0) > 0
                            and counters["gpu_normal_kernel_calls"] <= 0):
                        raise ValueError("GPU normal rows have no recorded normal kernel work")
                    if ("gpu_tree_builds" in adapter_constants
                            and (counters.get("gpu_tree_builds", 0) <= 0
                                or counters.get("gpu_tree_build_kernel_calls", 0)
                                < counters["gpu_tree_builds"]
                                or not 0 <= counters.get("native_tree_builds", -1)
                                <= counters["gpu_tree_builds"])):
                        raise ValueError("GPU tree construction accounting is inconsistent")
                    if ("gpu_packed_reference_rows" in adapter_constants
                            and (counters.get("gpu_packed_reference_rows", 0) <= 0
                                 or counters["gpu_packed_reference_rows"]
                                 < counters.get("gpu_normal_rows", 0)
                                 or counters.get("gpu_packed_reference_bytes", 0)
                                 != 24 * counters["gpu_packed_reference_rows"])):
                        raise ValueError("GPU packed reference accounting is inconsistent")
                    if ("gpu_preprocess_calls" in adapter_constants
                            and (counters.get("gpu_preprocess_calls", 0) <= 0
                                or counters.get("gpu_preprocess_kernel_calls", 0)
                                < counters["gpu_preprocess_calls"]
                                or not 0 <= counters.get("gpu_preprocess_output_rows", -1)
                                <= counters.get("gpu_preprocess_input_rows", 0)
                                or counters.get("native_preprocess_calls", -1) < 0)):
                        raise ValueError("GPU preprocessing accounting is inconsistent")
                    if "gpu_gyro_rotation_calls" in adapter_constants:
                        calls = counters.get("gpu_gyro_rotation_calls", -1)
                        rows = counters.get("gpu_gyro_rotation_rows", -1)
                        translations = counters.get("gpu_gyro_translation_calls", -1)
                        repairs = counters.get("native_gyro_point_repairs", -1)
                        if (calls < 0 or rows < calls or bool(calls) != bool(rows)
                                or not calls <= translations
                                <= counters.get("gpu_preprocess_calls", -1)
                                or not 0 <= repairs
                                <= counters.get("gpu_preprocess_input_rows", -1)):
                            raise ValueError("GPU gyro preprocessing accounting is inconsistent")
                    if ("calibrex/solvers/scipy_spatial_snapshot.py" in executed
                            and counters.get("serialized_tree_captures", 0)
                            + counters.get("public_node_tree_captures", 0)
                            + counters.get("gpu_tree_builds", 0) <= 0):
                        raise ValueError("GPU worker did not record its actual tree capture path")
                    if (counters.get("scalar_motion_repairs", 0)
                            > counters.get("scalar_registration_repairs", 0)
                            or counters.get("scalar_registration_repairs", 0)
                            > counters["gpu_fused_system_calls"]):
                        raise ValueError("GPU repair accounting is inconsistent")
        if not (
            attempt["whole_seconds"] >= record["pipeline_seconds"] >= record["fit_seconds"] > 0
        ):
            raise ValueError("timing scopes are inconsistent")
        if (protocol.get("comparison") == "cuda-cuda"
                and hasattr(module, "_time_normal_preparation")
                and not {"normal_preparation_seconds", "normal_preparation_calls"}
                <= record.keys()):
            raise ValueError("normal preparation observation is missing")
        if ("normal_preparation_seconds" in record
                and not (0 <= record["normal_preparation_seconds"] <= record["pipeline_seconds"]
                         and record["normal_preparation_calls"] > 0
                         and provenance.get("odometry_backend") == "cuda")):
            raise ValueError("normal preparation timing scope is inconsistent")
        for scope in ("spatial_build", "preprocessing"):
            seconds_key, calls_key = scope + "_seconds", scope + "_calls"
            if (protocol.get("comparison") == "cuda-cuda" and hasattr(module, "_time_method")
                    and not {seconds_key, calls_key} <= record.keys()):
                raise ValueError(f"{scope} observation is missing")
            if (seconds_key in record
                    and not (0 <= record[seconds_key] <= record["pipeline_seconds"]
                             and record[calls_key] > 0
                             and provenance.get("odometry_backend") == "cuda")):
                raise ValueError(f"{scope} timing scope is inconsistent")
        if (hasattr(module, "_time_controls")
                and not {"controls_seconds", "controls_calls"} <= record.keys()):
            raise ValueError("controls observation is missing")
        if ("controls_seconds" in record
                and not (0 < record["controls_seconds"] <= record["pipeline_seconds"]
                         and record["controls_calls"] == 1)):
            raise ValueError("controls timing scope is inconsistent")
        if (provenance.get("odometry_backend") == "cuda"
                and hasattr(module, "_time_registration")
                and not {"registration_seconds", "registration_calls"} <= record.keys()):
            raise ValueError("registration observation is missing")
        if ("registration_seconds" in record
                and not (0 < record["registration_seconds"] <= record["pipeline_seconds"]
                         and record["registration_calls"] > 0
                         and provenance.get("odometry_backend") == "cuda")):
            raise ValueError("registration timing scope is inconsistent")
        if hasattr(module, "_time_odometry") and not {
            "odometry_seconds", "odometry_calls",
        } <= record.keys():
            raise ValueError("odometry observation is missing")
        if (hasattr(module, "_time_odometry")
                and provenance.get("odometry_backend") == "cuda"):
            adapter_source = source_bytes[variant][
                "calibrex/solvers/cuda_scan_odometry_adapter.py"
            ]
            expected_scope = (
                "standard __call__ path; excludes device gyro_deskew"
                if b"class CudaScanPreprocessor" in adapter_source
                else "native preprocess_scan; device preprocessor unavailable"
            )
            if record.get("preprocessing_scope") != expected_scope:
                raise ValueError("preprocessing timing scope is not declared correctly")
            if b"def gyro_deskew(" in adapter_source and not {
                "gyro_preprocessing_seconds", "gyro_preprocessing_calls",
            } <= record.keys():
                raise ValueError("gyro_preprocessing observation is missing")
        for scope in ("odometry", "gyro_preprocessing"):
            if (scope + "_seconds" in record
                    and not (0 < record[scope + "_seconds"] <= record["pipeline_seconds"]
                             and record[scope + "_calls"] > 0)):
                raise ValueError(f"{scope} timing scope is inconsistent")
        if len(record["rounds"]) != len(result["rounds"]):
            raise ValueError("worker did not retain every fit round")
        for fitted, reported in zip(record["rounds"], result["rounds"], strict=True):
            if (
                fitted["status"] != reported["fit_status"]
                or fitted["iterations"] != reported["iterations"]
            ):
                raise ValueError("full fit record differs from reported round")
            covariance = np.asarray(fitted["calibration_covariance"])
            if (
                covariance.ndim != 2
                or covariance.shape[0] != covariance.shape[1]
                or not np.all(np.isfinite(covariance))
            ):
                raise ValueError("fit covariance is incomplete or non-finite")
        key = (attempt["repeat"], variant)
        results[key], measurements[key] = result, record
    quality = [
        module._compare(results[(r, "prior")], results[(r, "current")])
        for r in range(protocol["repeats"])
    ]
    if fit_compare is not None:
        for repeat, compared in enumerate(quality):
            compared["fits"] = [fit_compare(left, right) for left, right in zip(
                measurements[(repeat, "prior")]["rounds"],
                measurements[(repeat, "current")]["rounds"], strict=True,
            )]
            compared["passed"] = compared["passed"] and all(
                item["passed"] for item in compared["fits"]
            )
    expected_quality = summary["quality"]
    if len(quality) != len(expected_quality) or not all(q["passed"] for q in quality):
        raise ValueError("complete quality comparison differs or failed")
    for actual_quality, declared_quality in zip(quality, expected_quality, strict=True):
        # SciPy versions may normalize nearly identical quaternions with
        # slightly different rounding. Preserve the physical check and all
        # other exact fields; allow only sub-pic degree recomputation noise.
        actual_errors = dict(actual_quality["errors"])
        declared_errors = dict(declared_quality["errors"])
        actual_rotation = actual_errors.pop("rotation_deg")
        declared_rotation = declared_errors.pop("rotation_deg")
        actual_other = {k: v for k, v in actual_quality.items() if k != "errors"}
        declared_other = {k: v for k, v in declared_quality.items() if k != "errors"}
        if (
            actual_errors != declared_errors
            or actual_other != declared_other
            or not np.isclose(actual_rotation, declared_rotation, rtol=1e-6, atol=1e-12)
        ):
            raise ValueError("complete quality comparison differs or failed")
    for repeat in range(protocol["repeats"]):
        left = measurements[(repeat, "prior")]["rounds"]
        right = measurements[(repeat, "current")]["rounds"]
        for prior_fit, current_fit in zip(left, right, strict=True):
            if (
                prior_fit["status"] != current_fit["status"]
                or prior_fit["residual_count"] != current_fit["residual_count"]
                or prior_fit["parameter_count"] != current_fit["parameter_count"]
            ):
                raise ValueError("fit round status or dimensions differ")
            for field in ("calibration_covariance", "trajectory_positions", "trajectory_rotations"):
                if not np.allclose(prior_fit[field], current_fit[field], rtol=1e-4, atol=1e-8):
                    raise ValueError(f"full per-round {field} differs")
            for field in ("initial_cost", "final_cost", "residual_variance"):
                if not np.isclose(prior_fit[field], current_fit[field], rtol=1e-6, atol=1e-5):
                    raise ValueError(f"fit round {field} differs")
    for scope in ("whole", "fit"):
        medians = {
            v: statistics.median(
                a["whole_seconds"]
                if scope == "whole"
                else measurements[(a["repeat"], v)]["fit_seconds"]
                for a in attempts
                if a["variant"] == v
            )
            for v in sources
        }
        if (
            medians != summary[f"{scope}_median_seconds"]
            or medians["prior"] / medians["current"] != summary[f"{scope}_speedup"]
        ):
            raise ValueError(f"{scope} speed ratio differs")
    for name in ("protocol", "attempts"):
        if summary["provenance"][f"{name}_sha256"] != _sha((root / f"{name}.json").read_bytes()):
            raise ValueError("summary input binding differs")
    _audit_rejected_pair(root, module._compare)
    if (root / "previous_cuda").exists():
        audit(root / "previous_cuda")
    observation_summary = {}
    if (root / "repeat-v2").exists():
        observation_summary["monitored_repeat"] = audit(root / "repeat-v2")
    if (root / "observation.json").exists():
        observed = json.loads((root / "observation.json").read_bytes())
        observation_schema = root / "observation.schema.json"
        jsonschema.validate(observed, json.loads(observation_schema.read_bytes()))
        binding = observed["provenance"]
        if (binding["observer_sha256"] != _sha((root / "observer.py.txt").read_bytes())
                or binding["schema_sha256"] != _sha(observation_schema.read_bytes())
                or binding["launcher_sha256"] != protocol["launcher_sha256"]
                or observed["exit_code"] != 0
                or binding.get("terminated_on_interference", False)):
            raise ValueError("GPU observation source/schema/launcher binding differs")
        command = binding["benchmark_command"]
        command_values = [("--comparison", protocol["comparison"])]
        # CPU/CUDA already requires equal effective revisions above. Its
        # launcher resolves the requested baseline, then replaces it with HEAD.
        if protocol["comparison"] != "cpu-cuda":
            command_values.append(("--baseline-commit", protocol["baseline_commit"]))
        for option, value in command_values:
            if command[command.index(option) + 1] != value:
                raise ValueError("GPU observation benchmark command differs")
        for option, value in (("--max-seconds", protocol["max_seconds"]),
                              ("--max-rounds", protocol["max_rounds"]),
                              ("--repeats", protocol["repeats"])):
            if float(command[command.index(option) + 1]) != value:
                raise ValueError("GPU observation benchmark command differs")
        for option, key in (("--odometry-voxel-size-m", "voxel_size_m"),
                            ("--odometry-local-map-scans", "local_map_scans"),
                            ("--odometry-geometry-quantum-m", "geometry_quantum_m"),
                            ("--odometry-motion-translation-quantum-m",
                             "motion_translation_quantum_m"),
                            ("--odometry-motion-rotation-quantum-rad",
                             "motion_rotation_quantum_rad")):
            if float(command[command.index(option) + 1]) != protocol["odometry_options"][key]:
                raise ValueError("GPU observation geometry command differs")
        offsets = [row["offset_seconds"] for row in observed["samples"]]
        if offsets != sorted(offsets) or binding["sampling_interval_seconds"] != 10:
            raise ValueError("GPU observation sampling scope differs")
        observation_summary.update({"gpu_observation_samples": len(offsets),
                               "foreign_gpu_samples": sum(row["foreign"] > 0
                                                          for row in observed["samples"]),
                               "unknown_gpu_samples": sum(row["unknown"] > 0
                                                          for row in observed["samples"])})
    normal_summary = {}
    if all("normal_preparation_seconds" in record for record in measurements.values()):
        normal_medians = {
            variant: statistics.median(measurements[(r, variant)]["normal_preparation_seconds"]
                                       for r in range(protocol["repeats"]))
            for variant in sources
        }
        if any(value <= 0 for value in normal_medians.values()):
            raise ValueError("normal preparation median is not positive")
        normal_summary = {"normal_median_seconds": normal_medians,
                          "normal_speedup": normal_medians["prior"] / normal_medians["current"]}
    front_summary = {}
    for scope in ("spatial_build", "preprocessing", "controls", "registration", "odometry"):
        if (scope == "preprocessing"
                and any("gyro_preprocessing_seconds" in record
                        for record in measurements.values())):
            # Device gyro preprocessing bypasses the timed __call__ path. The
            # complete odometry scope remains comparable across both versions.
            continue
        if all(scope + "_seconds" in record for record in measurements.values()):
            medians = {
                variant: statistics.median(measurements[(r, variant)][scope + "_seconds"]
                                           for r in range(protocol["repeats"]))
                for variant in ("prior", "current")
            }
            if any(value <= 0 for value in medians.values()):
                raise ValueError(f"{scope} median is not positive")
            front_summary[scope + "_median_seconds"] = medians
            front_summary[scope + "_speedup"] = medians["prior"] / medians["current"]
    return {
        "workers": len(attempts),
        "files": len(index["files"]),
        "whole_speedup": summary["whole_speedup"],
        "fit_speedup": summary["fit_speedup"],
        "quality_passed": True,
        **normal_summary,
        **front_summary,
        **observation_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "archive", nargs="?", type=Path, default=Path("docs/assets/bspline_lidar_imu_acceleration")
    )
    print(json.dumps(audit(parser.parse_args().archive), indent=2))


if __name__ == "__main__":
    main()
