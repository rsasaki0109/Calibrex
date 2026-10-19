"""Publish and audit complete resident-refinement evidence without CUDA imports."""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import statistics
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import jsonschema
import yaml

from calibrex.core.benchmark import BenchmarkArtifact, BenchmarkDefinition
from calibrex.core.probabilistic_correspondence import ProbabilisticRefinementResultArtifact

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.benchmark_probabilistic_refinement_ablation import (  # noqa: E402
    comparable_benchmark,
    comparable_result,
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoded(value: object) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def canonical(value: object) -> str:
    return digest(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def source_manifest(commit: str) -> dict[str, str]:
    """Verify source bytes against local Git history, without extracting files."""
    data = subprocess.check_output(["git", "archive", commit, "src/calibrex"], cwd=ROOT)
    result = {}
    with tarfile.open(fileobj=io.BytesIO(data)) as bundle:
        for member in bundle.getmembers():
            if member.isfile() and member.name.endswith(".py"):
                file = bundle.extractfile(member)
                assert file is not None
                result[PurePosixPath(member.name).relative_to("src").as_posix()] = digest(
                    file.read()
                )
    return result


def audit(path: Path, *, verify_git_sources: bool = False) -> dict:
    """Recompute every result/quality digest and timing using a bounded archive."""
    files = {}
    size = 0
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            size += member.size
            if (
                not member.isfile()
                or name.is_absolute()
                or ".." in name.parts
                or "\\" in member.name
                or ":" in member.name
                or member.name in files
                or size > 32 * 1024**2
            ):
                raise ValueError("unexpected publication entry")
            stream = archive.extractfile(member)
            assert stream is not None
            files[member.name] = stream.read()

    def read(name: str):
        return json.loads(files[name])

    index = read("publication-index.json")
    if index["schema_version"] != "calibrex.probabilistic_ablation_publication/v1":
        raise ValueError("unknown publication version")
    expected = {
        name: digest(data) for name, data in files.items() if name != "publication-index.json"
    }
    if expected != index["files"]:
        raise ValueError("publication inventory or bytes differ")
    protocol, attempts, summary = (
        read(name + ".json") for name in ("protocol", "attempts", "summary")
    )
    schema = json.loads(
        (ROOT / "tools/schemas/probabilistic_ablation_benchmark.schema.json").read_bytes()
    )
    if schema != read("comparison.schema.json"):
        raise ValueError("comparison schema differs from the measured source version")
    publication_schema = json.loads(
        (ROOT / "tools/schemas/probabilistic_ablation_publication.schema.json").read_bytes()
    )
    if publication_schema != read("publication.schema.json"):
        raise ValueError("publication schema differs")
    jsonschema.validate(index, publication_schema)
    if index["provenance"]["generator_sha256"] != digest(files["publication-tool.py.txt"]):
        raise ValueError("publication generator source differs")
    for value in (protocol, attempts, summary):
        jsonschema.validate(value, schema)
    if index["provenance"]["protocol_sha256"] != digest(files["protocol.json"]):
        raise ValueError("publication protocol binding differs")
    if (
        digest(files["benchmark.py.txt"]) != protocol["launcher_sha256"]
        or digest(files["source-freezer.py.txt"]) != protocol["freezer_sha256"]
    ):
        raise ValueError("benchmark source binding differs")
    current_launcher = ROOT / "tools/benchmark_probabilistic_refinement_ablation.py"
    if ast.dump(ast.parse(current_launcher.read_text(encoding="utf-8-sig"))) != ast.dump(
        ast.parse(files["benchmark.py.txt"].decode("utf-8-sig"))
    ):
        raise ValueError("normalization implementation differs from measured launcher")
    for variant in ("prior", "current"):
        name = f"sources-{variant}-manifest.json"
        if digest(files[name]) != protocol["source_manifests"][variant]:
            raise ValueError("source manifest binding differs")
        manifest = read(name)
        for expected_digest in manifest.values():
            if digest(files[f"source-blobs/{expected_digest}.py.txt"]) != expected_digest:
                raise ValueError("frozen source blob differs from the measured manifest")
        if verify_git_sources and manifest != source_manifest(protocol["commits"][variant]):
            raise ValueError("source manifest differs from pinned Git bytes")
    if (
        len(attempts) != 6
        or [a["variant"] for a in attempts] != protocol["order"]
        or [a["index"] for a in attempts] != list(range(6))
        or any(a["exit_code"] != 0 for a in attempts)
    ):
        raise ValueError("comparison omits attempts or changes execution order")
    manifest = read("sources-current-manifest.json")
    module = manifest["calibrex/solvers/cuda_probabilistic_camera_lidar_adapter.py"]
    tree = ast.parse(files[f"source-blobs/{module}.py.txt"].decode("utf-8-sig"))
    kernel = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "_KERNEL" for target in node.targets)
    )
    kernel_digest = digest(kernel.encode())
    records = []
    for attempt in attempts:
        variant = attempt["variant"]
        folder = f"{attempt['index']:02d}-{variant}"
        record = read(folder + "/measurement.json")
        jsonschema.validate(record, schema)
        source = "prior" if variant == "prior" else "current"
        provenance = record["provenance"]
        if (
            record["variant"] != variant
            or provenance["source_commit"] != protocol["commits"][source]
            or provenance["source_manifest_sha256"] != protocol["source_manifests"][source]
            or provenance["protocol_sha256"] != digest(files["protocol.json"])
            or provenance["launcher_sha256"] != protocol["launcher_sha256"]
        ):
            raise ValueError("worker source or protocol binding differs")
        definition = BenchmarkDefinition.model_validate(read(folder + "/definition.json"))
        benchmark = BenchmarkArtifact.model_validate(read(folder + "/benchmark.json"))
        assert len(definition.trials) == len(protocol["split_seeds"]) * 4
        results, execution = {}, []
        for trial in definition.trials:
            seed = int(trial.split_id.removeprefix("seed-"))
            name = f"{trial.method_id}.seed-{seed:04d}.yaml"
            data = files[folder + "/results/" + name]
            if digest(data) != trial.provenance.output_sha256:
                raise ValueError("saved result differs from its trial binding")
            value = yaml.safe_load(data)
            result = ProbabilisticRefinementResultArtifact.model_validate(value)
            if (
                result.provenance.git_commit != protocol["commits"][source]
                or result.provenance.correspondence_artifact_sha256
                != protocol["inputs"]["correspondence"]
                or result.provenance.initialization_problem_sha256 != protocol["inputs"]["problem"]
            ):
                raise ValueError("result input or source binding differs")
            info = result.provenance.execution
            if variant == "cuda":
                assert info is not None and info.backend == "cuda" and info.cuda_calls > 0
                assert info.numpy_fallback_calls == 0 and info.kernel_launches < info.cuda_calls
                assert info.kernel_sha256 == kernel_digest
                execution.append(info.model_dump(mode="json"))
            else:
                assert info is None
            results[name] = canonical(comparable_result(value))
        if (
            record["complete_results"] != results
            or record["execution"] != execution
            or record["definition_sha256"]
            != canonical(
                comparable_benchmark(definition.model_dump(mode="json"), cuda=variant == "cuda")
            )
            or record["benchmark_sha256"]
            != canonical(
                comparable_benchmark(benchmark.model_dump(mode="json"), cuda=variant == "cuda")
            )
        ):
            raise ValueError("worker complete results or aggregates differ")
        records.append(record)
    exact = all(
        len({canonical(r[key]) for r in records}) == 1
        for key in ("complete_results", "definition_sha256", "benchmark_sha256")
    )
    medians = {
        v: statistics.median(a["whole_seconds"] for a in attempts if a["variant"] == v)
        for v in ("prior", "numpy", "cuda")
    }
    ratios = {
        name: medians[v] / medians["cuda"]
        for name, v in (("prior_cuda_whole_ratio", "prior"), ("numpy_cuda_whole_ratio", "numpy"))
    }
    passed = exact and all(r >= 2.0 for r in ratios.values())
    if (
        summary["exact_complete_results"] != exact
        or summary["median_whole_seconds"] != medians
        or any(summary[k] != v for k, v in ratios.items())
        or summary["passes_gate"] != passed
        or summary["provenance"]
        != {
            "protocol_sha256": digest(files["protocol.json"]),
            "attempts_sha256": digest(files["attempts.json"]),
        }
    ):
        raise ValueError("summary differs from complete measured attempts")
    fixture = read("fixture-protocol.json")
    if (
        fixture["correspondence_sha256"] != protocol["inputs"]["correspondence"]
        or fixture["problem_sha256"] != protocol["inputs"]["problem"]
        or digest(files["fixture-builder.py.txt"]) != fixture["builder_sha256"]
    ):
        raise ValueError("execution fixture provenance differs")
    return {
        "files": len(files),
        "git_sources_verified": verify_git_sources,
        "workers": len(attempts),
        "fits_per_worker": len(protocol["split_seeds"]) * 4,
        "exact_complete_results": exact,
        "passes_gate": passed,
        **ratios,
        "provenance": {
            "archive_sha256": digest(path.read_bytes()),
            "protocol_sha256": digest(files["protocol.json"]),
        },
    }


def publish(run: Path, fixture: Path, output: Path) -> dict:
    """Archive every measured artifact and log, excluding caches and raw inputs."""
    if output.exists():
        raise ValueError("publication destination already exists")
    files = {
        name: (run / name).read_bytes()
        for name in ("protocol.json", "attempts.json", "summary.json")
    }
    protocol = json.loads(files["protocol.json"])
    for name in ("prior", "current"):
        files[f"sources-{name}-manifest.json"] = (
            run / f"source-{name}" / "manifest.json"
        ).read_bytes()
        for module, source_digest in json.loads(files[f"sources-{name}-manifest.json"]).items():
            data = (run / f"source-{name}" / "src" / module).read_bytes()
            if digest(data) != source_digest:
                raise ValueError("source changed after measurement")
            files[f"source-blobs/{source_digest}.py.txt"] = data
    for index, variant in enumerate(protocol["order"]):
        folder = f"{index:02d}-{variant}"
        for p in (run / folder).rglob("*"):
            if p.is_file():
                files[p.relative_to(run).as_posix()] = p.read_bytes()
        for suffix in ("stdout.log", "stderr.log"):
            files[f"{folder}.{suffix}"] = (run / f"{folder}.{suffix}").read_bytes()
    files.update(
        {
            "fixture-protocol.json": fixture.read_bytes(),
            "publication-tool.py.txt": Path(__file__).read_bytes(),
            "publication.schema.json": (
                ROOT / "tools/schemas/probabilistic_ablation_publication.schema.json"
            ).read_bytes(),
            "comparison.schema.json": (
                ROOT / "tools/schemas/probabilistic_ablation_benchmark.schema.json"
            ).read_bytes(),
            "benchmark.py.txt": (
                ROOT / "tools/benchmark_probabilistic_refinement_ablation.py"
            ).read_bytes(),
            "source-freezer.py.txt": (ROOT / "tools/benchmark_bspline_lidar_imu.py").read_bytes(),
            "fixture-builder.py.txt": (
                ROOT / "tools/benchmark_probabilistic_refinement.py"
            ).read_bytes(),
        }
    )
    for number in range(1, 6):
        pilot = ROOT / f"outputs/probabilistic-ablation-screen-v{number}"
        for name in ("protocol.json", "attempts.json", "summary.json", "post-run-summary.json"):
            if (pilot / name).is_file():
                files[f"pilots/v{number}/{name}"] = (pilot / name).read_bytes()
    source_backup = ROOT / "outputs/probabilistic-private-pilot-sources-v1"
    if source_backup.is_dir():
        for file in source_backup.iterdir():
            if file.is_file():
                files["pilots/sources/" + file.name + ".txt"] = file.read_bytes()
    files["publication-index.json"] = encoded(
        {
            "schema_version": "calibrex.probabilistic_ablation_publication/v1",
            "files": {name: digest(data) for name, data in sorted(files.items())},
            "provenance": {
                "generator": "tools/audit_probabilistic_refinement_ablation.py",
                "generator_sha256": digest(Path(__file__).read_bytes()),
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "protocol_sha256": digest(files["protocol.json"]),
            },
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for name, data in sorted(files.items()):
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            entry.mode = 0o644
            archive.addfile(entry, io.BytesIO(data))
    return audit(output, verify_git_sources=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--publish-run", type=Path)
    parser.add_argument("--fixture-protocol", type=Path)
    parser.add_argument("--verify-git-sources", action="store_true")
    args = parser.parse_args()
    if args.publish_run:
        if args.fixture_protocol is None:
            parser.error("--publish-run requires --fixture-protocol")
        result = publish(args.publish_run, args.fixture_protocol, args.archive)
    else:
        result = audit(args.archive, verify_git_sources=args.verify_git_sources)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
