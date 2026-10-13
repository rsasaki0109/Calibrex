"""Validate the retained rejected experiment without enabling its retired code."""
from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import statistics
from pathlib import Path

from calibrex.core.benchmark import BenchmarkDefinition, aggregate_benchmark_definition
from calibrex.core.continuous_time_lidar_artifacts import ContinuousTimeLidarPairArtifact
from calibrex.core.io import read_mapping


def audit(root: Path) -> dict[str, int]:
    index = json.loads((root / "publication-index.json").read_bytes())
    for name, binding in index["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != binding["sha256"]:
            raise ValueError(f"archive digest mismatch: {name}")
    source = root / "sources/measured.benchmark_adaptive_plane_search.py.txt"
    loader = importlib.machinery.SourceFileLoader("_retired_projection_benchmark", str(source))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    assert spec is not None
    measured = importlib.util.module_from_spec(spec)
    loader.exec_module(measured)
    models = trials = 0
    for cohort in ("gpu-dense", "cpu-dense", "default-smoke"):
        definition = BenchmarkDefinition.model_validate(read_mapping(
            root / cohort / "pipeline.definition.json"))
        aggregate = read_mapping(root / cohort / "pipeline.benchmark.json")
        assert aggregate_benchmark_definition(definition).model_dump(
            mode="json", exclude_none=True) == aggregate
        baseline = None
        for trial in definition.trials:
            path = root / cohort / f"{trial.method_id}.{trial.split_id}.measurement.json"
            row = json.loads(path.read_bytes())
            model = ContinuousTimeLidarPairArtifact.model_validate(read_mapping(
                path.with_name(path.name.replace(".measurement.json", ".result.yaml"))))
            payload = model.model_dump(mode="json", exclude_none=True)
            assert measured._canonical(row["result"]) == measured._canonical(payload)
            comparable = measured._comparison_canonical(payload)
            baseline = comparable if baseline is None else baseline
            assert comparable == baseline
            trials += 1
            models += 1
        medians = {method: statistics.median(t.runtime_seconds for t in definition.trials
                                           if t.method_id == method)
                   for method in ("baseline", "sparse")}
        assert medians["baseline"] / medians["sparse"] == index["summaries"][cohort][
            "ratio_of_medians"]
        for folder in (("diagnostic-gpu", "failed-diagnostic") if cohort == "gpu-dense" else
                       ("diagnostic-cpu",) if cohort == "cpu-dense" else ()):
            row = json.loads((root / folder / "measurement.json").read_bytes())
            model = ContinuousTimeLidarPairArtifact.model_validate(read_mapping(
                root / folder / "result.yaml"))
            payload = model.model_dump(mode="json", exclude_none=True)
            assert measured._canonical(payload) == measured._canonical(row["result"])
            assert measured._comparison_canonical(payload) == baseline
            models += 1
    assert (trials, models) == (14, 17)
    return {"formal_trials": trials, "complete_models": models, "digest_files": len(index["files"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("docs/assets/initial_lidar_projection_reuse"))
    args = parser.parse_args()
    print(json.dumps(audit(args.root), sort_keys=True))


if __name__ == "__main__":
    main()
