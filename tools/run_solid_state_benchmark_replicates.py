#!/usr/bin/env python3
"""Materialize repeated split/seed runs for the solid-state benchmark."""

from __future__ import annotations

import argparse
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.solid_state_cross_dataset_benchmark import (
    SOLID_STATE_CROSS_DATASET_BENCHMARK_CONFIG_SCHEMA_VERSION,
    SolidStateCrossDatasetBenchmarkSpec,
    SolidStateCrossDatasetBenchmarkSpecReplicate,
)

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.run_continuous_time_lidar_ablation import run_ablation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path, help="base benchmark declaration YAML")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="directory receiving generated configs and repeated ablations",
    )
    parser.add_argument(
        "--output-spec",
        type=Path,
        required=True,
        help="generated benchmark spec with materialized replicate paths",
    )
    parser.add_argument(
        "--append-spec",
        type=Path,
        help="existing generated spec whose replicate declarations are preserved",
    )
    parser.add_argument(
        "--split-id",
        action="append",
        dest="split_ids",
        help="materialize only this split; repeat for multiple splits",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        dest="seeds",
        help="materialize only this sampling seed; repeat for multiple seeds",
    )
    parser.add_argument(
        "--reuse-only",
        action="store_true",
        help="fail when a requested variant result cannot be reused",
    )
    parser.add_argument(
        "--adaptive-no-uniform-fallback",
        action="store_true",
        help="materialize adaptive_mad with only adaptive plane correspondences",
    )
    return parser


def _resolve(path_text: str, *, base: Path) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "replicate"


def _variant_config(
    base: dict[str, Any],
    *,
    output_dir: Path,
    holdout_start_fraction: float,
    sampling_seed: int,
) -> dict[str, Any]:
    config = deepcopy(base)
    project = dict(config.get("project") or {})
    project["output_dir"] = str(output_dir)
    config["project"] = project
    pipeline = dict(config.get("pipeline") or {})
    factors = dict(pipeline.get("factors") or {})
    factor = dict(factors.get("lidar_rig_point_to_plane") or {})
    options = dict(factor.get("options") or {})
    options["continuous_time_holdout_start_fraction"] = holdout_start_fraction
    options["continuous_time_sampling_seed"] = sampling_seed
    factor["options"] = options
    factors["lidar_rig_point_to_plane"] = factor
    pipeline["factors"] = factors
    config["pipeline"] = pipeline
    return config


def _display_path(path: Path, *, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path.resolve())


def run_replicates(
    spec_path: Path,
    output_root: Path,
    output_spec: Path,
    *,
    split_ids: list[str] | None = None,
    seeds: list[int] | None = None,
    append_spec: Path | None = None,
    reuse_only: bool = False,
    adaptive_use_uniform_fallback: bool = True,
) -> SolidStateCrossDatasetBenchmarkSpec:
    """Run every requested paired split/seed and write a reusable spec."""

    base_dir = Path.cwd().resolve()
    spec_path = spec_path.resolve()
    output_root = output_root.resolve()
    output_spec = output_spec.resolve()
    spec = SolidStateCrossDatasetBenchmarkSpec.model_validate(read_mapping(spec_path))
    requested_splits = split_ids or list(spec.protocol.split_ids)
    requested_seeds = seeds or list(spec.protocol.seed_values)
    missing_fractions = [
        split_id
        for split_id in requested_splits
        if split_id not in spec.protocol.holdout_start_fractions
    ]
    if missing_fractions:
        raise ValueError(
            "protocol.holdout_start_fractions is missing split ids: "
            + ", ".join(missing_fractions)
        )

    append_model = (
        SolidStateCrossDatasetBenchmarkSpec.model_validate(read_mapping(append_spec))
        if append_spec is not None
        else None
    )
    raw_spec = spec.model_dump(mode="json")
    if append_model is not None:
        appended_by_id = {
            dataset["id"]: dataset.get("replicates", [])
            for dataset in append_model.model_dump(mode="json")["datasets"]
        }
        for dataset in raw_spec["datasets"]:
            dataset["replicates"] = appended_by_id.get(dataset["id"], [])
    raw_spec["schema_version"] = SOLID_STATE_CROSS_DATASET_BENCHMARK_CONFIG_SCHEMA_VERSION
    for dataset_index, dataset_spec in enumerate(spec.datasets):
        if dataset_spec.replicates and append_model is None:
            raise ValueError(
                f"{dataset_spec.id} already declares replicates; use a fresh base spec"
            )
        base_config_path = _resolve(dataset_spec.config_path, base=base_dir)
        base_config = read_mapping(base_config_path)
        existing_replicates = list(
            next(
                dataset["replicates"]
                for dataset in raw_spec["datasets"]
                if dataset["id"] == dataset_spec.id
            )
            or []
        )
        existing_ids = {item["id"] for item in existing_replicates}
        generated_replicates: list[dict[str, Any]] = []
        for split_id in requested_splits:
            fraction = spec.protocol.holdout_start_fractions[split_id]
            for seed in requested_seeds:
                replicate_id = (
                    f"{dataset_spec.id}__{_safe_name(split_id)}__seed_{seed}"
                )
                if replicate_id in existing_ids:
                    continue
                replicate_dir = output_root / dataset_spec.id / replicate_id
                config_path = replicate_dir / "config.yaml"
                write_mapping(
                    config_path,
                    _variant_config(
                        base_config,
                        output_dir=replicate_dir,
                        holdout_start_fraction=fraction,
                        sampling_seed=seed,
                    ),
                )
                run_ablation(
                    config_path,
                    replicate_dir,
                    variant_ids=["adaptive_mad", "uniform_none"],
                    reuse_only=reuse_only,
                    adaptive_use_uniform_fallback=adaptive_use_uniform_fallback,
                )
                generated_replicates.append(
                    SolidStateCrossDatasetBenchmarkSpecReplicate(
                        id=replicate_id,
                        split_id=split_id,
                        seed=seed,
                        ablation_manifest_path=_display_path(
                            replicate_dir / "ablation_manifest.yaml",
                            base=base_dir,
                        ),
                        notes=[
                            f"holdout_start_fraction={fraction}",
                            f"sampling_seed={seed}",
                        ],
                    ).model_dump(mode="json")
                )
        raw_spec["datasets"][dataset_index]["replicates"] = [
            *existing_replicates,
            *generated_replicates,
        ]
    write_mapping(output_spec, raw_spec)
    return SolidStateCrossDatasetBenchmarkSpec.model_validate(raw_spec)


def main() -> int:
    args = _parser().parse_args()
    spec = run_replicates(
        args.spec,
        args.output_root,
        args.output_spec,
        split_ids=args.split_ids,
        seeds=args.seeds,
        append_spec=args.append_spec,
        reuse_only=args.reuse_only,
        adaptive_use_uniform_fallback=not args.adaptive_no_uniform_fallback,
    )
    replicate_count = sum(len(dataset.replicates) for dataset in spec.datasets)
    print(f"materialized_replicates={replicate_count}")
    print(f"spec: {args.output_spec}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
