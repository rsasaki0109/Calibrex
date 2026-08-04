#!/usr/bin/env python3
"""Run a fair uniform-vs-adaptive continuous-time LiDAR ablation."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from calibrex import __version__
from calibrex.core.continuous_time_lidar_ablation import (
    ContinuousTimeLidarAblationManifest,
    ContinuousTimeLidarAblationPolicy,
    ContinuousTimeLidarAblationVariant,
)
from calibrex.core.continuous_time_lidar_artifacts import (
    ContinuousTimeLidarPairArtifact,
)
from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.provenance import sha256_path
from calibrex.pipelines.online import evaluate_continuous_time_lidar_pair

_VARIANTS: dict[
    str,
    tuple[Literal["uniform", "adaptive"], Literal["none", "mad"]],
] = {
    "uniform_none": ("uniform", "none"),
    "adaptive_mad": ("adaptive", "mad"),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", type=Path, help="base rosbag1/rosbag2 continuous-time config"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory receiving variant configs, artifacts, and manifest",
    )
    parser.add_argument(
        "--variant",
        choices=sorted(_VARIANTS),
        action="append",
        help="run only the requested variant; repeat for multiple variants",
    )
    parser.add_argument(
        "--reuse-only",
        action="store_true",
        help="fail instead of recomputing a variant whose config digest changed",
    )
    parser.add_argument(
        "--adaptive-no-uniform-fallback",
        action="store_true",
        help="run adaptive_mad with only adaptive plane correspondences",
    )
    return parser


def _variant_config(
    base: dict[str, Any],
    *,
    output_dir: Path,
    voxel_strategy: str,
    outlier_policy: str,
    adaptive_use_uniform_fallback: bool = True,
) -> dict[str, Any]:
    config = deepcopy(base)
    project = dict(config.get("project") or {})
    project["output_dir"] = str(output_dir)
    config["project"] = project
    pipeline = dict(config.get("pipeline") or {})
    factors = dict(pipeline.get("factors") or {})
    factor = dict(factors.get("lidar_rig_point_to_plane") or {})
    options = dict(factor.get("options") or {})
    options["continuous_time_voxel_strategy"] = voxel_strategy
    options["continuous_time_outlier_policy"] = outlier_policy
    options["continuous_time_adaptive_use_uniform_fallback"] = (
        adaptive_use_uniform_fallback if voxel_strategy == "adaptive" else True
    )
    factor["options"] = options
    factors["lidar_rig_point_to_plane"] = factor
    pipeline["factors"] = factors
    config["pipeline"] = pipeline
    return config


def _load_reusable(
    result_path: Path,
    *,
    config_path: Path,
    config_sha256: str,
) -> ContinuousTimeLidarPairArtifact | None:
    if not result_path.exists():
        return None
    try:
        artifact = ContinuousTimeLidarPairArtifact.model_validate(
            read_mapping(result_path)
        )
    except (OSError, ValueError):
        return None
    source_digest = artifact.provenance.source_sha256.get(str(config_path))
    if source_digest != config_sha256:
        source_digest = artifact.provenance.source_sha256.get(str(config_path.resolve()))
    if source_digest != config_sha256:
        return None
    return artifact


def _variant_record(
    variant_id: str,
    *,
    config_path: Path,
    config_sha256: str,
    result_path: Path,
    artifact: ContinuousTimeLidarPairArtifact | None,
) -> ContinuousTimeLidarAblationVariant:
    voxel_strategy, outlier_policy = _VARIANTS[variant_id]
    return ContinuousTimeLidarAblationVariant(
        id=variant_id,
        config_path=str(config_path),
        config_sha256=config_sha256,
        result_path=str(result_path) if artifact is not None else None,
        result_sha256=sha256_path(result_path) if artifact is not None else None,
        voxel_strategy=voxel_strategy,
        outlier_policy=outlier_policy,
        status=artifact.status if artifact is not None else "not_run",
        estimated_time_offset_sec=(
            artifact.estimated_time_offset_sec if artifact is not None else None
        ),
        final_train_rmse_m=artifact.final_train_rmse_m if artifact is not None else None,
        final_holdout_rmse_m=(
            artifact.final_holdout_rmse_m if artifact is not None else None
        ),
        observability_rank=(
            artifact.observability.rank if artifact is not None else None
        ),
        outlier_rejected_count=(
            artifact.outlier_rejected_count if artifact is not None else None
        ),
        quality_grade=(
            artifact.refined_transform.quality.grade if artifact is not None else None
        ),
    )


def run_ablation(
    config_path: Path,
    output_dir: Path,
    *,
    variant_ids: list[str],
    reuse_only: bool,
    adaptive_use_uniform_fallback: bool = True,
) -> ContinuousTimeLidarAblationManifest:
    """Materialize and execute the declared fair comparison variants."""

    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    base = read_mapping(config_path)
    base_digest = sha256_path(config_path)
    if base_digest is None:
        raise RuntimeError(f"could not hash base config: {config_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[ContinuousTimeLidarAblationVariant] = []
    for variant_id in variant_ids:
        voxel_strategy, outlier_policy = _VARIANTS[variant_id]
        variant_dir = output_dir / variant_id
        variant_config_path = variant_dir / "config.yaml"
        write_mapping(
            variant_config_path,
            _variant_config(
                base,
                output_dir=variant_dir,
                voxel_strategy=voxel_strategy,
                outlier_policy=outlier_policy,
                adaptive_use_uniform_fallback=adaptive_use_uniform_fallback,
            ),
        )
        variant_digest = sha256_path(variant_config_path)
        if variant_digest is None:
            raise RuntimeError(f"could not hash variant config: {variant_config_path}")
        result_path = variant_dir / "continuous_time_lidar_pair.yaml"
        artifact = _load_reusable(
            result_path,
            config_path=variant_config_path,
            config_sha256=variant_digest,
        )
        if artifact is None:
            if reuse_only:
                raise RuntimeError(f"no reusable result for {variant_id}: {result_path}")
            artifact = evaluate_continuous_time_lidar_pair(variant_config_path)
            write_mapping(
                result_path,
                artifact.model_dump(mode="json", exclude_none=True),
            )
        records.append(
            _variant_record(
                variant_id,
                config_path=variant_config_path,
                config_sha256=variant_digest,
                result_path=result_path,
                artifact=artifact,
            )
        )

    manifest = ContinuousTimeLidarAblationManifest(
        tool="tools/run_continuous_time_lidar_ablation.py",
        tool_version=__version__,
        base_config_path=str(config_path),
        base_config_sha256=base_digest,
        policy=ContinuousTimeLidarAblationPolicy(
            same_capture_windows=True,
            same_temporal_holdout=True,
            same_solver_budget=True,
            baseline_definition=(
                "uniform voxel planes with no deterministic pre-LM MAD rejection"
            ),
        ),
        variants=records,
    )
    write_mapping(
        output_dir / "ablation_manifest.yaml",
        manifest.model_dump(mode="json", exclude_none=True),
    )
    return manifest


def main() -> int:
    args = _parser().parse_args()
    variant_ids = args.variant or sorted(_VARIANTS)
    manifest = run_ablation(
        args.config,
        args.output_dir,
        variant_ids=variant_ids,
        reuse_only=args.reuse_only,
        adaptive_use_uniform_fallback=not args.adaptive_no_uniform_fallback,
    )
    for variant in manifest.variants:
        print(
            f"{variant.id}: status={variant.status} "
            f"offset={variant.estimated_time_offset_sec} "
            f"train_rmse={variant.final_train_rmse_m} "
            f"holdout_rmse={variant.final_holdout_rmse_m}"
        )
    print(f"manifest: {args.output_dir / 'ablation_manifest.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
