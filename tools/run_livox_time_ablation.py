#!/usr/bin/env python3
"""Run reproducible Livox point-time/deskew ablations on a public ROS1/ROS2 config.

The tool materializes one validated config and one result directory per variant.
Injected offsets are validation-only controls; they are never treated as a
calibration estimate.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from calibrex import __version__
from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.livox_time_ablation import LivoxTimeAblationManifest
from calibrex.core.provenance import sha256_path
from calibrex.core.result import CalibrationResult, load_result
from calibrex.pipelines.online import OnlineCalibrationRunOptions, run_online_calibration


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="base online ROS1 YAML config")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory receiving variant configs, results, and manifest",
    )
    parser.add_argument(
        "--offset-s",
        type=float,
        nargs="+",
        default=[0.0, -0.005, 0.005, -0.010, 0.010, -0.020, 0.020],
        help="signed target capture-time shifts used as validation controls",
    )
    parser.add_argument(
        "--include-no-deskew",
        action="store_true",
        help="also run with use_point_time_offsets=false",
    )
    parser.add_argument(
        "--reuse-only",
        action="store_true",
        help="fail instead of replaying a variant whose result is not reusable",
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--accumulation-batches", type=int, default=3)
    parser.add_argument("--max-accumulated-train-points", type=int, default=8000)
    return parser


def _offset_label(offset_s: float) -> str:
    milliseconds = round(offset_s * 1000.0)
    sign = "p" if milliseconds >= 0 else "m"
    return f"{sign}{abs(milliseconds):g}ms"


def _variant_config(
    base: dict[str, Any],
    *,
    output_dir: Path,
    use_point_time_offsets: bool,
    inject_time_offset_s: float,
) -> dict[str, Any]:
    config = dict(base)
    project = dict(config.get("project") or {})
    project["output_dir"] = str(output_dir)
    config["project"] = project
    pipeline = dict(config.get("pipeline") or {})
    factors = dict(pipeline.get("factors") or {})
    factor = dict(factors.get("lidar_rig_point_to_plane") or {})
    factor_options = dict(factor.get("options") or {})
    factor_options["use_point_time_offsets"] = use_point_time_offsets
    factor_options["inject_time_offset_s"] = inject_time_offset_s
    factor["options"] = factor_options
    factors["lidar_rig_point_to_plane"] = factor
    pipeline["factors"] = factors
    config["pipeline"] = pipeline
    return config


def _metric_value(result: Any, name: str) -> float | None:
    metric = result.metrics.get(name)
    if metric is None:
        return None
    value = metric.value
    return float(value) if isinstance(value, (int, float)) else None


def _load_reusable_result(
    result_path: Path, *, config_sha256: str
) -> CalibrationResult | None:
    """Load a completed variant only when its config binding still matches."""

    if not result_path.exists():
        return None
    try:
        result = load_result(result_path)
    except (OSError, ValueError):
        return None
    if result.run.config_sha256 != config_sha256:
        return None
    return result


def _point_time_metadata(base: Mapping[str, Any]) -> tuple[str, str]:
    """Return the target sensor's declared point-time field and reference."""

    sensors = base.get("sensors")
    if not isinstance(sensors, Mapping):
        return "offset_time", "Livox CustomMsg timebase"
    frames = base.get("frames")
    target_name: str | None = None
    if isinstance(frames, Mapping):
        for name, frame in frames.items():
            if not isinstance(frame, Mapping) or frame.get("parent") is None:
                continue
            transform = frame.get("transform")
            if isinstance(transform, Mapping) and transform.get("estimate") is True:
                target_name = str(name)
                break
    ordered_names = [target_name] if target_name is not None else []
    ordered_names.extend(str(name) for name in sensors if str(name) != target_name)
    for name in ordered_names:
        sensor = sensors.get(name)
        if not isinstance(sensor, Mapping):
            continue
        field = sensor.get("point_time_field")
        solid_state = sensor.get("solid_state")
        if field is None and isinstance(solid_state, Mapping):
            field = solid_state.get("point_time_field")
        if not isinstance(field, str) or not field:
            continue
        reference = (
            solid_state.get("point_time_reference")
            if isinstance(solid_state, Mapping)
            else None
        )
        return field, str(reference or "unknown")
    return "offset_time", "Livox CustomMsg timebase"


def run_ablation(
    config_path: Path,
    output_dir: Path,
    offsets_s: Iterable[float],
    *,
    include_no_deskew: bool,
    reuse_only: bool,
    batch_size: int,
    accumulation_batches: int,
    max_accumulated_train_points: int,
) -> dict[str, Any]:
    """Materialize and execute the requested time-ablation variants."""

    base = read_mapping(config_path)
    point_time_field, point_time_reference = _point_time_metadata(base)
    output_dir.mkdir(parents=True, exist_ok=True)
    variants: list[dict[str, Any]] = []
    modes = [True, False] if include_no_deskew else [True]
    for use_point_time_offsets in modes:
        for offset_s in offsets_s:
            label = (
                "deskew_on" if use_point_time_offsets else "deskew_off"
            ) + f"_offset_{_offset_label(offset_s)}"
            variant_dir = output_dir / label
            variant_config_path = variant_dir / "config.yaml"
            variant_config = _variant_config(
                base,
                output_dir=variant_dir,
                use_point_time_offsets=use_point_time_offsets,
                inject_time_offset_s=offset_s,
            )
            write_mapping(variant_config_path, variant_config)
            variant_config_sha256 = sha256_path(variant_config_path)
            if variant_config_sha256 is None:
                raise RuntimeError(f"failed to hash variant config: {variant_config_path}")
            result_path = variant_dir / "result.yaml"
            result = _load_reusable_result(
                result_path, config_sha256=variant_config_sha256
            )
            if result is None:
                if reuse_only:
                    raise RuntimeError(
                        "reuse-only ablation is missing a reusable result for "
                        f"{label}: {result_path}"
                    )
                result = run_online_calibration(
                    variant_config_path,
                    OnlineCalibrationRunOptions(
                        output_dir=variant_dir,
                        batch_size=batch_size,
                        accumulation_batches=accumulation_batches,
                        max_accumulated_train_points=max_accumulated_train_points,
                    ),
                )
            variants.append(
                {
                    "id": label,
                    "config_path": str(variant_config_path),
                    "config_sha256": variant_config_sha256,
                    "result_path": str(result_path) if result is not None else None,
                    "result_sha256": sha256_path(result_path)
                    if result_path.exists()
                    else None,
                    "use_point_time_offsets": use_point_time_offsets,
                    "inject_time_offset_s": offset_s,
                    "run_status": result.run.status if result is not None else "dry_run",
                    "final_gate_status": (
                        result.run.provenance.get("online_final_gate_status")
                        if result is not None
                        else None
                    ),
                    "holdout_rmse_m": (
                        _metric_value(result, "lidar_pair_holdout_point_to_plane_rmse_m")
                        if result is not None
                        else None
                    ),
                    "known_bad_detectable_fraction": (
                        _metric_value(
                            result, "lidar_pair_known_bad_detectable_fraction"
                        )
                        if result is not None
                        else None
                    ),
                    "final_rolling_rmse_m": (
                        _metric_value(
                            result, "online_calibration_final_rolling_rmse_m"
                        )
                        if result is not None
                        else None
                    ),
                    "quality_grade": result.quality.grade if result is not None else None,
                    "trajectory_gate_status": (
                        _trajectory_gate_status(result) if result is not None else None
                    ),
                    "deskew_applied": (
                        result.run.provenance.get("deskew_applied")
                        if result is not None
                        else None
                    ),
                    "point_time_mapping_status": (
                        _point_time_mapping_value(result, "status")
                        if result is not None
                        else None
                    ),
                    "point_time_mapping_residual_max_abs_s": (
                        _float_value(
                            _point_time_mapping_value(
                                result, "residual_max_abs_s"
                            )
                        )
                        if result is not None
                        else None
                    ),
                }
            )
    manifest_payload = {
        "schema_version": "slac.livox_time_ablation/v0.1",
        "tool": "tools/run_livox_time_ablation.py",
        "tool_version": __version__,
        "base_config_path": str(config_path),
        "base_config_sha256": sha256_path(config_path),
        "policy": {
            "injected_offsets_are_validation_only": True,
            "point_time_field": point_time_field,
            "point_time_reference": point_time_reference,
            "deskew_off_means": (
                "decode/observe offsets but do not consume them for pose interpolation"
            ),
        },
        "variants": variants,
    }
    manifest = LivoxTimeAblationManifest.model_validate(manifest_payload)
    write_mapping(
        output_dir / "ablation_manifest.yaml",
        manifest.model_dump(mode="json", exclude_none=True),
    )
    return manifest.model_dump(mode="json", exclude_none=True)


def _trajectory_gate_status(result: CalibrationResult) -> str | None:
    """Return the trajectory verdict recorded by an ablation result."""

    metric = result.metrics.get("trajectory_gate_verdict")
    return metric.grade if metric is not None else None


def _point_time_mapping_value(result: CalibrationResult, key: str) -> object:
    """Read a shared Livox mapping field from result provenance."""

    mappings = result.run.provenance.get("point_time_clock_mapping_by_sensor")
    if not isinstance(mappings, dict):
        return None
    target_sensor = result.run.provenance.get("online_target_sensor")
    mapping = mappings.get(target_sensor) if isinstance(target_sensor, str) else None
    if not isinstance(mapping, dict):
        mapping = mappings.get("livox_horizon")
    if not isinstance(mapping, dict):
        mapping = next(
            (candidate for candidate in mappings.values() if isinstance(candidate, dict)),
            None,
        )
    if not isinstance(mapping, dict):
        return None
    return mapping.get(key)


def _float_value(value: object) -> float | None:
    """Convert a numeric provenance value without accepting booleans."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def main() -> int:
    args = _parser().parse_args()
    run_ablation(
        args.config,
        args.output_dir,
        args.offset_s,
        include_no_deskew=args.include_no_deskew,
        reuse_only=args.reuse_only,
        batch_size=args.batch_size,
        accumulation_batches=args.accumulation_batches,
        max_accumulated_train_points=args.max_accumulated_train_points,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
