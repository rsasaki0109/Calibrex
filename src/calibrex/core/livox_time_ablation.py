"""Schema-versioned provenance for Livox point-time ablation runs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from calibrex.core.result import Grade, StrictModel

LIVOX_TIME_ABLATION_SCHEMA_VERSION: Literal[
    "slac.livox_time_ablation/v0.1"
] = "slac.livox_time_ablation/v0.1"


class LivoxTimeAblationPolicy(StrictModel):
    """Declared interpretation policy shared by all ablation variants."""

    injected_offsets_are_validation_only: bool = True
    point_time_field: str = "offset_time"
    point_time_reference: str = "Livox CustomMsg timebase"
    deskew_off_means: str


class LivoxTimeAblationVariant(StrictModel):
    """One materialized deskew/clock-shift online replay."""

    id: str
    config_path: str
    config_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    result_path: str | None = None
    result_sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    use_point_time_offsets: bool
    inject_time_offset_s: float
    run_status: str
    final_gate_status: str | None = None
    holdout_rmse_m: float | None = None
    known_bad_detectable_fraction: float | None = Field(default=None, ge=0.0, le=1.0)
    final_rolling_rmse_m: float | None = None
    quality_grade: Grade | None = None
    trajectory_gate_status: str | None = None
    deskew_applied: bool | None = None
    point_time_mapping_status: str | None = None
    point_time_mapping_residual_max_abs_s: float | None = None


class LivoxTimeAblationManifest(StrictModel):
    """Machine-readable manifest for a complete Livox time ablation."""

    schema_version: Literal[
        "slac.livox_time_ablation/v0.1"
    ] = LIVOX_TIME_ABLATION_SCHEMA_VERSION
    tool: str
    tool_version: str
    base_config_path: str
    base_config_sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")
    policy: LivoxTimeAblationPolicy
    variants: list[LivoxTimeAblationVariant] = Field(min_length=1)


def livox_time_ablation_json_schema() -> dict[str, Any]:
    """Return the JSON schema for an ablation manifest."""

    return LivoxTimeAblationManifest.model_json_schema()
