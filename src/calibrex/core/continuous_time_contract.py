"""Typed continuous-time body-pose trajectory contract.

This is a versioned extension of the discrete ``slac.trajectory/v0.1``
artifact: the discrete artifact records recorded poses for motion compensation,
while this contract additionally declares the interpolation model, the knot
domain validity, and the clock/capture-time convention so that analytic
trajectory factors and sparse optimization can consume the knots directly.

The ``piecewise_linear_slerp/v0.1`` interpolation model is byte-for-byte the
same as :class:`calibrex.core.continuous_trajectory.PiecewiseSE3Trajectory`;
``to_piecewise()`` converts this contract into that adapter-facing
representation without changing its meaning.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from calibrex.core.continuous_trajectory import (
    ContinuousTrajectoryPose,
    PiecewiseSE3Trajectory,
)
from calibrex.core.io import read_mapping, write_mapping
from calibrex.core.result import StrictModel, TransformResult

CONTINUOUS_TIME_TRAJECTORY_SCHEMA_VERSION: Literal[
    "slac.continuous_time_trajectory/v0.1"
] = "slac.continuous_time_trajectory/v0.1"

InterpolationModel = Literal["piecewise_linear_slerp/v0.1", "screw_linear/v0.1"]


class ContinuousTimeTrajectoryProvenance(StrictModel):
    """Generator and immutable input lineage for the trajectory contract."""

    generator: str
    generator_version: str
    git_commit: str | None = None
    command: list[str] = Field(default_factory=list)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ContinuousTimeKnotPose(StrictModel):
    """One timestamped ``T_world_body`` trajectory knot."""

    timestamp_sec: float
    transform_world_body: TransformResult

    @model_validator(mode="after")
    def check_timestamp(self) -> ContinuousTimeKnotPose:
        """Reject non-finite knot timestamps."""

        if not math.isfinite(self.timestamp_sec):
            raise ValueError("trajectory knot timestamp must be finite")
        return self


class ContinuousTimeKnotDomain(StrictModel):
    """Valid temporal support of the trajectory."""

    minimum_time_sec: float
    maximum_time_sec: float
    span_sec: float = Field(ge=0.0)

    @model_validator(mode="after")
    def check_domain(self) -> ContinuousTimeKnotDomain:
        """Require a non-empty ordered domain."""

        if not math.isfinite(self.minimum_time_sec) or not math.isfinite(
            self.maximum_time_sec
        ):
            raise ValueError("knot domain endpoints must be finite")
        if self.minimum_time_sec > self.maximum_time_sec:
            raise ValueError(
                "knot domain minimum must not exceed the maximum"
            )
        if not math.isclose(
            self.span_sec,
            self.maximum_time_sec - self.minimum_time_sec,
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        ):
            raise ValueError("knot domain span must equal the endpoint gap")
        return self


class ContinuousTimeClockSemantics(StrictModel):
    """Clock domain and capture-time convention for the knots."""

    time_unit: Literal["seconds"] = "seconds"
    epoch_reference: str = (
        "unix epoch (1970-01-01T00:00:00Z) unless otherwise stated"
    )
    monotonic: bool = True
    note: str = (
        "knot timestamps are the sensor-clock times at which the body pose "
        "T_world_body is exact; interpolation is only defined inside the "
        "declared knot domain and queries outside it are rejected"
    )


class ContinuousTimeTrajectoryContract(StrictModel):
    """Schema-valid typed continuous body-pose trajectory."""

    schema_version: Literal[
        "slac.continuous_time_trajectory/v0.1"
    ] = CONTINUOUS_TIME_TRAJECTORY_SCHEMA_VERSION
    trajectory_id: str
    world_frame: str
    body_frame: str
    interpolation: InterpolationModel
    extrapolation: Literal["reject"] = "reject"
    knot_domain: ContinuousTimeKnotDomain
    clock: ContinuousTimeClockSemantics
    knots: list[ContinuousTimeKnotPose] = Field(min_length=2)
    provenance: ContinuousTimeTrajectoryProvenance

    @model_validator(mode="after")
    def check_trajectory(self) -> ContinuousTimeTrajectoryContract:
        """Require ordered knots with consistent frames and matching domain."""

        timestamps = [item.timestamp_sec for item in self.knots]
        if any(
            right <= left
            for left, right in pairwise(timestamps)
        ):
            raise ValueError("trajectory knot timestamps must be strictly increasing")
        for knot in self.knots:
            if (
                knot.transform_world_body.parent != self.world_frame
                or knot.transform_world_body.child != self.body_frame
            ):
                raise ValueError("trajectory knot frame IDs are inconsistent")
        if not math.isclose(
            timestamps[0],
            self.knot_domain.minimum_time_sec,
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        ) or not math.isclose(
            timestamps[-1],
            self.knot_domain.maximum_time_sec,
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "knot domain endpoints must match the first and last knot times"
            )
        return self

    @property
    def knot_count(self) -> int:
        """Return the number of knots."""

        return len(self.knots)

    def to_piecewise(self) -> PiecewiseSE3Trajectory:
        """Convert to the adapter-facing piecewise linear-slerp trajectory.

        Only valid for the ``piecewise_linear_slerp/v0.1`` interpolation
        model; the conversion preserves the exact pose and timing semantics.
        """

        if self.interpolation != "piecewise_linear_slerp/v0.1":
            raise ValueError(
                f"interpolation model {self.interpolation!r} cannot be "
                "converted to the piecewise linear-slerp adapter"
            )
        return PiecewiseSE3Trajectory(
            tuple(
                ContinuousTrajectoryPose(
                    timestamp_sec=item.timestamp_sec,
                    transform_world_body=item.transform_world_body.as_se3(),
                )
                for item in self.knots
            )
        )

    def save(self, path: str | Path) -> None:
        """Save the trajectory contract as YAML or JSON."""

        write_mapping(
            Path(path),
            self.model_dump(mode="json", exclude_none=True),
        )

    @classmethod
    def from_piecewise(
        cls,
        trajectory: PiecewiseSE3Trajectory,
        *,
        trajectory_id: str,
        world_frame: str,
        body_frame: str,
        provenance: ContinuousTimeTrajectoryProvenance,
    ) -> ContinuousTimeTrajectoryContract:
        """Build a contract from the adapter-facing piecewise trajectory."""

        return cls(
            trajectory_id=trajectory_id,
            world_frame=world_frame,
            body_frame=body_frame,
            interpolation="piecewise_linear_slerp/v0.1",
            knot_domain=ContinuousTimeKnotDomain(
                minimum_time_sec=trajectory.minimum_time_sec,
                maximum_time_sec=trajectory.maximum_time_sec,
                span_sec=(
                    trajectory.maximum_time_sec
                    - trajectory.minimum_time_sec
                ),
            ),
            clock=ContinuousTimeClockSemantics(),
            knots=[
                ContinuousTimeKnotPose(
                    timestamp_sec=item.timestamp_sec,
                    transform_world_body=TransformResult(
                        parent=world_frame,
                        child=body_frame,
                        translation_m=list(item.transform_world_body.translation_m),
                        rotation_quat_xyzw=list(
                            item.transform_world_body.rotation_quat_xyzw
                        ),
                    ),
                )
                for item in trajectory.poses
            ],
            provenance=provenance,
        )


def continuous_time_trajectory_json_schema() -> dict[str, Any]:
    """Return the JSON schema for continuous-time trajectory contracts."""

    return ContinuousTimeTrajectoryContract.model_json_schema()


def load_continuous_time_trajectory(
    path: str | Path,
) -> ContinuousTimeTrajectoryContract:
    """Load and validate a continuous-time trajectory contract."""

    return ContinuousTimeTrajectoryContract.model_validate(
        read_mapping(Path(path))
    )