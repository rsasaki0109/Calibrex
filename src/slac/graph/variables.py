"""Backend-neutral graph variable descriptors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

VariableKind = Literal[
    "pose",
    "trajectory",
    "extrinsic",
    "intrinsic",
    "time_offset",
    "imu_bias",
    "map",
    "velocity",
    "control_grid",
]


@dataclass(frozen=True)
class VariableDescriptor:
    """A variable block to be estimated or held fixed."""

    name: str
    kind: VariableKind
    dimension: int
    estimate: bool = True
    frame: str | None = None
    sensor: str | None = None
    initial: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable representation."""

        return {
            "name": self.name,
            "kind": self.kind,
            "dimension": self.dimension,
            "estimate": self.estimate,
            "frame": self.frame,
            "sensor": self.sensor,
            "initial": self.initial,
        }
