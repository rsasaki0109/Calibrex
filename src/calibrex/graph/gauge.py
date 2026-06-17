"""Gauge constraint descriptors."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GaugeConstraint:
    """A constraint that fixes an otherwise free gauge direction."""

    name: str
    variable: str
    constraint_type: str
    reason: str

    def as_dict(self) -> dict[str, str]:
        """Return a serializable representation."""

        return {
            "name": self.name,
            "variable": self.variable,
            "constraint_type": self.constraint_type,
            "reason": self.reason,
        }
