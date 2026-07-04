"""Backend-neutral prior descriptors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PriorDescriptor:
    """A prior attached to a graph variable."""

    name: str
    variable: str
    prior_type: str
    sigma: dict[str, float] = field(default_factory=dict)
    value: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable representation."""

        return {
            "name": self.name,
            "variable": self.variable,
            "prior_type": self.prior_type,
            "sigma": self.sigma,
            "value": self.value,
        }
