"""Feasible clock shifts shared by every fitted IMU observation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from calibrex.core.result import StrictModel


class ClockDomainEvidence(StrictModel):
    """Bounds and rejected candidates; not a constrained-optimality certificate."""

    lower_sec: float = Field(allow_inf_nan=False)
    upper_sec: float = Field(allow_inf_nan=False)
    rejected_steps: int = Field(ge=0)
    derivative_step_sec: float = Field(allow_inf_nan=False)
    opposite_gradient_norm: float | None = Field(default=None, ge=0, allow_inf_nan=False)


@dataclass(frozen=True)
class ClockOffsetDomain:
    """Intersection of start <= IMU_time - offset <= end for all observations."""

    start: float
    end: float
    first_observation: float
    last_observation: float

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(v)
            for v in (
                self.start,
                self.end,
                self.first_observation,
                self.last_observation,
                self.lower,
                self.upper,
            )
        ):
            raise ValueError("clock domain and observation times must be finite")
        if self.end <= self.start or self.last_observation < self.first_observation:
            raise ValueError("clock domain and observation times must be ordered")
        if self.upper <= self.lower:
            raise ValueError("clock estimation requires a nonempty, positive-width time domain")

    @property
    def lower(self) -> float:
        """Smallest clock offset keeping the last observation inside the trajectory."""
        return self.last_observation - self.end

    @property
    def upper(self) -> float:
        """Largest clock offset keeping the first observation inside the trajectory."""
        return self.first_observation - self.start

    def contains(self, offset: float) -> bool:
        """Check both computed bounds and actual floating-point timestamp subtraction."""
        return (
            math.isfinite(offset)
            and self.lower <= offset <= self.upper
            and self.start <= self.first_observation - offset
            and self.last_observation - offset <= self.end
        )

    def difference_step(
        self,
        offset: float,
        epsilon: float = 1.0e-6,
        *,
        direction: Literal[-1, 1] | None = None,
    ) -> float:
        """Choose one feasible signed perturbation shared by all factors.

        Prefer the historical positive-clock derivative when a full stencil fits;
        otherwise use the inward negative side. On very narrow domains, use the
        larger available side. Return the representable increment actually used.
        """
        if not self.contains(offset) or not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("clock derivative requires a feasible offset and positive step")
        positive, negative = self.upper - offset, offset - self.lower
        targets: tuple[float, ...]
        if direction == 1:
            targets = (offset + min(epsilon, positive),)
        elif direction == -1:
            targets = (offset - min(epsilon, negative),)
        elif positive >= epsilon:
            targets = (offset + epsilon, offset - min(epsilon, negative))
        elif negative >= epsilon:
            targets = (offset - epsilon, offset + min(epsilon, positive))
        elif positive >= negative:
            targets = (self.upper, self.lower)
        else:
            targets = (self.lower, self.upper)
        for target in targets:
            # The subtraction that formed a bound may round in the other direction
            # from the timestamp subtraction. Move inward, never expand the domain.
            if not self.contains(target):
                target = math.nextafter(target, offset)
            if target != offset and self.contains(target):
                return target - offset
        raise ValueError("clock domain has no representable finite-difference perturbation")
