"""Static problem observability diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ObservabilityDiagnostic:
    """Static diagnostic computed before numerical optimization."""

    estimated_dimension: int
    fixed_dimension: int
    factor_count: int
    prior_count: int
    gauge_count: int
    weak_directions: list[str] = field(default_factory=list)

    @property
    def grade(self) -> str:
        """Return a coarse pre-solver observability grade."""

        if self.factor_count == 0 or self.gauge_count == 0:
            return "fail"
        if self.weak_directions:
            return "warn"
        return "pass"

    def as_dict(self) -> dict[str, object]:
        """Return a serializable representation."""

        return {
            "estimated_dimension": self.estimated_dimension,
            "fixed_dimension": self.fixed_dimension,
            "factor_count": self.factor_count,
            "prior_count": self.prior_count,
            "gauge_count": self.gauge_count,
            "weak_directions": list(self.weak_directions),
            "grade": self.grade,
        }
