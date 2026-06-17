"""Common solver adapter contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from calibrex.core.config import CalibrationConfig
from calibrex.core.frames import FrameGraph
from calibrex.core.geometry import SE3
from calibrex.core.result import MetricResult
from calibrex.data.inspect import DatasetInspection


@dataclass(frozen=True)
class SolverAdapterResult:
    """Backend-neutral solver output used before conversion to result.yaml."""

    backend: str
    available: bool
    status: str
    metrics: dict[str, MetricResult] = field(default_factory=dict)
    transforms: dict[str, SE3] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class SolverAdapter(ABC):
    """Base class for optional solver adapters."""

    backend: str

    @abstractmethod
    def solve(
        self,
        config: CalibrationConfig,
        frame_graph: FrameGraph,
        inspection: DatasetInspection,
    ) -> SolverAdapterResult:
        """Run or describe a solver backend."""
