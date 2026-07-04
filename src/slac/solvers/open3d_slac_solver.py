"""Open3D SLAC adapter boundary."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

from slac.core.config import CalibrationConfig
from slac.core.frames import FrameGraph
from slac.core.result import MetricResult
from slac.data.inspect import DatasetInspection
from slac.data.manifest import load_manifest
from slac.solvers.base import SolverAdapter, SolverAdapterResult


@dataclass(frozen=True)
class Open3DSLACInputs:
    """Open3D SLAC input summary extracted from a slac dataset."""

    fragment_count: int | None
    pose_graph_edges: int | None
    has_fragments: bool
    has_pose_graph: bool


class Open3DSLACSolver(SolverAdapter):
    """Optional adapter for Open3D tensor SLAC pipelines."""

    backend = "open3d_slac"

    def solve(
        self,
        config: CalibrationConfig,
        frame_graph: FrameGraph,
        inspection: DatasetInspection,
    ) -> SolverAdapterResult:
        """Inspect Open3D SLAC readiness and return slac metrics."""

        del frame_graph
        inputs = _summarize_inputs(config, inspection)
        available = importlib.util.find_spec("open3d") is not None
        metrics = {
            "open3d_slac_backend_available": MetricResult(
                value=1.0 if available else 0.0,
                grade="pass" if available else "warn",
                reason=(
                    "Open3D is importable"
                    if available
                    else "Open3D optional dependency is not installed; adapter was not executed"
                ),
            ),
            "rgbd_fragment_count": _count_metric(
                metric_name="rgbd_fragment_count",
                count=inputs.fragment_count,
                has_stream=inputs.has_fragments,
                stream_label="RGB-D fragments",
            ),
            "pose_graph_edges": _count_metric(
                metric_name="pose_graph_edges",
                count=inputs.pose_graph_edges,
                has_stream=inputs.has_pose_graph,
                stream_label="pose graph edges",
            ),
        }
        warnings: list[str] = []
        if not available:
            warnings.append("install slac[open3d] to execute Open3D SLAC")
        if not inputs.has_fragments:
            warnings.append("provide RGB-D fragments or fragment metadata")
        if not inputs.has_pose_graph:
            warnings.append("provide an Open3D-compatible pose graph")
        status = (
            "ready"
            if available and inputs.has_fragments and inputs.has_pose_graph
            else "not_executed"
        )
        return SolverAdapterResult(
            backend=self.backend,
            available=available,
            status=status,
            metrics=metrics,
            provenance={
                "open3d_available": available,
                "open3d_slac_status": status,
                "fragment_count": inputs.fragment_count,
                "pose_graph_edges": inputs.pose_graph_edges,
            },
            warnings=warnings,
        )


def _summarize_inputs(
    config: CalibrationConfig,
    inspection: DatasetInspection,
) -> Open3DSLACInputs:
    if inspection.manifest is None:
        return Open3DSLACInputs(
            fragment_count=0,
            pose_graph_edges=0,
            has_fragments=False,
            has_pose_graph=False,
        )
    manifest = load_manifest(Path(inspection.manifest))
    fragment_count: int | None = 0
    pose_graph_edges: int | None = 0
    has_fragments = False
    has_pose_graph = False
    for name, stream in manifest.streams.items():
        stream_name = name.lower()
        if stream.kind in {"rgbd", "depth_image"} or "fragment" in stream_name:
            has_fragments = True
            fragment_count = _accumulate_optional_count(fragment_count, stream.count)
        if stream.kind == "pose_graph" or "pose_graph" in stream_name:
            has_pose_graph = True
            pose_graph_edges = _accumulate_optional_count(pose_graph_edges, stream.count)
    if config.pipeline.factors.get("open3d_slac") is not None:
        has_fragments = has_fragments or _positive_count(fragment_count)
    return Open3DSLACInputs(
        fragment_count=fragment_count,
        pose_graph_edges=pose_graph_edges,
        has_fragments=has_fragments,
        has_pose_graph=has_pose_graph,
    )


def _accumulate_optional_count(total: int | None, value: int | None) -> int | None:
    if value is None:
        return None
    if total is None:
        return None
    return total + value


def _positive_count(value: int | None) -> bool:
    return value is not None and value > 0


def _count_metric(
    metric_name: str,
    count: int | None,
    has_stream: bool,
    stream_label: str,
) -> MetricResult:
    if count is None:
        return MetricResult(
            value=None,
            grade="warn",
            reason=f"{stream_label} are declared but count is unknown",
        )
    if count > 0:
        return MetricResult(
            value=float(count),
            grade="pass",
            reason=f"{count} {stream_label} declared",
        )
    return MetricResult(
        value=0.0 if has_stream else None,
        grade="warn",
        reason=(
            f"{stream_label} stream is declared but no files are counted"
            if has_stream
            else f"no {stream_label} declared in dataset manifest"
        ),
    )
