"""Backend-neutral multi-start basin and symmetry diagnostics."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MultiStartSolution:
    """One deterministic local-search outcome from a declared start."""

    start_id: str
    parameters: tuple[float, ...]
    objective: float
    converged: bool = True

    def __post_init__(self) -> None:
        if not self.start_id or not self.parameters or not math.isfinite(self.objective):
            raise ValueError("multi-start solution requires ID, parameters, and finite objective")
        if any(not math.isfinite(value) for value in self.parameters):
            raise ValueError("multi-start parameters must be finite")


@dataclass(frozen=True)
class MultiStartCluster:
    """One scale-normalized solution basin represented by its best member."""

    cluster_id: str
    representative_start_id: str
    representative_parameters: tuple[float, ...]
    best_objective: float
    member_start_ids: tuple[str, ...]
    basin_fraction: float

    def as_dict(self) -> dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "representative_start_id": self.representative_start_id,
            "representative_parameters": list(self.representative_parameters),
            "best_objective": self.best_objective,
            "member_start_ids": list(self.member_start_ids),
            "basin_fraction": self.basin_fraction,
        }


@dataclass(frozen=True)
class MultiStartEvaluation:
    """Clustered local minima with explicit competitive-basin ambiguity."""

    declared_start_count: int
    converged_start_count: int
    cluster_count: int
    competitive_cluster_count: int
    best_cluster_id: str | None
    best_objective: float | None
    best_to_second_objective_gap: float | None
    max_competitive_normalized_separation: float | None
    ambiguous: bool
    parameter_scales: tuple[float, ...]
    cluster_radius: float
    objective_absolute_tolerance: float
    objective_relative_tolerance: float
    clusters: tuple[MultiStartCluster, ...]
    nonconverged_start_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "declared_start_count": self.declared_start_count,
            "converged_start_count": self.converged_start_count,
            "cluster_count": self.cluster_count,
            "competitive_cluster_count": self.competitive_cluster_count,
            "best_cluster_id": self.best_cluster_id,
            "best_objective": self.best_objective,
            "best_to_second_objective_gap": self.best_to_second_objective_gap,
            "max_competitive_normalized_separation": (self.max_competitive_normalized_separation),
            "ambiguous": self.ambiguous,
            "parameter_scales": list(self.parameter_scales),
            "cluster_radius": self.cluster_radius,
            "objective_absolute_tolerance": self.objective_absolute_tolerance,
            "objective_relative_tolerance": self.objective_relative_tolerance,
            "clusters": [cluster.as_dict() for cluster in self.clusters],
            "nonconverged_start_ids": list(self.nonconverged_start_ids),
        }


def evaluate_multistart_solutions(
    solutions: tuple[MultiStartSolution, ...],
    *,
    parameter_scales: tuple[float, ...],
    cluster_radius: float = 1.0,
    objective_absolute_tolerance: float = 1.0e-6,
    objective_relative_tolerance: float = 0.01,
) -> MultiStartEvaluation:
    """Cluster converged solutions and flag separated objective-equivalent basins."""

    if not parameter_scales or any(scale <= 0.0 for scale in parameter_scales):
        raise ValueError("multi-start parameter scales must be positive")
    if (
        cluster_radius <= 0.0
        or min(objective_absolute_tolerance, objective_relative_tolerance) < 0.0
    ):
        raise ValueError("multi-start radius must be positive and tolerances nonnegative")
    if len({solution.start_id for solution in solutions}) != len(solutions):
        raise ValueError("multi-start solution IDs must be unique")
    if any(len(solution.parameters) != len(parameter_scales) for solution in solutions):
        raise ValueError("multi-start parameter dimensions must match scales")
    converged = sorted(
        (solution for solution in solutions if solution.converged),
        key=lambda solution: (solution.objective, solution.start_id),
    )
    members: list[list[MultiStartSolution]] = []
    for solution in converged:
        distances = [
            _normalized_distance(solution.parameters, cluster[0].parameters, parameter_scales)
            for cluster in members
        ]
        if distances and min(distances) <= cluster_radius:
            members[distances.index(min(distances))].append(solution)
        else:
            members.append([solution])
    clusters = tuple(
        MultiStartCluster(
            cluster_id=f"basin-{index:03d}",
            representative_start_id=cluster[0].start_id,
            representative_parameters=cluster[0].parameters,
            best_objective=cluster[0].objective,
            member_start_ids=tuple(item.start_id for item in cluster),
            basin_fraction=len(cluster) / len(converged),
        )
        for index, cluster in enumerate(members)
    )
    best = clusters[0] if clusters else None
    tolerance = (
        objective_absolute_tolerance + objective_relative_tolerance * abs(best.best_objective)
        if best is not None
        else 0.0
    )
    competitive = tuple(
        cluster
        for cluster in clusters
        if best is not None and cluster.best_objective <= best.best_objective + tolerance
    )
    separations = [
        _normalized_distance(
            left.representative_parameters,
            right.representative_parameters,
            parameter_scales,
        )
        for left_index, left in enumerate(competitive)
        for right in competitive[left_index + 1 :]
    ]
    return MultiStartEvaluation(
        declared_start_count=len(solutions),
        converged_start_count=len(converged),
        cluster_count=len(clusters),
        competitive_cluster_count=len(competitive),
        best_cluster_id=best.cluster_id if best is not None else None,
        best_objective=best.best_objective if best is not None else None,
        best_to_second_objective_gap=(
            clusters[1].best_objective - best.best_objective
            if best is not None and len(clusters) > 1
            else None
        ),
        max_competitive_normalized_separation=(max(separations) if separations else None),
        ambiguous=len(competitive) > 1,
        parameter_scales=parameter_scales,
        cluster_radius=cluster_radius,
        objective_absolute_tolerance=objective_absolute_tolerance,
        objective_relative_tolerance=objective_relative_tolerance,
        clusters=clusters,
        nonconverged_start_ids=tuple(
            sorted(solution.start_id for solution in solutions if not solution.converged)
        ),
    )


def _normalized_distance(
    left: tuple[float, ...],
    right: tuple[float, ...],
    scales: tuple[float, ...],
) -> float:
    return math.sqrt(
        sum(
            ((left_value - right_value) / scale) ** 2
            for left_value, right_value, scale in zip(left, right, scales, strict=True)
        )
    )
