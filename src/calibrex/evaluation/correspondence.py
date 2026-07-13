"""Backend-neutral correspondence stability diagnostics for iterative registration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, order=True)
class CorrespondenceAssignment:
    """One deterministic query-to-target association."""

    query_id: str
    target_id: str

    def __post_init__(self) -> None:
        if not self.query_id or not self.target_id:
            raise ValueError("correspondence query_id and target_id must be non-empty")


@dataclass(frozen=True)
class CorrespondenceStabilityEvaluation:
    """Exact set and query-conditional changes between two association rounds."""

    reference_count: int
    rematched_count: int
    common_query_count: int
    retained_query_fraction: float | None
    query_jaccard: float | None
    pair_jaccard: float | None
    same_target_fraction: float | None
    dropped_query_ids: tuple[str, ...]
    added_query_ids: tuple[str, ...]
    reassigned_query_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a schema-safe diagnostic payload."""

        return {
            "reference_count": self.reference_count,
            "rematched_count": self.rematched_count,
            "common_query_count": self.common_query_count,
            "retained_query_fraction": self.retained_query_fraction,
            "query_jaccard": self.query_jaccard,
            "pair_jaccard": self.pair_jaccard,
            "same_target_fraction": self.same_target_fraction,
            "dropped_query_ids": list(self.dropped_query_ids),
            "added_query_ids": list(self.added_query_ids),
            "reassigned_query_ids": list(self.reassigned_query_ids),
        }


def evaluate_correspondence_stability(
    reference: tuple[CorrespondenceAssignment, ...],
    rematched: tuple[CorrespondenceAssignment, ...],
) -> CorrespondenceStabilityEvaluation:
    """Compare unique query assignments without hiding drops or reassignments."""

    reference_by_query = _by_query(reference, label="reference")
    rematched_by_query = _by_query(rematched, label="rematched")
    reference_queries = set(reference_by_query)
    rematched_queries = set(rematched_by_query)
    common_queries = reference_queries & rematched_queries
    query_union = reference_queries | rematched_queries
    reference_pairs = {(item.query_id, item.target_id) for item in reference}
    rematched_pairs = {(item.query_id, item.target_id) for item in rematched}
    pair_union = reference_pairs | rematched_pairs
    same = {
        query_id
        for query_id in common_queries
        if reference_by_query[query_id] == rematched_by_query[query_id]
    }
    reassigned = common_queries - same
    return CorrespondenceStabilityEvaluation(
        reference_count=len(reference),
        rematched_count=len(rematched),
        common_query_count=len(common_queries),
        retained_query_fraction=(
            len(common_queries) / len(reference_queries) if reference_queries else None
        ),
        query_jaccard=(len(common_queries) / len(query_union) if query_union else None),
        pair_jaccard=(
            len(reference_pairs & rematched_pairs) / len(pair_union) if pair_union else None
        ),
        same_target_fraction=(len(same) / len(common_queries) if common_queries else None),
        dropped_query_ids=tuple(sorted(reference_queries - rematched_queries)),
        added_query_ids=tuple(sorted(rematched_queries - reference_queries)),
        reassigned_query_ids=tuple(sorted(reassigned)),
    )


def _by_query(assignments: tuple[CorrespondenceAssignment, ...], *, label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for assignment in assignments:
        if assignment.query_id in result:
            raise ValueError(f"{label} correspondence query IDs must be unique")
        result[assignment.query_id] = assignment.target_id
    return result
