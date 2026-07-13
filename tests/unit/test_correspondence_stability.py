import pytest

from calibrex.data.livox import (
    LivoxPointRecord,
    build_voxel_plane_map,
    nearest_voxel_plane_match,
)
from calibrex.evaluation.correspondence import (
    CorrespondenceAssignment,
    evaluate_correspondence_stability,
)


def _assignment(query: str, target: str) -> CorrespondenceAssignment:
    return CorrespondenceAssignment(query, target)


def test_correspondence_stability_separates_drop_add_and_reassignment() -> None:
    evaluation = evaluate_correspondence_stability(
        (
            _assignment("q0", "a"),
            _assignment("q1", "b"),
            _assignment("q2", "c"),
            _assignment("q3", "d"),
        ),
        (
            _assignment("q0", "a"),
            _assignment("q1", "x"),
            _assignment("q3", "d"),
            _assignment("q4", "e"),
        ),
    )

    assert evaluation.reference_count == 4
    assert evaluation.rematched_count == 4
    assert evaluation.common_query_count == 3
    assert evaluation.retained_query_fraction == pytest.approx(0.75)
    assert evaluation.query_jaccard == pytest.approx(3.0 / 5.0)
    assert evaluation.pair_jaccard == pytest.approx(2.0 / 6.0)
    assert evaluation.same_target_fraction == pytest.approx(2.0 / 3.0)
    assert evaluation.dropped_query_ids == ("q2",)
    assert evaluation.added_query_ids == ("q4",)
    assert evaluation.reassigned_query_ids == ("q1",)
    assert evaluation.as_dict()["pair_jaccard"] == pytest.approx(1.0 / 3.0)


def test_empty_correspondence_stability_is_explicitly_undefined() -> None:
    evaluation = evaluate_correspondence_stability((), ())

    assert evaluation.retained_query_fraction is None
    assert evaluation.query_jaccard is None
    assert evaluation.pair_jaccard is None
    assert evaluation.same_target_fraction is None


def test_correspondence_stability_rejects_duplicate_query_ids() -> None:
    with pytest.raises(ValueError, match=r"reference.*unique"):
        evaluate_correspondence_stability((_assignment("q0", "a"), _assignment("q0", "b")), ())


def test_voxel_plane_match_exposes_stable_target_identity() -> None:
    records = [
        LivoxPointRecord((0.01 * index, 0.0, 1.0, 0.0), (0.0, 0.0, 1.0)) for index in range(8)
    ]
    plane_map = build_voxel_plane_map(records, 0.5)

    match = nearest_voxel_plane_match(
        (0.04, 0.02, 1.01),
        plane_map,
        voxel_size_m=0.5,
        correspondence_gate_m=0.1,
    )

    assert match is not None
    assert match.voxel_key == (0, 0, 2)
    assert match.target_id == "voxel:0:0:2"
    assert match.normal == pytest.approx((0.0, 0.0, 1.0))
    assert match.centroid_distance_m < 0.03
