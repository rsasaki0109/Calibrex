import pytest

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
