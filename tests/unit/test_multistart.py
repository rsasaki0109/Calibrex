import pytest

from calibrex.evaluation.multistart import (
    MultiStartSolution,
    evaluate_multistart_solutions,
)


def _solution(name: str, x: float, y: float, objective: float) -> MultiStartSolution:
    return MultiStartSolution(name, (x, y), objective)


def test_multistart_clusters_duplicate_starts_into_one_basin() -> None:
    result = evaluate_multistart_solutions(
        (
            _solution("center", 0.0, 0.0, 1.0),
            _solution("near-a", 0.01, -0.02, 1.01),
            _solution("near-b", -0.02, 0.01, 1.02),
        ),
        parameter_scales=(0.1, 0.1),
        cluster_radius=0.5,
    )

    assert result.cluster_count == 1
    assert result.competitive_cluster_count == 1
    assert result.ambiguous is False
    assert result.clusters[0].basin_fraction == 1.0


def test_multistart_flags_separated_objective_equivalent_symmetry() -> None:
    result = evaluate_multistart_solutions(
        (
            _solution("positive", 1.0, 0.0, 0.5),
            _solution("negative", -1.0, 0.0, 0.5001),
            _solution("worse", 0.0, 2.0, 1.0),
        ),
        parameter_scales=(0.1, 0.2),
        objective_absolute_tolerance=0.001,
        objective_relative_tolerance=0.0,
    )

    assert result.cluster_count == 3
    assert result.competitive_cluster_count == 2
    assert result.ambiguous is True
    assert result.best_to_second_objective_gap == pytest.approx(0.0001)
    assert result.max_competitive_normalized_separation == pytest.approx(20.0)


def test_multistart_excludes_nonconverged_and_validates_dimensions() -> None:
    result = evaluate_multistart_solutions(
        (
            _solution("good", 0.0, 0.0, 0.0),
            MultiStartSolution("failed", (5.0, 5.0), -1.0, converged=False),
        ),
        parameter_scales=(1.0, 1.0),
    )
    assert result.converged_start_count == 1
    assert result.nonconverged_start_ids == ("failed",)

    with pytest.raises(ValueError, match="dimensions"):
        evaluate_multistart_solutions((_solution("bad", 0.0, 0.0, 0.0),), parameter_scales=(1.0,))
