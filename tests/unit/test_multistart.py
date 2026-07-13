import pytest

from calibrex.evaluation.multistart import (
    MultiStartSolution,
    PatternSearchOptions,
    coordinate_pattern_search,
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


def test_coordinate_pattern_search_recovers_quadratic_minimum() -> None:
    result = coordinate_pattern_search(
        lambda x: (x[0] - 1.0) ** 2 + 2.0 * (x[1] + 0.5) ** 2,
        start_id="far",
        start=(-2.0, 2.0),
        options=PatternSearchOptions(
            initial_steps=(1.0, 1.0),
            minimum_steps=(1.0e-4, 1.0e-4),
            max_sweeps=50,
        ),
    )

    assert result.solution.converged is True
    assert result.solution.parameters == pytest.approx((1.0, -0.5), abs=2.0e-4)
    assert result.solution.objective < 1.0e-7
    assert result.evaluation_count == 1 + 4 * result.completed_sweeps


def test_pattern_search_preserves_two_symmetric_basins() -> None:
    def objective(x: tuple[float, ...]) -> float:
        return (x[0] * x[0] - 1.0) ** 2 + x[1] ** 2

    options = PatternSearchOptions(
        initial_steps=(0.5, 0.5),
        minimum_steps=(1.0e-4, 1.0e-4),
        max_sweeps=40,
    )
    positive = coordinate_pattern_search(
        objective, start_id="positive", start=(2.0, 0.3), options=options
    )
    negative = coordinate_pattern_search(
        objective, start_id="negative", start=(-2.0, -0.3), options=options
    )
    evaluation = evaluate_multistart_solutions(
        (positive.solution, negative.solution),
        parameter_scales=(0.1, 0.1),
        objective_absolute_tolerance=1.0e-6,
        objective_relative_tolerance=0.0,
    )

    assert positive.solution.parameters[0] == pytest.approx(1.0, abs=2.0e-4)
    assert negative.solution.parameters[0] == pytest.approx(-1.0, abs=2.0e-4)
    assert evaluation.ambiguous is True
    assert evaluation.competitive_cluster_count == 2
