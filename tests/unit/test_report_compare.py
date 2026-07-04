import copy

import pytest

from slac.core.result import (
    CalibrationResult,
    FrameGraphSnapshot,
    MetricResult,
    RunInfo,
    TransformEstimateProvenance,
    TransformResult,
)
from slac.evaluation.report_compare import (
    REPORT_COMPARISON_SCHEMA_VERSION,
    compare_reports,
    report_comparison_json_schema,
)


def _livox_pair_provenance(metrics_origin: str = "cached") -> dict[str, object]:
    return {
        "metrics_origin": metrics_origin,
        "data_verified": False,
        "livox_pair_evidence": {
            "candidate_transform": "T_source_target",
            "transform_convention": (
                "T_source_target maps target PCD points into the source PCD frame"
            ),
            "known_bad_case_count": 24,
            "holdout_geometry": {
                "split_policy": "single_pair_source_map_target_query",
                "independent_holdout": False,
                "support_population_id": (
                    "livox_pair_support:train=base:holdout=target:"
                    "eligible_points=1000:voxel_m=1:gate_m=1.5"
                ),
                "support_definition": "eligible population is every finite target PCD point",
                "voxel_size_m": 1.0,
                "correspondence_gate_m": 1.5,
                "inlier_threshold_m": 0.25,
                "plane_normal_source": "pcd_normal_fields_or_local_fallback",
            },
        },
    }


def _result(
    run_id: str,
    *,
    rmse_holdout: float,
    support_ratio: float | None = None,
    provenance: dict[str, object] | None = None,
    transform_provenance: TransformEstimateProvenance | None = None,
) -> CalibrationResult:
    metrics = {
        "lidar_point_to_plane_rmse_m": MetricResult(
            train=rmse_holdout,
            holdout=rmse_holdout,
            grade="pass",
            unit="m",
        ),
    }
    if support_ratio is not None:
        metrics["lidar_pair_holdout_point_to_plane_support_ratio"] = MetricResult(
            value=support_ratio,
            grade="pass",
        )
    return CalibrationResult(
        run=RunInfo(
            id=run_id,
            slac_version="0.1.0",
            provenance=provenance or {},
        ),
        frame_graph=FrameGraphSnapshot(root="base", frames={"base": None, "lidar0": "base"}),
        transforms={
            "T_base_lidar0": TransformResult(
                parent="base",
                child="lidar0",
                translation_m=[0.0, 0.0, 0.0],
                rotation_quat_xyzw=[0.0, 0.0, 0.0, 1.0],
                provenance=transform_provenance or TransformEstimateProvenance(),
            )
        },
        metrics=metrics,
    )


def test_compare_reports_ranks_compatible_results_across_all_pairs() -> None:
    best = _result(
        "best",
        rmse_holdout=0.01,
        support_ratio=0.9,
        provenance=_livox_pair_provenance(),
    )
    middle = _result(
        "middle",
        rmse_holdout=0.02,
        support_ratio=0.8,
        provenance=_livox_pair_provenance(),
    )
    worst = _result(
        "worst",
        rmse_holdout=0.03,
        support_ratio=0.7,
        provenance=_livox_pair_provenance(),
    )

    report = compare_reports(
        [("best", best), ("middle", middle), ("worst", worst)],
        paths={"best": "best.yaml", "middle": "middle.yaml", "worst": "worst.yaml"},
    )

    assert report.schema_version == REPORT_COMPARISON_SCHEMA_VERSION
    assert report.summary.entry_count == 3
    assert report.summary.comparison_mode == "all_pairs"
    assert report.summary.reference_label is None
    assert report.summary.pairwise_comparison_count == 3
    assert report.summary.protocol_compatibility_status == "compatible"
    assert report.summary.compatible_pair_count == 3
    assert report.summary.not_comparable_pairs == []
    assert set(report.pairwise) == {"best__middle", "best__worst", "middle__worst"}
    assert report.entries["best"].path == "best.yaml"
    assert report.pairwise["best__middle"].comparison.left.run_id == "best"
    assert report.pairwise["best__middle"].comparison.right.run_id == "middle"

    lidar = report.metric_family_rankings["lidar"]
    assert lidar.metric_count == 2
    assert [entry.label for entry in lidar.entries] == ["best", "middle", "worst"]
    assert [entry.rank for entry in lidar.entries] == [1, 2, 3]
    assert lidar.entries[0].win_count == 4
    assert lidar.entries[0].loss_count == 0
    assert lidar.entries[2].win_count == 0
    assert lidar.entries[2].loss_count == 4


def test_compare_reports_reference_mode_only_compares_against_reference() -> None:
    reference = _result("reference", rmse_holdout=0.02)
    candidate_a = _result("candidate_a", rmse_holdout=0.01)
    candidate_b = _result("candidate_b", rmse_holdout=0.03)

    report = compare_reports(
        [
            ("reference", reference),
            ("candidate_a", candidate_a),
            ("candidate_b", candidate_b),
        ],
        reference_label="reference",
    )

    assert report.summary.comparison_mode == "reference_vs_each"
    assert report.summary.reference_label == "reference"
    assert report.summary.pairwise_comparison_count == 2
    assert set(report.pairwise) == {
        "reference__candidate_a",
        "reference__candidate_b",
    }
    assert report.entries["reference"].is_reference is True
    assert report.entries["candidate_a"].is_reference is False


def test_compare_reports_surfaces_incompatible_protocols_without_silent_ranking() -> None:
    declared = _result(
        "declared",
        rmse_holdout=0.02,
        support_ratio=0.8,
        provenance=_livox_pair_provenance(),
    )
    undeclared = _result("undeclared", rmse_holdout=0.01, support_ratio=0.9)

    report = compare_reports([("declared", declared), ("undeclared", undeclared)])

    assert report.summary.protocol_compatibility_status == "not_comparable"
    assert report.summary.not_comparable_pair_count == 1
    assert report.summary.not_comparable_pairs == ["declared__undeclared"]
    pair = report.pairwise["declared__undeclared"]
    assert pair.comparison.protocol_compatibility.status == "not_comparable"
    assert pair.comparison.protocol_compatibility.reasons == [
        "no shared evidence protocol IDs"
    ]
    gated = pair.comparison.metrics["lidar_pair_holdout_point_to_plane_support_ratio"]
    assert gated.winner == "not_comparable"
    assert gated.not_comparable_reason is not None
    assert "not_comparable" in gated.not_comparable_reason
    lidar = report.metric_family_rankings["lidar"]
    by_label = {entry.label: entry for entry in lidar.entries}
    assert by_label["declared"].not_comparable_count == 1
    assert by_label["undeclared"].not_comparable_count == 1


def test_compare_reports_mixed_protocol_statuses_roll_up_to_worst() -> None:
    declared_left = _result(
        "declared_left",
        rmse_holdout=0.02,
        provenance=_livox_pair_provenance(),
    )
    declared_right = _result(
        "declared_right",
        rmse_holdout=0.03,
        provenance=_livox_pair_provenance(),
    )
    undeclared = _result("undeclared", rmse_holdout=0.01)

    report = compare_reports(
        [
            ("declared_left", declared_left),
            ("declared_right", declared_right),
            ("undeclared", undeclared),
        ]
    )

    assert report.summary.protocol_compatibility_status == "not_comparable"
    assert report.summary.compatible_pair_count == 1
    assert report.summary.not_comparable_pair_count == 2
    assert (
        report.pairwise["declared_left__declared_right"]
        .comparison.protocol_compatibility.status
        == "compatible"
    )


def test_compare_reports_distinguishes_entry_provenance_roles() -> None:
    reference = _result(
        "dataset_reference",
        rmse_holdout=0.02,
        transform_provenance=TransformEstimateProvenance(
            producer="dataset_provider",
            execution_mode="dataset_reference",
            role_in_comparison="selected_reference",
            evidence_level="dataset_provided",
        ),
    )
    external = _result(
        "koide_baseline",
        rmse_holdout=0.02,
        transform_provenance=TransformEstimateProvenance(
            producer="external_tool",
            execution_mode="imported",
            role_in_comparison="comparison_baseline",
            evidence_level="algorithmically_refined",
            tool_name="direct_visual_lidar_calibration",
        ),
    )
    native = _result(
        "slac_output",
        rmse_holdout=0.02,
        transform_provenance=TransformEstimateProvenance(
            producer="slac_native",
            execution_mode="offline_batch",
            role_in_comparison="output",
            evidence_level="algorithmically_refined",
        ),
    )

    report = compare_reports(
        [
            ("reference", reference),
            ("external", external),
            ("native", native),
        ],
        reference_label="reference",
    )

    assert report.entries["reference"].provenance.dominant_producer == "dataset_provider"
    assert report.entries["reference"].provenance.dominant_role == "selected_reference"
    assert (
        report.entries["reference"].provenance.dominant_evidence_level
        == "dataset_provided"
    )
    assert report.entries["external"].provenance.dominant_producer == "external_tool"
    assert report.entries["external"].provenance.dominant_role == "comparison_baseline"
    assert report.entries["native"].provenance.dominant_producer == "slac_native"
    assert report.entries["native"].provenance.dominant_role == "output"
    assert report.entries["native"].provenance.provenance_group == "transforms"


def test_compare_reports_falls_back_to_candidate_extrinsics_provenance() -> None:
    result = _result("candidate_only", rmse_holdout=0.02)
    result.candidate_extrinsics = copy.deepcopy(result.transforms)
    result.candidate_extrinsics["T_base_lidar0"].provenance = TransformEstimateProvenance(
        producer="human",
        execution_mode="manual",
        role_in_comparison="candidate",
        evidence_level="imported_without_documented_derivation",
    )
    result.transforms = {}
    other = _result("other", rmse_holdout=0.03)

    report = compare_reports([("candidate", result), ("other", other)])

    provenance = report.entries["candidate"].provenance
    assert provenance.provenance_group == "candidate_extrinsics"
    assert provenance.dominant_producer == "human"
    assert provenance.dominant_role == "candidate"
    assert report.entries["other"].provenance.provenance_group == "transforms"
    assert report.entries["other"].provenance.dominant_producer == "unknown"


def test_compare_reports_rejects_invalid_inputs() -> None:
    result = _result("only", rmse_holdout=0.02)
    with pytest.raises(ValueError, match="at least two"):
        compare_reports([("only", result)])
    with pytest.raises(ValueError, match="duplicate"):
        compare_reports([("dup", result), ("dup", result)])
    with pytest.raises(ValueError, match="reference label"):
        compare_reports(
            [("a", result), ("b", result)],
            reference_label="missing",
        )


def test_report_comparison_json_schema_names_versioned_artifact() -> None:
    schema = report_comparison_json_schema()
    assert schema["properties"]["schema_version"]["const"] == (
        "slac.report_comparison/v0.1"
    )
