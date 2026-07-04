from slac.core.assessment import (
    AssessmentArtifact,
    AssessmentPolicy,
    AssessmentRuleResult,
    assess_report_evidence,
)
from slac.core.report_artifacts import (
    EvidenceCaseItem,
    EvidenceInputFileItem,
    EvidenceMaterializationInfo,
    EvidenceProtocolItem,
    EvidenceSummaryItem,
    ReportEvidenceArtifact,
    ReportRunInfo,
)


def test_raw_recomputation_requires_verified_raw_inputs() -> None:
    assert _raw_recomputation_status(data_verified=True, input_files=True) == "pass"
    assert _raw_recomputation_status(data_verified=True, input_files=False) == "inconclusive"
    assert _raw_recomputation_status(data_verified=False, input_files=True) == "inconclusive"
    assert _raw_recomputation_status(data_verified=None, input_files=True) == "inconclusive"


def _raw_recomputation_status(*, data_verified: bool | None, input_files: bool) -> str:
    evidence = ReportEvidenceArtifact(
        run=ReportRunInfo(
            id="assessment-unit",
            status="success",
            domain="robotics",
            slac_version="0.1.0",
            created_at="2026-06-19T00:00:00Z",
        ),
        materialization=EvidenceMaterializationInfo(
            metrics_origin="recomputed",
            data_verified=data_verified,
        ),
        input_files=[
            EvidenceInputFileItem(
                path="raw.pcd",
                sha256="0" * 64,
                size_bytes=128,
            )
        ]
        if input_files
        else [],
    )
    assessment = assess_report_evidence(evidence)
    for rule in assessment.rules:
        if rule.rule_id == "raw_recomputation":
            return rule.status
    raise AssertionError("raw_recomputation rule was not emitted")


def test_known_bad_controls_fail_when_only_support_collapse_is_detected() -> None:
    evidence = _known_bad_evidence(
        support_ratio=0.02,
        accepted_correspondence_count=10.0,
        point_to_plane_delta_m=0.10,
    )

    assessment = assess_report_evidence(evidence)
    rule = _rule_status(assessment, "known_bad_controls")

    assert rule.status == "fail"
    assert rule.observed["support_scored_case_count"] == 1
    assert rule.observed["supported_detection_count"] == 0
    assert rule.observed["support_collapse_count"] == 1


def test_known_bad_controls_pass_with_supported_geometry_detection() -> None:
    evidence = _known_bad_evidence(
        support_ratio=0.72,
        accepted_correspondence_count=720.0,
        point_to_plane_delta_m=0.10,
    )

    assessment = assess_report_evidence(evidence)
    rule = _rule_status(assessment, "known_bad_controls")

    assert rule.status == "pass"
    assert rule.observed["support_scored_case_count"] == 1
    assert rule.observed["supported_detection_count"] == 1
    assert rule.observed["supported_pass_fraction"] == 1.0


def test_holdout_support_gate_requires_candidate_independent_denominator() -> None:
    evidence = _known_bad_evidence(
        support_ratio=0.72,
        accepted_correspondence_count=720.0,
        point_to_plane_delta_m=0.10,
        missing_protocol_parameters={
            "eligible_point_count",
            "accepted_correspondence_count",
            "support_ratio",
        },
    )

    assessment = assess_report_evidence(evidence)
    rule = _rule_status(assessment, "holdout_support_gate")

    assert rule.status == "inconclusive"
    assert "candidate-independent eligible population" in rule.reason
    assert rule.observed["map_voxel_count"] == 30.0
    assert rule.observed["matched_point_count"] == 720.0
    assert rule.observed["eligible_point_count"] is None
    assert rule.observed["accepted_correspondence_count"] is None
    assert rule.observed["support_ratio"] is None


def test_known_bad_controls_inconclusive_when_mandatory_challenge_incomplete() -> None:
    evidence = _known_bad_evidence(
        support_ratio=0.72,
        accepted_correspondence_count=720.0,
        point_to_plane_delta_m=0.10,
        challenge={
            "challenge_id": "livox_pair_mandatory_6dof_large_controls/v0.1",
            "mandatory_case_count": 6,
            "mandatory_supported_detection_count": 6,
            "mandatory_support_collapse_count": 0,
        },
    )

    assessment = assess_report_evidence(evidence)
    rule = _rule_status(assessment, "known_bad_controls")

    assert rule.status == "inconclusive"
    assert rule.reason == "mandatory known-bad challenge is incomplete"
    assert rule.observed["mandatory_case_count"] == 6.0


def test_known_bad_controls_fail_when_mandatory_challenge_is_weak() -> None:
    evidence = _known_bad_evidence(
        support_ratio=0.72,
        accepted_correspondence_count=720.0,
        point_to_plane_delta_m=0.10,
        mandatory_supported_case_count=7,
        challenge={
            "challenge_id": "livox_pair_mandatory_6dof_large_controls/v0.1",
            "mandatory_case_count": 12,
            "mandatory_supported_detection_count": 7,
            "mandatory_support_collapse_count": 0,
        },
    )

    assessment = assess_report_evidence(evidence)
    rule = _rule_status(assessment, "known_bad_controls")

    assert rule.status == "fail"
    assert "mandatory known-bad controls were not rejected" in rule.reason
    assert rule.observed["mandatory_supported_detection_count"] == 7.0
    assert rule.observed["materialized_mandatory_supported_detection_count"] == 7


def test_custom_policy_parameters_are_applied_to_mandatory_challenge() -> None:
    evidence = _known_bad_evidence(
        support_ratio=0.72,
        accepted_correspondence_count=720.0,
        point_to_plane_delta_m=0.10,
        mandatory_supported_case_count=7,
        challenge={
            "challenge_id": "livox_pair_mandatory_6dof_large_controls/v0.1",
            "mandatory_case_count": 12,
            "mandatory_supported_detection_count": 7,
            "mandatory_support_collapse_count": 0,
        },
    )
    policy = AssessmentPolicy(
        parameters={
            **AssessmentPolicy().parameters,
            "min_mandatory_supported_detection_count": 7,
        }
    )

    assessment = assess_report_evidence(evidence, policy=policy)
    rule = _rule_status(assessment, "known_bad_controls")

    assert assessment.policy.parameters["min_mandatory_supported_detection_count"] == 7
    assert rule.status == "pass"
    assert rule.thresholds["min_mandatory_supported_detection_count"] == 7.0


def test_known_bad_controls_inconclusive_when_declaration_exceeds_materialized_cases() -> None:
    evidence = _known_bad_evidence(
        support_ratio=0.72,
        accepted_correspondence_count=720.0,
        point_to_plane_delta_m=0.10,
        mandatory_supported_case_count=7,
        challenge={
            "challenge_id": "livox_pair_mandatory_6dof_large_controls/v0.1",
            "mandatory_case_count": 12,
            "mandatory_supported_detection_count": 8,
            "mandatory_support_collapse_count": 0,
        },
    )

    assessment = assess_report_evidence(evidence)
    rule = _rule_status(assessment, "known_bad_controls")

    assert rule.status == "inconclusive"
    assert "supported detection declaration does not match" in rule.reason
    assert rule.observed["mandatory_supported_detection_count"] == 8.0
    assert rule.observed["materialized_mandatory_supported_detection_count"] == 7


def test_known_bad_controls_pass_with_mandatory_challenge_support() -> None:
    evidence = _known_bad_evidence(
        support_ratio=0.72,
        accepted_correspondence_count=720.0,
        point_to_plane_delta_m=0.10,
        mandatory_supported_case_count=8,
        challenge={
            "challenge_id": "livox_pair_mandatory_6dof_large_controls/v0.1",
            "mandatory_case_count": 12,
            "mandatory_supported_detection_count": 8,
            "mandatory_support_collapse_count": 0,
        },
    )

    assessment = assess_report_evidence(evidence)
    rule = _rule_status(assessment, "known_bad_controls")

    assert rule.status == "pass"
    assert rule.observed["mandatory_supported_detection_count"] == 8.0
    assert rule.observed["materialized_mandatory_case_count"] == 12
    assert rule.observed["materialized_mandatory_supported_detection_count"] == 8


def _known_bad_evidence(
    *,
    support_ratio: float,
    accepted_correspondence_count: float,
    point_to_plane_delta_m: float,
    challenge: dict[str, object] | None = None,
    missing_protocol_parameters: set[str] | None = None,
    mandatory_supported_case_count: int | None = None,
) -> ReportEvidenceArtifact:
    parameters: dict[str, object] = {
        "map_voxel_count": 30,
        "matched_point_count": 720,
        "eligible_point_count": 1000,
        "accepted_correspondence_count": 720,
        "support_ratio": 0.72,
        "unmatched_fraction": 0.28,
    }
    if challenge is not None:
        parameters.update(challenge)
    for key in missing_protocol_parameters or set():
        parameters.pop(key, None)
    cases = _mandatory_known_bad_cases(
        supported_count=mandatory_supported_case_count,
        support_ratio=support_ratio,
        accepted_correspondence_count=accepted_correspondence_count,
        point_to_plane_delta_m=point_to_plane_delta_m,
    )
    return ReportEvidenceArtifact(
        run=ReportRunInfo(
            id="assessment-known-bad-unit",
            status="success",
            domain="robotics",
            slac_version="0.1.0",
            created_at="2026-06-19T00:00:00Z",
        ),
        materialization=EvidenceMaterializationInfo(
            metrics_origin="recomputed",
            data_verified=True,
        ),
        input_files=[
            EvidenceInputFileItem(
                path="raw.pcd",
                sha256="0" * 64,
                size_bytes=128,
            )
        ],
        protocols=[
            EvidenceProtocolItem(
                family="lidar_pair",
                protocol_id="livox_pair_single_pair_holdout_point_to_plane/v0.1",
                status="scored",
                split_policy="temporal_block_holdout",
                independent_holdout=True,
                candidate_transform="T_source_target",
                transform_convention=(
                    "T_source_target maps target PCD points into the source PCD frame"
                ),
                known_bad_case_count=len(cases),
                parameters=parameters,
            )
        ],
        summaries=[
            EvidenceSummaryItem(
                family="lidar_pair",
                check="Known-Bad Controls",
                status="pass",
                evidence="known-bad controls separated",
                interpretation="controls are distinguishable",
                metric_ids=["lidar_pair_known_bad_detectable_fraction"],
            ),
            EvidenceSummaryItem(
                family="lidar_pair",
                check="Decision Boundary",
                status="pass",
                evidence="candidate vs controls",
                interpretation="supported under protocol",
            ),
        ],
        cases=cases,
    )


def _mandatory_known_bad_cases(
    *,
    supported_count: int | None,
    support_ratio: float,
    accepted_correspondence_count: float,
    point_to_plane_delta_m: float,
) -> list[EvidenceCaseItem]:
    if supported_count is None:
        return [
            _known_bad_case(
                dof="x_m",
                amount=0.1,
                unit="m",
                status="pass",
                support_ratio=support_ratio,
                accepted_correspondence_count=accepted_correspondence_count,
                point_to_plane_delta_m=point_to_plane_delta_m,
            )
        ]
    probes = [
        ("roll_deg", 1.0, "deg"),
        ("roll_deg", -1.0, "deg"),
        ("pitch_deg", 1.0, "deg"),
        ("pitch_deg", -1.0, "deg"),
        ("yaw_deg", 1.0, "deg"),
        ("yaw_deg", -1.0, "deg"),
        ("x_m", 0.1, "m"),
        ("x_m", -0.1, "m"),
        ("y_m", 0.1, "m"),
        ("y_m", -0.1, "m"),
        ("z_m", 0.1, "m"),
        ("z_m", -0.1, "m"),
    ]
    cases: list[EvidenceCaseItem] = []
    for index, (dof, amount, unit) in enumerate(probes):
        supported = index < supported_count
        cases.append(
            _known_bad_case(
                dof=dof,
                amount=amount,
                unit=unit,
                status="pass" if supported else "warn",
                support_ratio=support_ratio,
                accepted_correspondence_count=accepted_correspondence_count,
                point_to_plane_delta_m=point_to_plane_delta_m if supported else 0.0,
            )
        )
    return cases


def _known_bad_case(
    *,
    dof: str,
    amount: float,
    unit: str,
    status: str,
    support_ratio: float,
    accepted_correspondence_count: float,
    point_to_plane_delta_m: float,
) -> EvidenceCaseItem:
    return EvidenceCaseItem(
        family="lidar_pair",
        case_id=f"{dof}:{amount:+g}{unit}",
        check="Known-Bad Controls",
        status=status,
        dof=dof,
        amount=amount,
        unit=unit,
        metric_values={
            "lidar_pair_holdout_point_to_plane_support_ratio": support_ratio,
            "lidar_pair_holdout_point_to_plane_accepted_correspondence_count": (
                accepted_correspondence_count
            ),
        },
        delta_values={
            "point_to_plane_p90_delta_m": point_to_plane_delta_m,
        },
    )


def _rule_status(
    assessment: AssessmentArtifact,
    rule_id: str,
) -> AssessmentRuleResult:
    for rule in assessment.rules:
        if rule.rule_id == rule_id:
            return rule
    raise AssertionError(f"{rule_id} rule was not emitted")
