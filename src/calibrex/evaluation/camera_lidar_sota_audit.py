"""Execute a frozen Camera--LiDAR SOTA claim audit."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from calibrex import __version__
from calibrex.core.benchmark import load_benchmark
from calibrex.core.camera_lidar_sota_audit import (
    CameraLidarClaimRequirement,
    CameraLidarClaimRequirementResult,
    CameraLidarSotaAuditProvenance,
    CameraLidarSotaAuditResult,
    load_camera_lidar_sota_audit_protocol,
)
from calibrex.core.provenance import git_commit, sha256_path


def audit_camera_lidar_sota_claim(
    protocol_path: str | Path,
    *,
    audit_id: str | None = None,
    command: list[str] | None = None,
) -> CameraLidarSotaAuditResult:
    """Evaluate all frozen requirements without tuning or omission."""

    path = Path(protocol_path)
    protocol = load_camera_lidar_sota_audit_protocol(path)
    protocol_digest = _required_digest(path)
    results = [
        _evaluate_requirement(item, base_path=path.parent)
        for item in protocol.requirements
    ]
    achieved_ids = {
        item.requirement_id
        for item in results
        if item.status == "achieved"
    }
    families = sorted(
        {
            item.dataset_family
            for item in protocol.requirements
            if item.requirement_id in achieved_ids
            and item.required
            and item.dataset_family is not None
        }
    )
    rigs = {
        item.rig_id
        for item in protocol.requirements
        if item.requirement_id in achieved_ids
        and item.required
        and item.independent_rig
        and item.rig_id is not None
    }
    required_results = [item for item in results if item.required]
    coverage_achieved = (
        len(families) >= protocol.minimum_dataset_families
        and len(rigs) >= protocol.minimum_independent_rigs
    )
    verdict: Literal["supported", "refuted", "incomplete"]
    if all(item.status == "achieved" for item in required_results) and coverage_achieved:
        verdict = "supported"
        summary = (
            "Every frozen required gate and declared dataset/rig coverage "
            "criterion is achieved."
        )
    elif any(item.status == "contradicted" for item in required_results):
        verdict = "refuted"
        summary = (
            "At least one frozen required numerical or integrity gate is "
            "contradicted; the declared SOTA claim is not supported."
        )
    else:
        verdict = "incomplete"
        summary = (
            "Required evidence or dataset/rig coverage is incomplete; no SOTA "
            "claim may be made."
        )
    source_sha256 = {"protocol": protocol_digest}
    for item in results:
        if item.evidence_path is not None and item.evidence_sha256 is not None:
            source_sha256[item.requirement_id] = item.evidence_sha256
    return CameraLidarSotaAuditResult(
        audit_id=audit_id or f"{protocol.protocol_id}-audit",
        protocol_id=protocol.protocol_id,
        declared_category=protocol.declared_category,
        claim_text=protocol.claim_text,
        verdict=verdict,
        achieved_dataset_families=families,
        achieved_independent_rig_count=len(rigs),
        minimum_dataset_families=protocol.minimum_dataset_families,
        minimum_independent_rigs=protocol.minimum_independent_rigs,
        requirements=results,
        summary=summary,
        provenance=CameraLidarSotaAuditProvenance(
            generator=__name__,
            generator_version=__version__,
            git_commit=git_commit(),
            command=command or [],
            source_sha256=source_sha256,
        ),
    )


def _evaluate_requirement(
    requirement: CameraLidarClaimRequirement,
    *,
    base_path: Path,
) -> CameraLidarClaimRequirementResult:
    if requirement.evidence_path is None:
        return _result(
            requirement,
            "incomplete",
            "no evidence artifact is assigned to this frozen requirement",
        )
    evidence_path = Path(requirement.evidence_path)
    if not evidence_path.is_absolute():
        evidence_path = base_path / evidence_path
    if not evidence_path.is_file():
        return _result(
            requirement,
            "missing",
            "declared evidence artifact is missing",
            evidence_path=evidence_path,
        )
    digest = sha256_path(evidence_path)
    if digest != requirement.evidence_sha256:
        return _result(
            requirement,
            "contradicted",
            (
                "evidence SHA-256 differs from the frozen protocol: "
                f"expected {requirement.evidence_sha256}, observed {digest}"
            ),
            evidence_path=evidence_path,
            evidence_sha256=digest,
        )
    try:
        benchmark = load_benchmark(evidence_path)
    except Exception as exc:
        return _result(
            requirement,
            "contradicted",
            f"evidence is not a valid benchmark artifact: {exc}",
            evidence_path=evidence_path,
            evidence_sha256=digest,
        )
    if not benchmark.provenance.data_verified:
        return _result(
            requirement,
            "contradicted",
            "benchmark provenance declares data_verified=false",
            evidence_path=evidence_path,
            evidence_sha256=digest,
        )
    observed: float | None
    if requirement.statistic in {
        "paired_improvement_ci95_low",
        "paired_improvement_ci95_high",
    }:
        comparison = next(
            (
                item
                for item in benchmark.paired_comparisons
                if item.reference_method_id
                == requirement.reference_method_id
                and item.candidate_method_id == requirement.method_id
                and item.metric == requirement.metric
            ),
            None,
        )
        if comparison is None:
            return _result(
                requirement,
                "missing",
                (
                    "paired comparison is absent: "
                    f"{requirement.reference_method_id} -> "
                    f"{requirement.method_id}, metric {requirement.metric}"
                ),
                evidence_path=evidence_path,
                evidence_sha256=digest,
            )
        observed = (
            comparison.improvement_ci95_low
            if requirement.statistic == "paired_improvement_ci95_low"
            else comparison.improvement_ci95_high
        )
    else:
        summary = benchmark.method_summaries.get(requirement.method_id or "")
        if summary is None:
            return _result(
                requirement,
                "missing",
                f"benchmark method is absent: {requirement.method_id}",
                evidence_path=evidence_path,
                evidence_sha256=digest,
            )
        if requirement.statistic == "failure_rate":
            observed = summary.failure_rate
        elif requirement.statistic == "runtime_mean":
            observed = summary.runtime_seconds.mean
        elif requirement.statistic == "runtime_p95":
            observed = summary.runtime_seconds.p95
        elif requirement.statistic == "peak_memory_mean":
            observed = summary.peak_memory_mb.mean
        elif requirement.statistic == "peak_memory_p95":
            observed = summary.peak_memory_mb.p95
        else:
            metric = summary.metrics.get(requirement.metric or "")
            distribution = metric.distribution if metric is not None else None
            observed = (
                {
                    "metric_mean": distribution.mean,
                    "metric_median": distribution.median,
                    "metric_p90": distribution.p90,
                    "metric_p95": distribution.p95,
                    "metric_maximum": distribution.maximum,
                }.get(requirement.statistic or "")
                if distribution is not None
                else None
            )
    if observed is None or not math.isfinite(observed):
        return _result(
            requirement,
            "incomplete",
            "declared benchmark statistic has no finite value",
            evidence_path=evidence_path,
            evidence_sha256=digest,
        )
    achieved = _compare(
        observed,
        requirement.threshold or 0.0,
        requirement.comparison or "equal",
    )
    return _result(
        requirement,
        "achieved" if achieved else "contradicted",
        (
            f"observed {observed:.12g} "
            f"{'meets' if achieved else 'does not meet'} "
            f"{requirement.comparison} {requirement.threshold:.12g}"
        ),
        observed_value=observed,
        evidence_path=evidence_path,
        evidence_sha256=digest,
    )


def _compare(
    observed: float,
    threshold: float,
    comparison: str,
) -> bool:
    if comparison == "greater_equal":
        return observed >= threshold
    if comparison == "greater_than":
        return observed > threshold
    if comparison == "less_equal":
        return observed <= threshold
    if comparison == "less_than":
        return observed < threshold
    return math.isclose(observed, threshold, rel_tol=0.0, abs_tol=1.0e-12)


def _result(
    requirement: CameraLidarClaimRequirement,
    status: str,
    reason: str,
    *,
    observed_value: float | None = None,
    evidence_path: Path | None = None,
    evidence_sha256: str | None = None,
) -> CameraLidarClaimRequirementResult:
    return CameraLidarClaimRequirementResult(
        requirement_id=requirement.requirement_id,
        required=requirement.required,
        status=status,  # type: ignore[arg-type]
        observed_value=observed_value,
        threshold=requirement.threshold,
        reason=reason,
        evidence_path=str(evidence_path) if evidence_path is not None else None,
        evidence_sha256=evidence_sha256,
    )


def _required_digest(path: Path) -> str:
    digest = sha256_path(path)
    if digest is None:
        raise ValueError(f"audit protocol is not readable: {path}")
    return digest
