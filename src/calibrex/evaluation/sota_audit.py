"""Evaluate frozen, pair-agnostic SOTA claim audits.

The same engine evaluates Camera--LiDAR protocols (through their pair-agnostic
form) and every other claim scope, so a verdict never depends on which sensor
pair the claim is about.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

from calibrex import __version__
from calibrex.core.benchmark import load_benchmark
from calibrex.core.provenance import git_commit, sha256_path
from calibrex.core.sota_audit import (
    SotaAuditProtocol,
    SotaAuditProvenance,
    SotaAuditResult,
    SotaClaimRequirement,
    SotaClaimRequirementResult,
    SotaVerdict,
    load_sota_audit_protocol,
)


def audit_sota_claim(
    protocol_path: str | Path,
    *,
    audit_id: str | None = None,
    command: list[str] | None = None,
) -> SotaAuditResult:
    """Evaluate every frozen requirement of a pair-agnostic protocol."""

    path = Path(protocol_path)
    return evaluate_sota_protocol(
        load_sota_audit_protocol(path),
        protocol_sha256=required_sha256(path),
        base_path=path.parent,
        audit_id=audit_id,
        command=command,
    )


def evaluate_sota_protocol(
    protocol: SotaAuditProtocol,
    *,
    protocol_sha256: str,
    base_path: Path,
    audit_id: str | None = None,
    command: list[str] | None = None,
    generator: str = __name__,
) -> SotaAuditResult:
    """Evaluate all frozen requirements without tuning or omission."""

    results = [
        evaluate_sota_requirement(item, base_path=base_path) for item in protocol.requirements
    ]
    families, rig_count = achieved_coverage(protocol.requirements, results)
    verdict, summary = decide_sota_verdict(
        results,
        achieved_dataset_families=families,
        achieved_independent_rig_count=rig_count,
        minimum_dataset_families=protocol.minimum_dataset_families,
        minimum_independent_rigs=protocol.minimum_independent_rigs,
    )
    source_sha256 = {"protocol": protocol_sha256}
    for item in results:
        if item.evidence_path is not None and item.evidence_sha256 is not None:
            source_sha256[item.requirement_id] = item.evidence_sha256
    return SotaAuditResult(
        audit_id=audit_id or f"{protocol.protocol_id}-audit",
        protocol_id=protocol.protocol_id,
        scope=protocol.scope,
        claim_text=protocol.claim_text,
        verdict=verdict,
        achieved_dataset_families=families,
        achieved_independent_rig_count=rig_count,
        minimum_dataset_families=protocol.minimum_dataset_families,
        minimum_independent_rigs=protocol.minimum_independent_rigs,
        requirements=results,
        summary=summary,
        provenance=SotaAuditProvenance(
            generator=generator,
            generator_version=__version__,
            git_commit=git_commit(),
            command=command or [],
            source_sha256=source_sha256,
        ),
    )


def achieved_coverage(
    requirements: Sequence[SotaClaimRequirement],
    results: Sequence[SotaClaimRequirementResult],
) -> tuple[list[str], int]:
    """Return dataset families and independent rigs backed by achieved required gates."""

    achieved_ids = {item.requirement_id for item in results if item.status == "achieved"}
    families = sorted(
        {
            item.dataset_family
            for item in requirements
            if item.requirement_id in achieved_ids
            and item.required
            and item.dataset_family is not None
        }
    )
    rigs = {
        item.rig_id
        for item in requirements
        if item.requirement_id in achieved_ids
        and item.required
        and item.independent_rig
        and item.rig_id is not None
    }
    return families, len(rigs)


def decide_sota_verdict(
    results: Sequence[SotaClaimRequirementResult],
    *,
    achieved_dataset_families: Sequence[str],
    achieved_independent_rig_count: int,
    minimum_dataset_families: int,
    minimum_independent_rigs: int,
) -> tuple[SotaVerdict, str]:
    """Return the verdict and summary implied by required gates and coverage."""

    required_results = [item for item in results if item.required]
    coverage_achieved = (
        len(achieved_dataset_families) >= minimum_dataset_families
        and achieved_independent_rig_count >= minimum_independent_rigs
    )
    if all(item.status == "achieved" for item in required_results) and coverage_achieved:
        return (
            "supported",
            "Every frozen required gate and declared dataset/rig coverage criterion is achieved.",
        )
    if any(item.status == "contradicted" for item in required_results):
        return (
            "refuted",
            "At least one frozen required numerical or integrity gate is "
            "contradicted; the declared SOTA claim is not supported.",
        )
    return (
        "incomplete",
        "Required evidence or dataset/rig coverage is incomplete; no SOTA claim may be made.",
    )


def evaluate_sota_requirement(
    requirement: SotaClaimRequirement,
    *,
    base_path: Path,
) -> SotaClaimRequirementResult:
    """Evaluate one frozen requirement against its digest-pinned benchmark."""

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
                if item.reference_method_id == requirement.reference_method_id
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
    requirement: SotaClaimRequirement,
    status: str,
    reason: str,
    *,
    observed_value: float | None = None,
    evidence_path: Path | None = None,
    evidence_sha256: str | None = None,
) -> SotaClaimRequirementResult:
    return SotaClaimRequirementResult(
        requirement_id=requirement.requirement_id,
        required=requirement.required,
        status=status,  # type: ignore[arg-type]
        observed_value=observed_value,
        threshold=requirement.threshold,
        reason=reason,
        evidence_path=str(evidence_path) if evidence_path is not None else None,
        evidence_sha256=evidence_sha256,
    )


def required_sha256(path: Path) -> str:
    """Return the SHA-256 of a readable audit protocol."""

    digest = sha256_path(path)
    if digest is None:
        raise ValueError(f"audit protocol is not readable: {path}")
    return digest
