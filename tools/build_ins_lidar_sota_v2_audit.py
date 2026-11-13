#!/usr/bin/env python3
"""Build the INS-LiDAR rotation audit v2 from a run of ``ins_lidar_sota_v2_evaluate``.

Each declared case is in exactly one state:

* **scored**: the method finished and its artifact is scored by the v1 case scorer;
* **method failure**: the method attempt finished without a usable result (a crash, a
  timeout, no convergence, missing evidence). It fails the completion gate;
* **missing**: no method attempt finished, because the inputs could not be acquired or
  the run has not reached the case. Its family's gates get no evidence, and the audit is
  incomplete.

Gates, per dataset family (thresholds in ``ins_lidar_sota_v2_protocol``):

* completion: no method failure;
* agreement: the worst estimated axis of every recording with an estimated axis;
* coverage: the mean unobserved-axis fraction over the scored recordings, a recording
  with no estimated axis counting as fully unobserved;
* controls: the missed-control fraction of every recording with an estimated axis.

::

    python tools/build_ins_lidar_sota_v2_audit.py RUNS/run.json PREREGISTRATION OUT
    python tools/build_ins_lidar_sota_v2_audit.py RUNS/run.json --development OUT
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.build_ins_lidar_sota_audit import _benchmark  # noqa: E402
from tools.ins_lidar_sota_acquire import file_digest  # noqa: E402
from tools.ins_lidar_sota_protocol import verify_frozen_source  # noqa: E402
from tools.ins_lidar_sota_v2_evaluate import development_plan  # noqa: E402
from tools.ins_lidar_sota_v2_protocol import (  # noqa: E402
    CLAIM,
    CONTROL_MISS_FRACTION_MAX,
    ERROR_MAX_DEG,
    METHOD_FAILURE_FRACTION_MAX,
    METHOD_ID,
    PROTOCOL_ID,
    SCOPE,
    UNOBSERVED_FRACTION_MAX,
    RunManifest,
    load_preregistration,
)
from tools.score_ins_lidar_sota import CaseSpec, score_artifacts  # noqa: E402

from calibrex import __version__  # noqa: E402
from calibrex.core.benchmark import aggregate_benchmark_definition  # noqa: E402
from calibrex.core.sota_audit import (  # noqa: E402
    SotaAuditProtocol,
    SotaAuditProvenance,
    SotaClaimRequirement,
    SotaClaimScope,
)
from calibrex.evaluation.sota_audit import audit_sota_claim  # noqa: E402

GENERATOR = "tools.build_ins_lidar_sota_v2_audit"
FAILURE_POLICY = (
    "All cases retained. A method failure fails completion. A recording with no estimated "
    "rotation axis is not a failure: it counts as fully unobserved in coverage and has no "
    "agreement or control trial. A recording whose inputs could not be acquired leaves "
    "its family's gates without evidence."
)
LIMITATIONS = (
    "The zero anchor is not an empirical competitor or a targetless method.",
    "Only estimated rotation axes are gated for agreement and controls; coverage retains "
    "all axes of every scored recording.",
    "Different KITTI dates share one platform; Boreas supplies the second platform.",
    "No absolute clock, translation, full-extrinsic or general superiority claim.",
)
FAMILIES = (("kitti_raw", "kitti-platform"), ("boreas", "boreas-platform"))
METRICS = (
    "unobserved_rotation_fraction",
    "rotation_reference_difference_deg",
    "missed_observed_control_fraction",
)
GATES = (
    ("completion", "unobserved_rotation_fraction", "failure_rate", METHOD_FAILURE_FRACTION_MAX),
    ("agreement", "rotation_reference_difference_deg", "metric_maximum", ERROR_MAX_DEG),
    ("coverage", "unobserved_rotation_fraction", "metric_mean", UNOBSERVED_FRACTION_MAX),
    (
        "controls",
        "missed_observed_control_fraction",
        "metric_maximum",
        CONTROL_MISS_FRACTION_MAX,
    ),
)


def case_states(manifest: dict[str, Any]) -> dict[str, str]:
    """``scored`` candidate, ``method_failure`` or ``missing`` for every declared case."""

    states = {}
    for case in manifest["cases"]:
        attempt = manifest["attempts"].get(case["case_id"])
        if attempt is None or attempt["status"] == "RUNNING":
            states[case["case_id"]] = "missing"
        elif attempt["status"] == "PASS" and case.get("artifact_path"):
            states[case["case_id"]] = "scored"
        else:
            states[case["case_id"]] = "method_failure"
    return states


def family_rows(
    rows: list[dict[str, Any]], states: dict[str, str]
) -> dict[str, dict[str, float | str]]:
    """The trials of one family's three benchmarks; a string value is a failed trial."""

    coverage: dict[str, float | str] = {}
    agreement: dict[str, float | str] = {}
    controls: dict[str, float | str] = {}
    for row in rows:
        case_id = row["case"]["case_id"]
        if states[case_id] == "missing":
            continue
        if states[case_id] == "method_failure" or not row["completed"]:
            coverage[case_id] = "; ".join(row["reasons"]) or "the method attempt failed"
            continue
        estimated = row["estimated_axes"]
        coverage[case_id] = 1 - estimated / 3
        if estimated:
            agreement[case_id] = max(row["errors_deg"])
            controls[case_id] = row["control_misses"] / estimated
    return {
        "unobserved_rotation_fraction": coverage,
        "rotation_reference_difference_deg": agreement,
        "missed_observed_control_fraction": controls,
    }


def build(
    manifest_path: Path, prereg_path: Path | None, out: Path, *, source_root: Path = ROOT
) -> dict[str, Any]:
    """Score every scored case, build three benchmarks per family, then audit the gates."""

    manifest = RunManifest.model_validate_json(manifest_path.read_text()).model_dump(mode="json")
    if prereg_path is None:
        plan = development_plan(source_root)
        if not manifest["development"]:
            raise ValueError("an evaluation run needs its pre-registration")
        identity, frozen = plan.identity_sha256, manifest["frozen_code_commit"]
        expected = plan.evaluation
    else:
        prereg = load_preregistration(prereg_path, source_root)
        verify_frozen_source(prereg, source_root, require_main=False)  # type: ignore[arg-type]
        identity, frozen = file_digest(prereg_path), prereg.frozen_code_commit
        expected = prereg.evaluation
        if manifest["development"] or manifest["frozen_code_commit"] != frozen:
            raise ValueError("run does not match the frozen pre-registration")
    if manifest["preregistration_sha256"] != identity:
        raise ValueError("run does not match its plan")
    cases = [CaseSpec.model_validate(c) for c in manifest["cases"]]
    for declared, case in zip(expected, cases, strict=True):
        if (declared.case_id, declared.drive_ids) != (case.case_id, case.drive_ids):
            raise ValueError("declared recording IDs differ")
    states = case_states(manifest)
    scores = score_artifacts(cases, manifest_path.parent)
    sources = {str(manifest_path): file_digest(manifest_path)}
    if prereg_path is not None:
        sources[str(prereg_path)] = file_digest(prereg_path)
    # A benchmark is built only for a family whose every case has a finished attempt, and
    # an attempt starts only after acquire_case verified each input against its pin (the
    # worker re-hashes them before and after the fit). So what is built is verified.
    verified = True
    if out.exists():
        raise ValueError("preserve the previous audit output; choose a new directory")
    out.mkdir(parents=True)
    (out / "scores.json").write_text(
        json.dumps(
            {
                "schema_version": "calibrex.ins_lidar_audit_scores/v0.2",
                "provenance": {
                    "generator": GENERATOR,
                    "frozen_code_commit": frozen,
                    "source_sha256": sources,
                    "development": manifest["development"],
                },
                "case_states": states,
                "interrupted_attempts": len(manifest["interrupted_attempts"]),
                "acquisition_errors": manifest["acquisition_errors"],
                **scores,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    requirements = []
    for family, rig in FAMILIES:
        group = [r for r in scores["cases"] if r["case"]["dataset_family"] == family]
        missing = [r["case"]["case_id"] for r in group if states[r["case"]["case_id"]] == "missing"]
        trials = family_rows(group, states)
        paths: dict[str, Path | None] = {}
        for metric in METRICS:
            rows = trials[metric]
            if missing or not rows:
                paths[metric] = None  # no evidence: an undeclared subset must not be gated
                continue
            definition = _benchmark(
                family,
                metric,
                rows,
                file_digest(manifest_path),
                verified,
                sources,
                frozen,
                protocol_id=PROTOCOL_ID,
                failure_policy=FAILURE_POLICY,
                limitations=LIMITATIONS,
                generator=GENERATOR,
            )
            path = out / f"{family}_{metric}.yaml"
            aggregate_benchmark_definition(definition).save(path)
            paths[metric] = path
        for name, metric, statistic, threshold in GATES:
            path = paths[metric]
            description = f"{family}: {name} under the frozen per-judged-axis contract"
            if path is None:
                why = (
                    f"no method attempt finished for {', '.join(missing)}"
                    if missing
                    else "no recording of this family has an estimated axis"
                )
                requirements.append(  # a gate without evidence is incomplete, never achieved
                    SotaClaimRequirement(
                        requirement_id=f"{family}-{name}",
                        phase="phase2",
                        description=f"{description} ({statistic} <= {threshold}): {why}",
                        dataset_family=family,
                        independent_rig=True,
                        rig_id=rig,
                    )
                )
                continue
            requirements.append(
                SotaClaimRequirement(
                    requirement_id=f"{family}-{name}",
                    phase="phase2",
                    description=description,
                    dataset_family=family,
                    independent_rig=True,
                    rig_id=rig,
                    evidence_path=path.name,
                    evidence_sha256=file_digest(path),
                    method_id=METHOD_ID,
                    metric=metric if statistic != "failure_rate" else None,
                    statistic=statistic,
                    comparison="less_equal",
                    threshold=threshold,
                )
            )
    protocol = SotaAuditProtocol(
        protocol_id=PROTOCOL_ID,
        scope=SotaClaimScope.model_validate(SCOPE),
        claim_text=CLAIM,
        minimum_dataset_families=2,
        minimum_independent_rigs=2,
        requirements=requirements,
        provenance=SotaAuditProvenance(
            generator=GENERATOR,
            generator_version=__version__,
            git_commit=frozen,
            command=["python", "tools/build_ins_lidar_sota_v2_audit.py", *sys.argv[1:]],
            source_sha256=sources,
        ),
    )
    protocol_path = out / "ins_lidar_sota_protocol.yaml"
    protocol.save(protocol_path)
    audit = audit_sota_claim(protocol_path)
    audit.save(out / "ins_lidar_sota_audit.yaml")
    return {
        "verdict": audit.verdict,
        "development": manifest["development"],
        "case_states": states,
        "families": scores["families"],
        "achieved": sum(r.status == "achieved" for r in audit.requirements),
        "requirements": len(audit.requirements),
        "statuses": {r.requirement_id: r.status for r in audit.requirements},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path)
    parser.add_argument("paths", nargs="+", type=Path, help="[PREREGISTRATION] OUT")
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args(argv)
    if len(args.paths) != (1 if args.development else 2):
        parser.error("give RUN PREREGISTRATION OUT, or RUN --development OUT")
    result = build(
        args.manifest, None if args.development else args.paths[0], args.paths[-1]
    )
    print(json.dumps(result, indent=1))
    return 0 if result["verdict"] == "supported" else 2


if __name__ == "__main__":
    raise SystemExit(main())
