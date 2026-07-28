# Calibration CI

Calibration CI applies the same validation, falsification, bundle-integrity,
and protocol-compatibility gates locally and in GitHub Actions. It never
re-solves a calibration or treats optimizer convergence as acceptance.

## Local use

```bash
calibrex ci calibration/candidate.yaml \
  --baseline calibration/baseline.yaml \
  --output-dir outputs/calibration-ci \
  --enforce \
  --json
```

`--enforce` returns nonzero unless the final status is `PASS`. When a baseline
is supplied, incompatible evidence protocols also fail the decision. Use
`--allow-incompatible-protocol` only when a comparison is intentionally
informational.

The output directory contains:

```text
calibration-ci.json    schema-valid CI decision and provenance
summary.md             GitHub-compatible check summary
evidence.json          candidate evidence extracted without recomputation
assessment.json        falsification policy decision
protocol.json          declared evidence protocols
policy.json            evidence-contract policy
transforms.json        typed candidate transforms
bundle.json            SHA-256 evidence manifest
verification.json      materialized bundle verification
evidence-card.svg      provenance-bound visual summary
comparison.json        baseline comparison, when requested
comparison-table.svg   provenance-bound comparison visual
```

Validate the decision independently:

```bash
calibrex validate outputs/calibration-ci/calibration-ci.json
calibrex verify outputs/calibration-ci/bundle.json --json
```

## GitHub Action

```yaml
name: Calibration

on:
  pull_request:

permissions:
  contents: read

jobs:
  evidence:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Check calibration evidence
        id: calibrex
        uses: rsasaki0109/Calibrex@v0.4.0
        with:
          candidate: calibration/candidate.yaml
          baseline: calibration/baseline.yaml
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: calibration-evidence
          path: calibrex-ci/
```

The action installs Calibrex from the exact action revision, writes
`summary.md` to the GitHub Step Summary, and exposes `status`, `artifact`,
`summary`, `evidence_card`, and `comparison` outputs.

The default inputs enforce both a `PASS` assessment and compatible
candidate/baseline evidence protocols. Set `enforce: "false"` to collect an
informational result while initially adopting the workflow.

## Custom policy

Pass a schema-valid `slac.policy/v0.1` artifact:

```yaml
with:
  candidate: calibration/candidate.yaml
  policy: calibration/policy.yaml
```

The CI artifact records the policy ID, source path, and SHA-256 digest. The
original evidence-contract policy remains separately available so the
generated evidence bundle stays independently verifiable.

The action adds no ROS, GPL solver, or external calibration-tool dependency to
the Calibrex core. External solver execution remains an explicit adapter or
subprocess concern.
