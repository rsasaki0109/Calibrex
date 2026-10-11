# Calibration CI workflow examples

Copy these workflows into your repository to gate pull requests on
schema-valid calibration evidence.

## Quick start

1. Commit a **baseline** result from a trusted calibration run
   (`result.yaml` from `calibrex calibrate`).
2. Commit the **candidate** result produced on the PR branch (or generate it in
   CI before the check step).
3. Copy [`calibration-ci-pr.yml`](calibration-ci-pr.yml) to
   `.github/workflows/calibration-ci.yml` and edit the `candidate` / `baseline`
   paths.

```bash
mkdir -p .github/workflows
cp examples/ci/calibration-ci-pr.yml .github/workflows/calibration-ci.yml
```

Pin the action to a release tag (for example `@v0.5.1`) rather than `@main`
once you adopt the workflow.

## What the check does

The [Calibrex Calibration CI action](../../action.yml) runs `calibrex ci`, which:

- validates candidate (and optional baseline) result artifacts;
- extracts falsification evidence without re-solving;
- compares baseline vs candidate when both are supplied;
- writes `calibration-ci.json`, `summary.md`, and provenance-bound visuals.

It does **not** treat optimizer convergence as acceptance.

## Outputs

| Output | Meaning |
| --- | --- |
| `status` | Final `pass`, `fail`, or `inconclusive` (lowercase) |
| `artifact` | Path to `calibration-ci.json` |
| `summary` | Path to Markdown summary (also appended to the GitHub step summary) |
| `evidence_card` | Provenance-bound SVG |
| `comparison` | `comparison.json` when a baseline was supplied |

## Lifecycle registry workflow

[`calibration-ci-lifecycle.yml`](calibration-ci-lifecycle.yml) records each
decision in an append-only lifecycle registry kept in the repository: on a push
to `main` it evaluates the merged candidate and promotes it when the decision
is PASS and the promotion and smoke evidence admit it; a weekly run re-checks
the same file and rolls the edge back when the decision fails the promoted
incumbent. INCONCLUSIVE decisions are held and never change the incumbent. See
[`docs/reference/lifecycle_registry.md`](../../docs/reference/lifecycle_registry.md).

## Adoption tip

Set `enforce: "false"` on the first PR while you tune paths and baseline
storage. Switch to `enforce: "true"` once a green baseline is checked in.

Full reference: [`docs/tutorials/calibration_ci.md`](../../docs/tutorials/calibration_ci.md).
