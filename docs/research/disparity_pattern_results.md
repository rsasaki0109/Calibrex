# Coordinate polling does not recover perturbed disparity controls

Frozen `3c966459`, formatting-only revision `da10b9bb`. See
[protocol](disparity_pattern_protocol.md),
[active plan](../assets/disparity_pattern_20261030/plan.json),
[171 results](../assets/disparity_pattern_20261030/evaluation.json) and
[schema](../assets/disparity_pattern_20261030/evaluation.schema.json).

Reject the coordinate-search replacement. Despite measured descent opportunities
in earlier offline paths, this search from the original seeds does not recover
any perturbed case. A descent path to truth is not a deployment initialization.

| Condition | Start | Archived TRF joint recovery | Coordinate search |
| --- | --- | ---: | ---: |
| Clean | Perturbed | 12/36 | 0/36 |
| Clean | Reference | 6/18 | 16/18 |
| Seed 730 | Perturbed | 0/36 | 0/36 |
| Seed 730 | Reference | 12/18 | 16/18 |
| Seed 731 | Perturbed | 0/36 | 0/36 |
| Seed 731 | Reference | 12/18 | 16/18 |

Clean perturbed controls lose all twelve prior successes. Reference-start controls
gain ten clean and four per noise seed, with no losses in those groups. Those
18 gains are not recovery from perturbed poses. Pose-only perturbed recovery is
also zero in all conditions. Convergence/support passes are clean 3/36 perturbed,
17/18 reference; seed 730 12/36 and 17/18; seed 731 8/36 and 18/18. All nine
parallel-plane controls remain rejected.

Every accepted update has nonincreasing measured cost. Calls range 85–800;
all actual-call caps hold, and incomplete polls never declare convergence. Total
solve time including diagnostics is 429.78 seconds, excluding report serialization.
Termination is a complete minimum-step coordinate poll without improvement, not
TRF stationarity, and does not establish accuracy or a joint global optimum.

## Execution record and validation

The initial freeze contained one Ruff line-length violation after an automatic
literal rewrite. The initial execution was explicitly interrupted, preserving
[original plan](../assets/disparity_pattern_20261030/plan.initial.json),
[partial results](../assets/disparity_pattern_20261030/evaluation.interrupted.json)
and [interruption log](../assets/disparity_pattern_20261030/initial_interruption.txt).
Revision `da10b9bb` changes formatting only and updates that source hash. No search
rule, threshold, data, denominator or gate changes. The final run starts afresh
and completes all 171 cases; initial rows are not pooled into results.

Three focused tests, repository Ruff and strict documentation build pass. Checks
verify schema, all frozen source hashes, 171-case completion, each paired depth
hash against archived TRF, actual-call budgets and monotonic cost traces. No core
or production default changed; no fresh real data or holdouts were consumed.

## Decision

Close this sequence of metric weighting, direct-disparity and coordinate-search
variants. None improves noisy perturbed recovery on the frozen controls, and
full-cloud real-data metric transfer already failed. Retain the existing MI
real-data baseline (6/12) and preserve all negative evidence. The next substantial
direction should establish independent geometric camera/LiDAR correspondences
and a matched external-method comparison before another local optimizer variant.
Review reusable external adapters and measurement assumptions first, then freeze
that distinct evaluation. These results do not prove correspondence methods work
or establish SOTA; they justify reallocating development effort.
