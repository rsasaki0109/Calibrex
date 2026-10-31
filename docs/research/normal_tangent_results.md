# Tangent angular approximation fails exact-plane coverage

Frozen `f349514f`, import-order-only rerun `febdaa2b`. See
[protocol](normal_tangent_protocol.md),
[plan](../assets/normal_tangent_20261030/plan.json),
[27 exact-plane scenarios](../assets/normal_tangent_20261030/evaluation.planes.json),
[27 surface controls](../assets/normal_tangent_20261030/evaluation.json) and
[schema](../assets/normal_tangent_20261030/evaluation.schema.json).

Reject this tangent confidence approximation. Nine of27 exact-plane scenarios
fail the predeclared >=94% empirical coverage gate. Minimum coverage is78.47%,
median98.005%, maximum98.59%; the median hides the failed frontal-plane stratum.
All nine failures have zero disparity slopes, across center disparities6/12/24
and all three tested image positions. Tightening the previous bound by a local
linearization is not calibrated throughout this parameter range.

| Exact frontal-plane center disparity | Principal-point coverage | Off-center coverage range |
| --- | ---: | ---: |
| 6px | 78.47% | 85.67–85.94% |
| 12px | 83.18% | 85.36–85.71% |
| 24px | 88.58% | 89.61–89.79% |

The enclosing-ball baseline has minimum97.90% coverage on the same27 scenarios.
The tangent method uses the covariance of a normalized noisy normal and a95%
2D ellipse/circle approximation; finite-noise nonlinearity and plug-in length/
direction mean that this approximation need not have nominal coverage. These
results are not a failure of the known linear coefficient covariance itself.
Each scenario uses20000 independent coefficient draws from the frozen .5px model.
Image-patch overlap is not involved in that exact-plane validation.

## Curved/boundary controls remain diagnostic

The surface controls were executed as explicitly planned even though the plane
coverage gate failed. All27 depth-array hashes match the prior experiment.
Clean data retains zero points in all nine frames with radius<=15degrees.
For noisy seeds730/731, only near_foreground frame2 has nonempty selection:
10 and2 points respectively, both with100% joint surface correctness. Every other
noisy frame retains zero. These tiny conditional successes do not pass the >=10%
retention and spatial support screens. Null empty-population precision is preserved.

Across all-valid noisy patches, observed coverage ranges78.80–81.95%; within the
previous depth/planarity gate it ranges89.16–94.61%. Curvature/boundary model error
adds another limitation; exact-plane calibration must be fixed before trusting
these intervals. There is no registration run or revised calibration recovery.

## Execution and verification

An initial completed run used a test file with unsorted imports. The test import
order alone was corrected, its source hash updated, and the full deterministic
validation rerun. Original plan and outputs remain under `initial-*` in the same
asset directory. Final surface case records exactly match the initial case records;
no numerical algorithm, model, threshold, noise, seed or denominator was changed.

Four focused tests, repository Ruff and strict documentation build pass. Tests
check the analytic tangent variance at a principal-point frontal plane, missing
patch handling, the previous exact-plane confidence sanity check and far-plane
behavior. Validation checks schema, frozen hashes, case counts, paired depth
identity and deterministic rerun equality. Core/defaults and MI baseline remain.

Close the5x5 local confidence shortcuts. Neither a broad enclosing ball nor the
uncalibrated tangent approximation provides a usable measurement-only gate here.
A next measurement-level direction is coherent surface estimation with larger
spatial support, explicitly balancing reduced disparity-slope variance against
curvature and discontinuity bias. Freeze those model assumptions and independent
coverage tests before any real-data filtering or ICP rerun; do not select window
sizes or angular thresholds from the18 failed real frames. This work establishes
no production readiness or SOTA claim.
