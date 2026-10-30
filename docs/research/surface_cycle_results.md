# Fixed reverse consistency does not resolve the remaining surface failures

Frozen `d6bcae4e`, evaluated2026-10-30. See
[protocol](surface_cycle_protocol.md),
[plan](../assets/surface_cycle_20261030/plan.json),
[all18 conditional diagnoses](../assets/surface_cycle_20261030/evaluation.json)
and [schema](../assets/surface_cycle_20261030/evaluation.schema.json).

Do not use the1px reverse mask as a remedy for the three failed surface frames.
All three remain below80% normal agreement after cycle selection. This is a new
conditional diagnosis of predictive surfaces, not a rerun or revised verdict of
the earlier translation-refinement experiment.

| Previously failed frame | Before cycle agreement | Cycle-pass agreement | Selected pixels retained |
| --- | ---: | ---: | ---: |
| 0005/frame38 | 42.70% | 45.71% | 93.78% |
| 0005/frame114 | 76.28% | 75.50% | 95.83% |
| 0009/frame111 | 76.43% | 78.83% | 93.69% |

Normal agreement improves in13/18 frames and worsens in five. Across-frame median
agreement is88.16% before and87.95% in the cycle-pass subset: paired improvement
counts and medians summarize different aspects and must both be disclosed. Median
distance-failure fraction decreases9.14→8.57%. Median selected-domain retention
changes24.77→24.37%; individual retained fractions remain above12.39% of the domain.
These modest changes do not establish calibration improvement.

Rejected-or-reverse-missing selected points have median normal failure30.04%,
versus12.05% among cycle-pass points, and median distance failure27.16% versus8.57%.
This indicates useful association with some errors but does not make accepted
points reliable: reverse-consistent surfaces can still disagree strongly with
LiDAR. The removed group combines inconsistency and absent reverse support; it
must not be interpreted as a labeled set of incorrect correspondences.

## Reproduction and scope

All18 forward/restored SGBM depth arrays exactly reproduce the archived joint
stereo outputs. All12 previously generated cycle-filtered depths also reproduce
array-for-array. The other six frames belong to the already-spent0013/0015 drives.
Threshold, matcher settings, reverse direction, rectification and restoration are
unchanged. Candidate plane estimates were not refitted on masked depth; the test
only partitions the already selected points using independent reverse processing.

Five focused tests, repository Ruff and strict documentation build pass. Validation
checks schema,18-frame completion, all source/input hashes, partition count totals,
normal-support totals and saved filtered-depth hashes. No new optimizer runs,
source masks, ICP adoption, production defaults or calibration verdicts change.
Filtered arrays remain under /workspace/work/surface-cycle-20261030, with artifact
paths and hashes in the report. Reference distance/normal maps are scoring-only.

## Next direction

Close reverse-threshold/filter adjustments. Review the feasibility of temporal
consistency using independent camera motion and overlapping observed surfaces.
The three archived frames per drive are separated in time, so overlap and motion
availability must be audited before prescribing a temporal filter. Identify pose
source, camera/IMU timing/extrinsics, scale and frame conventions; do not estimate
camera motion from reference camera/LiDAR extrinsics or silently apply a truth
transform in selection. If independent motion or overlap is inadequate, report
that limitation rather than manufacturing a successful temporal test. Freeze the
distinct measurement comparison before any new fit or threshold selection. No
fresh holdouts or SOTA claim are authorized by this spent-data diagnostic.
