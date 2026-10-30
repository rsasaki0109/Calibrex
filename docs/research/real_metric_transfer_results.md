# Metric calibration does not transfer to the spent real-data controls

Frozen at `1ceb2d52` before evaluation on 2026-10-30. See
[protocol](real_metric_transfer_protocol.md),
[plan and input hashes](../assets/real_metric_transfer_20261030/plan.json),
[all 54 fits](../assets/real_metric_transfer_20261030/evaluation.json) and
[schema](../assets/real_metric_transfer_20261030/evaluation.schema.json).

**Reject production adoption of this metric path.** None of the three methods
recovers any of the 12 perturbed real-data starts. Synthetic gains did not
transfer under the frozen sampling and SGBM depth conditions. There are no new
recordings, fresh holdouts or SOTA claims.

| Outcome | Bilinear metric | Edge metric | Edge + scalar rescue |
| --- | ---: | ---: | ---: |
| Accepted perturbed pose recovery | 0/12 | 0/12 | 0/12 |
| Accepted reference-start pose control | 1/6 | 3/6 | 3/6 |
| Perturbed fits passing convergence/support | 8/12 | 11/12 | 11/12 |
| Reference fits passing convergence/support | 6/6 | 6/6 | 6/6 |
| Perturbed median rotation error (deg) | 1.5741 | 1.5010 | 1.5010 |
| Perturbed median translation error (m) | .3089 | .5069 | .5069 |
| Reference-start median rotation drift (deg) | .5594 | .4768 | .4768 |
| Reference-start median translation drift (m) | .2684 | .1540 | .1540 |

Convergence/support successes above also pass local identifiability. They are
frequently inaccurate; acceptance is not an accuracy certificate. Edge sampling
gains two reference-start passes (0009 and 0013), losing none, but worsens median
translation error for perturbed starts. The successful reference controls still
have nonzero drift.

All 18 rescue baseline records exactly reproduce the separately run edge fits;
all poses remain identical. Seventeen eligible fits trigger and adopt scalar
correction with reduced cost, but this cannot repair their fixed poses. Estimated
bias is not measured bias error: there is no true nuisance-bias label in KITTI.
Rescue estimates range .1317–.5000 m, including estimates near the upper bound;
do not interpret cost reduction as evidence of a physical constant depth offset.

All original three-frame observations and hashes are retained. This run fixes
4096 file-order points per frame for every method, so the historical joint-MI
SGBM 6/12 score is **not** a matched comparator: its objective, nuisance parameters
and point selection differ. The present three metric methods are paired. Reference
extrinsics enter scoring and the labeled reference-start controls only. SGBM
arrays were restored to original camera rays before archiving, matching intrinsics.

## All starts

Edge + rescue has exactly the edge pose in every row. Errors are degrees/metres;
pass means converged, identifiable and within both pose gates.

| Case | Bilinear deg / m | Pass | Edge deg / m | Pass |
| --- | ---: | --- | ---: | --- |
| 2011_09_26_0005-positive | 0.2934 / 0.2365 | False | 0.6055 / 0.5076 | False |
| 2011_09_26_0005-reference | 0.4382 / 0.1973 | True | 0.1518 / 0.0753 | True |
| 2011_09_26_0005-negative | 1.5764 / 0.2581 | False | 1.4991 / 0.1097 | False |
| 2011_09_28_0002-positive | 1.8709 / 0.5044 | False | 1.3216 / 0.5062 | False |
| 2011_09_28_0002-reference | 0.4791 / 0.2301 | False | 0.5826 / 0.1375 | False |
| 2011_09_28_0002-negative | 2.5773 / 0.3597 | False | 2.8320 / 0.3674 | False |
| 2011_09_26_0009-positive | 0.6417 / 0.3896 | False | 1.4245 / 0.5878 | False |
| 2011_09_26_0009-reference | 0.6398 / 0.3561 | False | 0.3143 / 0.1640 | True |
| 2011_09_26_0009-negative | 1.5718 / 0.1231 | False | 1.4746 / 0.1236 | False |
| 2011_09_26_0011-positive | 1.9307 / 0.5251 | False | 2.3515 / 1.0925 | False |
| 2011_09_26_0011-reference | 1.0356 / 0.3068 | False | 0.7271 / 0.2372 | False |
| 2011_09_26_0011-negative | 3.0365 / 0.2253 | False | 2.2528 / 0.8808 | False |
| 2011_09_26_0013-positive | 0.3386 / 0.3956 | False | 0.6967 / 0.6028 | False |
| 2011_09_26_0013-reference | 0.4578 / 0.2212 | False | 0.4808 / 0.1439 | True |
| 2011_09_26_0013-negative | 2.8468 / 0.1517 | False | 2.2712 / 0.2376 | False |
| 2011_09_26_0015-positive | 1.0465 / 0.4554 | False | 1.5566 / 0.5285 | False |
| 2011_09_26_0015-reference | 1.3197 / 0.3383 | False | 0.4729 / 0.2399 | False |
| 2011_09_26_0015-negative | 1.4692 / 0.1018 | False | 1.5028 / 0.1111 | False |

## Accounting and next decision

Optimizer calls: bilinear 81–192, edge 57–151, rescue 90–170, all below 800.
Total solve time including diagnostics: 5.62, 5.93, 7.64 seconds respectively;
input loading is excluded and this single run is not a rigorous speed benchmark.
Ten focused tests, repository Ruff and strict documentation build pass. Checks
cover all 102 input hashes and frozen source hashes, report schema, 54-case
completion, budgets, exact edge-baseline reproduction, unchanged rescue poses
and cost monotonicity. No core or default settings changed.

Stop tuning the synthetic sampler or scalar threshold. Next isolate the real-data
failure with a frozen diagnostic: compare the metric cost at reference and fitted
poses, retain support/coverage and signed residual distributions, and establish a
matched MI control using the same selected inputs. This distinguishes objective/
measurement mismatch from the change in point population before choosing another
optimizer or stereo model. Reference-based diagnostics must remain offline and
must not become a truth-dependent deployment gate. Fresh holdouts remain untouched.
