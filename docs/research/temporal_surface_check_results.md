# Temporal agreement does not resolve the wrong real surfaces

Frozen `53c39a61`, evaluated2026-10-30. See
[protocol](temporal_surface_check_protocol.md),
[plan](../assets/temporal_surface_check_20261030/plan.json),
[18 reprojection controls](../assets/temporal_surface_check_20261030/evaluation.controls.json),
[all18 centers/36 neighbor checks](../assets/temporal_surface_check_20261030/evaluation.json)
and [schema](../assets/temporal_surface_check_20261030/evaluation.schema.json).

Reject this temporal verification as a remedy for the remaining surface failures.
All18 exact-render translation/visibility controls pass, minimum visible-point
consistency98.34%, above the frozen95% threshold. Real data, however, retains
incorrect surfaces after both-neighbor temporal agreement and loses useful support.

| Across18 centers | Predictive surface baseline | Verified in both neighbors |
| --- | ---: | ---: |
| Centers passing declared quality/support screen | 15/18 | 13/18 |
| Median domain retained | 24.77% | 19.43% |
| Median normal failure | 11.84% | 11.70% among17 nonempty centers |
| Median distance failure | 9.14% | 11.08% among17 nonempty centers |

The candidate has one empty center, so its conditional precision median excludes
that undefined value and must not be read as an all-case improvement. Full screen
counts include the empty center as a failure. Dense points are not independent
recordings; no significance or calibration improvement claim follows.

## Original failures remain

| Frame | Both-neighbor selected points | Domain retained | Normal agreement | Distance failure |
| --- | ---: | ---: | ---: | ---: |
| 0005/frame38 | 49861 | 11.49% | 45.24% | 20.69% |
| 0005/frame114 | 99418 | 22.92% | 76.02% | 34.75% |
| 0009/frame111 | 80656 | 18.59% | 76.98% | 10.26% |

All three remain below80% normal agreement. Additional failures are0013/frame107,
with both motion pairs unsupported and therefore zero verified points, and
0015/frame74, whose selected distance failure is21.79% despite adequate normal
agreement. No missing support is relabeled as success or omitted.

The report partitions every selected point per neighbor into invalid projection,
out-of-view, missing depth, consistent, nearer observed surface, farther observed
surface or unsupported motion. Category totals equal baseline counts in all36
checks. Nearer observed surface can be real occlusion; rejected temporal support
is not automatically incorrect geometry. Both-neighbor verification deliberately
requires both tests; no one-sided or reference-assisted fallback was introduced.

## Interpretation

Temporal projection uses measured predictive surface XYZ, estimated camera-only
motion and raw neighbor stereo depth. The target camera/LiDAR reference transform
is never used in that classification. Reference distance/normal maps score the
result afterward. Motion and depth nevertheless share stereo images and errors;
a coherent geometry error can be temporally consistent. Current results do not
prove which failures arise from motion error, correlated stereo bias, occlusion,
LiDAR normals or reference/timing assumptions. Exact controls validate projection
on static clean scenes, not every real dynamic/noisy condition.

Four focused tests, repository Ruff and strict documentation build pass. Tests
check projection direction, positive-depth/out-of-view/missing categories, nearer/
farther surface sign and separate error denominators. Checks validate report/schema,
all hashes,18 control outcomes,18 real centers,36 neighbor-count partitions and
baseline-versus-selected population totals. No extrinsic fits, ICP runs, fresh
data, default changes or SOTA claims occurred.

## Decision

Close the current self-consistency-filter sequence: plane predictability, reverse
stereo and temporal agreement cannot establish reliable geometry on all18 frames.
Do not tune tolerances or add another mask based on these reference-error maps.
The next useful work is reference-assisted error attribution using independent
measurement channels, explicitly evaluation-only. First audit available OXTS
motion/timing and the camera-frame conversion uncertainty against the estimated
motion; reference-based conversion is allowed for diagnosis but must never feed
selection. Such a proxy is not independent metric ground truth and cannot reopen
an accuracy/SOTA claim. If this cannot localize the error, pursue a different
measurement construction rather than extending the filter chain or ICP iterations.
