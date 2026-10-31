# Camera-parametric surface transfer improves fifteen-frame quality screen

Frozen `22453494`, evaluated2026-10-30. See
[protocol](real_surface_prediction_protocol.md),
[plan and camera geometry](../assets/real_surface_prediction_20261030/plan.json),
[all36 paired rows](../assets/real_surface_prediction_20261030/evaluation.json) and
[schema](../assets/real_surface_prediction_20261030/evaluation.schema.json).

The generic implementation reproduces every output array exactly for all27
analytic measurements. Real construction explicitly receives original camera
intrinsics, calibration-derived fB and assumed .5px noise. Analytic constants are
not used for the real frames. Invalid camera geometry is rejected.

| Across18 real frames | Baseline21 | Predictive21 |
| --- | ---: | ---: |
| Frames passing all declared quality/support gates | 13/18 | 15/18 |
| Median retained domain | 26.81% | 24.77% |
| Retention range | 15.22–34.28% | 13.21–32.74% |
| Median target fraction within1m of LiDAR | 88.54% | 90.86% |
| Median normal agreement within15deg | 86.12% | 88.16% |
| Normal agreement range | 42.52–94.95% | 42.70–96.76% |
| Selected point count range | 66014–148728 | 57301–142016 |

Normal agreement is conditional on reference-pose nearest-LiDAR distance<=1m
and a valid12-neighbor LiDAR normal. It is an evaluation proxy, not exact physical
correspondence truth or calibration precision. Dense target pixels are scored;
this is not the earlier voxel-target/source-FOV population. Spatial cells remain
13–16; valid normal pairs range45943–114861 for the candidate. Selection uses no
reference or LiDAR normals, and predictive output is a subset of baseline points.

## Every predictive frame

| Drive / frame | Retention | Within1m | Normal agreement | Pass |
| --- | ---: | ---: | ---: | --- |
| 2011_09_26_0005 / frame-38 | 13.21% | 80.57% | 42.70% | False |
| 2011_09_26_0005 / frame-76 | 21.56% | 91.81% | 84.21% | True |
| 2011_09_26_0005 / frame-114 | 25.52% | 68.62% | 76.28% | False |
| 2011_09_28_0002 / frame-93 | 26.33% | 87.84% | 88.50% | True |
| 2011_09_28_0002 / frame-187 | 30.65% | 88.92% | 91.39% | True |
| 2011_09_28_0002 / frame-281 | 31.35% | 87.82% | 93.26% | True |
| 2011_09_26_0009 / frame-111 | 26.01% | 91.15% | 76.43% | False |
| 2011_09_26_0009 / frame-223 | 29.54% | 87.08% | 85.73% | True |
| 2011_09_26_0009 / frame-334 | 24.68% | 92.41% | 81.68% | True |
| 2011_09_26_0011 / frame-58 | 31.26% | 90.98% | 93.47% | True |
| 2011_09_26_0011 / frame-116 | 24.86% | 95.51% | 96.00% | True |
| 2011_09_26_0011 / frame-174 | 32.74% | 93.83% | 96.76% | True |
| 2011_09_26_0013 / frame-35 | 24.21% | 90.75% | 87.82% | True |
| 2011_09_26_0013 / frame-71 | 23.79% | 88.33% | 93.50% | True |
| 2011_09_26_0013 / frame-107 | 20.68% | 95.89% | 81.07% | True |
| 2011_09_26_0015 / frame-74 | 23.46% | 88.69% | 92.41% | True |
| 2011_09_26_0015 / frame-148 | 21.24% | 92.82% | 84.94% | True |
| 2011_09_26_0015 / frame-222 | 20.19% | 91.77% | 91.46% | True |

All-frame continuation fails. The remaining frames are0005/frame38,
0005/frame114 and0009/frame111; normal agreement is42.70%,76.28%,76.43% respectively.
Frame114 also has only68.62% within1m. No frame is dropped, and the80% screen is
not relaxed. No ICP rerun is performed as specified by the prior protocol.

The surrogate disparity is fB/z on restored original-camera depths. Rectification,
nearest-neighbor restoration and stereo matching can violate the independent
Gaussian assumption. Consequently the predictive chi-square threshold is a fixed
transfer hypothesis, not calibrated real99% confidence. LiDAR normals can also be
wrong at boundaries or with motion; these results do not uniquely attribute the
three failures to stereo, timing or source geometry. Surface similarity alone
does not demonstrate observable or recoverable six-DoF extrinsics.

Five focused tests, repository Ruff and strict documentation build pass. Checks
verify schema, all36 rows, every source/input/calibration hash and27-case bitwise
analytic parity for every estimator output array. Tests cover changed focal/stereo
scale, analytic plane depth/normal, invalid geometry and the prior predictive gate.
No core/default change, newly acquired data, calibration accuracy or SOTA claim.

Next inspect the spatial distribution of failures in all18 frames, emphasizing the
three failed ones, using fixed reference-only distance/normal residual maps and
measurement-only support masks. Determine whether errors concentrate at dynamic/
depth boundaries or large regions before changing the measurement model. Preserve
the failed frames and avoid making truth-dependent exclusion masks. A matched
registration experiment remains pending; the synthetic pass and15/18 quality
screen do not establish a real-data calibration improvement.
