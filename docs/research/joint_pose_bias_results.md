# Joint pose/depth-bias results

[Protocol](joint_pose_bias_protocol.md) frozen at `b58e9d93`. All 63 new fits
completed: 36 perturbed sphere-scene trials, 18 sphere truth-start controls and
9 separately reported parallel-plane controls. The archived six-parameter baseline
is hash-pinned; it was not rerun.

| Injected bias | Baseline perturbed pose pass | Joint-fit pose pass | Joint pose+bias pass | Truth-start joint pose+bias pass |
| --- | ---: | ---: | ---: | ---: |
| 0 m | 12/12 | 12/12 | 4/12 | 2/6 |
| +0.25 m | 0/12 | 12/12 | 4/12 | 2/6 |
| -0.25 m | 12/12 | 12/12 | 4/12 | 2/6 |

## Complete outputs

| Scene | Population | Bias m | Seed | Rotation deg | Translation m | Estimated bias m | Joint pass | Locally identifiable |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| original | exact_z_visible | +0.00 | +1 | 0.2366 | 0.0821 | -0.0631 | False | True |
| original | exact_z_visible | +0.00 | -1 | 0.2384 | 0.0820 | -0.0626 | False | True |
| original | exact_z_visible | +0.00 | +0 | 0.2355 | 0.0828 | -0.0640 | False | True |
| original | exact_z_visible | +0.25 | +1 | 0.2392 | 0.0829 | +0.1863 | False | True |
| original | exact_z_visible | +0.25 | -1 | 0.2424 | 0.0823 | +0.1875 | False | True |
| original | exact_z_visible | +0.25 | +0 | 0.2222 | 0.0811 | +0.1865 | False | True |
| original | exact_z_visible | -0.25 | +1 | 0.2579 | 0.0869 | -0.3168 | False | True |
| original | exact_z_visible | -0.25 | -1 | 0.2468 | 0.0834 | -0.3136 | False | True |
| original | exact_z_visible | -0.25 | +0 | 0.2498 | 0.0849 | -0.3150 | False | True |
| original | exact_z_all | +0.00 | +1 | 0.2627 | 0.0860 | -0.0653 | False | True |
| original | exact_z_all | +0.00 | -1 | 0.2627 | 0.0860 | -0.0653 | False | True |
| original | exact_z_all | +0.00 | +0 | 0.2528 | 0.0864 | -0.0667 | False | True |
| original | exact_z_all | +0.25 | +1 | 0.2252 | 0.0793 | +0.1889 | False | True |
| original | exact_z_all | +0.25 | -1 | 0.2627 | 0.0860 | +0.1847 | False | True |
| original | exact_z_all | +0.25 | +0 | 0.2568 | 0.0843 | +0.1857 | False | True |
| original | exact_z_all | -0.25 | +1 | 0.2627 | 0.0860 | -0.3153 | False | True |
| original | exact_z_all | -0.25 | -1 | 0.2631 | 0.0863 | -0.3157 | False | True |
| original | exact_z_all | -0.25 | +0 | 0.2319 | 0.0826 | -0.3143 | False | True |
| near_foreground | exact_z_visible | +0.00 | +1 | 0.1771 | 0.0664 | -0.0586 | False | True |
| near_foreground | exact_z_visible | +0.00 | -1 | 0.1814 | 0.0679 | -0.0602 | False | True |
| near_foreground | exact_z_visible | +0.00 | +0 | 0.1815 | 0.0682 | -0.0605 | False | True |
| near_foreground | exact_z_visible | +0.25 | +1 | 0.1809 | 0.0679 | +0.1898 | False | True |
| near_foreground | exact_z_visible | +0.25 | -1 | 0.1791 | 0.0674 | +0.1904 | False | True |
| near_foreground | exact_z_visible | +0.25 | +0 | 0.1813 | 0.0680 | +0.1898 | False | True |
| near_foreground | exact_z_visible | -0.25 | +1 | 0.1736 | 0.0655 | -0.3078 | False | True |
| near_foreground | exact_z_visible | -0.25 | -1 | 0.1814 | 0.0679 | -0.3102 | False | True |
| near_foreground | exact_z_visible | -0.25 | +0 | 0.1384 | 0.0583 | -0.3005 | False | True |
| near_foreground | exact_z_all | +0.00 | +1 | 0.1746 | 0.0639 | -0.0547 | False | True |
| near_foreground | exact_z_all | +0.00 | -1 | 0.1746 | 0.0639 | -0.0547 | False | True |
| near_foreground | exact_z_all | +0.00 | +0 | 0.1695 | 0.0631 | -0.0540 | False | True |
| near_foreground | exact_z_all | +0.25 | +1 | 0.1609 | 0.0626 | +0.1957 | False | True |
| near_foreground | exact_z_all | +0.25 | -1 | 0.2024 | 0.0671 | +0.1932 | False | True |
| near_foreground | exact_z_all | +0.25 | +0 | 0.1604 | 0.0626 | +0.1958 | False | True |
| near_foreground | exact_z_all | -0.25 | +1 | 0.1600 | 0.0624 | -0.3039 | False | True |
| near_foreground | exact_z_all | -0.25 | -1 | 0.1646 | 0.0634 | -0.3050 | False | True |
| near_foreground | exact_z_all | -0.25 | +0 | 0.1625 | 0.0622 | -0.3035 | False | True |
| staggered | exact_z_visible | +0.00 | +1 | 0.0599 | 0.0446 | -0.0432 | True | True |
| staggered | exact_z_visible | +0.00 | -1 | 0.0718 | 0.0447 | -0.0428 | True | True |
| staggered | exact_z_visible | +0.00 | +0 | 0.0595 | 0.0446 | -0.0432 | True | True |
| staggered | exact_z_visible | +0.25 | +1 | 0.0718 | 0.0447 | +0.2072 | True | True |
| staggered | exact_z_visible | +0.25 | -1 | 0.0718 | 0.0447 | +0.2072 | True | True |
| staggered | exact_z_visible | +0.25 | +0 | 0.0718 | 0.0447 | +0.2072 | True | True |
| staggered | exact_z_visible | -0.25 | +1 | 0.0604 | 0.0448 | -0.2935 | True | True |
| staggered | exact_z_visible | -0.25 | -1 | 0.0718 | 0.0447 | -0.2928 | True | True |
| staggered | exact_z_visible | -0.25 | +0 | 0.0660 | 0.0442 | -0.2923 | True | True |
| staggered | exact_z_all | +0.00 | +1 | 0.0689 | 0.0445 | -0.0427 | True | True |
| staggered | exact_z_all | +0.00 | -1 | 0.0735 | 0.0437 | -0.0420 | True | True |
| staggered | exact_z_all | +0.00 | +0 | 0.0684 | 0.0442 | -0.0426 | True | True |
| staggered | exact_z_all | +0.25 | +1 | 0.0709 | 0.0445 | +0.2074 | True | True |
| staggered | exact_z_all | +0.25 | -1 | 0.0561 | 0.0449 | +0.2063 | True | True |
| staggered | exact_z_all | +0.25 | +0 | 0.0526 | 0.0425 | +0.2092 | True | True |
| staggered | exact_z_all | -0.25 | +1 | 0.0594 | 0.0446 | -0.2934 | True | True |
| staggered | exact_z_all | -0.25 | -1 | 0.0709 | 0.0445 | -0.2926 | True | True |
| staggered | exact_z_all | -0.25 | +0 | 0.0701 | 0.0445 | -0.2927 | True | True |
| plane | parallel_planes | +0.00 | +1 | 1.0879 | 0.3518 | -0.0333 | False | False |
| plane | parallel_planes | +0.00 | -1 | 1.2208 | 0.2644 | +0.0333 | False | False |
| plane | parallel_planes | +0.00 | +0 | 0.0000 | 0.0000 | +0.0000 | False | False |
| plane | parallel_planes | +0.25 | +1 | 0.7940 | 0.2093 | +0.0500 | False | False |
| plane | parallel_planes | +0.25 | -1 | 1.4158 | 0.4333 | +0.1167 | False | False |
| plane | parallel_planes | +0.25 | +0 | 0.0311 | 0.1674 | +0.0833 | False | False |
| plane | parallel_planes | -0.25 | +1 | 0.4962 | 0.4719 | -0.1167 | False | False |
| plane | parallel_planes | -0.25 | -1 | 0.5692 | 0.3402 | -0.0500 | False | False |
| plane | parallel_planes | -0.25 | +0 | 0.0372 | 0.1705 | -0.0833 | False | False |

## Gains and failure

Pose-only perturbed recovery improves **24/36 → 36/36**, driven by the +.25 m
subset (0/12→12/12). All 18 sphere truth-start poses stay within the original
accuracy gates. All 54 sphere-scene fits converge, retain support and pass the
provisional local-identifiability check. However, joint pose AND bias recovery is
only **12/36**, with **6/18** truth-start controls. Every joint success belongs to
the staggered layout; neither original nor near-foreground layout passes the
bias-accuracy gate for any perturbation.

Across all 54 sphere outputs, estimated minus injected bias is **-0.0668 to
-0.0408 m**. Even zero-bias data acquire an artificial negative correction. Median
absolute bias error over perturbed trials is .0566 m at zero injection, .0582 m
at +.25 m and .0564 m at -.25 m. Do not relax the preregistered .05 m bias-error
gate or call pose-only gains complete calibration recovery. A locally identifiable
parameter can still have systematic estimation error under a misspecified model.

All nine plane controls numerically converge but are flagged locally degenerate
and receive no accepted joint result. Different bias/translation combinations
explain the same parallel planes; some poses happen to meet geometric gates but
must not be reported as uniquely recovered. This validates the intended negative
control, not a general guarantee against all unobservable configurations.

Actual optimizer residual calls are 8–566, below 800. No retries or omitted failures.
Fifteen sensitivity passes per fit are reported separately; wall times include them.
Raw and effective-robust singular spectra are in the full evidence. Local rank is
not a calibrated confidence interval and does not certify bias accuracy. The scene
conditions and seed pairs are dependent synthetic development evidence.

## Decision and next diagnostic

Keep joint pose/bias fitting experimental. It solves the large positive-bias pose
failure here, but the systematic nonzero correction on clean measurements blocks
promotion. Do not tune a bias prior, shift the reported bias by the observed error,
or change visibility saturation to make these cases pass.

Next isolate the camera observation approximation: compare current pixel/bilinear
depth sampling with an analytic first-surface depth sampler on the same known
geometry and fixed seeds/objective. This is an oracle measurement diagnostic,
not a deployable estimator or real-data result. Preserve positive/negative/zero
bias and occlusion controls, and separately retain the current sampler baseline.
If the offset disappears it implicates sampling/model mismatch; if it remains,
inspect the asymmetric likelihood/optimization rather than blaming interpolation
without evidence. Freeze the comparison before execution and keep bias and pose
scores separate. No real-data promotion or SOTA claim yet.

[Full evidence](../assets/joint_pose_bias_20261029/evaluation.json) and
[schema](../assets/joint_pose_bias_20261029/evaluation.schema.json) preserve every
fit, archived baseline, raw/robust diagnostic and score. Eleven focused tests,
repository Ruff, strict documentation build, frozen-hash checks, denominator and
budget validation pass. Core/default behavior remains unchanged.
