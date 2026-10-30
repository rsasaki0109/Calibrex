# Full clouds restore MI recovery but do not rescue the metric objective

Frozen at `07398b48` before evaluation on 2026-10-30. See
[protocol](full_cloud_transfer_protocol.md),
[plan](../assets/full_cloud_transfer_20261030/plan.json),
[all 54 fits](../assets/full_cloud_transfer_20261030/evaluation.json) and
[schema](../assets/full_cloud_transfer_20261030/evaluation.schema.json).

**Retain the existing MI path; reject adoption of the current metric candidate.**
Full-cloud MI exactly reproduces all 12 historical SGBM result records, including
traces, transforms and metrics, recovering 6/12 perturbed starts. Both metric
methods still recover 0/12. File-order thinning harmed MI, but does not explain
away the metric method's failure on these real recordings.

| Full-cloud method | Perturbed recovery | Reference-start controls | Perturbed convergence/support | Perturbed median rotation / translation |
| --- | ---: | ---: | ---: | ---: |
| MI | 6/12 | 5/6 | 12/12 | .3096 deg / .2190 m |
| Bilinear metric | 0/12 | 1/6 | 11/12 | 1.3615 deg / .2611 m |
| Edge + scalar rescue | 0/12 | 0/6 | 11/12 | 1.4104 deg / .7032 m |

All six reference-start fits terminate successfully for each method, yet metric
outputs frequently drift outside the pose gates. Median reference drift is MI
.1675 deg/.1531 m, bilinear .7662 deg/.2356 m, edge+rescue .7624 deg/.2434 m.
Metric convergence/support successes also pass the local rank gate; neither
termination nor local rank certifies accuracy. The nested edge-only baseline has
identical poses to the rescue output. Sixteen scalar triggers/adoptions lower
cost without changing any pose; this does not produce a pose recovery gain.

Compared with the selected-point experiments, MI perturbed recovery improves
1/12→6/12, bilinear remains 0/12 and edge+rescue remains 0/12. Reference-start
scores change MI 5/6→5/6, bilinear 1/6→1/6 and edge+rescue 3/6→0/6. The complete
input contains 107739–124044 points per frame; every point is passed through,
without a new visibility mask, sampling or revised frontend. Algorithms retain
their own projection and correspondence rules, so this comparison of full methods
does not isolate the objective formula from the search algorithm.

## Cost preference survives full input

At equal fitted bias, fitted cost is below reference cost in 18/18 bilinear cases,
17 with inaccurate poses. On identical common-support point indices the same
18/18 and 17 inaccurate cases remain. Edge+rescue has lower total cost in 17/18,
all inaccurate, and lower common-support cost in 15/18, all inaccurate. Thus
support membership changes alone do not account for the wrong-pose cost preference.
These are fixed-state probes, not globally optimized reference-bias comparisons.

Reference-zero supported populations are 41918–46841 points per three-frame
batch, out of 335032–368538 total points. Depending on drive, about 14.8–29.5%
of supported signed depth residuals exceed 1 m in magnitude. This error is present
at reference poses with both samplers. It warrants studying the measurement-error
model; it does not by itself prove which of stereo, occlusion, timing or search
causes every failure. Real nuisance-bias accuracy is unknown.

## Every full-cloud outcome

Each cell gives rotation in degrees / translation in metres / recovery.

| Case | MI | Bilinear | Edge + rescue |
| --- | ---: | ---: | ---: |
| 2011_09_26_0005-positive | 0.8773 / 0.2577 / False | 0.7516 / 0.2053 / False | 0.7037 / 0.6965 / False |
| 2011_09_26_0005-reference | 0.1875 / 0.0904 / True | 0.4446 / 0.1630 / True | 0.4700 / 0.2171 / False |
| 2011_09_26_0005-negative | 0.3149 / 0.1803 / True | 1.3356 / 0.1669 / False | 1.3042 / 0.7980 / False |
| 2011_09_28_0002-positive | 0.1835 / 0.0400 / True | 1.3671 / 0.1065 / False | 2.9478 / 1.4668 / False |
| 2011_09_28_0002-reference | 0.1474 / 0.0265 / True | 0.7066 / 0.1723 / False | 0.5408 / 0.1623 / False |
| 2011_09_28_0002-negative | 0.5553 / 0.3536 / False | 1.5672 / 0.4403 / False | 2.2532 / 0.4755 / False |
| 2011_09_26_0009-positive | 0.9009 / 0.2658 / False | 0.5184 / 0.3571 / False | 0.8894 / 0.5641 / False |
| 2011_09_26_0009-reference | 0.1406 / 0.1602 / True | 0.3681 / 0.2027 / False | 0.6607 / 0.2522 / False |
| 2011_09_26_0009-negative | 0.1985 / 0.1250 / True | 2.8131 / 0.2393 / False | 0.9686 / 0.7099 / False |
| 2011_09_26_0011-positive | 0.1371 / 0.0468 / True | 1.0653 / 0.2409 / False | 1.5167 / 0.8016 / False |
| 2011_09_26_0011-reference | 0.0781 / 0.1579 / True | 1.6247 / 0.3347 / False | 0.8792 / 0.2930 / False |
| 2011_09_26_0011-negative | 0.2782 / 0.3197 / False | 3.1093 / 0.3811 / False | 1.6354 / 0.6155 / False |
| 2011_09_26_0013-positive | 0.6802 / 0.3085 / False | 0.7051 / 0.4026 / False | 0.5839 / 0.4333 / False |
| 2011_09_26_0013-reference | 0.6320 / 0.1945 / False | 1.8058 / 0.4621 / False | 0.9784 / 0.2346 / False |
| 2011_09_26_0013-negative | 0.8442 / 0.4727 / False | 2.4919 / 0.2814 / False | 1.7396 / 0.2276 / False |
| 2011_09_26_0015-positive | 0.3044 / 0.1281 / True | 1.3559 / 0.8844 / False | 1.0671 / 1.0793 / False |
| 2011_09_26_0015-reference | 0.2210 / 0.1484 / True | 0.8259 / 0.2685 / False | 0.8642 / 0.5137 / False |
| 2011_09_26_0015-negative | 0.1214 / 0.0904 / True | 2.4066 / 0.2045 / False | 1.9391 / 0.7968 / False |

## Validation and decision

MI uses 157–421 evaluations; bilinear 97–315 and edge+rescue 105–208 actual
residual calls, all below their respective 800 caps. Total solve times including
solver diagnostics are 68.63, 177.60 and 188.90 seconds respectively, excluding
loading and separate cost probes. MI evaluations and metric calls have different
semantics; this is not a rigorous speed benchmark.

Nine focused tests, repository Ruff and strict documentation build pass. Validation
checks all frozen source/input hashes, report schema, all 54 cases, full input
counts, hard budgets, unchanged rescue poses, cost descent, reproduction of fitted
costs and exact historical MI equality for all 12 perturbations. No core or default
changes. No new recordings or fresh holdouts were consumed.

Close the sequence of edge-sampler/scalar-rescue tuning. They improve synthetic
bias controls but do not provide a viable real-data calibration path. Next build
an independently specified stereo uncertainty control: disparity error maps into
range-dependent depth error, whereas the current fixed metric residual scale and
single additive bias assume a simpler model. First test that hypothesis on known
geometry with predeclared disparity noise and occlusions; compare uncertainty-aware
residuals to this fixed baseline without fitting noise scales on these failed real
cases. Keep MI as the real-data baseline and preserve all truth-start/regression
controls. This is a next experiment, not a claim that uncertainty weighting already
solves calibration or that MI is SOTA.
