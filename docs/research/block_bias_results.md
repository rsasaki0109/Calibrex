# Separated bias/pose block results

[Protocol](block_bias_protocol.md) frozen at `91e7aecb`; a serialization-only repair
was committed at `c2409b5c` before the completed run. All 54 cases executed.

| Cohort | Joint TRF joint pass | Alternating blocks joint pass | Block numerical convergence |
| --- | ---: | ---: | ---: |
| Perturbed starts | 33/36 | **0/36** | 0/36 |
| Truth starts | 11/18 | **17/18** | 17/18 |
| Negative-bias truth starts (subset) | 0/6 | **5/6** | 5/6 |

## Complete cases

| Scene | Population | Bias m | Seed | Rotation deg | Translation m | Bias error m | Joint pass | Calls | Cycles |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| original | exact_z_visible | +0.00 | +1 | 0.015630 | 0.032513 | 0.035058 | False | 413 | 8 |
| original | exact_z_visible | +0.00 | -1 | 0.060162 | 0.136974 | 0.148616 | False | 768 | 7 |
| original | exact_z_visible | +0.00 | +0 | 0.000000 | 0.000000 | 0.000000 | True | 19 | 1 |
| original | exact_z_visible | +0.25 | +1 | 0.015630 | 0.032513 | 0.035058 | False | 405 | 8 |
| original | exact_z_visible | +0.25 | -1 | 0.071833 | 0.130073 | 0.140345 | False | 773 | 8 |
| original | exact_z_visible | +0.25 | +0 | 0.000000 | 0.000000 | 0.000000 | True | 59 | 2 |
| original | exact_z_visible | -0.25 | +1 | 0.015630 | 0.032514 | 0.035058 | False | 412 | 8 |
| original | exact_z_visible | -0.25 | -1 | 0.073828 | 0.137532 | 0.148213 | False | 776 | 7 |
| original | exact_z_visible | -0.25 | +0 | 0.000000 | 0.000000 | 0.000000 | True | 57 | 2 |
| original | exact_z_all | +0.00 | +1 | 0.015613 | 0.032391 | 0.034920 | False | 469 | 8 |
| original | exact_z_all | +0.00 | -1 | 0.075145 | 0.130537 | 0.141189 | False | 745 | 8 |
| original | exact_z_all | +0.00 | +0 | 0.000061 | 0.000020 | 0.000005 | True | 221 | 3 |
| original | exact_z_all | +0.25 | +1 | 0.015613 | 0.032391 | 0.034920 | False | 460 | 8 |
| original | exact_z_all | +0.25 | -1 | 0.063728 | 0.129934 | 0.140744 | False | 739 | 8 |
| original | exact_z_all | +0.25 | +0 | 0.000061 | 0.000020 | 0.000005 | True | 210 | 3 |
| original | exact_z_all | -0.25 | +1 | 0.015613 | 0.032391 | 0.034920 | False | 462 | 8 |
| original | exact_z_all | -0.25 | -1 | 0.055370 | 0.130290 | 0.141019 | False | 756 | 8 |
| original | exact_z_all | -0.25 | +0 | 0.000061 | 0.000020 | 0.000005 | True | 207 | 3 |
| near_foreground | exact_z_visible | +0.00 | +1 | 0.028792 | 0.036422 | 0.039663 | False | 544 | 8 |
| near_foreground | exact_z_visible | +0.00 | -1 | 0.131463 | 0.120556 | 0.133663 | False | 778 | 7 |
| near_foreground | exact_z_visible | +0.00 | +0 | 0.000000 | 0.000000 | 0.000000 | True | 19 | 1 |
| near_foreground | exact_z_visible | +0.25 | +1 | 0.028792 | 0.036421 | 0.039663 | False | 528 | 8 |
| near_foreground | exact_z_visible | +0.25 | -1 | 0.109615 | 0.120612 | 0.134540 | False | 760 | 7 |
| near_foreground | exact_z_visible | +0.25 | +0 | 0.000000 | 0.000000 | 0.000000 | True | 59 | 2 |
| near_foreground | exact_z_visible | -0.25 | +1 | 0.028792 | 0.036421 | 0.039663 | False | 541 | 8 |
| near_foreground | exact_z_visible | -0.25 | -1 | 0.115357 | 0.120437 | 0.133742 | False | 776 | 7 |
| near_foreground | exact_z_visible | -0.25 | +0 | 0.000000 | 0.000000 | 0.000000 | True | 57 | 2 |
| near_foreground | exact_z_all | +0.00 | +1 | 0.021680 | 0.035892 | 0.039124 | False | 610 | 8 |
| near_foreground | exact_z_all | +0.00 | -1 | 0.111903 | 0.119933 | 0.133213 | False | 772 | 7 |
| near_foreground | exact_z_all | +0.00 | +0 | 0.000073 | 0.000109 | 0.000043 | True | 215 | 3 |
| near_foreground | exact_z_all | +0.25 | +1 | 0.021680 | 0.035892 | 0.039124 | False | 603 | 8 |
| near_foreground | exact_z_all | +0.25 | -1 | 0.120004 | 0.119022 | 0.132381 | False | 763 | 7 |
| near_foreground | exact_z_all | +0.25 | +0 | 0.000073 | 0.000109 | 0.000043 | True | 206 | 3 |
| near_foreground | exact_z_all | -0.25 | +1 | 0.021680 | 0.035892 | 0.039124 | False | 609 | 8 |
| near_foreground | exact_z_all | -0.25 | -1 | 0.099837 | 0.113470 | 0.126173 | False | 784 | 8 |
| near_foreground | exact_z_all | -0.25 | +0 | 0.000073 | 0.000109 | 0.000043 | True | 203 | 3 |
| staggered | exact_z_visible | +0.00 | +1 | 0.022344 | 0.047321 | 0.049801 | False | 544 | 8 |
| staggered | exact_z_visible | +0.00 | -1 | 0.100778 | 0.142021 | 0.150689 | False | 760 | 7 |
| staggered | exact_z_visible | +0.00 | +0 | 0.000000 | 0.000000 | 0.000000 | True | 19 | 1 |
| staggered | exact_z_visible | +0.25 | +1 | 0.022344 | 0.047321 | 0.049801 | False | 533 | 8 |
| staggered | exact_z_visible | +0.25 | -1 | 0.055059 | 0.134618 | 0.143744 | False | 778 | 8 |
| staggered | exact_z_visible | +0.25 | +0 | 0.000000 | 0.000000 | 0.000000 | True | 59 | 2 |
| staggered | exact_z_visible | -0.25 | +1 | 0.022344 | 0.047321 | 0.049801 | False | 537 | 8 |
| staggered | exact_z_visible | -0.25 | -1 | 0.057983 | 0.135202 | 0.144272 | False | 785 | 8 |
| staggered | exact_z_visible | -0.25 | +0 | 0.000000 | 0.000000 | 0.000000 | True | 57 | 2 |
| staggered | exact_z_all | +0.00 | +1 | 0.022359 | 0.047203 | 0.049663 | False | 532 | 8 |
| staggered | exact_z_all | +0.00 | -1 | 0.096076 | 0.142429 | 0.151121 | False | 757 | 7 |
| staggered | exact_z_all | +0.00 | +0 | 0.000463 | 0.000229 | 0.000050 | True | 507 | 6 |
| staggered | exact_z_all | +0.25 | +1 | 0.022359 | 0.047203 | 0.049662 | False | 512 | 8 |
| staggered | exact_z_all | +0.25 | -1 | 0.080236 | 0.141877 | 0.150908 | False | 759 | 7 |
| staggered | exact_z_all | +0.25 | +0 | 0.000449 | 0.000215 | 0.000045 | True | 345 | 5 |
| staggered | exact_z_all | -0.25 | +1 | 0.022359 | 0.047203 | 0.049662 | False | 530 | 8 |
| staggered | exact_z_all | -0.25 | -1 | 0.059873 | 0.135935 | 0.145012 | False | 775 | 8 |
| staggered | exact_z_all | -0.25 | +0 | 0.000461 | 0.000231 | 0.000050 | False | 748 | 8 |

## What worked, and why it is not an improvement

Scalar updates recover negative-bias correction at all six truth starts to within
0.000051 m. Five meet complete-cycle convergence; staggered/all-return remains a
failure despite small pose/bias error. This supports the prior diagnostic that a
smooth feasible bias descent was being missed by joint linearization near surface
switches. Zero/positive/negative truth-start performance overall rises 11/18→17/18.

But all 36 perturbed trials fail the declared budget/cycle convergence requirement,
versus 33 prior joint successes. Eighteen have geometric pose+bias errors inside the
thresholds if convergence is ignored, but none is counted as recovered. Half still
miss a geometric/bias gate as well. There are 33 regressions and six gains across
all 54 candidates. No increase in budget or relaxed stopping rule was applied.

Alternating coupled pose/bias coordinates under this allocation is therefore an
unsuccessful replacement. Monotonically decreasing cost is insufficient for useful
full calibration in the fixed budget. This comparison changes allocation/stopping
structure as declared, not the sampler or measurement model; it does not disprove
every possible block method. The narrower complete-cycle condition is intentionally
reported alongside actual geometric errors rather than disguised as equal stopping.

Actual optimizer calls span 19–785. Thirty-seven outputs stop at cycle/budget caps;
17 converge. Total new run time including sensitivity diagnostics is 51.85 s on
this CPU; the joint baseline was archived, not rerun as a timing comparison.
All recorded scalar/pose acceptances are nonincreasing in the identical objective.

## Execution incident

The initial attempt failed while serializing the first report: `stable_bias` was
a NumPy boolean. The [original plan](../assets/block_bias_20261029/plan.initial.json)
and [traceback](../assets/block_bias_20261029/initial_failure.txt) are retained.
The fix converts this flag to a Python bool and adds a JSON serialization assertion
to the unit test; the active plan updates only the solver file hash. No objective,
limits, gates or numerical branch changed. No first-attempt evaluation artifact
was overwritten; the repaired run starts the full denominator from the beginning.

## Decision

Do not adopt the full alternating strategy or move it into a real-data improvement
claim. Keep the joint TRF baseline. If testing a scalar rescue next, preserve the
completed joint result and use only its remaining declared budget for a bias-only
cost-decreasing update and bounded refinement. Trigger it from residual information,
not knowledge of negative bias or a truth-start label. Retain a fallback and report
any regressions; objective decrease still does not guarantee accurate pose.
Freeze that separate algorithm before scoring. Do not mix it with image-sampler
changes or retry this failed allocation with a larger cap after seeing the results.

[Evidence](../assets/block_bias_20261029/evaluation.json) and
[schema](../assets/block_bias_20261029/evaluation.schema.json) include every cycle,
status, stage acceptance, cumulative count and archived baseline. Seven focused
tests, repository Ruff, strict docs build, active-plan hashes, monotonic-cost,
count/denominator and score checks pass. No production/core/default changes or SOTA.
