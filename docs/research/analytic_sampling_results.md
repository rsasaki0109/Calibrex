# Analytic versus bilinear depth sampling results

[Protocol](analytic_sampling_protocol.md) frozen at `b355c938`. All 54 oracle-sampler
fits executed. Function definitions of fitting and sensitivity code were verified
identical to the previous solver; only the depth sampler import differs. Bilinear
results are hash-pinned archived controls, not rerun.

| Population | Perturbed joint recovery: bilinear → analytic | Truth-start joint: bilinear → analytic |
| --- | ---: | ---: |
| Oracle-visible input points | 6/18 → 17/18 | 3/9 → 6/9 |
| All LiDAR returns | 6/18 → 16/18 | 3/9 → 5/9 |
| Total | **12/36 → 33/36** | **6/18 → 11/18** |

## All cases

| Scene | Population | Injected bias m | Seed | Rotation deg | Translation m | Bias error m | Converged | Joint pass |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| original | exact_z_visible | +0.00 | +1 | 0.000000 | 0.000000 | 0.000000 | True | True |
| original | exact_z_visible | +0.00 | -1 | 0.000000 | 0.000000 | 0.000000 | True | True |
| original | exact_z_visible | +0.00 | +0 | 0.000000 | 0.000000 | 0.000000 | True | True |
| original | exact_z_visible | +0.25 | +1 | 0.000000 | 0.000000 | 0.000000 | True | True |
| original | exact_z_visible | +0.25 | -1 | 0.000000 | 0.000000 | 0.000000 | True | True |
| original | exact_z_visible | +0.25 | +0 | 0.000000 | 0.000000 | 0.000000 | True | True |
| original | exact_z_visible | -0.25 | +1 | 0.000000 | 0.000000 | 0.000000 | True | True |
| original | exact_z_visible | -0.25 | -1 | 0.000000 | 0.000000 | 0.000000 | True | True |
| original | exact_z_visible | -0.25 | +0 | 0.000050 | 0.000218 | 0.249896 | True | False |
| original | exact_z_all | +0.00 | +1 | 0.000079 | 0.000044 | 0.000040 | True | True |
| original | exact_z_all | +0.00 | -1 | 0.000049 | 0.000049 | 0.000066 | True | True |
| original | exact_z_all | +0.00 | +0 | 0.000061 | 0.000020 | 0.000000 | True | True |
| original | exact_z_all | +0.25 | +1 | 0.001159 | 0.000511 | 0.000514 | True | True |
| original | exact_z_all | +0.25 | -1 | 0.000067 | 0.000046 | 0.000058 | True | True |
| original | exact_z_all | +0.25 | +0 | 0.001465 | 0.000674 | 0.000785 | True | True |
| original | exact_z_all | -0.25 | +1 | 0.000043 | 0.000063 | 0.000066 | True | True |
| original | exact_z_all | -0.25 | -1 | 0.000051 | 0.000048 | 0.000066 | True | True |
| original | exact_z_all | -0.25 | +0 | 0.000051 | 0.000209 | 0.249900 | True | False |
| near_foreground | exact_z_visible | +0.00 | +1 | 0.000000 | 0.000000 | 0.000000 | True | True |
| near_foreground | exact_z_visible | +0.00 | -1 | 0.000000 | 0.000000 | 0.000000 | True | True |
| near_foreground | exact_z_visible | +0.00 | +0 | 0.000000 | 0.000000 | 0.000000 | True | True |
| near_foreground | exact_z_visible | +0.25 | +1 | 0.325075 | 0.150498 | 0.171496 | False | False |
| near_foreground | exact_z_visible | +0.25 | -1 | 0.000000 | 0.000000 | 0.000000 | True | True |
| near_foreground | exact_z_visible | +0.25 | +0 | 0.000000 | 0.000000 | 0.000000 | True | True |
| near_foreground | exact_z_visible | -0.25 | +1 | 0.000000 | 0.000000 | 0.000000 | True | True |
| near_foreground | exact_z_visible | -0.25 | -1 | 0.000000 | 0.000000 | 0.000000 | True | True |
| near_foreground | exact_z_visible | -0.25 | +0 | 0.001974 | 0.006481 | 0.246924 | True | False |
| near_foreground | exact_z_all | +0.00 | +1 | 0.002877 | 0.000391 | 0.000341 | True | True |
| near_foreground | exact_z_all | +0.00 | -1 | 0.000163 | 0.000111 | 0.000010 | True | True |
| near_foreground | exact_z_all | +0.00 | +0 | 0.000089 | 0.000108 | 0.000007 | True | True |
| near_foreground | exact_z_all | +0.25 | +1 | 0.314767 | 0.150083 | 0.172101 | False | False |
| near_foreground | exact_z_all | +0.25 | -1 | 0.000178 | 0.000110 | 0.000009 | True | True |
| near_foreground | exact_z_all | +0.25 | +0 | 0.001190 | 0.000243 | 0.000167 | True | True |
| near_foreground | exact_z_all | -0.25 | +1 | 0.016990 | 0.000629 | 0.000341 | True | True |
| near_foreground | exact_z_all | -0.25 | -1 | 0.000538 | 0.000174 | 0.000141 | True | True |
| near_foreground | exact_z_all | -0.25 | +0 | 0.001957 | 0.006484 | 0.246918 | True | False |
| staggered | exact_z_visible | +0.00 | +1 | 0.000000 | 0.000000 | 0.000000 | True | True |
| staggered | exact_z_visible | +0.00 | -1 | 0.000094 | 0.000014 | 0.000007 | True | True |
| staggered | exact_z_visible | +0.00 | +0 | 0.000000 | 0.000000 | 0.000000 | True | True |
| staggered | exact_z_visible | +0.25 | +1 | 0.000094 | 0.000014 | 0.000007 | True | True |
| staggered | exact_z_visible | +0.25 | -1 | 0.000000 | 0.000000 | 0.000000 | True | True |
| staggered | exact_z_visible | +0.25 | +0 | 0.000094 | 0.000014 | 0.000007 | True | True |
| staggered | exact_z_visible | -0.25 | +1 | 0.000000 | 0.000000 | 0.000000 | True | True |
| staggered | exact_z_visible | -0.25 | -1 | 0.000094 | 0.000014 | 0.000007 | True | True |
| staggered | exact_z_visible | -0.25 | +0 | 0.000463 | 0.001165 | 0.249439 | True | False |
| staggered | exact_z_all | +0.00 | +1 | 0.001510 | 0.000270 | 0.000101 | False | False |
| staggered | exact_z_all | +0.00 | -1 | 0.000497 | 0.000147 | 0.000058 | True | True |
| staggered | exact_z_all | +0.00 | +0 | 0.000252 | 0.000113 | 0.000009 | False | False |
| staggered | exact_z_all | +0.25 | +1 | 0.002568 | 0.001859 | 0.001982 | True | True |
| staggered | exact_z_all | +0.25 | -1 | 0.001207 | 0.001264 | 0.001375 | True | True |
| staggered | exact_z_all | +0.25 | +0 | 0.000813 | 0.000302 | 0.000283 | True | True |
| staggered | exact_z_all | -0.25 | +1 | 0.004712 | 0.000579 | 0.000410 | True | True |
| staggered | exact_z_all | -0.25 | -1 | 0.002032 | 0.000760 | 0.000830 | True | True |
| staggered | exact_z_all | -0.25 | +0 | 0.000434 | 0.001114 | 0.249463 | True | False |

## What changed

Across all 54 cases, median absolute bias error drops from **0.05590 m** with
bilinear images to **0.00000948 m** with continuous first-surface depth. At zero
injected bias, the median is 0.00000361 m and maximum 0.000341 m, versus the earlier
artificial 4–7 cm correction. This controlled substitution strongly implicates the
pixel/bilinear measurement approximation in the preceding systematic offset.
It does not prove that any particular image-based replacement will reproduce the
oracle or that real stereo error has the same cause.

Do not summarize only the median. Four fits reach max_nfev: near-foreground +.25 m
positive seeds in both visibility populations, and staggered zero-bias all-return
positive/true starts. The latter have tiny pose/bias errors but remain failures
because convergence was required. Six -.25 m truth-start fits numerically converge
without recovering the injected correction: maximum bias error is **0.24990 m**.
These six occur in every layout/population. The near-foreground failed perturbations
have bias errors up to **0.17210 m**. All other trials satisfy the joint gates.

Thus 50/54 converge, with 44/54 passing joint gates when both cohorts are counted;
report the meaningful denominators separately as 33/36 perturbations and 11/18
truth-start controls. No local degeneracy flag rescues the wrong-bias fits. Small
pose error or locally full rank again does not establish correct nuisance-parameter
estimation. Negative-bias truth-start stopping deserves a separate objective/
finite-difference investigation; the present experiment does not isolate its cause.

Optimizer residual calls span 8–639; all remain below 800 and no retries occur.
Total 54-case wall time including post-fit diagnostics is 17.90 s on this CPU.
The oracle uses known surface geometry and camera origins to generate measurements;
it is neither a deployable sensor frontend nor unseen real-data validation. Archived
baseline timing is not a simultaneous performance comparison.

## Next work

Keep the oracle as a diagnostic. Do not promote it or claim the earlier bias issue
is fully solved. Two distinct next questions now have evidence:

1. Can an image-only sampler preserve surface boundaries without interpolating
   foreground and background into fictitious depth? Compare a predeclared
   discontinuity-aware sampler against the unchanged bilinear and analytic controls,
   with fixed populations and explicit missing-support counts. Do not tune boundary
   thresholds to joint-success labels.
2. Why do -.25 m truth starts stop with wrong bias even under exact sampling?
   Examine the frozen objective and derivative behavior before changing optimizer
   tolerances or claiming that another sampler fixes this failure. Keep all six
   cases as mandatory controls.

These should be separate experiments; avoid changing sampling and optimizer at
once. Preserve zero/positive/negative bias, both initial directions and true-start
controls, original gates and the 800-call accounting. No real-data/SOTA promotion.

[Evidence](../assets/analytic_sampling_20261029/evaluation.json) and
[schema](../assets/analytic_sampling_20261029/evaluation.schema.json) include every
oracle output and associated archived baseline. Ten focused tests, repository
Ruff, strict docs build, solver-AST equality, source hashes, full-denominator and
budget checks pass. Core and default behavior remain unchanged.
