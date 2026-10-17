# Public TUM example execution

The complete public TUM example improves from **58.116 to 29.610 seconds
(1.963x, 49.0% less elapsed time)** in a direct comparison of the preceding
checked-in example and the updated example. Both execute every original
calibration and diagnostic step and produce exactly matching results.

The change enables the existing bounded three-window process execution in the
public example and reduces CPU work in native geometry and diagnostic scoring.
The core option still defaults to one worker for other configurations. These
measurements use default NumPy execution; CUDA kernel code and numerical policy
remain unchanged.

## Separate the execution and arithmetic effects

Three variants run three times each, with the same mean position in the
nine-worker sequence. The preceding runtime is
`c8b048f13986ee9255b0d93ac3ac2913d8b14ece`; the current runtime is
`5e5f10522ab792ed9713b2b5c101589156e60b99`.

| Variant | Source and configuration | Median whole worker seconds |
| --- | --- | ---: |
| prior | preceding checked-in example, one window worker | 58.116 |
| parallel | preceding runtime, three window workers | 37.836 |
| current | current checked-in example, three window workers | 29.610 |

The directly measured example-to-example gain is **1.963x**. The preceding
runtime's parallel-only control gives **1.536x** against its serial example.
With three workers on both sides, the arithmetic and geometry changes give
**1.278x**, or **21.7% less elapsed time**. These are observations from the same
experiment; earlier benchmark ratios are not multiplied.

| Fresh worker | Variant | Whole worker seconds | Calibration API seconds |
| --- | --- | ---: | ---: |
| 01-current | current | 28.281 | 23.598 |
| 02-prior | prior | 58.116 | 53.453 |
| 03-parallel | parallel | 39.104 | 33.632 |
| 04-prior | prior | 57.830 | 52.514 |
| 05-parallel | parallel | 37.836 | 32.642 |
| 06-current | current | 30.367 | 25.357 |
| 07-parallel | parallel | 34.633 | 29.647 |
| 08-current | current | 29.610 | 24.643 |
| 09-prior | prior | 58.245 | 52.799 |

The public [example](../assets/tum_scoped_evaluation/example-current.yaml)
now requests `window_workers: 3`. Execution is capped at three window processes
and available CPUs, with results aggregated in configured replication order.
The comparison's parallel workers all report three effective workers.

## Changes to native CPU work

Diagnostic scores use the existing bounded evaluation scope to share immutable
frame transforms, instead of constructing a new RGB-D residual plan per probe.
Every residual callback still executes, with its original parameter reads,
fixed population, ordered sum, unmatched penalty and error behavior. Factory
signature caches are scoped to each native reassociation builder and released
on return or error.

Captured native plane queries can use packed lookup from eight points because
their immutable index already exists. Custom numeric inputs and unsupported
indices retain scalar lookup. Native normal-free point arrays can construct
immutable plane geometry directly, preserving voxel insertion order, scalar
accumulation, the first twelve normal candidates, first-tie selection and all
centroid/normal float bits. Array construction is bounded to 65,536 points and
works in chunks of 4,096 voxels.

Ordered row sums share the already tested finite-difference accumulation
algorithm across ordinary RGB-D residuals, field probes and TUM point correction.
Each independent row retains CPython's original accumulation order: left-to-right
on 3.10/3.11 and compensated on 3.12. The native path avoids Python row-list
allocation, retains signed zero, and bounds both array extent and magnitude.
Small, unusual and unsupported-runtime inputs retain builtin summation.

## Scope and evidence

Every run retains all three map frames, eight query frames per window, three
replication starts, all samples and correspondence limits, ablations,
reassociation, known-bad probes, holdout, curvature, multistart and report saving.
The 35 actual input files are hashed before execution and again afterward
against the existing public inventory. BLAS/OpenMP use one thread.

Whole worker time starts before spawning each fresh calibration worker and
includes imports, input verification, window process startup, factor transport,
calibration, saved reports and evidence saving. The API time covers the
calibration call including its child windows and reports. Numerical tracing
hooks are absent from these nine production workers. The host was unreserved;
these observations are not a hardware-independent guarantee.

The input is the already spent public TUM `freiburg1_xyz` development sequence.
This execution comparison makes no accuracy, convergence or SOTA claim.

Separate supplementary traces retain **24 solves, 168 Jacobian evaluations,
1,624 residual evaluations, 150 frontend records, 1,498 diagnostic records and
492 fixed-population scoring events** per complete run. All complete results,
ordered residual/Jacobian array hashes and population scores match exactly.
Trace timings are not production latency evidence.

The [raw-file inventory](../assets/tum_scoped_evaluation/index.json) binds all
[attempts](../assets/tum_scoped_evaluation/attempts.json),
[summary](../assets/tum_scoped_evaluation/summary.json), source capsules,
checked-in examples, actual-input pins and raw worker records. The independent
auditor also compares every complete result to a hash-pinned preceding accepted
result, after verifying only configuration, execution-policy and output-path
differences. All other numerical and result fields are retained.

The [development archive](../assets/tum_scoped_evaluation/development-screens.tar.gz)
preserves the earlier four six-worker comparisons. Their same-worker gains were
1.111x, 1.223x, 1.223x and 1.144x on separate unreserved-host screens. A fused
matching-score snapshot added substantially more machinery without establishing
a useful additional whole-pipeline gain and was reverted. Its source capsule
is retained. These exploratory ratios should not be combined or applied to
the direct example comparison.

```powershell
python tools/audit_tum_scoped_evaluation.py docs/assets/tum_scoped_evaluation
```

The offline audit requires no GPU or dataset download. It checks every indexed
byte, source and input pins, checked-in example/configuration differences, all
three execution controls, original worker times, direct medians, complete
accepted results and supplementary numerical histories.

The frozen runtime passes 687 Windows tests, including 34 real-device cases,
and 653 Linux/Python 3.10 tests with the optional CUDA test module skipped.
Ten archive and rehashed-tampering tests pass on each platform. Strict typing
passes for all 394 source modules on both platform targets; the 129 public
schema exports and their ledger remain byte-identical. The
[validation record](../assets/tum_scoped_evaluation/validation.json) binds these
checks to the measured source capsule.
