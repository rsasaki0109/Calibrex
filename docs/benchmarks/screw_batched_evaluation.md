# Batched screw trajectory evaluation

The native `screw_linear` CPU fitter now batches fixed-state pose interpolation
and point-to-plane evaluation. It also shares accelerometer motion stencils and
skips the translation solve when only a rotation logarithm is consumed. These
changes preserve complete calibration results, including covariance, final
gradient, accepted iterations, holdouts and signed controls.

Against the preceding maintained implementation, the paired median complete-API
ratios are **2.023x for synthetic LiDAR–IMU recovery** and **4.709x for spent
public GLIM geometry**. Both meet the unchanged 2x median API gate in this
development cohort. The joint result is close to that threshold and includes
a 1.641x pair. Including process startup, imports, source checks and output,
the paired medians are **1.462x** and **2.189x**, respectively.

## Frozen comparison

| Item | Value |
| --- | --- |
| Baseline | `5278168adbe8d1cffc4abbcd13800d901d0139e7` |
| Numerical candidate | `df4144a4d135dea3b0863847f26bc778b56fc719` |
| Joint inputs | Already-spent synthetic seeds 20261010, 20261011, 20261012 |
| Geometry input | Eight already-spent AIST GLIM registration captures; 1,013 retained planes |
| Order per case | Baseline, candidate, candidate, baseline, baseline, candidate |
| Execution | Twelve fresh CPU workers, one BLAS thread, no warmups, unreserved host |
| Gate | Paired median complete API at least 2x for each case |

The maintained benchmark was unchanged. One complete cohort was run; all
twelve observations are retained. No inputs were downloaded, timing attempts
discarded, convergence thresholds relaxed or observations repeated to obtain
a better ratio.

| Pair | Joint API | Joint fresh worker | Geometry API | Geometry fresh worker |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2.023x | 1.462x | 5.264x | 2.257x |
| 2 | 1.641x | 1.248x | 4.709x | 2.189x |
| 3 | 2.057x | 1.555x | 3.908x | 1.802x |
| Paired median | **2.023x** | **1.462x** | **4.709x** | **2.189x** |

| Median elapsed seconds | Joint baseline | Joint candidate | Geometry baseline | Geometry candidate |
| --- | ---: | ---: | ---: | ---: |
| Complete API | 9.217 | 4.826 | 8.024 | 1.747 |
| Fresh worker | 13.641 | 10.663 | 12.100 | 5.529 |

Ratios are medians of paired ratios, rather than ratios of marginal medians.
The API interval includes fitting and covariance. For joint recovery it also
includes holdout metrics, separability and all 32 signed controls. The larger
fresh-worker interval includes imports, source verification, serialization and
process startup/exit. Parent source freezing and geometry preparation are
outside both intervals.

All complete input/options, fit payload and native outcome hashes agree within
every baseline/candidate pair, omitting only top-level provenance. Provenance
is retained separately. Joint fits retain 21/23/29 iterations, policy passes
and all 32 detected controls. Every geometry fit retains 15 iterations.

## Execution and correctness

Known pose queries are grouped by screw interval, with batch temporaries
limited to 256 queries. Constant plane coordinates and normalized normals are
snapshotted once per fit. Point transforms and support Jacobians are evaluated
in batches while residual order and sequential weighted objective sums stay
fixed. Tiny groups retain lazy scalar evaluation.

The batch expressions retain the scalar norms, libm calls, multiplication
order and quaternion normalizations. Shared state belongs to one objective or
linearization; cached arrays are read-only and never reused after a knot
update. Accelerometer residuals and derivatives share their nominal stencil.
The existing sampled clock derivative, gyro integration order, stopping rules,
public signatures and result schemas remain unchanged. No GPU backend or new
dependency is introduced.

The 42 new checks compare scalar and batched poses, every Jacobian bit, signed
zero, zero/tiny/near-pi rotations, strided points, empty and small batches,
chunk boundaries, cache refresh and complete sparse models across multiple
updates and damping settings. A targeted run passes 421 tests. The first
development smoke run exposed a local name shadowing an interpolation helper;
its failure and the corrected runs are retained.

## Replay and limits

The [complete evidence](../assets/screw_batched_evaluation/README.md) contains
all observations, both complete source capsules, original benchmark tools,
input/options/results, public geometry, schemas and test records. Replay the
hashes, schemas, whole-result comparisons and gates without executing archived
calibration code or downloading a recording:

```console
python tools/audit_screw_residual_archive.py docs/assets/screw_batched_evaluation/complete-evidence.tar.gz
```

To run a new comparison from committed sources, choose a fresh output directory:

```console
python tools/benchmark_screw_residual_evaluation.py --baseline-commit 5278168adbe8d1cffc4abbcd13800d901d0139e7 --geometry-folder outputs/odometry-search-profile-v3 --output outputs/new-batch-comparison
```

These are spent development fixtures. GLIM uses assigned timestamps and fixed
capture-local nearest planes, so it measures processing rather than the
recording's physical trajectory or calibration accuracy. The unreserved host
and three pairs per case do not establish a universal 2x improvement. Earlier
published speed ratios cannot be multiplied by these measurements.

This execution improvement supplies more room for the
[fixed-budget reliability study](../research/continuous_time_reliability.md).
Mixed-unit stopping, sampled clock sensitivity and prior-packing approximations
still need their own numerical work before an independent reliability claim.
