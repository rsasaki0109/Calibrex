# Shared IMU evaluation: a measured 2x target miss

The native `screw_linear` fitter now shares a gyro factor's integral between
its residual and mounting-rotation Jacobian. Previously a mounted factor
integrated the same samples three times per linearization. Objective and IMU
RMSE evaluation now compute values without knot/calibration derivatives or
the accelerometer's perturbed clock sample. The gyro product order, force
prediction, sampled clock derivatives, weighting and stopping rules stay fixed.

This removes redundant execution while preserving complete numerical results.
**It does not meet the declared 2x whole-API target.** The three-seed paired
median is 1.452x for the complete joint API and 1.292x including fresh-worker
startup and output. This is an incremental CPU optimization. It is not a new
GPU result, an independent accuracy study, or evidence of a large new speedup.

## Frozen comparison

| Item | Value |
| --- | --- |
| Baseline | `0f76b0ec6bda2d975ea58d4b4bbd52d174892dae` |
| Numerical candidate | `6db6df77403461e86813c5c6f207cd7c9943a24e` |
| Cases | Already-spent synthetic LiDAR–IMU seeds 20261010, 20261011, 20261012 |
| Order | Baseline, candidate, candidate, baseline, baseline, candidate |
| Execution | Six fresh CPU workers, one BLAS thread, no warmups, unreserved host |
| Whole-API target | Paired median at least 2x; **missed** |

The maintained joint-recovery worker includes fitting, covariance, holdout
metrics, separability and all 32 signed controls in its API interval. Imports,
source checks, JSON serialization and process startup/exit are included in the
fresh-worker interval. Parent source preparation is outside both intervals.
Input/options and full fitted/native payload hashes omit only their top-level
provenance, which remains retained separately. No data was downloaded.

| Seed | Complete API ratio | Fit ratio | Fresh-worker ratio |
| --- | ---: | ---: | ---: |
| 20261010 | 1.452x | 1.388x | 1.292x |
| 20261011 | 1.384x | 1.357x | 1.236x |
| 20261012 | 1.509x | 1.452x | 1.353x |
| Paired median | **1.452x** | **1.388x** | **1.292x** |

API medians across the three observations are 12.155 s baseline and 8.748 s
candidate; fit medians are 10.329 s and 7.609 s. The table uses medians of
paired ratios, rather than dividing these marginal medians. All three complete
inputs/options, fitted results and native outcomes match exactly between
variants, including covariance, final gradient, status, accepted iterations,
holdouts and controls. Fits retain 21/23/29 iterations. These results cannot
be multiplied by earlier published ratios or generalized to public recordings.

## Correctness and replay

New regression tests compare the previous gyro and force expressions at
nonidentity and identity mounting rotations, nonunit scales, nonzero clock
offset, nonuniform sample times, zero/tiny increments, clamped accelerometer
stencils and knot boundaries. They verify bit equality of all factor blocks,
one integration per linearized gyro factor, fresh state between calls,
derivative-free objective/RMSE evaluation, complete fitted payloads and the
sparse matrices produced across multiple updates and damping settings.
Public signatures and serialized schemas are unchanged.

The [complete evidence](../assets/screw_imu_evaluation/README.md) retains all
six workers, every full input/options/result, both 401-module source capsules,
the original benchmark toolchain, result schemas, source tests and producer.
Replay its bindings and whole-result comparisons without running calibration:

```console
python tools/benchmark_screw_imu_evaluation.py --archive docs/assets/screw_imu_evaluation/complete-evidence.tar.gz
```

Exit 2 reports the retained 2x target miss. It is distinct from a corrupt or
incomplete evidence error. The archive reader imports no archived calibration
code and rejects unsafe paths, duplicate members and digest mismatches.

To run a new development comparison from committed sources, choose a fresh
output directory:

```console
python tools/benchmark_screw_imu_evaluation.py --baseline-commit 0f76b0ec6bda2d975ea58d4b4bbd52d174892dae --output outputs/new-imu-comparison
```

The remaining work is structural batching of trajectory/measurement evaluation
and the numerical stopping audit described in the
[fixed-budget reliability plan](../research/continuous_time_reliability.md).
This execution cleanup does not resolve the mixed-unit stopping criterion or
the known finite-difference clock sensitivity.
