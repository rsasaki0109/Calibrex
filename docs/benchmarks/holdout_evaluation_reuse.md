# Reusing fixed-trajectory holdout evaluations

Joint LiDAR-IMU calibration scores 32 signed known-bad controls, plus baseline
gyro and accelerometer chi-square values. Previously, each of those 34 scores
prepared the trajectory and evaluated all three residual families, including
the unused LiDAR planes. Only one family's sum of squares was retained.

The evidence stage now uses a private fixed-trajectory residual session. It
evaluates only the requested family and shares prepared spline states with
the preceding holdout RMS evaluation. Calibration perturbations still
recompute the requested residual, while clock perturbations prepare the
original shifted timestamps and segment selections. Bit-identical gyro and
accelerometer timestamp arrays share their state recursion; unequal arrays
remain separate.

The session reuses the fitter's 64 MiB retained-array bound and eviction rules.
Train and holdout have independent sessions. Inputs stay fixed during evidence
evaluation, and the session ends with that evaluation. Public residual calls
still create fresh sessions and observe measurement edits between calls.
Public APIs, result schemas, samples, summation order, control magnitudes and
decision thresholds are unchanged. This change runs on CPU.

## Complete public-data comparison

| Scope | Previous | Reused holdout evaluations | Observed ratio |
| --- | ---: | ---: | ---: |
| Complete worker | 82.051 s | 83.963 s | 0.977x |
| Signed controls after holdout RMS | 2.073258 s | 0.013284 s | 156.076x |
| Unchanged four CPU fits, including covariance | 12.768 s | 13.567 s | 0.941x |

The complete worker is **2.3% slower** in these observations, despite removing
about 2.06 seconds from the signed-control scope. Unchanged fit and GPU normal
preparation timings also vary. These two pairs do not isolate the cause of the
whole-worker difference, and establish no full-calibration speed gain. The
implementation is retained for its reduced CPU evidence work; this cohort is
not counted as successful end-to-end acceleration.

Two alternating fresh-process pairs replay the original public GLIM Livox
bag's first six seconds with four map rounds. Both variants use the adopted
CUDA scan-odometry path, one BLAS thread and separate initially empty CuPy
compilation caches. The host's background load is unreserved. Whole time
includes all preparation, imports, decoding, odometry, fits/covariance,
evidence, hashes, saves and source verification.

The control timer starts after holdout RMS evaluation, including its existing
baseline calls and all signed scores. Prepared states retained from that RMS
are already available to the current implementation; this timing is a stage
measurement, not a cold standalone score or a complete-worker speedup.

The dense odometry policy remains 0.03 m voxels and three local-map scans, with
an opt-in 1 nm coordinate grid and returned-motion grids of 1 micrometre
translation and 0.1 microradian rotation-vector components. These grids remain
disabled by default. Both implementations execute the same GPU kernels.

The [bound archive](../assets/holdout_evaluation_reuse/README.md) retains all
four complete results, full fit records, measurements/logs, source capsules,
launcher, schema and publisher. `--require-exact-results` checks every physical
and evidence field and execution report after excluding only declared run
metadata. Every fit record must also match exactly, excluding wall time.
The archive audit independently checks equality and the 34 scored calls.

The real-data verdict remains INCONCLUSIVE and this clip supplies no
independent extrinsic truth. Stage and complete-worker ratios have different
scopes; neither is multiplied by observations from other cohorts.

```bash
python -m tools.benchmark_bspline_lidar_imu --comparison cuda-cuda \
  --baseline-commit e59600c1354dc14f0b5afd06f92d90bf61d56f41 \
  --require-exact-results --max-seconds 6 --max-rounds 4 --repeats 2 \
  --odometry-voxel-size-m .03 --odometry-local-map-scans 3 \
  --odometry-geometry-quantum-m 1e-9 \
  --odometry-motion-translation-quantum-m 1e-6 \
  --odometry-motion-rotation-quantum-rad 1e-7 \
  --output outputs/glim-holdout-evaluation-reuse-pairs
python -m tools.audit_bspline_lidar_imu_benchmark docs/assets/holdout_evaluation_reuse
```

Use committed runtime sources and a fresh output directory. Keep the launcher,
comparison schema and repository HEAD fixed until all workers finish.

Local validation found a separate, existing screw-linear synthetic recovery
test returning `max_iterations` instead of `converged`. Frozen baseline sources
reproduce the same assertion failure; its source-bound validation receipt/log
are retained in the archive. The B-spline evidence tests and both synthetic bag
calibration tests pass. No assertion or solver setting was relaxed.
