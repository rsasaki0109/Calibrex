# Shared rotations in LiDAR-IMU calibration

Native gyro deskew and joint cubic B-spline fitting reuse rotation terms to
reduce the remaining cost of `calibrex imu-lidar joint`. The implementation
uses the ordinary NumPy/SciPy CPU dependencies. Public callbacks, calibration
settings, result schemas and evidence policy keep their existing interfaces.

## What changes

Gyro deskew previously constructed an orientation and a bias Jacobian matrix
for every return, then composed several matrices before applying the result
to a point. For a fixed bias, the new internal model projects the cumulative
bias sums into three-vectors once. The rotation identity
`Q Exp(-Q.T ds) = Exp(-ds) Q` lets it apply the equivalent rotations directly
to points. Each scan reuses its rotated coordinates across deskew rounds,
while the translation follows the updated motion estimate. The original
matrix callback remains available. Extrapolated times, large biases,
non-orthogonal matrices and unusual inputs use that reference path.

The spline fitter computes each referenced adjacent control rotation, its
logarithm and inverse right Jacobian once, then gathers those terms for the
timestamps in that segment. Residuals and analytic Jacobians share their
exponential rotations. Plane residuals evaluate pose only; gyro residuals
also evaluate velocity; accelerometer residuals retain acceleration. The
plane orientation derivative accumulates adjacent 3-by-3 blocks through the
rotation suffix rather than propagating mostly empty 3-by-12 arrays.

These are algebraic changes to the same models. Sampling, feedback passes,
iteration caps, LM acceptance, sparse solve, covariance and signed controls
retain their existing settings. Temporary shared terms consume memory
proportional to the selected returns; there is no persistent trajectory cache.
Caller edits to controls are observed on the next evaluation.

## Public-data comparison

The comparison starts from the analytic-Jacobian implementation in
[the preceding benchmark](bspline_lidar_imu_acceleration.md). It reads the
same original public GLIM Livox `livox.bag`: the first 12 seconds of
`/livox/points` with point time `t`, and `/livox/imu` with acceleration in g.
Four map rounds retain the default sampling and 30-iteration fit caps. The
supplied GT trajectory is not used.

Three alternating fresh-process pairs use one BLAS thread and frozen source
trees. Whole time includes startup, bag reading, initial calibration,
gyro-deskewed odometry, association, all four fits and covariance, noise
updates, holdout, signed controls, result saving, hashing and verified imports.
The fit time includes covariance. Source preparation is outside the worker.

| Scope | Analytic-Jacobian baseline | Shared rotations | Speedup |
| --- | ---: | ---: | ---: |
| Complete worker | 142.930 s | 100.615 s | 1.421x |
| Four fits, including covariance | 50.106 s | 33.072 s | 1.515x |

The incremental whole-run gain is modest: 1.421x after the preceding
analytic-Jacobian optimization. It does not establish another large
end-to-end acceleration. This implementation is a candidate for review.

The [complete evidence archive](../assets/lidar_imu_shared_rotations/README.md)
retains every trial, result, full per-round covariance and trajectory, actual
module digests, frozen source trees, launcher and tool-only schemas. Raw bag
data is an external input. Exploratory profiles had changing background load;
their wall-time ratios are excluded from the reported comparison.

## Quality and limits

Tests compare direct point rotation with the preserved matrix callback,
including nonzero bias, epoch timestamps, input mutation and reference
fallback. An odometry integration test checks reuse across deskew rounds,
float32 point offsets and ordinary custom callbacks. Prepared spline states
match the public evaluator, including repeated timestamps and knots. The
orientation Jacobian is independently checked by central differences at
small, ordinary and near-pi rotations. Synthetic calibration recovery keeps
the existing accuracy and covariance assertions.

Public result comparisons use the fixed tolerances of the preceding
benchmark. Decisions must agree exactly; numerical estimates, evidence,
full per-round trajectories and covariance must agree within the declared
tolerances. Results remain **INCONCLUSIVE** on this capture. Its estimated
map and formal covariance do not provide independent extrinsic truth.
This is a regression and performance result for one clip; larger captures
and different point densities can have different speedups. Ratios from
separate benchmark cohorts should not be multiplied.

## Reproduce and audit

```bash
python -m tools.benchmark_bspline_lidar_imu \
  --baseline-commit 6df5aa2526b1fb4d5bc8511b7e9a0265d1266abc \
  --bag data/public/glim_versatile/livox.bag \
  --max-seconds 12 --max-rounds 4 --repeats 3 \
  --output outputs/glim-shared-rotations-comparison
python -m tools.audit_bspline_lidar_imu_benchmark \
  docs/assets/lidar_imu_shared_rotations
```

The output directory must be new and runtime sources must be committed.
The CPU-only archive audit checks all file digests, full result models,
imported sources and covariance, and recomputes quality and median ratios.
