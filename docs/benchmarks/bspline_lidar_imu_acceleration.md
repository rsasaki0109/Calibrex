# Joint B-spline LiDAR-IMU acceleration

The native joint B-spline fitter uses analytic control-point rotation
Jacobians and shares the evaluated trajectory states with the residuals and
calibration derivatives. This speeds up `calibrex imu-lidar joint` and
`calibrex trajectory recover-lidar-imu-joint --trajectory cubic_bspline`
on the CPU, with the same public interfaces and result schemas.

The previous fitter reevaluated every local spline 24 times per residual
family to differentiate four rotation controls along three axes. The new
fitter differentiates the cumulative SO(3) recursion once, using right
tangents. Plane returns need only the orientation derivative; gyro factors
also need angular velocity, and accelerometer factors need angular
acceleration. Bias, gravity, IMU mounting rotation and lever-arm derivatives
reuse the base states. The clock-offset column retains central differences
and independently selects the segment at each perturbed time, including
across knots. Residuals, LM settings, sparse solve, covariance and evidence
gates are unchanged.

## Public-data comparison

The input is the public AIST GLIM Livox Avia `livox.bag`, described in the
[checked stream manifest](https://github.com/rsasaki0109/Calibrex/blob/main/examples/public_datasets/glim_versatile/manifest.yaml).
The comparison reads its first 12 seconds: `/livox/points` with per-point
`t`, and `/livox/imu` with acceleration in g. It uses the native angular-rate
initial estimate and gyro-deskewed odometry, then four map rounds with the
default sampling, 30 fit iterations per round, gravity prior and interleaved
IMU holdout. The supplied GT trajectory is not used by this run.

Both complete source trees are frozen before measuring. Three alternating
pairs run in fresh processes with one BLAS thread. Whole time includes
startup, bag reading, initial estimates, map construction/association, all
fits, marginal covariance, noise updates, holdout, 32 signed controls,
saved results, input hashing and verified imports. Fit time is the sum of
the four full fits, including their covariance. Neither is a kernel-only
measurement. See the [complete evidence archive](../assets/bspline_lidar_imu_acceleration/README.md).

| Scope | Prior median | Analytic median | Speedup |
| --- | ---: | ---: | ---: |
| Complete worker | 325.065 s | 127.257 s | 2.554x |
| Four fits, including covariance | 251.629 s | 39.818 s | 6.320x |

## Quality and limits

Independent central differences test every local rotation axis at small,
ordinary and near-pi rotations and at the segment boundaries. The assembled
plane, gyro and accelerometer Jacobians are checked against perturbations
of the actual residuals. Synthetic joint recovery and bag tests retain
known truth, holdout and signed-control assertions.

The public comparison fixes physical tolerances before measuring: 0.001 deg
rotation, 10 micrometres lever arm, 0.1 microseconds clock offset and 1e-5
whitened holdout RMS. It also checks block estimates, formal std, last-round
changes, signed-control chi-square and correlations; all policy, block,
control, separability and convergence decisions must agree. Full per-round
covariance and trajectory arrays are retained for inspection.

This is a performance and regression result for one 12-second public clip.
The plane map is estimated from the same trajectory; its formal covariance
holds that map fixed. The final calibration remains **INCONCLUSIVE**, so
the timing result does not establish absolute extrinsic accuracy or a
conclusive calibration on this capture. Throughput depends on the number of
returns, knots, fit iterations and the cost of the initial odometry.

## Reproduce

Place the original public GLIM `livox.bag` under
`data/public/glim_versatile/`; the raw recording is an external input and is
not redistributed in this repository. With the ROS bag reader installed:

```bash
python -m tools.benchmark_bspline_lidar_imu \
  --baseline-commit b89e50bbd9e48a05a2acb6010ac46e50ece9220f \
  --bag data/public/glim_versatile/livox.bag \
  --max-seconds 12 --max-rounds 4 --repeats 3 \
  --output outputs/glim-joint-bspline-comparison
```

The output directory must not exist and runtime sources must be committed.
Each worker verifies every imported Calibrex module against its frozen
source tree. Failures leave their logs and attempt records intact. The
comparison deliberately uses the GLIM topic, point-time and IMU unit
conventions above.

Audit the published archive without downloading a bag or installing CUDA:

```bash
python -m tools.audit_bspline_lidar_imu_benchmark
```

The audit validates the complete models and file digests, checks actual
source imports, and recomputes quality and all reported median speed ratios.
