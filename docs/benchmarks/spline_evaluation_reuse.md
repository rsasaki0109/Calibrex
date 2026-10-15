# Reusing B-spline fit evaluations

Joint LiDAR-IMU fitting prepares a spline state for each residual family, then
uses the same accepted trajectory and calibration in its next linearization.
Gyroscope and accelerometer samples also commonly have identical timestamps.
The fitter now reuses this work within each fit:

- Accepted cost evaluations supply the next linearization's gathered controls,
  spline states and residuals.
- Bit-identical IMU timestamp arrays on the same clock share the order-two
  state and control-derivative recursion. Its rotation/rate prefix is identical
  to the gyro-only recursion. Pose-only derivatives retain their own order.
- Gather, state evaluation, control derivatives and position sensitivities use
  the same precomputed B-spline basis arrays.
- Clock finite differences retain their original positive/negative time shifts
  and segment selection; their prepared states are shared across IMU families.

The workspace is local to one fit and one trajectory object, including rejected
LM candidates. A different trajectory discards it. Timestamp and clock keys
distinguish signed zero. Residuals are reused only for the same calibration
object, so changed calibration values recompute them. Public residual calls use
fresh workspaces and observe edits to their measurement arrays.

Retained prepared-array bytes are capped at 64 MiB, counting views
conservatively. Oldest batches are evicted when the budget is exceeded;
over-budget batches still compute normally without retention. This is a bound
on the cache's retained work arrays, not process RSS or active solver workspace.
No optional dependency, worker setting or numerical tolerance is added.

## Complete public-data comparison

| Scope | Previous | Shared evaluations | Observed ratio |
| --- | ---: | ---: | ---: |
| Complete worker | 92.704 s | 86.305 s | 1.074x |
| Four CPU fits, including covariance | 20.395 s | 13.370 s | 1.525x |

Two alternating fresh-process pairs replay the original public GLIM Livox bag's
first six seconds with four map rounds. Both variants use the adopted CUDA
scan-odometry path and one BLAS thread, with a separate initially empty CuPy
compilation cache for every worker. Host background load is unreserved.
First initialization and compilation remain inside complete-worker time.
Fitting and evidence evaluation run on CPU.

The dense odometry policy is unchanged: 0.03 m voxels, three local-map scans,
a 1 nm coordinate grid and returned-motion grids of 1 micrometre translation
and 0.1 microradian rotation-vector components. The coordinate and motion grids
remain disabled by default. Return budgets, neighbors, convergence checks,
covariance, holdout and signed controls are unchanged.

The [bound archive](../assets/spline_evaluation_reuse/README.md) retains all four
complete results, every full fitted trajectory/covariance, measurements/logs,
source capsules/manifests, launcher, schema and publisher. The comparison's
`--require-exact-results` additionally requires every physical/evidence field,
execution report and complete fit record to match exactly, excluding only
declared provenance timestamps/commands/source revisions and fit wall times.
The archive audit independently enforces the same requirement.

The clip supplies no independent extrinsic truth and its final verdict remains
INCONCLUSIVE. These observations belong to this dense workload; fitter and
complete-worker ratios have different scopes and are not multiplied.

```bash
python -m tools.benchmark_bspline_lidar_imu --comparison cuda-cuda \
  --baseline-commit 28acb5819bc254f1f8c9889c0bd4b03f2deb3c3f \
  --require-exact-results --max-seconds 6 --max-rounds 4 --repeats 2 \
  --odometry-voxel-size-m .03 --odometry-local-map-scans 3 \
  --odometry-geometry-quantum-m 1e-9 \
  --odometry-motion-translation-quantum-m 1e-6 \
  --odometry-motion-rotation-quantum-rad 1e-7 \
  --output outputs/glim-spline-evaluation-reuse-pairs
python -m tools.audit_bspline_lidar_imu_benchmark docs/assets/spline_evaluation_reuse
```

Use committed runtime sources and a new output directory. Keep the benchmark
launcher and comparison schema fixed until all workers finish.
