# Selective B-spline holdout controls

The joint LiDAR-IMU pipeline checks 32 signed, deliberately incorrect calibration
perturbations on held-out IMU samples, plus two baseline scores. Previously each
score evaluated plane, gyro and acceleration residuals, although the score uses
only one IMU family. This repeated LiDAR spline evaluation 34 times after fitting.

`lidar_imu_residuals(..., families=("gyro",))` now evaluates only the requested
family. The signed controls request either gyro or acceleration residuals. All
32 perturbations, thresholds and decisions are retained. The public API still
evaluates all three families by default; core result schemas are unchanged.
Selected residual arrays retain the original arithmetic and input order.

## Complete public-data comparison

| Scope | Previous median | Selective median | Observed ratio |
| --- | ---: | ---: | ---: |
| Complete worker | 77.870 s | 77.507 s | 1.005x |
| Signed control suite | 2.425 s | 0.090 s | 26.920x |
| Unchanged fits/covariance | 17.555 s | 17.058 s | 1.029x |

The complete-worker median is effectively unchanged. This removes a clear
waste in validation but does not establish a meaningful whole-pipeline gain.
Differences in unchanged fit timings reflect run variation.

Three alternating previous/current pairs replay the first six seconds of the
public GLIM Livox bag, with the same CUDA odometry and four map rounds. Whole
timing includes fresh process startup, compilation, decoding, odometry,
association, fitting, covariance and every validation control. The control
observer measures the complete signed-control suite once per pipeline.

Each worker uses one BLAS thread and a separate initially empty CuPy compilation
cache. Runtime/driver caches can persist and background host load is unreserved.
Component gains are not multiplied or presented as whole-pipeline gains.
The complete calibration evidence, fitted arrays and source bindings are
retained in the [CPU-auditable archive](../assets/selective_bspline_controls/README.md).

This is a CPU computation reduction. It adds no GPU kernels or new dependency.
It does not reduce scan, point, neighbor, fit-iteration or validation budgets.
Real-data verdicts remain **INCONCLUSIVE**: this short clip has no independent
extrinsic ground truth.

```bash
python -m tools.benchmark_bspline_lidar_imu --comparison cuda-cuda \
  --baseline-commit 392602713e9fc98b9a9defa86c3c29f106dec809 \
  --max-seconds 6 --max-rounds 4 --repeats 3 \
  --odometry-voxel-size-m .03 --odometry-local-map-scans 3 \
  --odometry-geometry-quantum-m 1e-9 \
  --odometry-motion-translation-quantum-m 1e-6 \
  --odometry-motion-rotation-quantum-rad 1e-7 \
  --output outputs/selective-bspline-controls-comparison
python -m tools.audit_bspline_lidar_imu_benchmark docs/assets/selective_bspline_controls
```

Use committed source and a fresh output directory. Coordinate and returned-motion
grids are explicit shared benchmark settings and remain disabled by default.
Tests check selected residual bits and ordering, skipped unrelated timestamps,
synthetic bag recovery, the requested families of every signed control, and
rejection of missing, zero, impossible or repeated control timing evidence.
