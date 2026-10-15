# Compiled CPU plane evaluation for LiDAR-IMU fitting

The joint cubic B-spline fitter has an optional `numba_cpu` plane evaluator.
It combines pose evaluation, whitened plane residuals and their 24 local
Jacobian columns in a compiled loop, reusing small workspaces within each row
slice. Runtime v0.2 also uses scalar array access in its small matrix loops
to avoid creating per-return views and reduce JIT startup overhead.
Independent slices can run in a persistent CPU thread pool. The kernel
uses FP64 arithmetic with `fastmath=False` and disables Numba's automatic
parallel pass. NumPy keeps the referenced-edge SO(3) logarithm and near-pi guard.

Install the existing optional dependency and select it explicitly:

```bash
pip install 'calibrex[numba]'
calibrex imu-lidar joint PATH_TO_BAG \
  --lidar-topic /livox/points --imu-topic /livox/imu --acceleration-unit g \
  --dataset-family glim_versatile --dataset-license 'upstream dataset terms' \
  --fit-backend numba_cpu --fit-workers 4 --output calibration.json
```

The Python API accepts `BSplineLidarImuFitOptions(plane_backend="numba_cpu",
plane_workers=4)`. Small inputs use fewer workers to avoid thread overhead.
The fitter releases its pool when it finishes or fails. NumPy remains the
default and imports neither Numba nor CuPy. Bag artifacts retain the selected
backend, requested workers and actual runtime/version/arithmetic identity in
their existing schema-valid `options` field.

## Complete public bag validation

Two cohorts decode the original public GLIM bag's first six seconds, including
initial estimates, CPU odometry, four map rounds, fits/covariance, noise updates,
holdout, 32 signed controls, input hashing, saving and frozen-import checks.
Every compiled worker starts with an empty Numba cache. Both methods use the
same runtime source and geometry settings, with unchanged work/evaluation budgets.

| Cohort | Whole NumPy median | Whole compiled median | Observed ratio |
| --- | ---: | ---: | ---: |
| First variable-load cohort | 58.094 s | 78.003 s | 0.745x |
| Repeat with CPU execution-time accounting | 86.076 s | 60.834 s | 1.415x |

These opposite timing outcomes **do not establish reliable whole-pipeline
acceleration**. Unchanged odometry times also vary substantially; CPU execution
time still depends on frequency and other host conditions. CPU resources were
unreserved. NumPy remains the default.

All six complete calibration/evidence comparisons and all 24 paired full fit
records pass. Fit status, iterations, dimensions and parameter order agree
exactly; trajectories, all calibration blocks, covariance, costs and variance
pass the predeclared `rtol=1e-8`, `atol=1e-9` gates. Policy, holdout, signed
controls and separability decisions agree. The final verdict remains
**INCONCLUSIVE**, and the supplied GT is unused. This validates regression
behavior, with no independent absolute accuracy claim.

The [complete bag archive](../assets/numba_bspline_public_bag/README.md) retains
both cohorts, full results, fitted arrays, source trees and frozen quality
helpers. Its CPU-only audit recomputes every result comparison and ratio:

```bash
python -m tools.audit_bspline_lidar_imu_benchmark docs/assets/numba_bspline_public_bag
python -m tools.benchmark_bspline_lidar_imu \
  --baseline-commit HEAD --current-fit-backend numba_cpu --fit-workers 4 \
  --max-seconds 6 --max-rounds 4 --repeats 3 --output outputs/compiled-plane-bag
```

## Earlier v0.1 fixed-input fit comparison

| Scope | NumPy median | Compiled median | Observed speedup |
| --- | ---: | ---: | ---: |
| Full fit, first use with empty JIT cache | 27.068 s | 28.776 s | 0.941x |
| Full fit after that worker's own prime | 31.166 s | 11.176 s | 2.789x |

This earlier archive measures runtime v0.1, before the scalar-access startup
change. Three alternating fresh-process pairs use the same committed, frozen source
tree. Every worker first primes its own backend and then repeats the complete
fit. Each compiled worker begins with an empty private JIT cache. The fit
includes residuals, analytic Jacobians, LM, sparse solves and final covariance.
First-use and warm times are separate; compilation must be amortized for the
normal-run improvement. Host resources were unreserved.

All six comparisons preserve status and 22 iterations exactly and pass fixed
`rtol=1e-8`, `atol=1e-9` checks for all fitted arrays and numerical evidence.
Independent tests check every local control axis against finite differences,
knots, repeated/unsorted timestamps, large epochs, zero/small/near-pi rotations,
threaded repeatability and visible control edits. Both native and compiled
paths recover known calibration on synthetic data and pass the synthetic
rosbag CLI test with schema validation and held-out evidence.

The measured input is a locally saved public GLIM-derived fitting problem,
with 28,563 plane rows and 1,550 rows per IMU family. The extraction helper was
unversioned; the input hashes identify the exact consumed problem. These
numbers cover this fixed fitting stage, excluding bag decoding, odometry,
mapping and association. They establish neither a whole-bag speedup nor
independent extrinsic accuracy. The original sensor arrays remain external.
The [complete archive](../assets/numba_bspline_planes/README.md) contains the
protocol, all results, source snapshot, hashes and replayable audit.

```bash
python -m tools.benchmark_bspline_planes \
  --problem outputs/glim-joint-analytic-profile-v1/problem-01 \
  --workers 4 --repeats 3 --output outputs/compiled-plane-fit-comparison
python -m tools.audit_bspline_planes
```

The benchmark requires committed runtime sources and external `.npz`/`.json`
input pairs in its documented loader format. It hashes the inputs, verifies
actual frozen imports, saves complete fitted results and rejects numerical or
decision drift. It can also accept repeated `--problem` arguments to measure
several complete fits in each first-use/warm batch.
