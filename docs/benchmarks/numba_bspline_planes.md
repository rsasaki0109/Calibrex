# Compiled CPU plane evaluation for LiDAR-IMU fitting

The joint cubic B-spline fitter has an optional `numba_cpu` plane evaluator.
It combines pose evaluation, whitened plane residuals and their 24 local
Jacobian columns in a compiled loop, reusing small workspaces within each row
slice. Scalar array access avoids creating per-return views and reduces JIT
startup overhead.
Independent slices can run in a persistent CPU thread pool. The kernel
uses FP64 arithmetic with `fastmath=False` and disables Numba's automatic
parallel pass. NumPy keeps the referenced-edge SO(3) logarithm and near-pi guard.

Runtime v0.3 also accumulates the normal system directly. Plane contributions
use the seven-block control band instead of assembling a large sparse Jacobian
on every LM iteration. IMU control, calibration and cross products retain that
local structure. Aligned gyro/acceleration timestamps share spline states,
control derivatives and the two clock-offset probes within one linearization.
Different timestamp arrays retain separate evaluations. Plane accumulation uses
the existing worker pool; IMU products use a serial compiled kernel. The final
covariance still uses the complete assembled Jacobian and original Schur method.
The NumPy default retains its original sparse normal construction.

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

## Runtime v0.3 complete fixed-input fits

Three alternating fresh-process pairs compare NumPy with the structured
compiled backend on the same frozen source and public GLIM-derived problem.
Each worker performs a complete first fit, then a complete warm fit.

| Scope | NumPy median | Structured compiled median | Observed ratio |
| --- | ---: | ---: | ---: |
| First use, empty private JIT cache | 18.550 s | 20.963 s | 0.885x |
| Full fit after that worker's own first fit | 33.047 s | 3.180 s | 10.391x |

Host resources were unreserved and worker times varied substantially. These
descriptive medians establish neither a stable tenfold gain nor the incremental
gain over earlier compiled versions. The first-use regression is retained;
compilation must be amortized. Every fit includes LM, sparse solves and final
covariance, but excludes bag decoding, odometry, mapping and evidence controls.
Individual warm pair ratios range from 3.553x to 11.504x.

All six full comparisons pass `rtol=1e-8`, `atol=1e-9`; status and 22 iterations
agree exactly. Trajectories, six calibration blocks, covariance, costs and
residual variance are retained in the
[v0.3 evidence archive](../assets/numba_bspline_banded/README.md).
The problem has 28,563 plane rows and 1,550 rows per IMU family. Its original
extraction helper was unversioned; NPZ/metadata hashes bind the consumed inputs.
No independent extrinsic accuracy or whole-bag acceleration is claimed.

```bash
python -m tools.audit_bspline_planes docs/assets/numba_bspline_banded
```

## Runtime v0.3 complete bag regression and negative timing

A fixed-order v0.2/v0.3 pair processes the original public bag's first six
seconds, four map rounds, covariance, holdout and 32 signed controls with CPU
odometry. Both workers start with empty private JIT caches. All four paired
full fits pass `rtol=1e-8`, `atol=1e-9`, including exact status/iterations;
complete calibration and evaluation decisions agree. GT remains unused and
the result remains INCONCLUSIVE.

The whole worker takes 54.809 s versus 67.962 s (0.806x). All four fits take
13.826 s versus 12.719 s (1.087x); the first fit regresses from 5.884 s to
8.963 s, while the subsequent three together take 7.942 s versus 3.756 s.
Unchanged odometry also varies from 24.242 s to 36.494 s. This unreserved
single pair does not establish whole-pipeline acceleration. The
[complete regression archive](../assets/numba_bspline_banded_bag/README.md)
retains this negative timing, full results, source snapshots and a separate
strict full-fit audit. NumPy remains the default; the compiled path requires
amortizing startup on repeated fitting.

## Earlier v0.2 complete public bag validation

Two earlier v0.2 cohorts decode the original public GLIM bag's first six seconds, including
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
