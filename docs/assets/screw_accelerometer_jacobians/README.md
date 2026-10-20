# Native screw accelerometer derivative development

These complete `slac.continuous_time_lidar_imu_joint/v0.1` artifacts retain all
three already-spent synthetic seeds used for the
[reliability study](../../research/continuous_time_reliability.md#2-complete-state-derivatives).
This is a correctness and development record. It makes no execution-speed,
independent accuracy, or general convergence claim.

## Correctness change

The fitter previously omitted trajectory-knot columns from accelerometer
factors, although the residual depends on the trajectory's body kinematics.
It now differentiates the existing clamped three-sample model analytically.
The sparse assembly includes the union of all three samples' supporting knots,
including segment crossings and dense nonuniform grids. The lever-arm Jacobian
also applies the accelerometer scale after the mounting rotation.

All 45 regression checks fail against the previous source and pass after the
correction. They compare against independent all-knot finite differences, test
nonunit scale with and without mounting rotation, inspect the actual sparse
assembly and weighted objective gradients, and solve an acceleration-driven
trajectory update with every calibration state fixed. The residual model,
weights, iteration limits, quality gates and artifact schemas are unchanged.

## Frozen source and complete outcomes

The numerical source is commit
[`11cd1ca4e71779505ba071711eb2486b8f4f90d8`](https://github.com/rsasaki0109/Calibrex/commit/11cd1ca4e71779505ba071711eb2486b8f4f90d8).
All 401 Python source modules were hashed with normalized LF text before and
after the runs and remained unchanged. The
[receipt](receipt.json), [receipt schema](receipt.schema.json),
[source manifest](source-manifest.json) and [original launcher](generator.py.txt)
retain the commands, source and result bindings. Runtime: Python 3.12.10,
NumPy 2.3.4, SciPy 1.17.1, Pydantic 2.12.5. OpenBLAS/MKL/OMP limits were one;
concurrent host activity was allowed.

The previous results below are the existing
[measurement-weight correction records](../screw_weighted_objective/README.md),
source `634751653d1be1411b801f26ca50ce83a14d8b75`, rather than a new control run.
All three generated input problems and solver options were also checked for
exact canonical JSON equality with the archived development inputs.

| Seed | Previous fit / iterations | Corrected fit / iterations | Previous policy | Corrected policy | Complete result |
| --- | --- | --- | --- | --- | --- |
| 20261010 | converged / 28 | converged / 29 | pass | pass | [JSON](seed-20261010.json) |
| 20261011 | max_iterations / 31 | converged / 22 | fail | pass | [JSON](seed-20261011.json) |
| 20261012 | converged / 23 | converged / 25 | pass | pass | [JSON](seed-20261012.json) |

Every result retains calibration errors, covariance-derived uncertainty,
training and holdout residuals, all 32 signed known-bad controls, separability,
and seed/command/source provenance. All controls are detected in each case.

| Seed | Gyro holdout RMSE, previous → corrected (rad) | Maximum translation error, previous → corrected (m) | Maximum rotation error, previous → corrected (deg) |
| --- | --- | --- | --- |
| 20261010 | 0.000505911 → 0.000488303 | 0.00650239 → 0.00585998 | 0.175275 → 0.163572 |
| 20261011 | 0.000356604 → 0.000341163 | 0.00800098 → 0.00699457 | 0.0950941 → 0.0972515 |
| 20261012 | 0.000424025 → 0.000423084 | 0.00716335 → 0.00634399 | 0.121820 → 0.126098 |

The policy improves from two passes and one failure to three passes. Some
rotation maxima increase slightly, so these results do not establish uniform
accuracy improvement. The holdout oracle evaluates truth calibration on the
fitted trajectory and therefore changes when that trajectory changes.
For seed 20261011 its gyro RMSE changes from 0.000235692 to 0.000245887 rad;
the corrected holdout is below 1.5 times that value.

The native `converged` label can result from a small accepted damped step;
it does not by itself establish a small undamped gradient. Clock and gyro
derivatives, correlated-prior whitening, and convergence remain explicit audit
work before the fixed-budget comparison. Earlier failures remain unchanged in
their original records.

## Reproduce

Using the pinned numerical source and the thread limits above, run all three
commands separately. Retain every outcome; floating-point and platform
differences can affect convergence. Provenance timestamps and output paths
will differ on replay.

```bash
calibrex trajectory recover-lidar-imu-joint --trajectory screw_linear --seed 20261010 --output replay-20261010.json --json
calibrex trajectory recover-lidar-imu-joint --trajectory screw_linear --seed 20261011 --output replay-20261011.json --json
calibrex trajectory recover-lidar-imu-joint --trajectory screw_linear --seed 20261012 --output replay-20261012.json --json
```

## Integrity

| Artifact | SHA-256 |
| --- | --- |
| seed-20261010.json | `54b1b181283fe825912d6564f3220027a95f460e6a849df3708dd656692f8233` |
| seed-20261011.json | `24d24beb478bbb98d46adc3d2f5c309f9ef649cde152474ab76583af534c6137` |
| seed-20261012.json | `bd64c2b734f8211bd03690a0d8e0471af9e474e7761b3649b2ea1c912879d19d` |
| source-manifest.json | `789c677a7cd8a5211055086d75f0ae8dea6b40fb7130690db164a4dd3f6f21d0` |
| generator.py.txt | `36aa2f6721f05939e63a15150790cbd30a9a2529280b89a8fb9173d8eafa97a7` |

Maintained tests verify the full declared cohort, native schemas, byte hashes,
source/seed provenance, signed controls, and the receipt's bindings.
