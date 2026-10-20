# Native screw gyro derivative development

These complete `slac.continuous_time_lidar_imu_joint/v0.1` artifacts retain all
three already-spent synthetic seeds used by the
[reliability study](../../research/continuous_time_reliability.md#2-complete-state-derivatives).
This is a correctness and development record, with no execution-speed,
independent accuracy, or general convergence claim.

## Correctness change

The gyro factor previously used the opposite SO(3) Log differential for each
endpoint and the inverse Exp differential for integrated bias/scale updates.
Its bias derivative also omitted the diagonal gyro scale. It now distinguishes
left and right perturbations of the rotation-error product and chains each
increment's right Exp differential through subsequent rotations.

The historical `so3_left_jacobian` helper implements the right Exp differential;
its formula and the existing SE(3) retraction are preserved. Forward integration,
sampling, residual values, measurement weights, solver limits, quality gates
and artifact schemas are unchanged. The
[value-parity record](value-parity.json), [schema](value-parity.schema.json),
[original producer](verify-values.py.txt) and [fixture](fixture.py.txt) retain
exact equality of old/current residuals and integrated rotations on ten declared
mounted/unmounted windows. This finite fixture check is not a global proof.

Of the 57 maintained regression checks, 46 fail against the preceding source
and all 57 pass after correction. They cover nonzero residuals, nonidentity
mounting, nonunit scales, nonuniform sample times, zero and small increments,
the actual sparse assembly, weighted objective gradients and known constant-bias
recovery under a fixed five-iteration budget. At exact knot boundaries the
clock residual can have different one-sided derivatives; central differences
there are explicitly excluded as a smooth clock oracle. Other state blocks
at those windows remain tested.

## Frozen source and complete outcomes

The numerical source is commit
[`8a6b69d84c13003cceddc3c4921985424068f442`](https://github.com/rsasaki0109/Calibrex/commit/8a6b69d84c13003cceddc3c4921985424068f442).
All 401 Python source modules were hashed with normalized LF text before and
after the runs and remained unchanged. The [receipt](receipt.json),
[receipt schema](receipt.schema.json), [source manifest](source-manifest.json)
and [original launcher](generator.py.txt) bind the commands, source and outcomes.
Runtime: Python 3.12.10, NumPy 2.3.4, SciPy 1.17.1, Pydantic 2.12.5.
OpenBLAS/MKL/OMP limits were one; concurrent host activity was allowed.

The previous results are the existing
[accelerometer derivative records](../screw_accelerometer_jacobians/README.md),
source `11cd1ca4e71779505ba071711eb2486b8f4f90d8`, rather than a new control run.
The [input comparison](input-comparison.json),
[schema](input-comparison.schema.json) and [producer](verify-inputs.py.txt)
retain exact canonical JSON equality of each generated problem/options pair
with its archived development input. That audit reads the local archived
inputs named in the producer; those private copies are not required to replay
the public CLI below.

| Seed | Previous fit / iterations | Corrected fit / iterations | Previous policy | Corrected policy | Complete result |
| --- | --- | --- | --- | --- | --- |
| 20261010 | converged / 29 | converged / 21 | pass | pass | [JSON](seed-20261010.json) |
| 20261011 | converged / 22 | converged / 23 | pass | pass | [JSON](seed-20261011.json) |
| 20261012 | converged / 25 | converged / 29 | pass | pass | [JSON](seed-20261012.json) |

All three cases retain calibration errors, covariance-derived uncertainty,
training/holdout residuals, all 32 signed known-bad controls, separability and
seed/command/source provenance. Every control is detected in each case.

| Seed | Gyro holdout RMSE, previous → corrected (rad) | Maximum translation error, previous → corrected (m) | Maximum rotation error, previous → corrected (deg) |
| --- | --- | --- | --- |
| 20261010 | 0.000488302805 → 0.000488302268 | 0.00585998359 → 0.00585998271 | 0.163571927 → 0.163571973 |
| 20261011 | 0.000341162642 → 0.000341162463 | 0.00699456566 → 0.00699456513 | 0.0972514785 → 0.0972514742 |
| 20261012 | 0.000423083991 → 0.000423084156 | 0.00634399002 → 0.00634399006 | 0.126097571 → 0.126097612 |

Policy remains three passes; accuracy changes are small and mixed, and two
iteration counts increase. The holdout oracle evaluates truth calibration on
the fitted trajectory and therefore moves when that trajectory changes.
The native `converged` label can result from a small accepted damped step;
it does not establish a small undamped gradient. Clock behavior at boundaries,
correlated-prior whitening and convergence remain explicit audit work before
the fixed-budget comparison. Earlier records remain unchanged.

## Reproduce

At the pinned numerical source and thread limits above, run every command and
retain every outcome. Floating-point/platform differences can affect convergence;
provenance timestamps and output paths differ on replay.

```bash
calibrex trajectory recover-lidar-imu-joint --trajectory screw_linear --seed 20261010 --output replay-20261010.json --json
calibrex trajectory recover-lidar-imu-joint --trajectory screw_linear --seed 20261011 --output replay-20261011.json --json
calibrex trajectory recover-lidar-imu-joint --trajectory screw_linear --seed 20261012 --output replay-20261012.json --json
```

The archived producer scripts preserve their original bytes and working paths.
To replay a producer, copy it into its named `outputs/gyro-jacobian-development-v1/`
location at the pinned source, using a fresh output directory and retaining the
declared baseline commit for the value comparison. The production launcher
refuses to overwrite existing public result files. The source manifest uses LF
normalization; the fixture and producer SHA-256 values bind their original raw
checkout bytes.

## Integrity

| Artifact | SHA-256 |
| --- | --- |
| seed-20261010.json | `2457f2389af7597c3b6052189c76a5c9befa6052e63316f5ba8b57f4252138dc` |
| seed-20261011.json | `91dc8458687ec5af2cc7111e759b4e95943e4eb424a2a8f591c3974ebf86ea6b` |
| seed-20261012.json | `eac4cc41fdd682b204e8eda1cbe5732804efc003c98af26a00cf77899c88bda5` |
| source-manifest.json | `226f84e4496e8361362d02ef60f7429047a827c9adec157b73861e417af5c573` |
| generator.py.txt | `8b754767942aeb56a2574c11ba9d58cb152813056b2d11ee67b41d11d69392a4` |
| value-parity.json | `383ede2d1a9813c998951f70552e4cbba0d5789a64a037fe28a2c80a14b9625c` |
| verify-values.py.txt | `f218dae890ff3a15fa9756c16605c413d815becf012c070638927cca50ef3172` |
| fixture.py.txt | `026232879ce82d6bde6f0105c9842703c07da2d09c3cfc5d786b8850af4ae2b7` |
| input-comparison.json | `cece67044e73f5292fb63352cf418531ffe92c8925719599eb966b53e159276b` |
| verify-inputs.py.txt | `800dfdca389f1595c3f86ee56657658def2914ee29e14ddbaa4ab609780aac65` |

Maintained tests verify the full declared cohort, native and audit schemas,
byte hashes, source/seed provenance, signed controls, exact fixture values,
input digests and original producer bindings.
