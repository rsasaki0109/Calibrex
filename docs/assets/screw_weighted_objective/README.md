# Native screw measurement-loss development records

These are complete native `slac.continuous_time_lidar_imu_joint/v0.1` artifacts
for three already-read synthetic seeds. They support the
[measurement-loss correctness work](../../research/continuous_time_reliability.md#1-consistent-measurement-loss).
All three cases were fixed in advance and retained. Thresholds, iteration
limits, inputs and solver options remain those of the existing generator.

The corrected numerical source is commit
[`634751653d1be1411b801f26ca50ce83a14d8b75`](https://github.com/rsasaki0109/Calibrex/commit/634751653d1be1411b801f26ca50ce83a14d8b75).
All 401 Python source modules were hashed with normalized LF text before and
after the runs and remained identical. Runtime: Python 3.12.10, NumPy 2.3.4,
SciPy 1.17.1, Pydantic 2.12.5; OpenBLAS/MKL/OMP thread limits were one.
Other host activity was allowed; this record makes no timing claim.

## Every outcome

The prior column reproduces the archived CPU baseline in the
[previous measurement](../screw_residual_evaluation/README.md), source
`5a80e7470487ceab148f7d49b560f48c1158665e`. That measurement also established exact
quality equality with the pre-correction accelerated fitter. It is an existing
record, not a newly executed control.

| Seed | Prior fit / iterations | Corrected fit / iterations | Prior policy | Corrected policy | Complete corrected artifact |
| --- | --- | --- | --- | --- | --- |
| 20261010 | max_iterations / 25 | converged / 28 | pass | pass | [JSON](seed-20261010.json) |
| 20261011 | converged / 25 | max_iterations / 31 | fail | fail | [JSON](seed-20261011.json) |
| 20261012 | converged / 23 | converged / 23 | pass | pass | [JSON](seed-20261012.json) |

Seed 20261011 retains its gyro-holdout failure: 0.000357 rad exceeds 1.5 times
the truth-calibration 0.000236 rad. The returned `max_iterations` status also
covers an early stop when damping exceeds the solver's ceiling. A stalled
trial is not relabelled as convergence. All three files validate against the
existing artifact schema and retain parameter budgets, covariance-derived
standard deviations, holdout results, all 32 signed controls, separability,
and generator/seed/command/source provenance.

The quality-policy counts remain two passes and one failure. These spent
synthetic cases establish development impact and limits; independent public
accuracy and a general convergence improvement remain unestablished.

## Reproduce

Using the pinned source, run all three commands separately. Exit status 2 for
seed 20261011 is the expected policy failure, and its artifact is retained.

```bash
calibrex trajectory recover-lidar-imu-joint --trajectory screw_linear --seed 20261010 --output replay-20261010.json --json
calibrex trajectory recover-lidar-imu-joint --trajectory screw_linear --seed 20261011 --output replay-20261011.json --json
calibrex trajectory recover-lidar-imu-joint --trajectory screw_linear --seed 20261012 --output replay-20261012.json --json
```

Timestamps and command-output paths in provenance vary on replay. Compare the
complete calibration evidence after accounting for those provenance fields.
Platform arithmetic can also affect convergence. The archived values above
remain the original observations.

## File integrity

| Artifact | SHA-256 |
| --- | --- |
| seed-20261010.json | `a3046d0bdb7aa44666dec1f3f20e76eb7bd02d50ea72a1cd196cb85b78e1c62b` |
| seed-20261011.json | `84672c64cfeac1f509f0cd8d001349e067875f4e8210f138d5e923aeff40491c` |
| seed-20261012.json | `f539bf5a6dc67546c82be356bd813e77ece94b2a2468a3c105f3a6187426fd60` |

Maintained tests verify these bytes, their source/seed provenance, the complete
declared cohort, and native schema validity.
