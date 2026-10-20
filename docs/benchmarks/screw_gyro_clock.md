# Exact screw gyro clock derivatives

The native `screw_linear` gyro factor now differentiates time using its
piecewise screw rotation directly. Within an interval,
`R(t) = R_left Exp(alpha(t) w)`, so body angular velocity is `w / dt`.
Previously, a 1 ms forward rotation difference averaged adjacent intervals
near a knot and could not evaluate a domain shorter than that stencil.

This is a derivative-correctness change. It preserves gyro integration,
residual predictions, all nonclock Jacobian blocks, measurement weights,
stopping rules, public signatures and serialized schemas. It makes no new
speed, independent accuracy or general convergence claim.

## Boundary convention and tests

Increasing clock offset moves both factor endpoints left in body time.
At an internal knot, the clock column therefore uses the left time limit.
If a factor starts exactly at the trajectory's first endpoint, only the
opposite clock direction can stay in-domain; both endpoints then use their
right time limits. Global endpoints use their single inward interval.
Representable times immediately beside a knot retain their actual interval;
there is no tolerance-based knot snapping.

The 79 new checks cover closed-form piecewise z rotations, nonuniform knots,
short domains, exact knots and their representable neighbours, nonzero clock,
mounting rotation, nonunit gyro scales, one-sided residual differences and
the fitter's actual weighted clock-gradient column. Of these checks, 47 fail
against the previous frozen source and all 79 pass after correction. A
405-test targeted run and 402-module Linux/Win32 type checks also pass.

Across nine saved initial/baseline-final/candidate-final states, all 63 gyro
factors retain bit equality of residuals and every nonclock block. All 63 clock
blocks change. This diagnostic performs no optimizer steps; the full current
development cohort below does perform calibration.

## Complete development outcomes

| Item | Value |
| --- | --- |
| Baseline content | `c8001467f88eee880c6f21169ba858c572bdb068` |
| Numerical candidate | `3c2998cf84cbd1916a313ead72269c4e62cc5695` |
| Inputs | Already-spent synthetic seeds 20261010, 20261011, 20261012 |
| Current execution | One complete maintained CLI cohort, one BLAS thread |
| Comparison | Complete problem/options equality; previous stored outcomes reused |

The baseline is the stored candidate cohort from the
[batch execution comparison](screw_batched_evaluation.md), originally measured
at `df4144a4d135dea3b0863847f26bc778b56fc719`. Its complete 402-module source
capsule matches `c8001467` exactly. It was not rerun or treated as a new timing
control. The current cohort uses unchanged options and quality gates; every
outcome is retained.

| Seed | Previous status / iterations | Current status / iterations | Previous → current raw gradient norm | Policy / detected controls |
| --- | --- | --- | --- | --- |
| 20261010 | converged / 21 | converged / 20 | 0.003569 → 0.029755 | pass / 32 of 32 |
| 20261011 | converged / 23 | **max_iterations / 32** | 0.043092 → 0.042580 | pass / 32 of 32 |
| 20261012 | converged / 29 | converged / 19 | 0.074092 → 0.039789 | pass / 32 of 32 |

The middle case stops after 32 iterations with `max_iterations`, before the
configured budget of 60. In this solver, that status also covers early damping
exhaustion: unsuccessful steps increase damping until it exceeds `1e12`.
The stopping path is inferred from the frozen control flow and returned status;
no per-iteration damping trace was captured. Quality-policy passes refer to the
declared error, holdout and control checks; they do not assert optimizer
stationarity. All final raw gradients exceed the configured `1e-8` threshold,
and the existing small accepted-step criterion can still label a fit converged.

| Seed | Gyro holdout RMSE, previous → current (rad) | Maximum translation error, previous → current (m) | Maximum rotation error, previous → current (deg) |
| --- | --- | --- | --- |
| 20261010 | 0.000488302268 → 0.000488302519 | 0.005859982705 → 0.005859983082 | 0.163571973239 → 0.163571969257 |
| 20261011 | 0.000341162463 → 0.000341163323 | 0.006994565126 → 0.006994564494 | 0.097251474177 → 0.097251478319 |
| 20261012 | 0.000423084156 → 0.000423084300 | 0.006343990061 → 0.006343990130 | 0.126097611874 → 0.126097637626 |

Gyro holdout RMSE increases slightly in all three cases; other accuracy changes
and iteration counts are mixed. Full covariance, all metrics, final normal
models and signed controls remain in the
[complete development evidence](../assets/screw_gyro_clock/README.md).

## Status correction and CI

The original archive and `summary.json` are retained byte-for-byte. Its
`candidate_iteration_limit_cases` key incorrectly described seed 20261011 as
exhausting its iteration budget. The [schema-validated erratum](../assets/screw_gyro_clock/status-erratum.json)
supersedes that interpretation and binds the saved options, fit, source and
initial CI failures. No numerical result, stopping threshold or quality gate
was changed for this correction.

The initial full Python 3.11 and 3.12 CI jobs each reported 5295 passes,
104 skips and one failure: a legacy test required `converged` although the
quality policy passed and the actual fit returned `max_iterations`. The joint
recovery tests now check exact propagation of the real fitter status and
iteration count, retain all error/holdout/control gates and reject singular
fits. Controlled stopping tests distinguish budget exhaustion from early
damping exhaustion and require both to remain nonconverged. Platform-dependent
termination is disclosed rather than relabelled or rerun for a preferred result.

## Reproduce and remaining work

At the numerical candidate, use one BLAS thread and retain all outcomes:

```console
calibrex trajectory recover-lidar-imu-joint --trajectory screw_linear --seed 20261010 --output replay-20261010.json --json
calibrex trajectory recover-lidar-imu-joint --trajectory screw_linear --seed 20261011 --output replay-20261011.json --json
calibrex trajectory recover-lidar-imu-joint --trajectory screw_linear --seed 20261012 --output replay-20261012.json --json
```

The archive retains both source capsules and original producers. Platform and
floating-point differences can affect accepted steps and iteration counts.
No recording download is required.

A one-sided factor derivative does not implement a constrained optimizer.
Global clock-domain bounds and feasible stepping remain separate work. A
factor spanning both domain endpoints admits no nonzero in-domain clock shift;
these checks make no clock-identifiability claim for that case. Accelerometer
clock finite differences, mixed-unit stopping and prior-packing approximations
also remain unchanged. These limits must be addressed before the
[independent fixed-budget reliability study](../research/continuous_time_reliability.md).
