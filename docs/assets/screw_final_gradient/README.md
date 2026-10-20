# Final-state trajectory gradient reporting

The native sparse `screw_linear` fitter now reports `gradient_norm` at its
returned state. This is a reporting-correctness and development record on
already-spent synthetic data, with no timing, independent accuracy or SOTA claim.

## What changed

The normal equations use scalar square-root information, `weight = 1 / sigma`.
The reported quantity is `max(abs(J.T @ (weight**2 * residual)))`, half the
squared-objective gradient in the fitter's local coordinates. Information-form
priors are already whitened and carry unit row weights. The previous result
could retain this quantity from before the last accepted update.

The fitter tracks whether the last linearization still describes the current
state. An accepted update or restoration invalidates it. The final gradient
refreshes that model when necessary; otherwise the existing model and gradient
are reused. Covariance shares that same final model. A gradient stop, rejection,
or failed solve does not itself change the state and needs no refresh.

The maintained checks cover independent four-anchor pose gradients in identity
and rotated world frames, weighted targets, accepted updates at the iteration
limit, tiny accepted steps, rejected steps, acceptance followed by rejection,
the damping limit, and exceptions/nonfinite solves, with and without calibration
parameters. Of the 24 checks, 15 fail on the preceding fitter and all 24 pass
after correction. This count includes model-reuse checks as well as reporting.

Stopping criteria, iteration limits, accepted updates, the residual model,
covariance and serialized schemas are unchanged. A final accepted update in a
fit without calibration parameters can require one final linearization that
the old, stale report omitted. An already-current model is reused.

## Frozen paired evidence

The baseline is commit
[`1e6da103bf1d268d6469cb60dce269099dae53ea`](https://github.com/rsasaki0109/Calibrex/commit/1e6da103bf1d268d6469cb60dce269099dae53ea).
The corrected numerical source is
[`aa4a0c24069020624e4d6bb8c519c2d9d9838fcd`](https://github.com/rsasaki0109/Calibrex/commit/aa4a0c24069020624e4d6bb8c519c2d9d9838fcd).
All 401 source modules were hashed with normalized LF text before and after
each cohort. Only `src/calibrex/core/continuous_time_sparse.py` differs.

The [receipt](receipt.json), [schema](receipt.schema.json),
[source manifest](source-manifest.json), [original cohort producer](generator.py.txt)
and [archive producer](package.py.txt) bind all inputs, outputs and source.
The [complete archive](complete-development.tar.gz) retains all 171 declared
members, including both three-seed cohorts, full input/options and fit records,
native artifacts, final normal matrices and gradients, schemas, CLI logs,
the preceding source, and exact-test red/green receipts and logs.

The producer invokes `calibrex.cli.main.main` with the recorded CLI arguments in
its process. Pass-through wrappers call the original maintained fitter and
assembler, recording their arguments and results while returning their outputs
unchanged. Additional dense products record the final normal model, so this
instrumented cohort is not a latency measurement. No residual, Jacobian,
sampling, stopping, or state override is applied.

Runtime was Python 3.12.10, NumPy 2.3.4, SciPy 1.17.1 and Pydantic 2.12.5.
OpenBLAS/MKL/OMP limits were one; concurrent host activity was allowed.

| Seed | Previous reported gradient | Corrected final gradient | Iterations | Assembly count, previous/current | Current native artifact |
| --- | --- | --- | --- | --- | --- |
| 20261010 | 0.0650938792306 | 0.00356875260312 | 21 | 22 / 22 | [JSON](seed-20261010.json) |
| 20261011 | 0.0426090979809 | 0.0430917463485 | 23 | 24 / 24 | [JSON](seed-20261011.json) |
| 20261012 | 0.0719102156829 | 0.0740917147636 | 29 | 30 / 30 | [JSON](seed-20261012.json) |

Each pair has exactly equal canonical problem/options JSON. All fitted result
fields except `gradient_norm` match exactly, including knot poses, calibration,
covariance, objective, status and iteration count. The final normal matrices
and gradients match exactly. Native JSON excluding provenance also matches
exactly, including all holdout metrics and signed controls. All three policies
pass, with all 32 controls detected in each case. Every corrected report equals
the recorded final normal-gradient norm. The SHA-256 inventory binds every
archive member and all three current native artifacts.

## Limits and previous attempts

Every final raw gradient above exceeds the configured `1e-8` gradient threshold.
The existing `converged` label can also result from a small accepted damped
step. Correct reporting discloses this limitation; it does not certify
stationarity. These gradients mix physical parameter units, so normalized
stopping needs a separately declared contract. Clock behavior at boundaries
and discarded marginalization gradients/cross-knot information remain separate
work in the [reliability study](../../research/continuous_time_reliability.md).

The archive also retains six named historical diagnostic directories and their
index. The first aggregation harness referred to an incorrect SE3 field name;
its six raw fits and all six aggregation errors remain intact. The corrected
aggregation reuses exactly those same raw fit bytes without rerunning them.
The private stopping draft has three expected failing checks and two positive
controls. A reporting-only prototype fixes two failures while deliberately
leaving premature stopping red. Both later four-case model-reuse prototypes
and the three saved-window model diagnostics are retained. These private
stopping drafts are separate from the 24 maintained reporting/reuse checks.

Assembly counts do not establish a wall-clock speedup. The controlled prototype
shows that an already-current calibration model can avoid a repeated assembly;
the native three-seed assembly counts are unchanged. Earlier speed records and
SOTA verdicts remain bound to their original source and results.

## Replay

At the pinned corrected source, the direct CLI recipes are:

```bash
calibrex trajectory recover-lidar-imu-joint --trajectory screw_linear --seed 20261010 --output replay-20261010.json --json
calibrex trajectory recover-lidar-imu-joint --trajectory screw_linear --seed 20261011 --output replay-20261011.json --json
calibrex trajectory recover-lidar-imu-joint --trajectory screw_linear --seed 20261012 --output replay-20261012.json --json
```

For instrumented replay, copy the original cohort producer to its named
`outputs/final-gradient-development-v1/run.py` path in a checkout with fresh
output directories. At the baseline commit run it with `--stage baseline
--source 1e6da103bf1d268d6469cb60dce269099dae53ea`; at the numerical commit use
`--stage current --source aa4a0c24069020624e4d6bb8c519c2d9d9838fcd`. Keep the
thread limits above and retain every outcome. The producer refuses to overwrite
its stage directories. Timestamps, paths and platform floating-point behavior
can differ on replay. Historical producer paths and source bindings are retained
as originally recorded; none is a current stopping-policy implementation.
