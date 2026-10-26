# Calibration reliability under a compute budget

Calibrex develops multi-sensor calibration methods, demonstrates their benefits
and limits with reproducible experiments, and makes them available as usable
OSS. Research evidence and practical delivery are both project objectives.

## Research question

**How much calibration reliability can we obtain within a declared compute
budget?** Faster kernels and parallel execution make multiple initializations,
stronger failure checks, and uncertainty estimation affordable. Their value is
measured through the quality of the resulting calibration decision as well as
processing time.

The first study uses continuous-time LiDAR–IMU calibration. This plan orders
development work; an independent accuracy claim requires a separate frozen
protocol under the [SOTA audit rules](../sota_handoff.md).

## P0 implementation update (2026-10-26)

[Stationarity and independent uncertainty coverage](p0_reliability.md) implements
normalized final-state stopping diagnostics and explicit `stalled` outcomes for
both native fitters. It retains 20 known-plane and five full-pipeline synthetic
bag draws, including undercoverage and failed quality gates. The historical
observations below describe the recorded source at each milestone; the P0 update
supersedes their descriptions of current convergence labels.

[Screw-linear clock-domain reliability](screw_clock_domain.md) adds a shared
feasible interval, rejection before objective evaluation, consistent one-sided
derivatives, and an opposite-side final gradient check at exact gyro knot ties.
B-spline clock-domain, general nonsmooth stationarity, and marginalization audits
remain open.

## Empirical interval follow-up

[Independent-recording error intervals](empirical_intervals.md) adds frozen
finite-sample radii and separate evaluation, with explicit abstention and leakage
checks. A 20-recording calibration cohort and ten evaluation recordings improve
several relative-axis coverage counts, but three axes retain undercoverage. It
also corrects the P0 synthetic-bag gravity comparison: the fitted and generator
world frames were unaligned, so those component-wise rows are not admissible.

## 1. Consistent measurement loss

The native `screw_linear` fitter uses scalar square-root information:
`weight = 1 / sigma`. Its normal equations whiten both residuals and Jacobians,
and its covariance uses the same convention. The acceptance objective must
therefore minimize `sum(weight**2 * ||residual||**2)`. This applies to points,
point-to-plane distances, pose anchors, gyro pre-integration, and accelerometer
lever-arm factors. A pose measurement whitens each of its four anchor vectors.
Information-form knot priors contribute `delta.T @ information @ delta`.

Previously the acceptance objective used only one power of each measurement
weight. Different noise levels could make it reject the direction computed by
its own normal equations. For two translation targets at 0 and 2 with weights
1 and 3, inverse-variance least squares has optimum 1.8. Starting at 1.5, the
previous acceptance objective could stall there. The regression tests solve
this problem through point, plane, and pose factors with full pose constraints.
Other checks use directly calculable residual norms for all five families,
weights below and above one, a diagonal information prior, and a central
finite-difference directional derivative.

The correction changes `final_objective` and potentially accepted steps when
measurement weights differ from one. Public types and serialized schemas stay
the same. Convergence thresholds, iteration limits, holdouts, and quality gates
stay fixed. Existing speed measurements remain bound to their recorded source;
new numerical behavior needs its own evidence.

The [three retained development runs](../assets/screw_weighted_objective/README.md)
record the corrected source on already-read synthetic seeds, including the
remaining gyro-holdout failure. The quality-policy outcome remains two passes
and one failure. Convergence changes in both directions, so these runs support
the narrow measurement-loss correction rather than a general convergence or
accuracy improvement.

## 2. Complete state derivatives

Before claiming a reliable joint optimizer, check every residual family against
independent finite differences for every state block it depends on. Include
nonidentity poses, nonunit scales, mounting rotation, nonzero clock offset,
segment boundaries, and correlated information priors.

The accelerometer linearization now includes analytic trajectory-knot blocks
through its existing discrete body kinematics. Predictions and derivatives
share the same clamped three-sample stencil. Each sample contributes both of
its supporting knots; a stencil can touch three knots across a segment boundary
or as many as six on a dense nonuniform grid. Lever-arm derivatives also include
the diagonal accelerometer scale after the mounting rotation.

Independent checks perturb every knot, including inactive knots, at ten fixed
timestamps covering both ends and both sides of a knot boundary. Other tests
cover nonunit scales with and without mounting rotation, a dense nonuniform
grid, the actual fitter's sparse assembly, weighted objective gradients, and
a trajectory update driven by acceleration with all calibration states fixed.
This establishes the knot and lever derivatives of the current residual model.

The [three retained development outcomes](../assets/screw_accelerometer_jacobians/README.md)
use the same already-read synthetic seeds and options as the measurement-loss
records. Their policy counts change from two passes and one failure to three
passes, with all signed controls retained. Some rotation maxima increase;
these records establish development impact rather than independent accuracy
or a general convergence improvement.

Gyro factors now distinguish the left and right SO(3) log differentials of
the rotation-error product. Pre-integration chains the right Exp differential
of each increment through subsequent rotations, including diagonal gyro scale
in bias derivatives. Tests use nonzero residuals, nonidentity mounting rotation,
nonunit scales, nonuniform sample times, zero and small increments, the actual
sparse assembly, weighted objective gradients and known constant-bias recovery.
The historical `so3_left_jacobian` helper implements the right Exp differential;
its formula and existing SE(3) retraction are preserved.

The [complete gyro development records](../assets/screw_gyro_jacobians/README.md)
retain all three spent seeds, exact problem/options equality and old/current
residual/rotation equality on ten fixtures. All three policy outcomes remain
passes, with every signed control detected. Iterations change from 29/22/25
to 21/23/29; small accuracy changes are mixed. These are derivative-correctness
checks and development evidence, without a general speed or accuracy claim.

The accelerometer clock derivative still uses a forward time difference.
The [gyro clock derivative](../benchmarks/screw_gyro_clock.md) now uses exact
piecewise screw angular velocity instead of a sampled rate whose 1 ms stencil
could cross a knot. Its declared clock-side convention is checked against
closed-form rotations and independent one-sided residual differences. All
63 saved-state gyro factors preserve residual and nonclock-block bits.
The three current development policies pass, but one fit stops after 32 of its
configured 60 iterations with `max_iterations` (early damping exhaustion,
inferred from the frozen control flow). The original limit description is
superseded by a [status erratum](../assets/screw_gyro_clock/status-erratum.json).
Accuracy changes are mixed; this is a derivative fix,
without a general convergence or speed claim. At segment boundaries the model
can have different one-sided derivatives, so central differences are not a
smooth clock oracle. Global clock-domain constraints and feasible stepping
remain separate audit work.
The native `converged` label can come from a small damped accepted step rather
than the gradient threshold. Audit the undamped model before interpreting that
label as stationarity. Stalled line searches must retain their failure status
when the undamped model still has a significant gradient or step.

Information-prior whitening now uses rows of the square root, so its residual
norm and normal equations match `delta.T @ information @ delta`. Diagonal
equilibration makes validation independent of global scale and coordinate
units. Real finite 6x6 matrices must have a nonnegative diagonal, exact zero
rows/columns at zero diagonal entries, and a symmetric positive-semidefinite
normalized matrix. Symmetry and spectral checks allow 60 machine epsilons times
the respective normalized norm. Only tiny negative spectral roundoff is
projected; positive weak information is retained without a floor. A negative
diagonal is rejected rather than implicitly repaired.

The [complete prior-information development records](../assets/prior_information_whitening/README.md)
retain both historical attempts and the current production result. On the
same-input window fixture, the prior-bearing fit changes from 26 to 7 iterations
with smaller overlap errors; the overall maximum translation error is unchanged.
A separate native holdout/control fixture still passes. All current API inputs,
per-window results, schemas and original source bindings are retained. These
are correctness/development records, without a timing or independent accuracy
claim.

This correction preserves the existing zero-centered prior. The current
window-packing helpers still discard a nonzero marginalized gradient and
cross-knot information blocks; those approximations need a separate audit.
Aggregate sliding-window result fields also copy some batch statistics, so
development comparisons retain actual per-window fitter results and objectives.

The sparse fitter now reports its weighted normal-gradient norm at the returned
state. Previously, the last accepted update could leave that field describing
the preceding state. Its final gradient and covariance now share a current
linearization. When the last model still describes the returned state, it is
reused; after an accepted update or restoration it is refreshed. Stopping rules,
iteration limits, accepted states and serialized schemas remain unchanged.

The [complete final-gradient comparison](../assets/screw_final_gradient/README.md)
retains three paired native CLI cases, their full problems/options, fitted
states, covariance, final normal models and controls, together with the earlier
failed harness and private prototypes. Every fit payload field except the
reported gradient, and every native metric, matches exactly. All three policies
pass with all 32 signed controls detected. Assembly counts remain 22/24/30;
these records establish reporting correctness rather than a runtime speedup.

The final raw gradients in those cases exceed the configured gradient threshold
despite the existing `converged` label: a small accepted damped step is still a
separate stop reason. Reporting this quantity correctly makes that limit
visible; normalized stopping in mixed physical units and nonsmooth clock
behavior require separate work before the fixed-budget reliability study.

## 3. Fixed-budget comparison

The [shared IMU execution comparison](../benchmarks/screw_imu_evaluation.md)
removes repeated integration and discarded derivatives while preserving all
three complete development outcomes exactly. Its complete-API paired median
is 1.452x and misses the declared 2x target. This incremental CPU cleanup does
not address normalized stopping, clock sensitivity or independent reliability.
The subsequent [batched trajectory and plane evaluation](../benchmarks/screw_batched_evaluation.md)
measures complete-API paired medians of 2.023x for the same spent synthetic
cases and 4.709x for spent public GLIM geometry against the preceding maintained
implementation. Complete input/results remain equal; fresh-worker ratios are
1.462x and 2.189x. One joint pair is 1.641x, so the near-threshold joint median
does not establish a universal 2x improvement. This execution work creates
compute headroom while the numerical reliability questions remain open.

After the derivative audit, compare the maintained single-start fitter with
bounded multiple starts and empirical uncertainty using identical inputs and
total compute budgets. Freeze the cases, budgets, seeds, solver settings,
selection rule, and quality gates before measurement. Compare against the
strongest maintained CPU and GPU paths available for the same task.

Evaluate multiple starts and uncertainty estimation separately and together.
Selection, stopping, and uncertainty estimation use training data; the method
cannot inspect holdout scores when selecting a result. Execution-only speed
comparisons require the same objective and numerical quality contract.

Report all attempts, including failures:

- Calibration error against known synthetic truth, or explicitly labelled
  public dataset references; holdout performance and signed known-bad controls.
- Successful decisions and appropriate `inconclusive` outcomes under weak
  excitation; failure detection and false acceptance rates.
- Empirical uncertainty coverage over independent synthetic draws and transfer
  between recordings, with the assumptions stated.
- Complete processing latency, including initialization, preparation, device
  synchronization, and result output; peak RAM/VRAM and incremental disk use.

Keep cold and reused-worker scopes explicit. Pair comparisons on identical
inputs and retain complete outcomes and source/input/options provenance.
Choose thresholds during development and freeze them before evaluation.

## 4. Independent public evaluation

Reuse already-read public captures for development and performance work. Audit
recording exposure before selecting independent evaluation data. Pre-register
the claim, comparison, scoring, thresholds, and rerun policy, then run once
from frozen code and publish every outcome. Existing SOTA verdicts remain
bound to their original runs.

An initial study is complete when its hypothesis, maintained comparisons,
quality-versus-budget results and limitations are reproducible, and the
supported behavior is available through a documented OSS workflow. Large
downloads are unnecessary for the first correctness milestone.
