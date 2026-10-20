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

The accelerometer residual depends on trajectory knots through body kinematics,
while its current linearization inserts calibration blocks only. Its knot
derivatives and the whitening of correlated priors remain explicit follow-up
work. Measurement-weight tests alone establish a narrower property. Stalled
line searches must retain their failure status when the undamped model still
has a significant gradient or step.

## 3. Fixed-budget comparison

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
