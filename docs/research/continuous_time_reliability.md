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
The B-spline fitter now also intersects gyro and accelerometer time bounds,
rejects infeasible coupled updates before constructing or evaluating candidates,
and retries with higher damping. Central clock differences remain in the
interior; near a boundary all IMU factors use the same feasible inward step,
divided by its actual representable increment. Domain diagnostics use the
existing optional `fit_diagnostics.clock_domain` fields. For central differences,
`derivative_step_sec` denotes the positive probe increment, not the full stencil
width. Zero-width domains require fixing the clock rather than estimating it.

B-spline regression tests cover both boundaries with and without the evaluation
cache, independent inward derivative comparisons, rejection before candidate
construction, successful retries, invalid initial clocks, and fixed-clock fits.
Existing synthetic joint-recovery and derivative tests remain the accuracy
checks; this change makes no independent accuracy or speed claim. The guard
uses strict trajectory bounds even though the spline evaluation API tolerates
tiny timestamp overshoots. It does not change that API or implement a projected
optimizer. Boundary optima may therefore still be reported as stalled.
At exact internal B-spline IMU knot ties, the final stopping audit now uses a
feasible one-sided clock column and checks the opposite side when feasible.
Both full normalized gradients must meet the existing tolerance. The primary
gradient is recorded in `normalized_gradient_norm` and the other in
`clock_domain.opposite_gradient_norm`; the signed primary probe is recorded in
`derivative_step_sec`. A central-difference convergence candidate that fails
this audit becomes `stalled` / `stationarity_not_confirmed`.

A deterministic regression constructs residuals orthogonal to the complete
central Jacobian while retaining a nonzero one-sided clock gradient, with all
trajectory control points free. An independent objective evaluation confirms a
small positive clock shift reduces cost. The audit rejects convergence in this
case and preserves convergence for zero-residual data. Optimization proposals
and covariance retain their existing central-difference calculation in the
interior; this audit does not establish covariance validity at a nonsmooth knot.

This is a conservative check: both side gradients must vanish, so even a valid
nonsmooth minimum can be reported as stalled. Only exact floating-point knot
ties trigger it; nearby stencils crossing a knot are not covered by this audit.
General nonsmooth stationarity remains open.

## Marginalization gradient audit

The Schur routine computed both retained curvature and gradient, but conversion
to knot priors discarded the gradient. Priors now store the anchor-tangent
gradient (default zero for existing callers). Given information `H = W.T W`,
the residual is `W Log(anchor, pose) + b`, with `W.T b = g`. Its squared norm
retains `delta.T H delta + 2 g.T delta` up to an additive constant. Candidate
scoring and both normal-equation builders use the same residual. This matters
when the previous window stopped before reaching stationarity.

The offset solve scales by the information diagonal so weak coordinates are
not silently removed because another coordinate has large units. Invalid or
nonfinite gradients, including components outside the information range, are
rejected. Schur elimination uses a linear solve rather than an explicit inverse;
the exact-zero and range-checked coupled-nullspace extensions described below
add no artificial damping.

Regression tests compare the retained prior against independently profiled
quadratic costs and the retained part of the full dense optimum, verify a fit
accepts the gradient-driven update, and cover weak information and nullspace
gradients. Existing sliding-window holdout and known-bad controls remain checked.
These are development tests, not an independent accuracy claim.

The follow-up joint prior stores all retained anchor poses, knot indices, the
complete Schur information matrix, and its gradient. Its residual whitens the
concatenated anchor-tangent displacements together, so columns for different
knots retain their cross terms. Candidate scoring and both normal-equation
builders use the same joint factor. Sliding-window transfer now uses this
representation for every overlap size. The older single-knot API remains
available, as does the explicitly approximate `split_retained_information`
helper, which is no longer used by the sliding-window fitter.

Tests compare a two-knot retained prior against the independently profiled full
quadratic, verify the full-system optimum, and check both dense assembly and
the nonlinear fit with deliberately reversed knot indices. Sliding-window
checks cover one- and two-knot overlaps. Invalid index lists, shape mismatches,
invalid information, and gradients outside the information range are rejected.
These results establish preservation of the local quadratic at its anchors,
not global nonlinear equivalence.

The next follow-up removes overlapping point-to-plane factor reuse. Each
outgoing Schur prior contains every raw row admitted through the window's final
timestamp, including observations exactly at that endpoint. If the next window
uses that prior, it admits only rows strictly later than the summarized endpoint.
This assigns observations by timestamp rather than measurement ID, so distinct
rows sharing an ID remain distinct. The current implementation summarizes the
entire fitted window; it does not retain old raw factors for relinearization.

When the prior is deliberately omitted, raw overlap observations remain available.
Invalid nonadvancing windows fail before any fit.
Trajectory problems with only prior factors are also accepted; this does not
guarantee that new, unobserved knots are observable.

Tests track all input occurrences over three windows, cover exact endpoints and
their next representable timestamps, and verify no-prior controls. At fixed
linearization anchors, the final sequential normal equations agree with Schur
elimination from all original observations at the same anchors. This local
quadratic identity does not establish nonlinear batch equivalence.

## Pose constraints and the gauge audit

The sliding fitter previously inserted a weight-100 pose observation at an
initial knot, despite fitting observations against planes in a fixed world
frame, and ignored caller-supplied pose observations. This was a physical pose
penalty, not merely a coordinate convention: it could bias observable motion
toward the initial guess. It also made batch and window objectives differ.

The fitter now adds no synthetic pose constraint. Explicit pose observations
are routed into each window by timestamp, using the same once-only accounting
as plane observations when an incoming Schur prior is present. Without that
prior, raw pose observations in the overlap remain available. A three-window
pose-only regression verifies that identical measurement IDs do not cause rows
to be dropped and that the result reaches the supplied pose, away from the
initial guess. Truly unobservable directions are not repaired by inventing a
pose observation; coupled singular elimination requires the range checks below.

[Retained paired evidence](../assets/sliding_gauge_audit/manifest.json) uses the
already-read seed 20260820 and unchanged thresholds. Baseline and current sliding
holdout RMSE are both 0.0215409 m. The no-marginalization control's excess RMSE
changes from 0.0223418 m to approximately zero, so its gate now **fails**. The
previous result did not isolate marginalization benefit from the inserted pose
penalties. The generator is version 0.2 and the CLI retains this failure with
exit code 2. Tests assert the failed gate; no thresholds or seeds were retuned.

The evidence includes the exact runner, source hashes, and changed-source
snapshots. Reproduce the baseline from the manifest's revision and the current
run by overlaying the snapshots, then run the saved Python runner with a fresh
output directory and `OPENBLAS_NUM_THREADS=1`. These are development findings,
not an independent accuracy comparison.

## Sliding-result diagnostics

The sliding result previously copied its objective, plane RMSE, gradient, step
maxima and Jacobian block count from the batch reference, while reporting only
the last window's status. It now reevaluates the original plane and pose
observations at the stitched knots, without transferred priors (which already
summarize those observations). The full weighted objective, raw gradient,
normalized stationarity, plane/pose RMSE, and block count describe that returned
trajectory. Step maxima aggregate accepted steps across all window fits;
iteration counts remain summed. Every window result is retained separately.

Overall status preserves earlier failures, in precedence order: singular system,
iteration limit, then stalled. Only all-converged windows and a stationary
stitched trajectory produce `converged`. The diagnostics' stationary flag
describes the final global gradient independently of those historical failures.
This does not certify observability or globally optimal nonlinear estimates.

Recovery artifacts add optional `window_fit_statuses` and
`sliding_fit_diagnostics`; old artifacts remain readable. Generator 0.3 also
requires convergence of the sliding result before passing its quality policy.
Seed 20260820 reveals window statuses `[singular_system, converged]` and a
nonstationary stitched trajectory (normalized gradient approximately 0.142).
The earlier failure is now visible even though the last window converges.
The separate control failure recorded by generator 0.2 remains historical
evidence; generator 0.3 reports the singular sliding status first.

Tests deliberately alter the returned final knots, poison unrelated batch
metrics, and vary earlier window statuses. They verify costs against direct
pose residuals, gradients against the full normal equations, and preservation
of the earlier failure. The wrapper rejects unrouted point, IMU, or caller
prior factors before fitting rather than reporting diagnostics for silently
discarded inputs. Nonlinear relinearization and broader factor support remain
open audit items.

## Window information diagnostics

Each window now reports local information at its final returned knots, both for
its admitted observations alone and with the incoming marginal prior. Before
eigendecomposition, the normal matrix is divided by the square roots of its
positive diagonal entries on both axes. Zero-information coordinates remain
zero. Numerical rank counts eigenvalues greater than `1e-10` times the largest
scaled eigenvalue. The spectrum, relative tolerance and actual threshold are
recorded so the numerical decision can be inspected.

Exact zero diagonal columns identify coordinates with no local information;
global knot indices and local translation/rotation axis names locate them.
The diagonal of the numerical nullspace projector also records participation
in coupled unobservable directions, even if every individual diagonal is
positive. Participation is in the diagonally scaled tangent coordinates; it
is not a calibration covariance or a physical uncertainty magnitude.

Generator 0.4 diagnoses seed 20260820's first window as rank 18 of 24, with all
six coordinates of global knot 3 exactly unobserved. The second window is rank
18 of 18. This explains an information deficit behind the first singular solve,
while preserving that failure and the original measurements. No prior, damping,
or quality threshold is added to make the example pass.

Tests cover exact missing columns, coupled relative-pose null modes, rank lifted
by an absolute prior, independently rescaled coordinate units, weak but
independent information, invalid matrices, and the schema-valid seeded report.
This is a local numerical information assessment at one linearization, not a
global nonlinear observability proof. Incoming priors can restore numerical
rank without adding new measurements, which is why both information sources
are reported separately. A full-rank report does not establish accuracy or
rule out a separate numerical solver failure.

## Updating the observed coordinates

The sparse screw-linear solver now excludes exactly zero-information columns
from its damped linear solve and inserts zero increments for those coordinates.
The active columns are recomputed at every iteration; small positive information
is never discarded by a rank threshold. This fixes an entirely unobserved knot
making the whole linear solve singular. It adds neither an anchoring observation
nor a positive information floor. Fully active systems retain the original solve.

Fit results record the union of `frozen_parameter_indices` across iterations.
Sliding evidence maps those indices to global knot/axis names in
`frozen_parameters`; the aggregate result maps them to global parameter indices.
These are historical local zero-column events. A coordinate may become observed
in a later iteration or window, and its freezing is not permanent.

On reused development seed 20260820, generator 0.5 reports both windows and the
stitched trajectory stationary. The first window still has rank 18 of 24 and
leaves knot 3 unchanged; its six missing coordinates are explicitly reported.
Sliding holdout RMSE changes from approximately 0.02154 m to 0.01409 m, but the
unchanged no-marginalization control still fails. The overall evaluation remains
`fail`; this single development example is not an independent accuracy claim.

Tests verify recovery of an observed knot with an unobserved neighbor, unchanged
unobserved state, explicit reporting at zero residual, and retention of weak
positive information. This handles exact zero columns only: coupled null modes
with nonzero columns during optimization and general nonlinear observability
remain separate issues. Coupled singular elimination is addressed below. A stationary fit is not a claim that every
coordinate was measured.

An additional regression exposed numerical cancellation at screw-interpolation
endpoints: the nominally absent knot could receive tiny spurious Jacobian entries,
which the solver treated as information and amplified into a large update.
Scalar and batched endpoint Jacobians now return their exact analytic zero and
identity blocks at fractions zero and one. Interior fractions retain the
original calculation; no tolerance-based snapping or weak-information cutoff is
used. The multi-window regression verifies an unsupported knot remains unchanged
and its freeze history is mapped to the correct global indices.

## Eliminating entirely unobserved coordinates

Schur elimination now handles an eliminated prefix containing exactly
zero-information coordinates. Such a coordinate is omitted from the solve only
if its complete normal-matrix row and column and its gradient entry are all
exactly zero. Otherwise the normal equations are rejected instead of silently
discarding coupling or a linear term. The retained quadratic receives the usual
Schur contribution from the informative eliminated subblock; an entirely absent
prefix leaves retained information and gradient unchanged.

This exact-zero path introduces no positive diagonal floor, damping, or rank
cutoff; even very small positive coordinate information is retained. The coupled
active-block extension below uses a recorded numerical rank threshold. The
exact-zero rule extends the earlier update rule to removal of an
unobserved old knot, rather than only retaining an unobserved overlap knot.

Tests compare the result with independent least-squares elimination in residual
space, including changes in the profiled objective; cover an entirely absent
prefix and weak positive information; reject inconsistent coupling/gradient;
and run two sliding windows where an unobserved old knot is removed while all
observed knots recover their targets. The removed state keeps its initial value
and remains identified as unobserved in the first window's evidence. This does
not provide an estimate for that state or establish global observability.

## Coupled numerical nullspaces in Schur elimination

The informative eliminated block is diagonally scaled before symmetric
eigendecomposition. Its rank threshold is `60 * machine_epsilon *
max(1, eliminated_dimension) * max(1, max(abs(eigenvalues)))`. Directions at or
below that threshold are numerical null directions. Material negative
eigenvalues or asymmetry are rejected. Full-rank blocks use a direct solve in
these scaled coordinates.

For a deficient block, retained coupling columns and the eliminated gradient
must lie in its numerical range. Each nullspace projection is checked relative
to the maximum absolute entry of its own scaled right-hand-side column, using
tolerance `60 * machine_epsilon * max(1, eliminated_dimension)`. Inconsistent
inputs raise `ValueError`; consistent inputs use the inverse restricted to the
observed eigenspace. No damping, positive eigenvalue floor, or anchor is added.

`schur_complement_normal_with_evidence` returns the retained quadratic and its
elimination evidence; the original two-result API remains available. Sliding
evidence records eliminated dimension, rank, exact-zero count, coupled nullity,
rank threshold, range tolerance and maximum relative range error. The final
window has no outgoing elimination evidence. Generator 0.6 includes the optional
fields while keeping older artifacts readable.

Tests compare results with independent residual-space projection and profiled
costs, verify invariance to coordinate unit changes, reject incompatible coupling
and gradients, and exercise a sliding problem with an old knot pair observed
only through its midpoint. Its twelve eliminated coordinates have rank six;
information about the retained states still carries forward without inventing
information about each old knot individually.

This is a finite-precision rank decision, not exact symbolic observability.
Small positive eigen-directions below the threshold may be treated as numerical
nullspace, but a material right-hand side in those directions is rejected rather
than silently dropped. Weak independent coordinate information survives diagonal
scaling. Nonlinear relinearization, optimization of coupled null modes, and
interpretation of globally unobservable estimates remain open.

Both full-rank and rank-deficient elimination now contract the Schur correction
in scaled coordinates. With eliminated scale matrix `D`, the computation uses
`(H_re D^-1) solve(D^-1 H_ee D^-1, D^-1 [H_er, g_e])` (restricted to the
observed eigenspace when needed). It never constructs the potentially overflowing
unscaled eliminated solution. For example, `H=[[1e-320,5e-161],[5e-161,1]]` and
`g=[1,0]` previously returned negative infinities, although retained curvature
is approximately `0.75` and gradient approximately `-5e159`. Regression tests
cover this case, duplicated eliminated columns, equivalent coordinate units,
and explicit rejection when the retained result itself is nonfinite. These are
development arithmetic tests, not new accuracy or performance evidence. This
does not resolve subtractive cancellation in nearly zero retained curvature;
residual-space elimination is available separately as described below.

## Residual-space marginalization reference API

`calibrex.core.square_root_marginalization.marginalize_residual_prefix(J, r, n)`
accepts an already weighted Jacobian and residual and eliminates the first `n`
columns. It returns a retained Jacobian and residual, including constant cost,
so their squared norm represents the profiled least-squares objective. It does
not create an information matrix or subtract Schur terms. For example:

```python
import numpy as np
from calibrex.core.square_root_marginalization import marginalize_residual_prefix

factor = marginalize_residual_prefix(
    np.array([[1., 1.], [0., 1e-9]]), np.array([0., 1.]), 1,
)
# factor.jacobian.T @ factor.jacobian retains information 1e-18.
# Forming J.T @ J before Schur subtraction loses that term entirely.
```

The eliminated columns are independently normalized using a maximum-entry
scaling followed by a Euclidean norm. A thin SVD supplies the observed left
singular vectors. Raw QR stores their Householder reflectors, and LAPACK ORMQR
applies the transpose orthogonal transform to retained columns and residuals.
Rows after the observed rank form the retained factor; no square residual-space
basis or explicit projection subtraction is constructed.
The singular-value rank threshold is `60 * machine_epsilon *
max(involved_rows, nonzero_eliminated_columns) * largest_singular_value`.
Only rows with at least one exactly nonzero eliminated coefficient enter this
decision; exactly absent eliminated columns are also excluded. The result records
the total and involved row counts, this threshold, numerical rank and
the number of exactly absent eliminated columns. Below-threshold singular
directions are treated as null; this is a numerical rank approximation, not
an exact symbolic guarantee, and differs from thresholding normal eigenvalues.

Tests compare complete profiled costs against independent least-squares solves,
including coupled nullspaces, preserve weak independent columns under extreme
unit changes, and cover absent columns and elimination spanning every residual
row. No positive floor or invented pose constraint is added. Orthogonal
projection still has floating-point error when retained columns almost align
with eliminated columns; the API does not promise exact recovery at all scales.

For fixed column counts, this dense API now needs storage linear in the residual
row count. It still stores dense input and output factors and is not streaming.
Converting its result back to normal matrices can lose weak information again.
These development tests make no new trajectory accuracy or runtime claim.

## Opt-in residual-space window priors

`fit_sliding_window_trajectory(..., square_root_marginalization=True)` now uses
the residual-space API at window boundaries. The default remains normal-equation
Schur elimination. The CLI exposes the mode with:

```sh
calibrex trajectory recover-sliding-window --output result.json --square-root-marginalization
```

`SquareRootKnotMarginalizationPrior` retains a possibly rectangular weighted
Jacobian and residual in fixed anchor tangents. Both candidate cost and the next
window's linearization evaluate that factor directly, including constant cost
rows, without reconstructing it from information and gradient. The shared knot
factor assembler supplies weighted residual equations for outgoing elimination;
unsupported IMU factors are rejected rather than silently omitted. Observation
accounting, explicit pose constraints and known-bad controls remain unchanged.

Generator 0.8 records the selected mode. Each outgoing window records
`square_root_elimination` with eliminated dimension, numerical rank, exact-zero
column count and singular-value threshold; normal-path `elimination` is absent
for this mode. The final window has neither outgoing record. These optional
schema fields keep old artifacts readable.

Tests reproduce weak information `1e-18` lost by normal Schur subtraction and
verify that its factor and linear term reach the next window. They also cover
rectangular and zero-row priors, nonlinear tangent derivatives, three overlapping
windows, and CLI provenance/schema validation. The reused synthetic development
case converges but still fails the no-marginalization control gate; thresholds
and seeds have not been retuned.

Generator 0.9 compresses each outgoing factor before storing its prior.
`compress_residual_factor(J, r)` applies unpivoted QR to the augmented matrix
`[J, r]`, requesting only R. For `n` retained variables, the result has at most
`n + 1` rows. Including the residual column retains constant cost as well as
curvature and gradient; dropping orthogonal residual rows would change candidate
costs and normalized stationarity. No normal matrix or singular-value cutoff is
used for compression, and no weak positive direction is intentionally removed.
Short factors are copied unchanged. Optional evidence fields record
`retained_rows_before_compression` and `retained_rows_after_compression`.

Tests verify complete costs for full-rank and dependent factors, preservation
of weak independent information and irreducible constant cost, bounded storage
over repeated compression, and agreement with the uncompressed synthetic
development control. This bounds inherited rows, including constant terms; it
does not bound the new measurements admitted by a window. Compression QR only
constructs the triangular factor. Its finite-precision transformations can change
rounding and subsequent rank decisions; evidence reports the thresholds actually
used.

Generator 0.10 replaces full residual-space SVD projection with the thin-SVD and
implicit Householder method above. Column normalization and the rank threshold
formula remain unchanged. Different complement bases can change individual
residual entries and rounding, so regression comparisons check the full profiled
cost, retained shape and numerical rank rather than entrywise equality. Tests
compare against the previous full-basis method for tall, wide and coupled-rank
systems; weak-information, constant-cost and window-control tests still apply.
For `m` rows, `e` eliminated columns and `r` retained columns, storage is
`O(m(e+r+1))`, avoiding the former `m` by `m` left singular-vector matrix when
rows greatly exceed columns. This is a storage-complexity improvement, not a
claim of end-to-end trajectory speedup or improved accuracy.

Generator 0.11 excludes unrelated rows and absent columns from the rank decision
and its dimension-dependent tolerance. Previously, adding 100 all-zero rows to
`J=[[1,1,0],[0,1e-12,1]]`, eliminating its first two columns, changed numerical
rank from two to one and manufactured unit retained information. The new method
decomposes only the involved submatrix and appends unrelated retained observations
and constant residuals unchanged. It never thresholds small nonzero coefficients
to decide row involvement. Regression tests include unrelated measurements,
constant costs, absent eliminated columns and row permutations. Evidence adds
optional `residual_row_count` and `elimination_row_count` fields; the existing
zero-information count identifies excluded columns. This fixes invariance to
exactly unrelated padding, not sensitivity to all possible numerical scalings
or transformations of a nearly singular system.

The trajectory optimizer still forms normal equations for its own updates and
diagnostics. Thus this change protects factor transfer at window boundaries,
not every weak direction in optimization; it does not claim improved trajectory
accuracy, measured runtime, or global gauge consistency.

## Knot factor input and derivative audit

The knot-factor input audit also checks that residual-space assembly and the
native fitter describe the same weighted objective. A mixed development case
combines point, plane and pose observations, unequal weights, a single-knot
information prior, a reordered joint information prior and a rectangular
square-root prior away from its anchors. Tests compare initial and final costs,
the first optimizer normal gradient and diagonal, and central differences of
every knot tangent coordinate. Valid-input calculations agree without changing
weighting conventions.

The audit found input-validation gaps: nonfinite interior knot timestamps could
pass endpoint-only checks, dense assembly accepted extra knot states, and a
single-knot prior beyond the window could raise an indexing error before the
fitter's later validation. Timestamps must now all be finite, both dense builders
require exactly one state per timestamp, and single-knot prior indices are
checked before use. These are input-contract fixes, not new accuracy evidence.

## Marginalization input validation

Marginalization APIs require actual nonnegative integer dimensions or knot
counts, rejecting booleans, floating-point values and oversized requests before
dimension arithmetic. NumPy integer scalars remain supported, with knot counts
converted to Python integers before multiplication to avoid integer overflow.
Normal-equation Schur input symmetry is checked even when no variables are
eliminated. A valid zero-elimination call still returns independent copies,
including empty normal equations. This closes input-validation bypasses; it
does not add a full positive-semidefinite certification of retained normal
blocks or alter the existing numerical rank rules.

## Solver option validation

Fault-injection translations (`corrupt_marginalization_translation_m` and
`bad_overlap_translation_m`) must contain exactly three finite real components.
Provided values are checked before any fit, even if the selected settings would
not use that control. Scalar and single-component values are rejected rather
than broadcast over all three axes. Tests reject malformed shapes, complex and
nonfinite values before invoking the fitter, and check that valid controls shift
only the intended anchors or overlap initial states in both prior modes.

Sliding-window layouts are validated before the batch reference fit. Window
sizes and overlap counts must be actual integers rather than booleans or
floating-point values. Overlap must be positive, every window must advance
beyond it, and the complete layout must cover exactly the knot span. NumPy
integer counts are converted to Python integers before coverage arithmetic,
avoiding fixed-width overflow. Regression tests ensure malformed layouts never
invoke the fitter and verify equivalent Python/NumPy layouts in both normal
Schur and residual-space modes.

Solver options now reject nonfinite numerical settings before optimization.
The native fitter requires a positive integer iteration budget, finite positive
initial damping and convergence tolerances, a damping decrease in `(0, 1]`,
and an increase of at least one. Unit multipliers remain valid for callers that
keep damping fixed. B-spline fitting requires a nonnegative integer budget and
finite positive damping and gradient tolerance; its cost-change and step stall
tolerances may be zero to disable those stop conditions. The existing
zero-iteration B-spline evaluation path is preserved. Boolean or fractional
iteration counts are rejected. Tests cover invalid values and the retained
boundary behaviors; default numerical settings are unchanged.

## Opt-in updates in the observable subspace

Set `ContinuousTimeTrajectoryFitOptions(project_observable_subspace=True)` to
restrict native sparse updates to the measured numerical eigenspace. The synthetic
CLI exposes `calibrex trajectory recover-sliding-window --output result.json
--project-observable-subspace` and records the flag in provenance. The default
remains the existing solve. The optional mode forms a dense diagonally scaled
normal matrix, selects eigenvalues above `60 * machine_epsilon * parameter_count
* max(1, max(abs(eigenvalues)))`, and checks the gradient against the numerical
nullspace with relative tolerance `60 * machine_epsilon * parameter_count`.
A material nullspace gradient raises an error rather than being silently dropped.

The existing damping matrix is projected into the observed subspace. Its solve
therefore cannot create a step in a local null direction, even if unequal damping
would couple those directions in the unrestricted solve. Full-rank models retain
the direct damped solve, and exactly zero-information coordinates receive zero
increments. No observation or prior is added. The active eigenspace is recomputed
at each iteration; the objective acceptance and stationarity checks are unchanged.

`projected_step_diagnostics` records rank selection for each attempted update,
including candidates that may subsequently be rejected. Sliding artifacts expose
these records under each window's `projected_steps`. Generator 0.7 accepts the
same opt-in mode and records it explicitly; old reports remain readable. Tests
verify absence of null-direction motion, descent, coordinate-unit covariance
when both the normal and damping matrices are transformed consistently,
full-rank equivalence, inconsistent-gradient rejection, and a midpoint-only
coupled-gauge trajectory across two windows. Default fits produce no projected
step records.

This is an experimental local solver mode with dense decomposition cost, not a
runtime improvement or a global gauge-fixing method. Orthogonality is defined in
the diagonally scaled tangent coordinates. The existing block damping convention
is retained; this is not a claim of full-solver invariance to arbitrary units.
Changing linearization points can
change the local nullspace; nonlinear drift and global observability still need
separate assessment. Small positive modes below the numerical threshold may be
excluded only when the gradient passes the recorded range check.

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
