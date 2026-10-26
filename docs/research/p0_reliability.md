# P0: stationarity and uncertainty coverage

This change separates optimizer stationarity from calibration quality, and
measures whether formal uncertainty contains known truth across independent
synthetic recordings. These are development results, not a SOTA claim or a
real-sensor accuracy guarantee.

## Optimizer termination

Both native screw-linear and cubic B-spline fits now return `stalled` when a
small damped update, small cost change (B-spline), or exhausted damping fails
to establish stationarity. `max_iterations` means the iteration budget was
consumed. Singular solves remain `singular_system`. Only a final-state
normalized gradient below the configured threshold establishes `converged`.
The B-spline path previously called an entirely rejected line search converged.

The dimensionless gradient is
`max_j |(J.T r)_j| / (||J[:,j]|| * max(1, ||r||))`, with whitened residuals and
Jacobians. The step stagnation trigger is
`max_j |delta_j| ||J[:,j]|| / max(1, ||r||)`.
Both are invariant to independent linear changes of parameter units. Zero
Jacobian columns add no information and contribute zero; stationarity does not
imply observability, a unique solution, or correct calibration. The existing
numeric gradient threshold (1e-8) is now dimensionless in the screw fitter;
B-spline has an explicit `gradient_tolerance` with the same default. The raw
`gradient_norm` field retains its original units and meaning.

Optional `fit_diagnostics` fields retain the termination reason, final normalized
gradient, threshold and stationarity flag in direct fit and recovery artifacts,
including each bag map round. Historical artifacts remain readable without
these fields. `stalled` extends the status enum: consumers reading new results
must update their schemas. Sliding-window aggregates accept the new status but
do not pretend their copied batch statistics are per-window diagnostics.
A final nonconverged bag fit adds a warning. Existing synthetic holdout/error
quality gates remain separate: an acceptable calibration can be nonstationary,
and a stationary calibration can fail the quality policy.

The step itself, residual model, holdouts, signed controls, iteration budgets,
and covariance estimator are unchanged. Gradient stopping may now occur at a
different iteration. This is not a global convergence proof: screw-clock knot
nonsmoothness, feasible clock-domain stepping and marginalized gradient/cross-knot
information remain separate work.

## Independent uncertainty experiments

Run from an installed checkout, using a new output directory each time:

```bash
OPENBLAS_NUM_THREADS=1 python tools/evaluate_p0_uncertainty.py \
  --output-dir work/p0-known-planes
OPENBLAS_NUM_THREADS=1 python tools/evaluate_p0_uncertainty.py \
  --producer synthetic_bag --seeds 42001 42002 42003 42004 42005 \
  --duration-s 8 --map-rounds 4 --output-dir work/p0-bags
```

`calibrex validate <output-dir>/coverage.json` checks the registered report schema.

The runner writes the seed list, settings, source digests and command before
fitting. Every trial is retained, including failed quality gates, stalled fits,
and exceptions. The known-plane cohort uses seeds 41001–41020. The bag cohort
recreates the entire recording and reruns initialization, odometry, plane-map
construction, associations and fitting for each independent noise draw. Raw
synthetic bags stay outside the repository; specs and input digests remain in
the retained records.

For each calibration axis, the report evaluates the marginal Gaussian
`estimate ± 1.95996 * std` interval against injected truth. Rotation errors use
the same right tangent coordinates as the covariance. It reports the error
bias/RMS, mean std, conditional coverage, all-attempt coverage (missing intervals
count as undelivered), and a 95% Wilson interval for conditional coverage.
Only an upper Wilson bound below nominal is labelled `undercoverage`; all other
results remain `inconclusive`. Shrinking std to 0.1 times its value provides an
explicit overconfidence control. Intervals are never inflated using these draws.

The same motion and truth are shared across independent sensor-noise draws;
this is not a population of independent rigs or motions. Five bag runs are a
pilot, not a general coverage estimate. Per-axis Wilson intervals are not a
simultaneous multiple-comparison guarantee. Formal intervals from stalled fits
remain visible, not silently discarded or certified.

## Gravity-frame correction (follow-up)

The synthetic-bag gravity rows below compare different world frames and are
not valid undercoverage evidence. See [the empirical-interval follow-up](empirical_intervals.md)
for the correction. Historical artifacts are retained unchanged; sensor-relative
rotation, lever arm, IMU bias and clock-offset comparisons remain applicable.

## Retained outcomes

See [the evidence archive](../assets/p0_reliability/). It includes all 25 final
results, protocols, schema-valid coverage reports, individual trial records,
package versions, and SHA-256 manifests. `measured_source/` contains exact source
bytes differing from base commit `7ae0905d`; all other source hashes are recorded
in each protocol. The final measurements include CLI schema registration and final-model verification.

An earlier attempt used raw-unit step stagnation thresholds. Its complete
20+5 outcomes remain in `attempt1/`; final runs use normalized steps. These are
repeated development data, not independent confirmation. `attempt2/` retains the next 20+5 runs before the final backend-consistency
review. Final results are identical to attempt2 apart from provenance. The final
result is not selected by coverage score.

The known-plane cohort has 18–20 of 20 intervals covering truth per axis;
its Wilson intervals do not establish undercoverage. This does **not** validate
95% coverage. There are 14 quality-policy passes and 6 failures, all retained.

The full-bag cohort exposes overconfidence: rotation x and z, lever-arm y,
accelerometer-bias z cover truth in **0/5** cases. The historical gravity rows
also computed 0/5, but that comparison is invalid because the world gauges differ.
The 95% Wilson upper bound for 0/5 is approximately 0.434, well below 0.95.
Clock offset covers truth in 2/5. This supports the existing warning that
fixed-map covariance misses front-end/model error. No blanket covariance
correction is inferred from five runs. All five bag policies are `warn`.

The next research step is empirical full-pipeline uncertainty across varied
motions and longer recordings, with interval construction fitted on separate
development data and coverage measured on fresh, frozen evaluation draws.
