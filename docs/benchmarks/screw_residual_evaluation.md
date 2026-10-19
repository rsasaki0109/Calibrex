# Sparse screw trajectory acceleration

The native `screw_linear` fitter now evaluates point, plane and pose-anchor
residuals without building the derivatives that objective scoring discards.
Linearization and marginal covariance still use every analytic Jacobian.
Both paths share the original residual arithmetic. The SE(3) left Jacobian
uses explicit FP64 three-vector cross products in place of general NumPy
broadcast dispatch, preserving the multiply/subtract order.

This applies to `fit_continuous_trajectory`, artifact-backed trajectory fits,
and `calibrex trajectory recover-lidar-imu-joint --trajectory screw_linear`.
Public signatures, schemas, iteration limits, damping, reductions, sparse
solve, covariance, IMU factors and quality controls are unchanged. It adds
no dependency, approximate arithmetic, JIT compilation or result cache.
The separate cubic B-spline fitter is outside this timing claim.

## Complete comparisons

The baseline is the maintained implementation at
`5a80e7470487ceab148f7d49b560f48c1158665e`; the candidate source is
`38b98d03be01919abb5d7a12cab5c939001eef95`. Both full source trees and the
measurement tools were fixed before timing. Each case runs six fresh workers
in baseline/candidate, candidate/baseline, baseline/candidate order, with
one BLAS thread, no warmups and an unreserved host. The predeclared gate is
2x on the median of the three paired complete API ratios for each case.

The **synthetic joint recovery** case uses seeds 20261010, 20261011 and
20261012, one pair per seed. Each call includes the full sparse trajectory
and six calibration blocks, covariance, interleaved holdout, separability
and all 32 signed controls. The recorder retains references and brackets the
fit; input hashing and serialization take place after the API timer ends.

The **public geometry fit** case uses eight already-spent registration
captures from AIST GLIM Livox Avia. It selects 128 evenly spaced source
indices per capture, associates nearest target points within 1.5 m, and
retains 10-neighbour PCA planes with planarity at most 0.2. This produces
1,013 point-to-plane factors. Assigned timestamps span four seconds and
five initially perturbed knots; all fits keep the 30-iteration limit.
Associations and geometry are fixed before timing. These assigned times
and capture-local planes are a derived performance fixture, not a physical
trajectory reconstruction or an end-to-end bag/IMU calibration.

| Scope | Baseline median | Candidate median | Median paired speedup |
| --- | ---: | ---: | ---: |
| Synthetic joint recovery API | 88.606 s | 37.206 s | 2.367x |
| Synthetic joint fresh worker | 93.902 s | 42.527 s | 2.204x |
| Public geometry complete fit API | 93.289 s | 28.347 s | 3.143x |
| Public geometry fresh worker | 99.343 s | 33.465 s | 2.879x |

The ratio column is the median of the three pair ratios, rather than the
quotient of the separately reported timing medians. Joint API pair ratios
range from 2.342x to 2.381x; public geometry ranges from 3.029x to 4.048x on
this unreserved host. Both predeclared complete-API gates pass. All six
public geometry fits converge in 15 iterations with identical full results.

The full saved result, full fitted knot/calibration/covariance result,
complete input and options must agree exactly within every pair, excluding
provenance only. Exact equality includes stopping decisions, iteration
counts, holdout, signed controls and parameter uncertainty. The archive
retains all twelve complete observations and their logs, imported-source
digests, both 401-module source trees and the fixed tools.

API time excludes imports, source verification, input preparation and JSON
serialization. Fresh-worker time includes these tasks, process startup and
exit. Parent-side construction of the fixed geometry fixture and source
capsules is outside both clocks. The two cases are reported separately;
their speedups are not multiplied or extrapolated to bag decoding, map
association or the cubic B-spline pipeline.

The original Windows seed-20261010 case reports `max_iterations` after 25
iterations while its recovery policy passes. Its pre-existing convergence
assertion remains unchanged. Seed 20261011 converges, but **both variants
fail the same existing holdout gyro gate**: about 0.000357 rad against
1.5 times the truth-calibration 0.000236 rad. All its parameter budgets
and signed controls pass. Seed 20261012 converges and passes policy.
The complete results retain these outcomes. Timing does not establish new calibration
accuracy or change any SOTA verdict.

## Reproduce and audit

To collect another cohort, supply the eight original `registration-00.npz`
through `registration-07.npz` captures and their `.json` hash receipts.
Each capture contains FP64 `source` and `target` arrays with three columns.
Inputs are read in place; the tool neither fetches nor duplicates the bag.
Commit runtime and tool changes before measuring and use a new output path:

```bash
python tools/benchmark_screw_residual_evaluation.py \
  --geometry-folder outputs/odometry-search-profile-v3 \
  --output outputs/screw-residual-comparison
```

Audit the complete published cohort without running calibration or archived
Python code. It needs the ordinary validation dependencies, including
`jsonschema`, but no GPU or original bag:

```bash
python tools/audit_screw_residual_archive.py \
  docs/assets/screw_residual_evaluation/complete-evidence.tar.gz
```

See the [archive inventory and digests](../assets/screw_residual_evaluation/README.md).
The audit verifies full source capsules, actual imported-module bindings,
every result/input/measurement/log digest, complete artifact schemas, pair
equality and both clocks. It recomputes medians and gate decisions rather
than trusting the supplied summary. Git canonical-text verification also
runs during collection; archived sources use LF text, independent of the
checkout's line endings.
